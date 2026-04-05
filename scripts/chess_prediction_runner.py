#!/usr/bin/env python3
"""
Chess Prediction Runner — continuous ply-by-ply prediction loop.

Trains prediction models on the processed PGN dataset, then replays test
games one ply at a time, predicting each move before it is revealed:

  1. Initial training phase (configurable iterations).
  2. Prediction loop: pick a test game, step ply by ply.
     - Predict top-N next moves using motif + context + factorised + beam models.
     - Predict the game outcome (White win / Draw / Black win) from the prefix.
     - Write logs/chess_live_board.json consumed by the live_viz_server.
     - Pause `--ply-delay` seconds so the viz can animate.
     - Record correctness; do an online Hebbian-style weight boost on the move seen.
  3. After each completed game, run a mini re-training step on the game sequence.
  4. Print accuracy at every game boundary; write hourly checkpoint summaries.
  5. Auto-adjust learning rate if accuracy stagnates across an hourly window.

Usage:
  python scripts/chess_prediction_runner.py
  python scripts/chess_prediction_runner.py --train-iters 5 --ply-delay 0.6
  python scripts/chess_prediction_runner.py --max-games 4000 --export-priors logs/chess_ml_priors.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Import shared model utilities from the training loop ─────────────────────
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

try:
    import chess
except ImportError:
    raise SystemExit("python-chess required.  pip install python-chess")

# Thread cap before numpy import
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_k, str(min(os.cpu_count() or 1, 16)))

import numpy as np

from chess_training_loop import (
    BEAM_STEPS,
    BEAM_WIDTH,
    BOARD_FEATURE_DIM,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MOVE_HORIZONS,
    DEFAULT_OUTCOME_SCOPES,
    FACTOR_BLEND,
    FEATURE_DIM,
    LABEL_TO_RESULT,
    METADATA_DIM,
    PRIOR_BLEND,
    PRIOR_TOPK_MOTIFS,
    RESULT_TO_LABEL,
    TOTAL_FEATURE_DIM,
    GameRecord,
    OutcomeNetwork,
    ResidualRegressor,
    ReinforcementLedger,
    anneal_future_predictions,
    board_symbol_states,
    board_to_vector,
    build_move_datasets,
    build_outcome_features,
    compose_prefix_vector,
    factor_from_move,
    harvest_surprises,
    init_surprise_buffers,
    load_games,
    metadata_tokens,
    metadata_vector,
    motif_hash,
    sample_buffer,
    stable_bucket,
    train_move_models,
    update_residual_models,
    zone_bin_from_square,
)

ROOT = SCRIPTS.parent
BOARD_JSON = ROOT / "logs" / "chess_live_board.json"
METRICS_LOG = ROOT / "logs" / "chess_training_metrics.log"
HOURLY_LOG = ROOT / "logs" / "chess_hourly_checkpoints.jsonl"

# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_next_moves(
    board: chess.Board,
    last_move_hashes: List[int],
    context_window: int,
    move_counters: Dict,
    motif_counters: Dict,
    motif_global: Dict,
    factor_counters: Dict,
    factor_global: Dict,
    cluster_context_counters: Dict,
    cluster_motif_global: Dict,
    move_factor_lookup: Dict,
    anchor_counters: Dict,
    horizon: int = 1,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top_k predicted moves with probabilities for the current position."""
    context = tuple(last_move_hashes[-context_window:]) if last_move_hashes else ()

    # Motif for current position
    symbol_states = board_symbol_states(board)
    motif_prev = motif_hash(symbol_states)

    scores: Counter = Counter()
    candidates: set = set()

    ctx_counter = move_counters.get(horizon, {})
    motif_map = (motif_counters or {}).get(horizon, {})
    factor_map = (factor_counters or {}).get(horizon, {})

    # Motif-keyed priors
    if motif_prev:
        motif_counts = motif_map.get(context, {}).get(motif_prev)
        if motif_counts:
            for mv, c in motif_counts.items():
                scores[mv] += c * (1.0 + PRIOR_BLEND)
                candidates.add(mv)
        if motif_global:
            mg = motif_global.get(motif_prev)
            if mg:
                for mv, c in mg.items():
                    scores[mv] += c * (0.5 * PRIOR_BLEND)
                    candidates.add(mv)
        # Cluster priors
        from chess_training_loop import PRIOR_CLUSTERS
        cluster_id = stable_bucket(motif_prev, PRIOR_CLUSTERS)
        if cluster_motif_global:
            cm = cluster_motif_global.get(cluster_id, {}).get(motif_prev)
            if cm:
                for mv, c in cm.items():
                    scores[mv] += c * (0.5 * PRIOR_BLEND)
                    candidates.add(mv)
        if cluster_context_counters:
            ctx_cc = cluster_context_counters.get(horizon, {}).get(cluster_id, {}).get(context)
            if ctx_cc:
                for mv, c in ctx_cc.most_common(3):
                    scores[mv] += c
                    candidates.add(mv)
        if anchor_counters:
            ac = anchor_counters.get(horizon, {}).get(motif_prev)
            if ac:
                for mv, c in ac.most_common(3):
                    scores[mv] += c * PRIOR_BLEND
                    candidates.add(mv)

    # Context-frequency prior
    ctx_counts = ctx_counter.get(context)
    if ctx_counts:
        if not isinstance(ctx_counts, Counter):
            ctx_counts = Counter(ctx_counts)
        for mv, c in ctx_counts.most_common(5):
            scores[mv] += c
            candidates.add(mv)

    # Factorised move-attribute priors
    factor_counts_ctx = factor_map.get(context, {}) if factor_map else {}
    if move_factor_lookup and candidates:
        for mv in list(candidates):
            mf = move_factor_lookup.get(mv)
            if not mf:
                continue
            mv_score = 0.0
            for name, val in {
                "role": mf.role, "side": mf.side,
                "start": mf.start_bin, "end": mf.end_bin,
                "capture": mf.capture, "promo": mf.promo,
            }.items():
                ctx_fc = factor_counts_ctx.get(name)
                if ctx_fc:
                    mv_score += math.log1p(ctx_fc.get(val, 0))
                if factor_global:
                    mv_score += 0.5 * math.log1p(factor_global.get(name, {}).get(val, 0))
            if mv_score > 0:
                scores[mv] += mv_score * FACTOR_BLEND

    # Beam lookahead re-rank for longer horizons
    if scores and horizon > 1 and move_counters:
        base_h = 1 if 1 in move_counters else min(move_counters.keys())
        base_ctrs = move_counters.get(base_h, {})
        leaf: Counter = Counter()
        beam = [(list(context), 0.0)]
        for _ in range(min(BEAM_STEPS, horizon)):
            new_beam = []
            for seq, sc in beam:
                tail = tuple(seq[-context_window:])
                cc = base_ctrs.get(tail)
                if not cc:
                    continue
                for mv, c in cc.most_common(BEAM_WIDTH):
                    new_beam.append((seq + [stable_bucket(mv, FEATURE_DIM)], sc + math.log1p(c)))
            if not new_beam:
                break
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:BEAM_WIDTH]
        for seq, sc in beam:
            tail = tuple(seq[-context_window:])
            cc = base_ctrs.get(tail)
            if cc:
                best = cc.most_common(1)
                if best:
                    leaf[best[0][0]] += sc
        for mv, sc in leaf.items():
            scores[mv] += sc * 0.5

    if not scores:
        # Uniform fallback over legal moves
        legal = [board.san(m) for m in list(board.legal_moves)[:8]]
        for mv in legal:
            scores[mv] = 1.0

    total = sum(scores.values()) or 1.0
    results = []
    for move_san, raw in scores.most_common(top_k):
        prob = raw / total
        try:
            move = board.parse_san(move_san)
            ff = chess.square_file(move.from_square)
            fr = chess.square_rank(move.from_square)
            tf = chess.square_file(move.to_square)
            tr = chess.square_rank(move.to_square)
        except Exception:
            ff = fr = tf = tr = -1
        results.append({
            "san": move_san,
            "probability": round(prob, 4),
            "from_file": ff,
            "from_rank": fr,
            "to_file": tf,
            "to_rank": tr,
        })
    return results


def compute_square_heat(predictions: List[Dict], board: chess.Board) -> List[List[float]]:
    """8x8 float grid: how much prediction mass lands on each square."""
    heat = [[0.0] * 8 for _ in range(8)]
    for pred in predictions:
        prob = pred["probability"]
        tf, tr = pred.get("to_file", -1), pred.get("to_rank", -1)
        ff, fr = pred.get("from_file", -1), pred.get("from_rank", -1)
        if 0 <= tf <= 7 and 0 <= tr <= 7:
            heat[tr][tf] += prob
        if 0 <= ff <= 7 and 0 <= fr <= 7:
            heat[fr][ff] += prob * 0.4  # from squares lighter
    mx = max(max(row) for row in heat)
    if mx > 0:
        heat = [[v / mx for v in row] for row in heat]
    return heat


def board_pieces(board: chess.Board) -> List[Dict[str, Any]]:
    """Current piece list formatted for the viz server."""
    pieces = []
    for square, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        pieces.append({
            "id": f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}",
            "file": chess.square_file(square),
            "rank": chess.square_rank(square),
            "color": color,
            "piece": piece.symbol().upper(),
        })
    return pieces


def outcome_probs(board: chess.Board, model: OutcomeNetwork, hashed_moves: np.ndarray,
                   scope: int, meta_vec: np.ndarray) -> Dict[str, float]:
    length = min(scope, hashed_moves.shape[0])
    if length == 0:
        return {k: 1 / 3 for k in LABEL_TO_RESULT.values()}
    counts = np.bincount(hashed_moves[:length].astype(np.int64), minlength=FEATURE_DIM).astype(np.float32)
    bv = board_to_vector(board)
    vec = compose_prefix_vector(counts, length, meta_vec, bv)
    logits, _, _, _ = model._forward(vec[None, :])
    logits = logits.flatten()
    logits -= logits.max()
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    label_map = {v: k for k, v in RESULT_TO_LABEL.items()}
    return {label_map[i]: round(float(p), 4) for i, p in enumerate(probs)}


def write_board_json(
    data: Dict[str, Any],
    path: Path = BOARD_JSON,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


# ── Online reinforcement: boost move counter weights after seeing actual move ──

def online_boost(
    move_san: str,
    context: tuple,
    move_counters: Dict,
    boost: int = 3,
    horizon: int = 1,
) -> None:
    """Increment the observed move's count to steer future predictions."""
    ctr = move_counters.get(horizon)
    if ctr is None:
        return
    ctx_ctr = ctr.get(context)
    if ctx_ctr is None:
        ctr[context] = Counter({move_san: boost})
    else:
        ctx_ctr[move_san] += boost


# ── Accuracy tracking ─────────────────────────────────────────────────────────

@dataclass
class AccumulatedAccuracy:
    total: int = 0
    correct_top1: int = 0
    correct_top3: int = 0
    outcome_total: int = 0
    outcome_correct: int = 0

    def top1(self) -> float:
        return self.correct_top1 / self.total if self.total else 0.0

    def top3(self) -> float:
        return self.correct_top3 / self.total if self.total else 0.0

    def outcome_acc(self) -> float:
        return self.outcome_correct / self.outcome_total if self.outcome_total else 0.0

    def summary(self) -> str:
        return (
            f"move top-1={self.top1():.1%}  top-3={self.top3():.1%}  "
            f"outcome={self.outcome_acc():.1%}  "
            f"({self.total} plies, {self.outcome_total} games)"
        )


def check_hourly(
    last_check: float,
    acc_this_hour: AccumulatedAccuracy,
    acc_prev_hour: AccumulatedAccuracy,
    lr_boost: List[float],
    hourly_checkpoints: List[Dict],
    log_handle,
) -> Tuple[float, AccumulatedAccuracy, AccumulatedAccuracy]:
    """If an hour has passed, log checkpoint and auto-adjust if stagnating."""
    now = time.time()
    if now - last_check < 3600:
        return last_check, acc_this_hour, acc_prev_hour

    hour_num = len(hourly_checkpoints) + 1
    delta_top1 = acc_this_hour.top1() - acc_prev_hour.top1()
    delta_outcome = acc_this_hour.outcome_acc() - acc_prev_hour.outcome_acc()

    # Auto-adjust: if move accuracy hasn't improved by at least 0.2%, boost LR
    stagnating = delta_top1 < 0.002 and acc_this_hour.total > 200
    if stagnating:
        lr_boost[0] = min(lr_boost[0] * 1.25, 4.0)
        adj_msg = f"  → LR boost → {lr_boost[0]:.2f}x (stagnation detected)"
    else:
        lr_boost[0] = max(lr_boost[0] * 0.97, 1.0)
        adj_msg = ""

    record = {
        "hour": hour_num,
        "timestamp": now,
        "move_top1": round(acc_this_hour.top1(), 4),
        "move_top3": round(acc_this_hour.top3(), 4),
        "outcome_acc": round(acc_this_hour.outcome_acc(), 4),
        "delta_top1": round(delta_top1, 4),
        "delta_outcome": round(delta_outcome, 4),
        "plies_this_hour": acc_this_hour.total,
        "lr_boost": round(lr_boost[0], 3),
        "stagnating": stagnating,
    }
    hourly_checkpoints.append(record)
    HOURLY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HOURLY_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(
        f"\n{'=' * 60}\n"
        f"  HOURLY CHECKPOINT #{hour_num}\n"
        f"  Move top-1:  {acc_this_hour.top1():.1%}  (Δ {delta_top1:+.1%})\n"
        f"  Move top-3:  {acc_this_hour.top3():.1%}\n"
        f"  Outcome:     {acc_this_hour.outcome_acc():.1%}  (Δ {delta_outcome:+.1%})\n"
        f"  Plies seen:  {acc_this_hour.total:,}\n"
        + adj_msg +
        f"\n{'=' * 60}\n"
    )

    if log_handle:
        log_handle.write(json.dumps(record) + "\n")
        log_handle.flush()

    return now, AccumulatedAccuracy(), acc_this_hour


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-games", type=int, default=8000)
    ap.add_argument("--train-iters", type=int, default=3,
                    help="Initial training iterations before entering prediction loop")
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ply-delay", type=float, default=1.2,
                    help="Seconds between plies in the visualization (0 = as fast as possible)")
    ap.add_argument("--ply-delay-min", type=float, default=0.2,
                    help="Minimum ply delay after accuracy improves")
    ap.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    ap.add_argument("--outcome-scope", type=int, default=10,
                    help="Which outcome scope model to use for probability display")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--export-priors", type=Path, default=None,
                    help="Write ML priors JSON for the Rust annealer here")
    ap.add_argument("--online-boost", type=int, default=2,
                    help="Count increment applied to observed moves (online learning)")
    ap.add_argument("--log-file", type=Path, default=METRICS_LOG)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("Loading games …")
    all_games = load_games(args.max_games)
    train_games = [g for g in all_games if g.is_train]
    test_games = [g for g in all_games if not g.is_train]
    print(f"  {len(train_games)} train / {len(test_games)} test games")

    outcome_scopes = DEFAULT_OUTCOME_SCOPES
    scope = args.outcome_scope if args.outcome_scope in outcome_scopes else outcome_scopes[0]

    print("Building feature datasets …")
    outcome_data = build_outcome_features(all_games, outcome_scopes, multi_frame_windows=3)
    move_datasets, move_defaults, context_lookup, move_factor_lookup, anchor_counters = \
        build_move_datasets(all_games, DEFAULT_MOVE_HORIZONS, args.context_window, 2)
    (
        move_models, move_counters, move_motif_counters, move_motif_global,
        factor_counters, factor_global, cluster_context_counters, cluster_motif_global,
        factor_lookup, anchor_counters,
    ) = train_move_models(move_datasets, move_defaults, move_factor_lookup, anchor_counters,
                          prior_topk=PRIOR_TOPK_MOTIFS)

    outcome_models = {
        s: OutcomeNetwork(TOTAL_FEATURE_DIM, len(RESULT_TO_LABEL), args.hidden_dim)
        for s in outcome_scopes
    }
    residual_models = {s: ResidualRegressor(TOTAL_FEATURE_DIM) for s in outcome_scopes}
    surprise_buffers = init_surprise_buffers(outcome_scopes, 2048)
    ledger = ReinforcementLedger(outcome_scopes)

    # ── Initial training phase ────────────────────────────────────────────────
    print(f"\nInitial training ({args.train_iters} iterations) …")
    for it in range(1, args.train_iters + 1):
        lr_it = args.lr / (1.0 + 0.05 * (it - 1))
        for s in outcome_scopes:
            outcome_models[s].train_epoch(
                outcome_data[s]["train_X"], outcome_data[s]["train_y"], lr_it, args.batch_size
            )
        harvest_surprises(rng, outcome_models, outcome_data, surprise_buffers, 256)
        for s, buf in surprise_buffers.items():
            batch = sample_buffer(rng, buf, 256)
            if batch:
                bX, by = batch
                outcome_models[s].train_epoch(bX, by, lr_it * 1.25, min(args.batch_size, bX.shape[0]))
        update_residual_models(rng, outcome_models, residual_models, outcome_data, 256, args.lr * 0.3)

        acc_scope = outcome_models[scope].accuracy(outcome_data[scope]["test_X"], outcome_data[scope]["test_y"])
        print(f"  iter {it}/{args.train_iters}  outcome@{scope}={acc_scope:.1%}")

    print("\n─── Entering prediction loop ───\n")

    acc_global = AccumulatedAccuracy()
    acc_this_hour = AccumulatedAccuracy()
    acc_prev_hour = AccumulatedAccuracy()
    hourly_checkpoints: List[Dict] = []
    last_hour_check = time.time()
    lr_boost = [1.0]  # mutable scalar in a list so inner closure can modify
    game_count = 0

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.log_file.open("a", encoding="utf-8")

    # Shuffle test games for variety; cycle forever
    shuffled_test = list(test_games)
    rng.shuffle(shuffled_test)
    game_iter = iter(shuffled_test)

    ply_delay = args.ply_delay

    while True:
        # Get next test game (cycle)
        try:
            game = next(game_iter)
        except StopIteration:
            rng.shuffle(shuffled_test)
            game_iter = iter(shuffled_test)
            game = next(game_iter)

        game_count += 1
        board = chess.Board()
        move_hashes: List[int] = []
        hashed_moves = game.hashed_moves
        game_correct_top1 = 0
        game_correct_top3 = 0
        game_total = 0

        for ply_idx, move_san in enumerate(game.moves):
            # ── Predict ──────────────────────────────────────────────────────
            preds = predict_next_moves(
                board, move_hashes, args.context_window,
                move_counters, move_motif_counters, move_motif_global,
                factor_counters, factor_global,
                cluster_context_counters, cluster_motif_global,
                factor_lookup, anchor_counters,
                horizon=1, top_k=5,
            )
            heat = compute_square_heat(preds, board)
            pieces = board_pieces(board)

            # Outcome probabilities from the neural model
            opr = outcome_probs(board, outcome_models[scope], hashed_moves, scope, game.meta_vector)
            pred_outcome_label = max(opr, key=opr.get)

            # ── Mark actual move on predictions ───────────────────────────────
            pred_sans = [p["san"] for p in preds]
            correct_top1 = len(preds) > 0 and pred_sans[0] == move_san
            correct_top3 = move_san in pred_sans[:3]

            # Mark last-move squares
            try:
                actual_move_obj = board.parse_san(move_san)
                lm_ff, lm_fr = chess.square_file(actual_move_obj.from_square), chess.square_rank(actual_move_obj.from_square)
                lm_tf, lm_tr = chess.square_file(actual_move_obj.to_square), chess.square_rank(actual_move_obj.to_square)
                last_move_info = {"from_file": lm_ff, "from_rank": lm_fr, "to_file": lm_tf, "to_rank": lm_tr, "san": move_san}
            except Exception:
                last_move_info = {"san": move_san}

            # Mark which prediction was correct
            for p in preds:
                p["correct"] = p["san"] == move_san

            write_board_json({
                "game_count": game_count,
                "game_id": game.game_id,
                "ply": ply_idx,
                "total_plies": len(game.moves),
                "move_number": (ply_idx // 2) + 1,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "pieces": pieces,
                "last_move": None,           # will be filled after move
                "predictions": preds,
                "actual_move": last_move_info,
                "prediction_correct_top1": correct_top1,
                "prediction_correct_top3": correct_top3,
                "square_heat": heat,
                "outcome_probs": opr,
                "predicted_outcome": pred_outcome_label,
                "actual_outcome": game.result_label,
                "running": {
                    "move_top1": round(acc_global.top1(), 4),
                    "move_top3": round(acc_global.top3(), 4),
                    "outcome_acc": round(acc_global.outcome_acc(), 4),
                    "total_plies": acc_global.total,
                    "game_count": game_count,
                },
                "hourly_checkpoints": hourly_checkpoints[-10:],
                "reveal_phase": False,
            })

            # Brief pause so viz can show prediction
            if ply_delay > 0:
                time.sleep(ply_delay * 0.5)

            # ── Apply move, write reveal ──────────────────────────────────────
            board.push(actual_move_obj)
            new_pieces = board_pieces(board)

            write_board_json({
                "game_count": game_count,
                "game_id": game.game_id,
                "ply": ply_idx,
                "total_plies": len(game.moves),
                "move_number": (ply_idx // 2) + 1,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "pieces": new_pieces,
                "last_move": last_move_info,
                "predictions": preds,
                "actual_move": last_move_info,
                "prediction_correct_top1": correct_top1,
                "prediction_correct_top3": correct_top3,
                "square_heat": heat,
                "outcome_probs": opr,
                "predicted_outcome": pred_outcome_label,
                "actual_outcome": game.result_label,
                "running": {
                    "move_top1": round(acc_global.top1(), 4),
                    "move_top3": round(acc_global.top3(), 4),
                    "outcome_acc": round(acc_global.outcome_acc(), 4),
                    "total_plies": acc_global.total,
                    "game_count": game_count,
                },
                "hourly_checkpoints": hourly_checkpoints[-10:],
                "reveal_phase": True,
            })

            # ── Online reinforcement ──────────────────────────────────────────
            context_key = tuple(move_hashes[-args.context_window:]) if move_hashes else ()
            online_boost(move_san, context_key, move_counters, boost=args.online_boost)

            # Update hashes
            move_hashes.append(stable_bucket(move_san, FEATURE_DIM))

            # Accuracy tallying
            game_correct_top1 += int(correct_top1)
            game_correct_top3 += int(correct_top3)
            game_total += 1
            acc_global.total += 1
            acc_global.correct_top1 += int(correct_top1)
            acc_global.correct_top3 += int(correct_top3)
            acc_this_hour.total += 1
            acc_this_hour.correct_top1 += int(correct_top1)
            acc_this_hour.correct_top3 += int(correct_top3)

            if ply_delay > 0:
                time.sleep(ply_delay * 0.5)

        # ── End of game ───────────────────────────────────────────────────────
        label_map = {v: k for k, v in RESULT_TO_LABEL.items()}
        actual_result_str = label_map.get(game.result_label, "?")
        outcome_match = pred_outcome_label == actual_result_str
        acc_global.outcome_total += 1
        acc_global.outcome_correct += int(outcome_match)
        acc_this_hour.outcome_total += 1
        acc_this_hour.outcome_correct += int(outcome_match)

        game_top1 = game_correct_top1 / game_total if game_total else 0.0
        print(
            f"  game {game_count} [{game.game_id}]  "
            f"plies={game_total}  "
            f"move-top1={game_top1:.1%}  "
            f"outcome={'✓' if outcome_match else '✗'}  "
            f"│ global {acc_global.summary()}"
        )

        # Log to file
        log_record = {
            "game_count": game_count,
            "game_id": game.game_id,
            "timestamp": time.time(),
            "plies": game_total,
            "game_move_top1": round(game_top1, 4),
            "game_move_top3": round(game_correct_top3 / game_total if game_total else 0, 4),
            "outcome_correct": outcome_match,
            "global_move_top1": round(acc_global.top1(), 4),
            "global_move_top3": round(acc_global.top3(), 4),
            "global_outcome_acc": round(acc_global.outcome_acc(), 4),
            "lr_boost": lr_boost[0],
        }
        log_handle.write(json.dumps(log_record) + "\n")
        log_handle.flush()

        # ── Export ML priors for Rust annealer ───────────────────────────────
        if args.export_priors and game_count % 5 == 0:
            _export_priors(args.export_priors, outcome_models, outcome_scopes, game, rng)

        # ── Mini retraining on completed game ────────────────────────────────
        # Wrap the game into a tiny dataset and run one gradient step
        if len(game.hashed_moves) >= min(outcome_scopes):
            lr_now = args.lr * lr_boost[0] / (1.0 + 0.01 * game_count)
            for s in outcome_scopes:
                if game.hashed_moves.shape[0] < s:
                    continue
                bv = game.board_vectors[min(s, len(game.board_vectors)) - 1]
                counts = np.bincount(game.hashed_moves[:s].astype(np.int64), minlength=FEATURE_DIM).astype(np.float32)
                vec = compose_prefix_vector(counts, s, game.meta_vector, bv)
                outcome_models[s].train_epoch(
                    vec[None, :],
                    np.array([game.result_label], dtype=np.int64),
                    lr_now,
                    1,
                )

        # ── Hourly checkpoint ─────────────────────────────────────────────────
        last_hour_check, acc_this_hour, acc_prev_hour = check_hourly(
            last_hour_check, acc_this_hour, acc_prev_hour,
            lr_boost, hourly_checkpoints, log_handle,
        )

        # Adapt ply delay based on global accuracy (faster when better)
        current_acc = acc_global.top1()
        ply_delay = max(args.ply_delay_min, args.ply_delay * (1.0 - current_acc * 0.6))

    log_handle.close()


def _export_priors(
    path: Path,
    outcome_models: Dict,
    scopes: List[int],
    sample_game: GameRecord,
    rng: random.Random,
) -> None:
    """Write ML prior snapshot for the Rust annealer's w_ml_prior term."""
    scoped = []
    for s in scopes:
        if sample_game.hashed_moves.shape[0] < s:
            continue
        bv = sample_game.board_vectors[min(s, len(sample_game.board_vectors)) - 1]
        counts = np.bincount(sample_game.hashed_moves[:s].astype(np.int64), minlength=FEATURE_DIM).astype(np.float32)
        vec = compose_prefix_vector(counts, s, sample_game.meta_vector, bv)
        logits, _, _, _ = outcome_models[s]._forward(vec[None, :])
        logits = logits.flatten()
        logits -= logits.max()
        exp_l = np.exp(logits)
        probs = exp_l / exp_l.sum()
        label_map = {v: k for k, v in RESULT_TO_LABEL.items()}
        scoped.append({
            "scope": s,
            "probabilities": probs.tolist(),
            "label_map": label_map,
        })
    payload = {
        "timestamp": time.time(),
        "sample_game": sample_game.game_id,
        "outcome_states": scoped,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
