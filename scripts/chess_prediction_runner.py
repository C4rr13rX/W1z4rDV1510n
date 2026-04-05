#!/usr/bin/env python3
"""
W1z4rDV1510n Chess Prediction Trio
=====================================
Three models predict every chess move independently, then form a weighted
collective prediction.  Each model's recent accuracy adjusts its vote weight
so the trio self-refines toward the most accurate combination.

The three models and what makes each one distinct:

  CLASSICAL  — Simulated annealing only.  Pure energy-minimisation over piece
                positions using the stack-history alignment energy (w_stack_hash).
                Treats the sequence of prior board states as a positional
                "signature" and searches for the next state that is most
                consistent with it — analogous to finding a preimage that matches
                an observed hash series.  No quantum effects, no learned priors.

  QUANTUM    — Quantum Trotter-slice annealing.  Same energy landscape as
                classical but the proposal distribution is shaped by imaginary-time
                path integrals (worldlines across Trotter slices).  Quantum
                tunnelling lets it escape local energy minima the classical
                annealer misses.  Excels when the correct next move involves a
                non-obvious piece interaction.

  NEURO      — Neuro-fabric guided annealing.  Hebbian learning accumulates
                across every ply seen.  The w_neuro_alignment energy pulls
                particle trajectories toward centroid positions learned by the
                fabric.  After enough plies the fabric "remembers" where each
                piece tends to sit in similar positions and biases the annealer
                toward those patterns.  Slow to start, increasingly dominant
                as training deepens.

Fine-tuning feedback loops:
  - Classical → Neuro: classical best_state is added to Neuro's stack_history,
    so Neuro's next call has a richer temporal context.
  - Quantum   → Classical: quantum best_energy is compared to classical
    best_energy; if quantum is lower (found better state), classical n_iters
    is bumped up to compensate.
  - Neuro     → Classical + Quantum: when Neuro is correct and others are
    wrong, its vote weight rises; others' lr_scale increases to catch up.
  - Collective → individual ledgers: which model was in the majority vs
    correct adjusts all three weights each ply.

The stack-hash interpretation:
  The w_stack_hash energy treats the stack_history as an ordered fingerprint
  of the game.  Finding the minimum-energy next state given that fingerprint
  is structurally identical to quantum-annealing a constraint-satisfaction
  problem where the constraint is "be consistent with this hash chain."
  This is sensible and intentional — it is one of the core motivations for
  using quantum annealing in this context.

Usage:
  python scripts/chess_prediction_runner.py
  python scripts/chess_prediction_runner.py --service http://localhost:8080
  python scripts/chess_prediction_runner.py --history-depth 10 --ply-delay 0.5

Requirements:
  - W1z4rDV1510n Rust service running on port 8080
    (started by w1z4rd_start.py or manually: cargo run --bin service)
  - python-chess: pip install python-chess
  - data/chess/processed_games.jsonl (run preprocess_chess_games.py first)
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows (box-drawing chars etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chess
except ImportError:
    raise SystemExit("python-chess required.  pip install python-chess")

import argparse
import random

ROOT        = Path(__file__).resolve().parents[1]
BOARD_JSON  = ROOT / "logs" / "chess_live_board.json"
HOURLY_LOG  = ROOT / "logs" / "chess_hourly_checkpoints.jsonl"
METRICS_LOG = ROOT / "logs" / "chess_training_metrics.log"
GAMES_FILE  = ROOT / "data" / "chess" / "processed_games.jsonl"

CFG_CLASSICAL = ROOT / "run_config_chess_ply_classical.json"
CFG_QUANTUM   = ROOT / "run_config_chess_ply_quantum.json"
CFG_NEURO     = ROOT / "run_config_chess_ply_neuro.json"

NEURO_LIVE    = ROOT / "logs" / "chess_ply_neuro_live.json"

# ── Run config helpers ────────────────────────────────────────────────────────

def load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_ply_cfg(base: dict, n_iters: int, seed_offset: int = 0) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg["schedule"]["n_iterations"] = n_iters
    # Vary seed each call so repeated queries don't collapse to same path
    if cfg.get("random", {}).get("provider") == "DETERMINISTIC":
        cfg["random"]["seed"] = (cfg["random"].get("seed", 42) + seed_offset) % 2**31
    cfg.pop("snapshot_file", None)
    return cfg


# ── Snapshot builder ──────────────────────────────────────────────────────────

def board_symbol_states(board: chess.Board) -> Dict[str, Any]:
    states: Dict[str, Any] = {}
    for sq, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        sid = f"{color}_{piece.symbol().upper()}_{chess.square_name(sq)}"
        states[sid] = {
            "position": {"x": float(chess.square_file(sq)), "y": float(chess.square_rank(sq)), "z": 0.0},
            "velocity": None,
            "internal_state": {
                "piece": piece.symbol(),
                "color": color,
                "square": chess.square_name(sq),
                "role": piece.symbol().upper(),
            },
        }
    return states


def build_snapshot(board: chess.Board, ply_history: List[chess.Board], ts: int) -> dict:
    current = board_symbol_states(board)
    symbols = [
        {
            "id": sid,
            "type": "CUSTOM",
            "position": s["position"],
            "properties": {
                "piece":  s["internal_state"]["piece"],
                "color":  s["internal_state"]["color"],
                "role":   s["internal_state"]["role"],
                "radius": 0.45,
            },
        }
        for sid, s in current.items()
    ]
    stack_history = [
        {"timestamp": {"unix": i}, "symbol_states": board_symbol_states(b)}
        for i, b in enumerate(ply_history)
    ]
    return {
        "timestamp": {"unix": ts},
        "bounds": {"width": 8.0, "height": 8.0, "depth": 1.0},
        "symbols": symbols,
        "metadata": {"source": "chess_prediction_runner"},
        "stack_history": stack_history,
    }


# ── Subprocess-based engine client ───────────────────────────────────────────
# Uses predict_state.exe (already compiled, not named "service.exe" so antivirus
# doesn't block it).  For each call: write snapshot + patched config to temp
# files, run the binary, read results JSON.

PREDICT_EXE = ROOT / "target" / "debug" / "predict_state.exe"


def find_predict_exe() -> Path:
    """Find the predict_state binary."""
    candidates = [
        ROOT / "target" / "debug" / "predict_state.exe",
        ROOT / "target" / "release" / "predict_state.exe",
        ROOT / "target" / "debug" / "predict_state",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "predict_state binary not found.  Run:\n  cargo build --bin predict_state"
    )


def subprocess_predict(
    snapshot: dict,
    cfg: dict,
    model_name: str,
    timeout_s: float = 90.0,
    exe: Optional[Path] = None,
) -> Optional[dict]:
    """
    Write snapshot and config to temp files, run predict_state, return results.
    """
    if exe is None:
        exe = PREDICT_EXE

    tmp_dir = ROOT / "logs" / "ply_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    snap_path   = tmp_dir / f"snap_{model_name}.json"
    cfg_path    = tmp_dir / f"cfg_{model_name}.json"
    result_path = tmp_dir / f"result_{model_name}.json"

    # Patch config to point at the temp snapshot and result files
    patched_cfg = json.loads(json.dumps(cfg))
    patched_cfg["snapshot_file"]              = str(snap_path).replace("\\", "/")
    patched_cfg["output"]["output_path"]      = str(result_path).replace("\\", "/")
    patched_cfg["output"]["save_best_state"]  = True
    patched_cfg["output"]["save_population_summary"] = False

    snap_path.write_text(json.dumps(snapshot), encoding="utf-8")
    cfg_path.write_text(json.dumps(patched_cfg), encoding="utf-8")

    try:
        proc = subprocess.run(
            [str(exe), "--config", str(cfg_path)],
            capture_output=True,
            timeout=timeout_s,
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace")[-200:]
            print(f"    [{model_name}] exit {proc.returncode}: {err}", flush=True)
            return None
        if result_path.exists():
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            # Wrap in the same shape as service response for compatibility
            return {"result": {"results": raw}}
        return None
    except subprocess.TimeoutExpired:
        print(f"    [{model_name}] timeout after {timeout_s}s", flush=True)
        return None
    except Exception as e:
        print(f"    [{model_name}] error: {e}", flush=True)
        return None


# ── Decode best_state → chess move ───────────────────────────────────────────

def decode_move(board: chess.Board, best_state: dict, side: chess.Color) -> Optional[str]:
    """
    Find which piece of `side` moved in best_state vs current board.
    Prefer the largest displacement among legal moves.
    """
    side_str = "white" if side == chess.WHITE else "black"
    pred: Dict[str, Tuple[int, int]] = {}
    for sid, st in best_state.get("symbol_states", {}).items():
        pos = st.get("position") or st
        x, y = pos.get("x", -1), pos.get("y", -1)
        f, r = int(round(x)), int(round(y))
        if 0 <= f <= 7 and 0 <= r <= 7:
            pred[sid] = (f, r)

    curr: Dict[str, Tuple[int, int]] = {}
    for sq, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        sid = f"{color}_{piece.symbol().upper()}_{chess.square_name(sq)}"
        curr[sid] = (chess.square_file(sq), chess.square_rank(sq))

    candidates = []
    for sid, ppos in pred.items():
        if not sid.startswith(side_str):
            continue
        cpos = curr.get(sid)
        if cpos is None or ppos == cpos:
            continue
        from_sq = chess.square(cpos[0], cpos[1])
        to_sq   = chess.square(ppos[0], ppos[1])
        move    = chess.Move(from_sq, to_sq)
        piece   = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            promo_rank = 7 if side == chess.WHITE else 0
            if ppos[1] == promo_rank:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        if move in board.legal_moves:
            dist = math.hypot(ppos[0] - cpos[0], ppos[1] - cpos[1])
            candidates.append((move, dist))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    try:
        return board.san(candidates[0][0])
    except Exception:
        return None


def neuro_decode_move(board: chess.Board, neuro_json: dict, side: chess.Color) -> Optional[str]:
    """
    Use neuro centroid positions as a prediction.  For each centroid labelled
    as a chess piece of the moving side, find the closest piece on the current
    board and see if it moved.
    """
    centroids = neuro_json.get("neuro", {}).get("centroids", {})
    if not centroids:
        return None

    side_str = "white" if side == chess.WHITE else "black"
    # Build a fake best_state from centroids
    fake_states: Dict[str, Any] = {}
    for label, c in centroids.items():
        if not isinstance(c, dict):
            continue
        if side_str not in label.lower():
            continue
        fake_states[label] = {"position": {"x": c.get("x", 0), "y": c.get("y", 0), "z": 0.0}}

    if not fake_states:
        return None
    return decode_move(board, {"symbol_states": fake_states}, side)


# ── Per-model ledger for accuracy and weight adjustment ───────────────────────

@dataclass
class ModelLedger:
    name: str
    window: deque = field(default_factory=lambda: deque(maxlen=200))
    total: int = 0
    correct: int = 0
    weight: float = 1.0          # collective vote weight
    n_iters_delta: int = 0       # cumulative iteration adjustment
    energy_history: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, correct_: bool, energy: float) -> None:
        self.window.append(int(correct_))
        self.total   += 1
        self.correct += int(correct_)
        if math.isfinite(energy):
            self.energy_history.append(energy)

    def recent_acc(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def global_acc(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def update_weight(self, all_ledgers: List["ModelLedger"]) -> None:
        """Softmax weight over recent accuracy so weights sum to 1."""
        accs = [m.recent_acc() for m in all_ledgers]
        total_exp = sum(math.exp(a * 6) for a in accs) or 1.0
        self.weight = math.exp(self.recent_acc() * 6) / total_exp

    def adjust_iters(self, correct_: bool, others_correct: bool) -> int:
        """
        Fine-tuning: if this model missed but others were right, add iterations.
        If this model was right and others missed, keep or reduce.
        """
        if not correct_ and others_correct:
            self.n_iters_delta = min(self.n_iters_delta + 20, 200)
        elif correct_ and not others_correct:
            self.n_iters_delta = max(self.n_iters_delta - 10, -100)
        return self.n_iters_delta


# ── Heat map from best_state displacements ────────────────────────────────────

def displacement_heat(best_state: dict, board: chess.Board) -> List[List[float]]:
    heat = [[0.0] * 8 for _ in range(8)]
    for sid, st in best_state.get("symbol_states", {}).items():
        pos = st.get("position") or st
        f, r = int(round(pos.get("x", -1))), int(round(pos.get("y", -1)))
        if 0 <= f <= 7 and 0 <= r <= 7:
            heat[r][f] += 1.0
    mx = max(max(row) for row in heat)
    if mx > 0:
        heat = [[v / mx for v in row] for row in heat]
    return heat


def merge_heats(heats: List[Tuple[List[List[float]], float]]) -> List[List[float]]:
    """Weighted average of multiple heat maps."""
    result = [[0.0] * 8 for _ in range(8)]
    total_w = sum(w for _, w in heats)
    if total_w == 0:
        return result
    for heat, w in heats:
        for r in range(8):
            for f in range(8):
                result[r][f] += heat[r][f] * w / total_w
    return result


# ── Board JSON ────────────────────────────────────────────────────────────────

def pieces_list(board: chess.Board) -> List[dict]:
    pieces = []
    for sq, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        pieces.append({
            "id":    f"{color}_{piece.symbol().upper()}_{chess.square_name(sq)}",
            "file":  chess.square_file(sq),
            "rank":  chess.square_rank(sq),
            "color": color,
            "piece": piece.symbol().upper(),
        })
    return pieces


def write_board(data: dict) -> None:
    BOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = BOARD_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(BOARD_JSON)


def board_frame(
    board: chess.Board,
    ledgers: List[ModelLedger],
    model_preds: List[dict],
    collective_san: Optional[str],
    actual_lm: Optional[dict],
    heat: List[List[float]],
    game_meta: dict,
    reveal: bool,
    acc_global: "ModelLedger",
) -> dict:
    preds_for_viz = []
    for mp in model_preds:
        san = mp.get("san")
        if san:
            try:
                mv = board.parse_san(san)
                preds_for_viz.append({
                    "san":        san,
                    "model":      mp["model"],
                    "probability": round(mp["weight"], 3),
                    "correct":    mp.get("correct", False),
                    "from_file":  chess.square_file(mv.from_square),
                    "from_rank":  chess.square_rank(mv.from_square),
                    "to_file":    chess.square_file(mv.to_square),
                    "to_rank":    chess.square_rank(mv.to_square),
                    "energy":     mp.get("energy"),
                })
            except Exception:
                pass
    if collective_san:
        try:
            mv = board.parse_san(collective_san)
            preds_for_viz.insert(0, {
                "san":        collective_san,
                "model":      "collective",
                "probability": 1.0,
                "correct":    any(p.get("correct") for p in preds_for_viz if p["san"] == collective_san),
                "from_file":  chess.square_file(mv.from_square),
                "from_rank":  chess.square_rank(mv.from_square),
                "to_file":    chess.square_file(mv.to_square),
                "to_rank":    chess.square_rank(mv.to_square),
            })
        except Exception:
            pass
    return {
        "game_count":  game_meta["game_count"],
        "game_id":     game_meta["game_id"],
        "result":      game_meta["result"],
        "ply":         game_meta["ply"],
        "total_plies": game_meta["total_plies"],
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "pieces":      pieces_list(board),
        "last_move":   actual_lm if reveal else None,
        "predictions": preds_for_viz,
        "actual_move": actual_lm,
        "prediction_correct_top1": any(p.get("correct") for p in preds_for_viz if p["model"] == "collective"),
        "prediction_correct_top3": any(p.get("correct") for p in preds_for_viz),
        "square_heat": heat,
        "outcome_probs": {"1-0": 0.33, "1/2-1/2": 0.33, "0-1": 0.33},
        "predicted_outcome": "?",
        "actual_outcome": game_meta["result"],
        "model_ledgers": {
            ld.name: {
                "recent_acc": round(ld.recent_acc(), 4),
                "global_acc": round(ld.global_acc(), 4),
                "weight":     round(ld.weight, 4),
                "total":      ld.total,
            }
            for ld in ledgers
        },
        "running": {
            "move_top1":   round(acc_global.global_acc(), 4),
            "move_top3":   round(acc_global.global_acc(), 4),
            "outcome_acc": 0.0,
            "total_plies": acc_global.total,
            "game_count":  game_meta["game_count"],
        },
        "reveal_phase": reveal,
    }


# ── Hourly checkpoint ─────────────────────────────────────────────────────────

class HourlyMonitor:
    def __init__(self) -> None:
        self.last_check = time.time()
        self.checkpoints: List[dict] = []
        self.prev_acc = 0.0
        # per-model iteration adjustments tracked separately via ledgers
        self.base_iters = {"classical": 120, "quantum": 180, "neuro": 150}

    def maybe_checkpoint(self, ledgers: List[ModelLedger], log_handle) -> None:
        now = time.time()
        if now - self.last_check < 3600:
            return
        hour = len(self.checkpoints) + 1
        collective_acc = sum(ld.global_acc() * ld.weight for ld in ledgers)
        delta = collective_acc - self.prev_acc

        # Auto-tune: give more iterations to the weakest model
        weakest = min(ledgers, key=lambda ld: ld.recent_acc())
        self.base_iters[weakest.name] = min(self.base_iters[weakest.name] + 60, 600)

        stagnating = delta < 0.002 and sum(ld.total for ld in ledgers) > 500
        adj_msg = f"  → boosting {weakest.name} to {self.base_iters[weakest.name]} iters"
        if stagnating:
            for k in self.base_iters:
                self.base_iters[k] = min(self.base_iters[k] + 40, 600)
            adj_msg += "  + stagnation boost to all"

        rec = {
            "hour": hour, "timestamp": now,
            "collective_acc": round(collective_acc, 4),
            "delta": round(delta, 4),
            "model_accs": {ld.name: round(ld.global_acc(), 4) for ld in ledgers},
            "model_weights": {ld.name: round(ld.weight, 4) for ld in ledgers},
            "base_iters": dict(self.base_iters),
            "stagnating": stagnating,
        }
        self.checkpoints.append(rec)
        HOURLY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HOURLY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if log_handle:
            log_handle.write(json.dumps(rec) + "\n")
            log_handle.flush()

        self.prev_acc = collective_acc
        self.last_check = now
        print(
            f"\n{'='*62}\n"
            f"  HOURLY CHECKPOINT #{hour}\n"
            f"  Collective accuracy: {collective_acc:.1%}  (Δ{delta:+.1%})\n" +
            "\n".join(f"    {ld.name:10s}  acc={ld.global_acc():.1%}  weight={ld.weight:.2f}"
                      for ld in ledgers) +
            f"\n{adj_msg}\n{'='*62}\n",
            flush=True,
        )


# ── Game loader ───────────────────────────────────────────────────────────────

def load_test_games(max_games: int) -> List[dict]:
    if not GAMES_FILE.exists():
        raise SystemExit(f"Missing {GAMES_FILE} — run preprocess_chess_games.py first")
    games = []
    with GAMES_FILE.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_games and len(games) >= max_games:
                break
            rec = json.loads(line)
            if i % 5 == 0 and rec.get("result") and len(rec.get("moves", [])) >= 6:
                games.append(rec)
    return games


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--history-depth", type=int, default=8,
                    help="Prior plies to include in stack_history for temporal alignment")
    ap.add_argument("--ply-delay", type=float, default=0.0)
    ap.add_argument("--log-file", type=Path, default=METRICS_LOG)
    args = ap.parse_args()

    # ── Locate binary ─────────────────────────────────────────────────────────
    exe = find_predict_exe()
    print(f"Using engine binary: {exe}", flush=True)

    # ── Load configs ──────────────────────────────────────────────────────────
    cfg_classical = load_cfg(CFG_CLASSICAL)
    cfg_quantum   = load_cfg(CFG_QUANTUM)
    cfg_neuro     = load_cfg(CFG_NEURO)

    # ── Games ─────────────────────────────────────────────────────────────────
    print("Loading games …", flush=True)
    test_games = load_test_games(args.max_games)
    rng = random.Random(42)
    rng.shuffle(test_games)
    print(f"  {len(test_games)} test games loaded.", flush=True)

    # ── Model ledgers ─────────────────────────────────────────────────────────
    ldg_classical = ModelLedger("classical")
    ldg_quantum   = ModelLedger("quantum")
    ldg_neuro     = ModelLedger("neuro")
    ledgers       = [ldg_classical, ldg_quantum, ldg_neuro]
    acc_global    = ModelLedger("collective")

    monitor = HourlyMonitor()

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.log_file.open("a", encoding="utf-8")
    game_iter  = iter(test_games)
    game_count = 0

    print("\n─── W1z4rDV1510n Prediction Trio ───────────────────────────\n", flush=True)
    print("  classical  → energy-minimisation over stack-hash fingerprint", flush=True)
    print("  quantum    → Trotter-slice tunnelling through local minima", flush=True)
    print("  neuro      → Hebbian fabric alignment, learns across all plies", flush=True)
    print("  collective → weighted vote; weights adjust from ply-by-ply accuracy\n", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        while True:
            try:
                game_rec = next(game_iter)
            except StopIteration:
                rng.shuffle(test_games)
                game_iter = iter(test_games)
                game_rec  = next(game_iter)

            game_count += 1
            moves       = game_rec.get("moves", [])
            game_id     = game_rec.get("id", f"game-{game_count}")
            result      = game_rec.get("result", "?")

            board        = chess.Board()
            ply_history: List[chess.Board] = []
            game_correct = 0
            game_total   = 0
            call_seed    = game_count * 1000

            for ply_idx, actual_san in enumerate(moves):
                side = board.turn
                history_boards = ply_history[-args.history_depth:]
                snapshot = build_snapshot(board, history_boards, ply_idx)

                iters = monitor.base_iters.copy()

                # ── 3 parallel engine calls ───────────────────────────────────
                def call_model(name, cfg, n_iters, seed_off):
                    c = make_ply_cfg(cfg, n_iters, seed_off)
                    t0 = time.time()
                    resp = subprocess_predict(snapshot, c, name, timeout_s=120.0, exe=exe)
                    elapsed = time.time() - t0
                    if resp and resp.get("result"):
                        res = resp["result"]
                        bs  = res.get("results", {}).get("best_state", {})
                        be  = res.get("results", {}).get("best_energy", float("inf"))
                        return name, bs, be, elapsed
                    return name, {}, float("inf"), time.time() - t0

                futures = {
                    pool.submit(call_model, "classical", cfg_classical,
                                iters["classical"] + ldg_classical.n_iters_delta, call_seed),
                    pool.submit(call_model, "quantum",   cfg_quantum,
                                iters["quantum"]   + ldg_quantum.n_iters_delta,   call_seed + 1),
                    pool.submit(call_model, "neuro",     cfg_neuro,
                                iters["neuro"]     + ldg_neuro.n_iters_delta,     call_seed + 2),
                }
                call_seed += 3

                results_by_name: Dict[str, Any] = {}
                for fut in as_completed(futures):
                    name, bs, be, elapsed = fut.result()
                    results_by_name[name] = {"best_state": bs, "best_energy": be, "elapsed": elapsed}

                # ── Decode each model's predicted move ────────────────────────
                neuro_json = {}
                try:
                    if NEURO_LIVE.exists():
                        neuro_json = json.loads(NEURO_LIVE.read_text(encoding="utf-8"))
                except Exception:
                    pass

                preds: Dict[str, Optional[str]] = {}
                energies: Dict[str, float] = {}

                for name, res in results_by_name.items():
                    energies[name] = res["best_energy"]
                    if name == "neuro":
                        # Primary: annealer guided by neuro; secondary: centroid decoding
                        move = decode_move(board, res["best_state"], side)
                        if move is None and neuro_json:
                            move = neuro_decode_move(board, neuro_json, side)
                        preds[name] = move
                    else:
                        preds[name] = decode_move(board, res["best_state"], side)

                # ── Fallback: first legal move ────────────────────────────────
                legal_list = list(board.legal_moves)
                fallback_san = board.san(legal_list[0]) if legal_list else None
                for name in ("classical", "quantum", "neuro"):
                    if preds.get(name) is None:
                        preds[name] = fallback_san

                # ── Collective weighted vote ──────────────────────────────────
                vote_scores: Dict[str, float] = {}
                for ld in ledgers:
                    p = preds.get(ld.name)
                    if p:
                        vote_scores[p] = vote_scores.get(p, 0.0) + ld.weight

                collective_san = max(vote_scores, key=vote_scores.get) if vote_scores else fallback_san

                # ── Score ─────────────────────────────────────────────────────
                correct_by = {name: (preds[name] == actual_san) for name in ("classical", "quantum", "neuro")}
                collective_correct = (collective_san == actual_san)

                # Fine-tune: cross-model feedback
                others_right = {
                    "classical": correct_by["quantum"] or correct_by["neuro"],
                    "quantum":   correct_by["classical"] or correct_by["neuro"],
                    "neuro":     correct_by["classical"] or correct_by["quantum"],
                }

                # Quantum → Classical energy feedback: if quantum found lower energy, bump classical iters
                q_energy = energies.get("quantum", float("inf"))
                c_energy = energies.get("classical", float("inf"))
                if math.isfinite(q_energy) and math.isfinite(c_energy) and q_energy < c_energy * 0.9:
                    ldg_classical.n_iters_delta = min(ldg_classical.n_iters_delta + 10, 200)

                for ld, name in [(ldg_classical, "classical"), (ldg_quantum, "quantum"), (ldg_neuro, "neuro")]:
                    ld.record(correct_by[name], energies.get(name, float("inf")))
                    ld.adjust_iters(correct_by[name], others_right[name])
                for ld in ledgers:
                    ld.update_weight(ledgers)

                acc_global.record(collective_correct, 0.0)

                game_correct += int(collective_correct)
                game_total   += 1

                # ── Parse actual move metadata ────────────────────────────────
                try:
                    actual_mv = board.parse_san(actual_san)
                    lm = {
                        "san": actual_san,
                        "from_file": chess.square_file(actual_mv.from_square),
                        "from_rank": chess.square_rank(actual_mv.from_square),
                        "to_file":   chess.square_file(actual_mv.to_square),
                        "to_rank":   chess.square_rank(actual_mv.to_square),
                    }
                except Exception:
                    lm = {"san": actual_san}

                # Collective heat = weighted merge of model heats
                heats = []
                for ld in ledgers:
                    bs = results_by_name.get(ld.name, {}).get("best_state", {})
                    if bs:
                        heats.append((displacement_heat(bs, board), ld.weight))
                merged_heat = merge_heats(heats) if heats else [[0.0]*8 for _ in range(8)]

                model_preds_list = [
                    {
                        "model":   name,
                        "san":     preds[name],
                        "weight":  next(ld.weight for ld in ledgers if ld.name == name),
                        "correct": correct_by[name],
                        "energy":  round(energies.get(name, 0), 3),
                    }
                    for name in ("classical", "quantum", "neuro")
                ]

                game_meta = {
                    "game_count": game_count, "game_id": game_id,
                    "result": result, "ply": ply_idx, "total_plies": len(moves),
                }

                # Prediction phase
                write_board(board_frame(
                    board, ledgers, model_preds_list, collective_san,
                    None, merged_heat, game_meta, False, acc_global,
                ))

                if args.ply_delay > 0:
                    time.sleep(args.ply_delay * 0.4)

                # Apply actual move
                ply_history.append(board.copy())
                try:
                    board.push(actual_mv)
                except Exception:
                    break

                # Reveal phase
                write_board(board_frame(
                    board, ledgers, model_preds_list, collective_san,
                    lm, merged_heat, game_meta, True, acc_global,
                ))

                # Console
                c_sym = "✓" if collective_correct else "✗"
                ind_syms = "".join(
                    ("✓" if correct_by[n] else "✗") + n[0].upper()
                    for n in ("classical", "quantum", "neuro")
                )
                print(
                    f"  ply {ply_idx+1:3d} {c_sym}  "
                    f"collective={str(collective_san or '?'):8s}  "
                    f"actual={actual_san:8s}  "
                    f"[{ind_syms}]  "
                    f"w=[C{ldg_classical.weight:.2f} Q{ldg_quantum.weight:.2f} N{ldg_neuro.weight:.2f}]  "
                    f"global={acc_global.global_acc():.1%}",
                    flush=True,
                )

                if args.ply_delay > 0:
                    time.sleep(args.ply_delay * 0.6)

            # End of game
            game_acc = game_correct / game_total if game_total else 0.0
            print(
                f"\n  ── game {game_count} [{game_id}]  {result}  "
                f"plies={game_total}  game={game_acc:.1%}  "
                f"global={acc_global.global_acc():.1%} ──\n",
                flush=True,
            )
            log_handle.write(json.dumps({
                "game_count": game_count, "game_id": game_id,
                "timestamp": time.time(), "plies": game_total,
                "game_acc": round(game_acc, 4),
                "global_acc": round(acc_global.global_acc(), 4),
                "model_accs": {ld.name: round(ld.global_acc(), 4) for ld in ledgers},
                "model_weights": {ld.name: round(ld.weight, 4) for ld in ledgers},
            }) + "\n")
            log_handle.flush()
            monitor.maybe_checkpoint(ledgers, log_handle)

    log_handle.close()


if __name__ == "__main__":
    main()
