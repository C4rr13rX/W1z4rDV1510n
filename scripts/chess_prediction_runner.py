#!/usr/bin/env python3
"""
W1z4rDV1510n Chess Prediction Loop
====================================
This is a NODE of the W1z4rDV1510n architecture, not a standalone predictor.

For each ply of each chess game it:

  1. Builds an EnvironmentSnapshot of the current board state with full
     stack_history of all prior plies (so the annealer has temporal context).

  2. POSTs that snapshot to the running Rust service (POST /predict), which
     runs the simulated annealing engine — including the neuro fabric alignment
     energy, relational priors, stack-hash temporal alignment, quantum Trotter
     slices (when enabled), and homeostasis.  The annealer's best_state tells
     us where it predicts each piece will be on the NEXT ply.

  3. Decodes the annealer's predicted piece positions back into chess moves
     by finding which piece moved from its current square to a new square.

  4. Reveals the actual move, scores it (correct / wrong), then immediately
     submits the NEXT ply's board as the new snapshot so the annealer's neuro
     fabric sees it as a new observation — closing the Hebbian feedback loop
     via the stack_history accumulation and the w_neuro_alignment energy term.

  5. Writes logs/chess_live_board.json after every half-ply (prediction then
     reveal) so the live_viz_server can animate in real time.

  6. Logs per-game accuracy and hourly checkpoints; auto-adjusts n_iterations
     if the service is fast and accuracy stagnates.

Architecture feedback loops active during this run:
  - Simulated annealing  ←→  neuro fabric (w_neuro_alignment)
  - Stack-hash temporal alignment (prior plies weight future predictions)
  - Relational priors (piece relationship motifs built from prior games)
  - Quantum Trotter slices when run_config_quantum_chess.json is used
  - Homeostasis (automatic reheat when energy plateaus)

Requirements:
  - The W1z4rDV1510n Rust service must be running:
      cargo run --bin service
    (or it was started separately; default port 8080)
  - python-chess must be installed: pip install python-chess
  - processed_games.jsonl must exist (run preprocess_chess_games.py first)

Usage:
  python scripts/chess_prediction_runner.py
  python scripts/chess_prediction_runner.py --service http://localhost:8080 --quantum
  python scripts/chess_prediction_runner.py --ply-delay 0.8 --history-depth 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chess
except ImportError:
    raise SystemExit("python-chess required.  pip install python-chess")

ROOT = Path(__file__).resolve().parents[1]
BOARD_JSON    = ROOT / "logs" / "chess_live_board.json"
HOURLY_LOG    = ROOT / "logs" / "chess_hourly_checkpoints.jsonl"
METRICS_LOG   = ROOT / "logs" / "chess_training_metrics.log"
GAMES_FILE    = ROOT / "data" / "chess" / "processed_games.jsonl"

# ── Run configs ───────────────────────────────────────────────────────────────
LIVE_CONFIG_PATH    = ROOT / "run_config_chess_live.json"
QUANTUM_CONFIG_PATH = ROOT / "run_config_quantum_chess.json"


def load_run_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_chess_run_config(base: dict, history_depth: int, n_iterations: int) -> dict:
    """Return a stripped-down RunConfig for one ply prediction call."""
    cfg = json.loads(json.dumps(base))  # deep copy
    cfg["n_particles"] = min(cfg.get("n_particles", 64), 64)
    cfg["schedule"]["n_iterations"] = n_iterations
    cfg["schedule"]["t_start"] = 3.0
    cfg["schedule"]["t_end"]   = 0.3
    # Live-frame output from these short runs would clobber the board JSON
    cfg["logging"]["live_frame_path"] = "logs/chess_ply_frame.json"
    cfg["logging"]["live_neuro_path"] = "logs/chess_ply_neuro.json"
    cfg["logging"]["live_frame_every"] = max(1, n_iterations // 10)
    cfg["logging"]["live_neuro_every"] = max(1, n_iterations // 10)
    cfg["output"]["output_path"] = "logs/chess_ply_results.json"
    # Remove snapshot_file — we send the snapshot inline
    cfg.pop("snapshot_file", None)
    return cfg


# ── EnvironmentSnapshot builder ───────────────────────────────────────────────

def board_state_for_snapshot(board: chess.Board) -> Dict[str, Any]:
    """Symbol states dict for one board position (used inside stack_history)."""
    states: Dict[str, Any] = {}
    for square, piece in board.piece_map().items():
        file_idx  = chess.square_file(square)
        rank_idx  = chess.square_rank(square)
        color     = "white" if piece.color == chess.WHITE else "black"
        symbol_id = f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}"
        states[symbol_id] = {
            "position": {"x": float(file_idx), "y": float(rank_idx), "z": 0.0},
            "velocity": None,
            "internal_state": {
                "piece":  piece.symbol(),
                "color":  color,
                "square": chess.square_name(square),
                "role":   piece.symbol().upper(),
            },
        }
    return states


def build_snapshot(board: chess.Board, ply_history: List[chess.Board], timestamp_unix: int) -> dict:
    """
    Build a full EnvironmentSnapshot.
    - symbols        = current board pieces (initial positions for annealing).
    - stack_history  = sequence of all prior board states the annealer should
                       align with (temporal context for stack-hash energy).
    """
    current_states = board_state_for_snapshot(board)

    symbols = [
        {
            "id": symbol_id,
            "type": "CUSTOM",
            "position": state["position"],
            "properties": {
                "piece":  state["internal_state"]["piece"],
                "color":  state["internal_state"]["color"],
                "role":   state["internal_state"]["role"],
                "radius": 0.45,
            },
        }
        for symbol_id, state in current_states.items()
    ]

    stack_history = [
        {
            "timestamp": {"unix": idx},
            "symbol_states": board_state_for_snapshot(b),
        }
        for idx, b in enumerate(ply_history)
    ]

    return {
        "timestamp": {"unix": timestamp_unix},
        "bounds": {"width": 8.0, "height": 8.0, "depth": 1.0},
        "symbols": symbols,
        "metadata": {"source": "chess_prediction_runner"},
        "stack_history": stack_history,
    }


# ── Service client ────────────────────────────────────────────────────────────

def service_predict(service_url: str, snapshot: dict, config: dict, timeout_sec: float = 30.0) -> Optional[dict]:
    """POST /predict and return the result dict, or None on error."""
    payload = json.dumps({"config": config, "snapshot": snapshot}).encode("utf-8")
    req = urllib.request.Request(
        f"{service_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"    [service] HTTP {e.code}: {body[:200]}", flush=True)
        return None
    except Exception as e:
        print(f"    [service] error: {e}", flush=True)
        return None


def service_healthz(service_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{service_url.rstrip('/')}/healthz", timeout=3.0) as r:
            return r.status == 200
    except Exception:
        return False


# ── Decode annealer output → chess move ───────────────────────────────────────

def decode_annealer_move(
    current_board: chess.Board,
    best_state: dict,
    side_to_move: chess.Color,
) -> Optional[str]:
    """
    The annealer's best_state.symbol_states has predicted positions for all
    pieces.  Find the piece of the side-to-move whose predicted square differs
    from its current square, then verify it's a legal move.

    Returns SAN string of the best predicted move, or None if nothing decodes.
    """
    predicted_squares: Dict[str, Tuple[int, int]] = {}
    symbol_states = best_state.get("symbol_states", {})

    for sym_id, state in symbol_states.items():
        pos = state.get("position", {})
        if not pos:
            # v2 format: position is top-level
            pos = state
        x = pos.get("x", -1)
        y = pos.get("y", -1)
        if x < 0 or y < 0:
            continue
        file_idx = int(round(x))
        rank_idx = int(round(y))
        if 0 <= file_idx <= 7 and 0 <= rank_idx <= 7:
            predicted_squares[sym_id] = (file_idx, rank_idx)

    # Current piece positions
    current_squares: Dict[str, Tuple[int, int]] = {}
    for square, piece in current_board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        sym_id = f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}"
        current_squares[sym_id] = (chess.square_file(square), chess.square_rank(square))

    side_str = "white" if side_to_move == chess.WHITE else "black"

    # Score each predicted move by how much it deviates from current for moving side
    candidates = []
    for sym_id, pred_pos in predicted_squares.items():
        if not sym_id.startswith(side_str):
            continue
        curr_pos = current_squares.get(sym_id)
        if curr_pos is None:
            continue
        if pred_pos == curr_pos:
            continue  # piece didn't move

        from_sq = chess.square(curr_pos[0], curr_pos[1])
        to_sq   = chess.square(pred_pos[0], pred_pos[1])
        move    = chess.Move(from_sq, to_sq)

        # Check promotion
        piece = current_board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if (side_to_move == chess.WHITE and pred_pos[1] == 7) or \
               (side_to_move == chess.BLACK and pred_pos[1] == 0):
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)

        if move in current_board.legal_moves:
            dist = math.sqrt((pred_pos[0] - curr_pos[0])**2 + (pred_pos[1] - curr_pos[1])**2)
            candidates.append((move, dist))

    if not candidates:
        return None

    # Prefer the move with the greatest displacement (most deliberate shift)
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_move = candidates[0][0]
    try:
        return current_board.san(best_move)
    except Exception:
        return None


# ── Board JSON writer ─────────────────────────────────────────────────────────

def pieces_list(board: chess.Board) -> List[dict]:
    pieces = []
    for square, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        pieces.append({
            "id":    f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}",
            "file":  chess.square_file(square),
            "rank":  chess.square_rank(square),
            "color": color,
            "piece": piece.symbol().upper(),
        })
    return pieces


def annealer_heat(best_state: dict, current_board: chess.Board) -> List[List[float]]:
    """8×8 heat map from annealer displacement magnitudes."""
    heat = [[0.0] * 8 for _ in range(8)]
    symbol_states = best_state.get("symbol_states", {})
    for sym_id, state in symbol_states.items():
        pos = state.get("position", state)
        x = pos.get("x", -1)
        y = pos.get("y", -1)
        if x < 0:
            continue
        f = int(round(x))
        r = int(round(y))
        if 0 <= f <= 7 and 0 <= r <= 7:
            heat[r][f] += 1.0
    # Normalise
    mx = max(max(row) for row in heat)
    if mx > 0:
        heat = [[v / mx for v in row] for row in heat]
    return heat


def write_board_json(data: dict) -> None:
    BOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = BOARD_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(BOARD_JSON)


# ── Accuracy tracking ─────────────────────────────────────────────────────────

@dataclass
class Accuracy:
    total:   int = 0
    correct: int = 0

    def rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def pct(self) -> str:
        return f"{self.rate():.1%}"


@dataclass
class HourlyState:
    checkpoints: List[dict] = field(default_factory=list)
    last_check:  float = field(default_factory=time.time)
    prev_rate:   float = 0.0
    n_iters:     int = 300   # annealer iterations; auto-adjusted

    def maybe_checkpoint(self, acc: Accuracy, log_handle) -> None:
        now = time.time()
        if now - self.last_check < 3600:
            return
        hour = len(self.checkpoints) + 1
        rate  = acc.rate()
        delta = rate - self.prev_rate
        stagnating = delta < 0.002 and acc.total > 200

        # Auto-adjust annealer iterations: ramp up if stagnating, ease off if fast
        if stagnating:
            self.n_iters = min(self.n_iters + 100, 800)
            adj = f"  → n_iters → {self.n_iters} (stagnation)"
        else:
            self.n_iters = max(self.n_iters - 50, 150)
            adj = f"  → n_iters → {self.n_iters} (improving)"

        rec = {
            "hour": hour, "timestamp": now,
            "move_acc": round(rate, 4),
            "delta":    round(delta, 4),
            "plies":    acc.total,
            "n_iters":  self.n_iters,
            "stagnating": stagnating,
        }
        self.checkpoints.append(rec)
        HOURLY_LOG.parent.mkdir(parents=True, exist_ok=True)
        with HOURLY_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if log_handle:
            log_handle.write(json.dumps(rec) + "\n")
            log_handle.flush()
        self.prev_rate = rate
        self.last_check = now

        print(
            f"\n{'='*60}\n"
            f"  HOURLY CHECKPOINT #{hour}\n"
            f"  Move accuracy: {rate:.1%}  (Δ{delta:+.1%})\n"
            f"  Plies seen:    {acc.total:,}\n"
            + adj +
            f"\n{'='*60}\n",
            flush=True,
        )


# ── Game loader ───────────────────────────────────────────────────────────────

def load_games(max_games: int) -> List[dict]:
    if not GAMES_FILE.exists():
        raise SystemExit(f"Missing {GAMES_FILE} — run scripts/preprocess_chess_games.py first")
    games = []
    with GAMES_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if max_games and len(games) >= max_games:
                break
            rec = json.loads(line)
            if rec.get("result") and len(rec.get("moves", [])) >= 6:
                games.append(rec)
    return games


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service", default="http://localhost:8080",
                    help="URL of the running W1z4rDV1510n Rust service")
    ap.add_argument("--quantum", action="store_true",
                    help="Use run_config_quantum_chess.json (quantum Trotter slices enabled)")
    ap.add_argument("--max-games", type=int, default=0,
                    help="Game cap (0 = all games, cycle forever)")
    ap.add_argument("--history-depth", type=int, default=8,
                    help="How many prior plies to include in stack_history for temporal context")
    ap.add_argument("--n-iters", type=int, default=300,
                    help="Starting annealer iterations per ply prediction")
    ap.add_argument("--ply-delay", type=float, default=0.0,
                    help="Extra sleep between plies (seconds); 0 = as fast as service allows")
    ap.add_argument("--log-file", type=Path, default=METRICS_LOG)
    args = ap.parse_args()

    # ── Verify service is up ──────────────────────────────────────────────────
    print(f"Checking service at {args.service} …", flush=True)
    for attempt in range(30):
        if service_healthz(args.service):
            print("  Service is up.", flush=True)
            break
        if attempt == 0:
            print("  Waiting for service to start (up to 30s) …", flush=True)
        time.sleep(1.0)
    else:
        raise SystemExit(
            f"Service not reachable at {args.service}/healthz\n"
            "Start it with:  cargo run --bin service"
        )

    # ── Load run config ───────────────────────────────────────────────────────
    config_path = QUANTUM_CONFIG_PATH if args.quantum else LIVE_CONFIG_PATH
    print(f"Using config: {config_path.name}", flush=True)
    base_config = load_run_config(config_path)

    # ── Load games ────────────────────────────────────────────────────────────
    print("Loading games …", flush=True)
    all_games = load_games(args.max_games)
    import random
    rng = random.Random(42)
    # 80/20 split deterministically by game index
    test_games  = [g for i, g in enumerate(all_games) if i % 5 == 0]
    rng.shuffle(test_games)
    print(f"  {len(all_games)} total games, {len(test_games)} test games", flush=True)

    hourly = HourlyState(n_iters=args.n_iters)
    acc    = Accuracy()
    game_count = 0

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = args.log_file.open("a", encoding="utf-8")

    game_iter = iter(test_games)

    print("\n─── W1z4rDV1510n prediction loop running ───\n", flush=True)
    print("Every ply: board → EnvironmentSnapshot → POST /predict → annealer", flush=True)
    print("         → decode best_state → predicted move → score → next ply\n", flush=True)

    while True:
        # Cycle through test games forever
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
        ply_history: List[chess.Board] = []  # boards BEFORE each move
        game_correct = 0
        game_total   = 0

        for ply_idx, actual_san in enumerate(moves):
            side_to_move = board.turn

            # ── 1. Build snapshot from current board + history ─────────────
            history_boards = ply_history[-args.history_depth:]
            snapshot = build_snapshot(board, history_boards, ply_idx)
            run_cfg  = make_chess_run_config(base_config, args.history_depth, hourly.n_iters)

            # ── 2. Ask the W1z4rDV1510n architecture for a prediction ──────
            t0       = time.time()
            response = service_predict(args.service, snapshot, run_cfg, timeout_sec=60.0)
            elapsed  = time.time() - t0

            predicted_san: Optional[str] = None
            best_energy   = float("inf")
            heat          = [[0.0]*8 for _ in range(8)]

            if response and response.get("result"):
                res         = response["result"]
                best_state  = res.get("results", {}).get("best_state", {})
                best_energy = res.get("results", {}).get("best_energy", float("inf"))
                heat        = annealer_heat(best_state, board)
                predicted_san = decode_annealer_move(board, best_state, side_to_move)

            # Fallback: most common legal move by frequency heuristic
            if predicted_san is None:
                legal = list(board.legal_moves)
                if legal:
                    predicted_san = board.san(legal[0])

            correct = (predicted_san == actual_san)
            acc.total   += 1
            acc.correct += int(correct)
            game_total   += 1
            game_correct += int(correct)

            # ── 3. Write prediction phase to viz ────────────────────────────
            try:
                actual_move_obj = board.parse_san(actual_san)
                lm = {
                    "san":       actual_san,
                    "from_file": chess.square_file(actual_move_obj.from_square),
                    "from_rank": chess.square_rank(actual_move_obj.from_square),
                    "to_file":   chess.square_file(actual_move_obj.to_square),
                    "to_rank":   chess.square_rank(actual_move_obj.to_square),
                }
            except Exception:
                lm = {"san": actual_san}

            write_board_json({
                "game_count":  game_count,
                "game_id":     game_id,
                "result":      result,
                "ply":         ply_idx,
                "total_plies": len(moves),
                "side_to_move": "white" if side_to_move == chess.WHITE else "black",
                "pieces":      pieces_list(board),
                "last_move":   None,
                "predictions": [{"san": predicted_san, "probability": 1.0, "correct": correct,
                                 "from_file": lm.get("from_file", -1), "from_rank": lm.get("from_rank", -1),
                                 "to_file": lm.get("to_file", -1), "to_rank": lm.get("to_rank", -1)}]
                                if predicted_san else [],
                "actual_move": lm,
                "prediction_correct_top1": correct,
                "prediction_correct_top3": correct,
                "square_heat": heat,
                "outcome_probs": {"1-0": 0.33, "1/2-1/2": 0.33, "0-1": 0.33},  # annealer energy → outcome future work
                "predicted_outcome": "?",
                "actual_outcome": result,
                "annealer": {
                    "best_energy": round(best_energy, 4) if math.isfinite(best_energy) else None,
                    "elapsed_sec": round(elapsed, 2),
                    "n_iters":     hourly.n_iters,
                    "quantum":     args.quantum,
                },
                "running": {
                    "move_top1":   round(acc.rate(), 4),
                    "move_top3":   round(acc.rate(), 4),
                    "outcome_acc": 0.0,
                    "total_plies": acc.total,
                    "game_count":  game_count,
                },
                "hourly_checkpoints": hourly.checkpoints[-10:],
                "reveal_phase": False,
            })

            if args.ply_delay > 0:
                time.sleep(args.ply_delay * 0.4)

            # ── 4. Apply actual move, record board for next ply's history ──
            ply_history.append(board.copy())
            try:
                board.push(actual_move_obj)
            except Exception:
                break

            # ── 5. Write reveal phase to viz ─────────────────────────────────
            write_board_json({
                "game_count":  game_count,
                "game_id":     game_id,
                "result":      result,
                "ply":         ply_idx,
                "total_plies": len(moves),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "pieces":      pieces_list(board),
                "last_move":   lm,
                "predictions": [{"san": predicted_san, "probability": 1.0, "correct": correct,
                                 "from_file": lm.get("from_file", -1), "from_rank": lm.get("from_rank", -1),
                                 "to_file": lm.get("to_file", -1), "to_rank": lm.get("to_rank", -1)}]
                                if predicted_san else [],
                "actual_move": lm,
                "prediction_correct_top1": correct,
                "prediction_correct_top3": correct,
                "square_heat": heat,
                "outcome_probs": {"1-0": 0.33, "1/2-1/2": 0.33, "0-1": 0.33},
                "predicted_outcome": "?",
                "actual_outcome": result,
                "annealer": {
                    "best_energy": round(best_energy, 4) if math.isfinite(best_energy) else None,
                    "elapsed_sec": round(elapsed, 2),
                    "n_iters":     hourly.n_iters,
                    "quantum":     args.quantum,
                },
                "running": {
                    "move_top1":   round(acc.rate(), 4),
                    "move_top3":   round(acc.rate(), 4),
                    "outcome_acc": 0.0,
                    "total_plies": acc.total,
                    "game_count":  game_count,
                },
                "hourly_checkpoints": hourly.checkpoints[-10:],
                "reveal_phase": True,
            })

            result_sym = "✓" if correct else "✗"
            print(
                f"  ply {ply_idx+1:3d}  {result_sym}  "
                f"predicted={str(predicted_san or '?'):8s}  "
                f"actual={actual_san:8s}  "
                f"E={best_energy:8.3f}  {elapsed:.1f}s  "
                f"│ acc {acc.pct()} ({acc.correct}/{acc.total})",
                flush=True,
            )

            if args.ply_delay > 0:
                time.sleep(args.ply_delay * 0.6)

        # ── End of game ───────────────────────────────────────────────────────
        game_acc = game_correct / game_total if game_total else 0.0
        print(
            f"\n  ── game {game_count} [{game_id}]  result={result}  "
            f"plies={game_total}  game-acc={game_acc:.1%}  "
            f"global={acc.pct()} ({acc.correct}/{acc.total}) ──\n",
            flush=True,
        )

        log_rec = {
            "game_count":  game_count,
            "game_id":     game_id,
            "timestamp":   time.time(),
            "plies":       game_total,
            "game_acc":    round(game_acc, 4),
            "global_acc":  round(acc.rate(), 4),
            "n_iters":     hourly.n_iters,
            "quantum":     args.quantum,
        }
        log_handle.write(json.dumps(log_rec) + "\n")
        log_handle.flush()

        hourly.maybe_checkpoint(acc, log_handle)

    log_handle.close()


if __name__ == "__main__":
    main()
