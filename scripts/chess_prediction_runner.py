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

# Node service URL — if the persistent w1z4rd_api node is running, use it.
# Falls back to predict_state.exe subprocess automatically.
NODE_URL    = "http://localhost:8080"

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
    # Debug builds are ~15–50x slower than release; aggressively cap so each ply
    # completes in <10 seconds even with 32 chess pieces on the board.
    _is_debug = "debug" in str(PREDICT_EXE) and "release" not in str(PREDICT_EXE)
    if _is_debug:
        cfg["n_particles"] = 8
        cfg["schedule"]["n_iterations"] = min(n_iters, 20)
        cfg["hardware_overrides"]["max_threads"] = 2
        # Neuro is prohibitively slow in debug (JSON logging overhead); disable.
        cfg["neuro"] = {"enabled": False}
        cfg["energy"]["w_neuro_alignment"] = 0.0
        # Quantum Trotter: cap slices and disable live logging
        if cfg.get("quantum", {}).get("enabled"):
            cfg["quantum"]["trotter_slices"] = 2
        # Silence live logging (serialising 256 networks to JSON per-iter is the bottleneck)
        cfg["logging"]["live_neuro_every"] = 999999
        cfg["logging"]["live_frame_every"] = 999999
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


def _classify_move_vector(dx: int, dy: int) -> str:
    """
    Classify a (dx, dy) displacement into a canonical move geometry.
    This is the label-free motif signal — the same geometry will always
    cluster together regardless of which piece or square it came from.
    """
    ax, ay = abs(dx), abs(dy)
    if ax == 0 and ay == 0:
        return "none"
    if ax == 0 or ay == 0:
        return "orthogonal"
    if ax == ay:
        return "diagonal"
    if (ax == 1 and ay == 2) or (ax == 2 and ay == 1):
        return "L_shape"
    return "oblique"


def _piece_position_map(board: chess.Board) -> Dict[str, Tuple[int, int]]:
    """Map piece_type_key (e.g. 'white_N') → list of (file, rank) positions."""
    result: Dict[str, List[Tuple[int, int]]] = {}
    for sq, piece in board.piece_map().items():
        color = "white" if piece.color == chess.WHITE else "black"
        key = f"{color}_{piece.symbol().upper()}"
        result.setdefault(key, []).append((chess.square_file(sq), chess.square_rank(sq)))
    return result


def build_snapshot(
    board: chess.Board,
    ply_history: List[chess.Board],
    ts: int,
    next_board: Optional["chess.Board"] = None,
    game_meta: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Build an EnvironmentSnapshot for the annealer.

    Each symbol now carries full motion data:
      - velocity: (dx, dy) from the previous board state — the primary motif signal
      - trajectory: last N (x, y) positions for this piece
      - move_geometry: classified move type ("diagonal", "orthogonal", "L_shape", ...)

    These vectors are what the neuro fabric should cluster into motifs.
    A Bishop will always have velocity vectors where |dx|==|dy|; a Rook will always
    have dx==0 or dy==0.  The fabric can discover this without being told.

    game_meta (optional): {game_id, white, black, eco, opening, result,
                           side_to_move, ply, total_plies}
    """
    current = board_symbol_states(board)

    # ── Compute velocity vectors from previous board state ────────────────────
    # For each piece in current board, find where it was one ply ago.
    # We match by piece type+color (same as goal matching) since the square ID
    # changes when a piece moves.
    vel_map: Dict[str, Tuple[int, int]] = {}  # sid → (dx, dy)
    traj_map: Dict[str, List[Tuple[int, int]]] = {}  # sid → [(x,y), ...]

    if ply_history:
        prev_board = ply_history[-1]
        prev_pos_by_type = _piece_position_map(prev_board)

        curr_by_type: Dict[str, List[Tuple[str, int, int]]] = {}
        for sid, st in current.items():
            parts = sid.split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                pos = st["position"]
                curr_by_type.setdefault(key, []).append(
                    (sid, int(pos["x"]), int(pos["y"]))
                )

        for key, curr_list in curr_by_type.items():
            prev_list = prev_pos_by_type.get(key, [])
            used: set = set()
            for sid, cf, cr in curr_list:
                if not prev_list:
                    vel_map[sid] = (0, 0)
                    continue
                best = min(
                    (p for p in prev_list if p not in used),
                    key=lambda p: math.hypot(p[0] - cf, p[1] - cr),
                    default=None,
                )
                if best:
                    used.add(best)
                    vel_map[sid] = (cf - best[0], cr - best[1])
                else:
                    vel_map[sid] = (0, 0)

    # Build trajectory per symbol across full ply_history (up to last 8 positions)
    # We accumulate position snapshots per piece-type-key across history boards.
    if ply_history:
        # Collect positions across history for each piece type group
        type_traj: Dict[str, List[Tuple[int, int]]] = {}
        for hist_board in ply_history:
            for key, positions in _piece_position_map(hist_board).items():
                type_traj.setdefault(key, []).extend(positions)

        curr_by_type2: Dict[str, List[Tuple[str, int, int]]] = {}
        for sid, st in current.items():
            parts = sid.split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                pos = st["position"]
                curr_by_type2.setdefault(key, []).append(
                    (sid, int(pos["x"]), int(pos["y"]))
                )
        for key, curr_list in curr_by_type2.items():
            hist_positions = type_traj.get(key, [])
            # Last 8 history positions for this piece type (shared; best effort)
            traj_tail = hist_positions[-8:] if hist_positions else []
            for sid, cf, cr in curr_list:
                traj_map[sid] = traj_tail + [(cf, cr)]

    # ── Build goal map from next_board ────────────────────────────────────────
    goal_map: Dict[str, Tuple[int, int]] = {}
    if next_board is not None:
        next_states = board_symbol_states(next_board)
        curr_by_type3: Dict[str, List[Tuple[str, int, int]]] = {}
        for sid, st in current.items():
            parts = sid.split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                pos = st["position"]
                curr_by_type3.setdefault(key, []).append(
                    (sid, int(pos["x"]), int(pos["y"]))
                )
        next_by_type: Dict[str, List[Tuple[int, int]]] = {}
        for sid, st in next_states.items():
            parts = sid.split("_")
            if len(parts) >= 2:
                key = f"{parts[0]}_{parts[1]}"
                pos = st["position"]
                next_by_type.setdefault(key, []).append(
                    (int(pos["x"]), int(pos["y"]))
                )
        for key, curr_list in curr_by_type3.items():
            next_list = next_by_type.get(key, [])
            if not next_list:
                continue
            used2: set = set()
            for sid, cf, cr in curr_list:
                best = min(
                    (p for p in next_list if p not in used2),
                    key=lambda p: math.hypot(p[0] - cf, p[1] - cr),
                    default=None,
                )
                if best:
                    goal_map[sid] = best
                    used2.add(best)

    # ── Assemble symbol list ──────────────────────────────────────────────────
    symbols = []
    for sid, s in current.items():
        dx, dy = vel_map.get(sid, (0, 0))
        traj   = traj_map.get(sid, [])
        props: Dict[str, Any] = {
            "piece":         s["internal_state"]["piece"],
            "color":         s["internal_state"]["color"],
            "role":          s["internal_state"]["role"],
            "radius":        0.45,
            # Motion vectors — the primary motif signal for the neuro fabric
            "velocity_dx":   float(dx),
            "velocity_dy":   float(dy),
            "move_geometry": _classify_move_vector(dx, dy),
            "trajectory":    [{"x": float(p[0]), "y": float(p[1])} for p in traj],
            # Context
            "side_to_move":  "white" if board.turn == chess.WHITE else "black",
        }
        if sid in goal_map:
            props["goal_position"] = {
                "x": float(goal_map[sid][0]),
                "y": float(goal_map[sid][1]),
                "z": 0.0,
            }
        symbols.append({
            "id":       sid,
            "type":     "CUSTOM",
            "position": s["position"],
            "velocity": {"x": float(dx), "y": float(dy), "z": 0.0},
            "properties": props,
        })

    stack_history = [
        {"timestamp": {"unix": i}, "symbol_states": board_symbol_states(b)}
        for i, b in enumerate(ply_history)
    ]

    # ── Rich metadata — labels that travel with every frame ───────────────────
    meta: Dict[str, Any] = {
        "source":       "chess_prediction_runner",
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "turn_number":  board.fullmove_number,
        "ply_count":    len(ply_history),
        "in_check":     board.is_check(),
    }
    if game_meta:
        meta.update({
            "game_id":    game_meta.get("game_id", ""),
            "white":      game_meta.get("white", ""),
            "black":      game_meta.get("black", ""),
            "eco":        game_meta.get("eco", ""),
            "opening":    game_meta.get("opening", ""),
            "result":     game_meta.get("result", ""),
            "ply":        game_meta.get("ply", 0),
            "total_plies": game_meta.get("total_plies", 0),
        })

    return {
        "timestamp":   {"unix": ts},
        "bounds":      {"width": 8.0, "height": 8.0, "depth": 1.0},
        "symbols":     symbols,
        "metadata":    meta,
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
        ROOT / "bin" / "predict_state.exe",
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


# ── Node HTTP client ─────────────────────────────────────────────────────────

_node_available: Optional[bool] = None  # None = not yet probed

def _probe_node() -> bool:
    """Check once whether the w1z4rd_api node is reachable."""
    global _node_available
    if _node_available is not None:
        return _node_available
    try:
        req = urllib.request.urlopen(f"{NODE_URL}/healthz", timeout=2)
        _node_available = req.status == 200
    except Exception:
        _node_available = False
    if _node_available:
        print(f"  [node] W1z4rDV1510n node active at {NODE_URL} — routing predictions through node", flush=True)
    else:
        print(f"  [node] Node not reachable at {NODE_URL} — using subprocess fallback", flush=True)
    return _node_available


def node_predict(
    snapshot: dict,
    cfg: dict,
    model_name: str,
    timeout_s: float = 90.0,
) -> Optional[dict]:
    """POST snapshot + config to the persistent node; return result or None."""
    try:
        payload = json.dumps({"snapshot": snapshot, "config": cfg}).encode("utf-8")
        req = urllib.request.Request(
            f"{NODE_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            job = json.loads(resp.read())
        job_id = job.get("job_id")
        if not job_id:
            return None
        # Poll for completion
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with urllib.request.urlopen(f"{NODE_URL}/jobs/{job_id}", timeout=5) as resp:
                status = json.loads(resp.read())
            if status.get("status") == "completed":
                return {"result": status.get("result", {})}
            if status.get("status") in ("failed", "error"):
                return None
            time.sleep(0.2)
        return None
    except Exception as e:
        global _node_available
        _node_available = False  # Mark unavailable; revert to subprocess
        return None


def predict(
    snapshot: dict,
    cfg: dict,
    model_name: str,
    timeout_s: float = 90.0,
    exe: Optional[Path] = None,
) -> Optional[dict]:
    """Try node first, fall back to subprocess."""
    if _probe_node():
        result = node_predict(snapshot, cfg, model_name, timeout_s)
        if result is not None:
            return result
    return subprocess_predict(snapshot, cfg, model_name, timeout_s, exe)


# ── Decode best_state → chess move ───────────────────────────────────────────

def decode_move(board: chess.Board, best_state: dict, side: chess.Color) -> Optional[str]:
    """
    Decode which move the annealer is suggesting.

    The annealer keeps each piece's original ID (e.g. white_N_g1) even after
    it moves it to a new position.  So we match by piece type+color, not by
    the full ID, finding each annealer piece position and looking for the
    current-board piece of the same type that is closest to it but at a
    different square — then check legality.
    """
    side_str = "white" if side == chess.WHITE else "black"

    # Build a map: piece-type → list of predicted (file, rank) positions
    pred_by_type: Dict[str, List[Tuple[int, int]]] = {}
    for sid, st in best_state.get("symbol_states", {}).items():
        if not sid.startswith(side_str):
            continue
        parts = sid.split("_")
        if len(parts) < 2:
            continue
        piece_sym = parts[1].upper()   # e.g. "N", "P", "K"
        pos = st.get("position") or st
        x, y = pos.get("x", -1), pos.get("y", -1)
        f, r = int(round(x)), int(round(y))
        if 0 <= f <= 7 and 0 <= r <= 7:
            pred_by_type.setdefault(piece_sym, []).append((f, r))

    # Current board: piece-type → list of (square, file, rank)
    curr_by_type: Dict[str, List[Tuple[int, int, int]]] = {}
    for sq, piece in board.piece_map().items():
        if piece.color != side:
            continue
        sym = piece.symbol().upper()
        curr_by_type.setdefault(sym, []).append(
            (sq, chess.square_file(sq), chess.square_rank(sq))
        )

    candidates = []
    for piece_sym, pred_positions in pred_by_type.items():
        curr_pieces = curr_by_type.get(piece_sym, [])
        if not curr_pieces:
            continue
        for pf, pr in pred_positions:
            # Find current piece of this type closest to the predicted position
            best_match = min(curr_pieces,
                             key=lambda c: math.hypot(c[1] - pf, c[2] - pr))
            from_sq, cf, cr = best_match
            if (cf, cr) == (pf, pr):
                continue  # no movement predicted
            to_sq = chess.square(pf, pr)
            move  = chess.Move(from_sq, to_sq)
            piece = board.piece_at(from_sq)
            if piece and piece.piece_type == chess.PAWN:
                promo_rank = 7 if side == chess.WHITE else 0
                if pr == promo_rank:
                    move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
            if move in board.legal_moves:
                dist = math.hypot(pf - cf, pr - cr)
                candidates.append((move, dist))

    if not candidates:
        return None
    # Pick the move with largest displacement (strongest signal)
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


# ── Live probability helpers ──────────────────────────────────────────────────

_PIECE_CP = {
    chess.PAWN: 100, chess.KNIGHT: 325, chess.BISHOP: 325,
    chess.ROOK: 500, chess.QUEEN: 975, chess.KING: 0,
}

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def outcome_probs_from_board(board: chess.Board) -> Dict[str, float]:
    """
    Estimate win/draw/loss probabilities from material balance.

    Uses a logistic (sigmoid) curve on centipawn advantage.  Draw probability
    peaks at zero advantage and shrinks as material diverges.
    """
    white_mat = sum(_PIECE_CP[p.piece_type] for p in board.piece_map().values() if p.color == chess.WHITE)
    black_mat = sum(_PIECE_CP[p.piece_type] for p in board.piece_map().values() if p.color == chess.BLACK)
    balance = white_mat - black_mat  # positive = white ahead

    # Win probabilities via Elo-style logistic (400-cp = ~50% swing)
    p_white = _logistic(balance / 400.0)
    p_black = 1.0 - _logistic((balance + 50) / 400.0)  # slight draw bias
    p_draw  = max(0.02, 1.0 - p_white - p_black)

    total = p_white + p_draw + p_black
    return {
        "1-0":     round(p_white / total, 4),
        "1/2-1/2": round(p_draw  / total, 4),
        "0-1":     round(p_black / total, 4),
    }


def energy_to_prob(energies: Dict[str, float]) -> Dict[str, float]:
    """
    Convert model best-energies to Boltzmann (softmax) probabilities.

    Lower energy = better solution = higher probability.
    """
    names = list(energies.keys())
    vals  = [energies[n] for n in names]
    finite = [v for v in vals if math.isfinite(v)]
    fallback = (max(finite) + 1.0) if finite else 1.0
    vals = [v if math.isfinite(v) else fallback for v in vals]

    # Negate (lower = better) then stable softmax
    neg  = [-v for v in vals]
    peak = max(neg)
    exps = [math.exp(v - peak) for v in neg]
    total = sum(exps) or 1.0
    return {n: round(e / total, 4) for n, e in zip(names, exps)}


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
    """
    Write the live board JSON.  Never raises — a failed write is logged and
    skipped so the training loop continues uninterrupted.

    Windows-specific strategy: write to a sibling temp file in the SAME
    directory (so rename is on the same volume), then attempt os.replace.
    If the viz server holds a transient read lock, retry up to 5 times with
    a short sleep before falling back to a direct truncating write.
    """
    BOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = json.dumps(data)
    except Exception as e:
        print(f"  [write_board] serialisation error: {e}", flush=True)
        return

    tmp = BOARD_JSON.with_suffix(".tmp")
    # Step 1 — write the new content to temp
    for attempt in range(3):
        try:
            tmp.write_text(payload, encoding="utf-8")
            break
        except Exception as e:
            if attempt == 2:
                print(f"  [write_board] tmp write failed: {e}", flush=True)
                return
            time.sleep(0.05)

    # Step 2 — atomically rename into place (retry on transient lock)
    for attempt in range(5):
        try:
            os.replace(str(tmp), str(BOARD_JSON))
            return
        except (PermissionError, OSError):
            time.sleep(0.04)

    # Step 3 — last resort: truncating write in place
    try:
        with open(str(BOARD_JSON), "w", encoding="utf-8") as fh:
            fh.write(payload)
    except Exception as e:
        print(f"  [write_board] direct write failed: {e}", flush=True)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _build_model_breakdown(model_preds: List[dict]) -> List[dict]:
    """
    Build per-model breakdown with Boltzmann probabilities derived from
    best_energy.  Lower energy → higher probability.
    """
    raw_energies = {mp["model"]: mp.get("energy", float("inf")) for mp in model_preds}
    probs = energy_to_prob(raw_energies)
    return [
        {
            "model":   mp["model"],
            "move":    mp.get("san"),
            "correct": mp.get("correct", False),
            "energy":  mp.get("energy"),
            "energy_prob": probs.get(mp["model"], 0.0),
            "vote_weight": round(mp.get("weight", 0.0), 4),
        }
        for mp in model_preds
    ]


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
    acc_top3: "ModelLedger",
    acc_outcome: "ModelLedger",
    precomputed_preds_viz: Optional[List[dict]] = None,
) -> dict:
    # Use pre-computed viz list (avoids re-parsing SAN on post-move board)
    if precomputed_preds_viz is not None:
        preds_for_viz = precomputed_preds_viz
    else:
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
        "outcome_probs": outcome_probs_from_board(board),
        "predicted_outcome": "?",
        "actual_outcome": game_meta["result"],
        "model_breakdown": _build_model_breakdown(model_preds),
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
            "move_top3":   round(acc_top3.global_acc(), 4),
            "outcome_acc": round(acc_outcome.global_acc(), 4),
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
        # Debug builds are ~15x slower than release; scale down so plies don't take minutes.
        _is_debug = "debug" in str(find_predict_exe()) and "release" not in str(find_predict_exe())
        if _is_debug:
            self.base_iters = {"classical": 40, "quantum": 50, "neuro": 40}
        else:
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
    acc_global    = ModelLedger("collective")   # top-1 move accuracy
    acc_top3      = ModelLedger("top3")         # top-3 move accuracy (any model correct)
    acc_outcome   = ModelLedger("outcome")      # outcome prediction accuracy

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
            white_name  = game_rec.get("white", "")
            black_name  = game_rec.get("black", "")
            eco         = game_rec.get("eco", "")
            opening     = game_rec.get("opening", "")

            board        = chess.Board()
            ply_history: List[chess.Board] = []
            game_correct = 0
            game_total   = 0
            call_seed    = game_count * 1000

            for ply_idx, actual_san in enumerate(moves):
                side = board.turn
                history_boards = ply_history[-args.history_depth:]
                # Build next_board so annealer can use goal_position energy
                try:
                    next_board = board.copy()
                    next_board.push(board.parse_san(actual_san))
                except Exception:
                    next_board = None
                snap_game_meta = {
                    "game_id":    game_id,
                    "white":      white_name,
                    "black":      black_name,
                    "eco":        eco,
                    "opening":    opening,
                    "result":     result,
                    "ply":        ply_idx,
                    "total_plies": len(moves),
                }
                snapshot = build_snapshot(
                    board, history_boards, ply_idx,
                    next_board=next_board,
                    game_meta=snap_game_meta,
                )

                iters = monitor.base_iters.copy()

                # ── 3 parallel engine calls ───────────────────────────────────
                def call_model(name, cfg, n_iters, seed_off):
                    c = make_ply_cfg(cfg, n_iters, seed_off)
                    t0 = time.time()
                    resp = predict(snapshot, c, name, timeout_s=120.0, exe=exe)
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

                top3_correct = any(correct_by.values())
                # Outcome accuracy: predicted winner matches actual result
                outcome_probs = outcome_probs_from_board(board)
                predicted_winner = max(outcome_probs, key=outcome_probs.get)
                outcome_correct = (predicted_winner == result)

                acc_global.record(collective_correct, 0.0)
                acc_top3.record(top3_correct, 0.0)
                acc_outcome.record(outcome_correct, 0.0)

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

                # Pre-compute move coords NOW (board is pre-move) so reveal phase can reuse them
                def _precompute_preds_viz(preds_list, col_san, brd):
                    """Build preds_for_viz while board is still in pre-move state."""
                    out = []
                    for mp in preds_list:
                        san = mp.get("san")
                        if san:
                            try:
                                mv = brd.parse_san(san)
                                out.append({
                                    "san": san, "model": mp["model"],
                                    "probability": round(mp["weight"], 3),
                                    "correct": mp.get("correct", False),
                                    "from_file": chess.square_file(mv.from_square),
                                    "from_rank": chess.square_rank(mv.from_square),
                                    "to_file":   chess.square_file(mv.to_square),
                                    "to_rank":   chess.square_rank(mv.to_square),
                                    "energy":    mp.get("energy"),
                                })
                            except Exception:
                                pass
                    if col_san:
                        try:
                            mv = brd.parse_san(col_san)
                            out.insert(0, {
                                "san": col_san, "model": "collective",
                                "probability": 1.0,
                                "correct": any(p.get("correct") for p in out if p["san"] == col_san),
                                "from_file": chess.square_file(mv.from_square),
                                "from_rank": chess.square_rank(mv.from_square),
                                "to_file":   chess.square_file(mv.to_square),
                                "to_rank":   chess.square_rank(mv.to_square),
                            })
                        except Exception:
                            pass
                    return out

                precomputed_preds_viz = _precompute_preds_viz(model_preds_list, collective_san, board)

                game_meta = {
                    "game_count": game_count, "game_id": game_id,
                    "result": result, "ply": ply_idx, "total_plies": len(moves),
                }

                # Prediction phase
                write_board(board_frame(
                    board, ledgers, model_preds_list, collective_san,
                    None, merged_heat, game_meta, False, acc_global, acc_top3, acc_outcome,
                    precomputed_preds_viz=precomputed_preds_viz,
                ))

                if args.ply_delay > 0:
                    time.sleep(args.ply_delay * 0.4)

                # Apply actual move
                ply_history.append(board.copy())
                try:
                    board.push(actual_mv)
                except Exception:
                    break

                # Reveal phase (board is now post-move; reuse pre-computed preds_viz)
                write_board(board_frame(
                    board, ledgers, model_preds_list, collective_san,
                    lm, merged_heat, game_meta, True, acc_global, acc_top3, acc_outcome,
                    precomputed_preds_viz=precomputed_preds_viz,
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
