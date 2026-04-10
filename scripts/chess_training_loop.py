#!/usr/bin/env python3
"""
Iteratively train chess outcome + move prediction models on the processed PGN dataset.

Workflow:
1. Ensure PGNs are downloaded under data/chess and run preprocess_chess_games.py.
2. Launch this script. It loads the JSONL dataset, builds hashed feature vectors,
   and spins an infinite (or user-configured) loop that keeps training:
      - Softmax regressors predict the eventual game outcome from prefixes of moves.
      - Frequency-based predictors guess future moves up to 20 moves ahead.
3. After every loop the script prints integer accuracy percentages per scope/horizon.

The run is deterministic so repeated invocations are comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import threading
import urllib.request
import urllib.error
from typing import Iterable, Sequence, Dict, Tuple, List, Any, Optional

try:
    import chess
except ImportError as exc:  # pragma: no cover - dependency notice
    raise SystemExit(
        "python-chess is required. Install with `pip install python-chess`."
    ) from exc


def configure_thread_env() -> None:
    """Set sane defaults so NumPy/BLAS use the available CPU cores efficiently."""
    cpu_count = max(1, os.cpu_count() or 1)
    default_threads = str(min(cpu_count, 16))
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ.setdefault(key, default_threads)


configure_thread_env()

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "chess" / "processed_games.jsonl"
DEFAULT_LOG_PATH = ROOT / "logs" / "chess_training_metrics.log"
BOARD_JSON = ROOT / "logs" / "chess_live_board.json"
FEATURE_DIM = 512  # hashed bag-of-move features
METADATA_DIM = 128  # hashed player/opening features
BOARD_FEATURE_DIM = 12 * 64 + 8  # piece planes + extras
TOTAL_FEATURE_DIM = FEATURE_DIM + METADATA_DIM + BOARD_FEATURE_DIM
DEFAULT_OUTCOME_SCOPES = [6, 10, 14, 18, 22, 26, 30, 40]  # number of half-moves observed
DEFAULT_MOVE_HORIZONS = [1, 5, 10, 15, 20]  # moves to predict ahead
DEFAULT_CONTEXT_WINDOW = 6  # use last N moves as context for move prediction
DEFAULT_CONTEXT_STRIDE = 2  # step when iterating contexts to keep volume manageable
RESULT_TO_LABEL = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
LABEL_TO_RESULT = {0: "White win", 1: "Draw", 2: "Black win"}
RELATION_BINS = 8
RELATION_TOPK = 5
PRIOR_BLEND = 0.35  # how much weight to give priors in logit space
PRIOR_EPS = 1e-6
PRIOR_TOPK_MOTIFS = 50
ZONE_BINS = 4  # coarse spatial bins for role-aware motifs
PRIOR_CLUSTERS = 3  # number of style/cluster priors
FACTOR_BLEND = 0.4  # how much to trust factorized priors in move scoring
ANCHOR_STRIDE = 4  # interval for anchor motif counters
BEAM_WIDTH = 8
BEAM_STEPS = 3

# ── Node / service bridge ─────────────────────────────────────────────────────
# The w1z4rd_api service (port 8080) exposes:
#   POST /neuro/train    — observe an EnvironmentSnapshot → Hebbian weight update
#   POST /neuro/activate — input labels → target-stream activations
#   GET  /neuro/snapshot — current neuro state
# The node (port 8090) exposes:
#   POST /qa/ingest      — push Q&A examples for the QA runtime
# All calls are fire-and-forget (background queue) so training throughput is
# unaffected when the services are busy or unavailable.

SERVICE_URL = "http://localhost:8080"
NODE_URL    = "http://localhost:8090"
# How many games between neuro train bursts (1 = every game; higher = faster training loop)
NEURO_TRAIN_EVERY_N_GAMES = 1
# Max plies to send per game to /neuro/train (cap to avoid huge requests on long games)
NEURO_MAX_PLIES_PER_GAME  = 20
# Background queue capacity; excess drops silently to never block training.
_BRIDGE_QUEUE_MAXSIZE = 64

# psutil is optional but strongly preferred
try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _psutil = None  # type: ignore
    _PSUTIL_AVAILABLE = False


class ResourceMonitor:
    """
    Adaptive resource governor — no hard-coded limits anywhere.

    Polls system CPU and RAM usage for this process vs the whole machine.
    Keeps a reserved headroom for the OS and interactive use (mouse, windows).
    Ramps the training workload UP in small steps when resources are plentiful,
    ramps DOWN quickly when the machine is under pressure.

    Emitted scale factor (0 < scale ≤ 1.0) is applied by the training loop to
    batch_size, game_count, and sleep_secs.
    """

    # Fraction of total CPU / RAM the training loop is allowed to use at most.
    # These are soft ceilings derived from measurement, not static constants.
    _CPU_HEADROOM   = 0.20   # reserve at least 20% CPU for OS / user
    _RAM_HEADROOM   = 0.25   # reserve at least 25% RAM for OS / user
    _RAMP_UP_STEP   = 0.10   # increase scale by 10% per check when under budget
    _RAMP_DOWN_STEP = 0.30   # decrease scale by 30% per check when over budget
    _SCALE_MIN      = 0.05   # never go below 5% — keep the loop alive
    _SCALE_MAX      = 1.00

    def __init__(self) -> None:
        self._scale: float = 0.10  # start conservatively
        self._proc = _psutil.Process() if _PSUTIL_AVAILABLE else None
        self._total_ram: float = _psutil.virtual_memory().total if _PSUTIL_AVAILABLE else 8 * 1024 ** 3
        cpu_count = max(1, os.cpu_count() or 1)
        self._cpu_count: int = cpu_count
        # Initialise CPU percent measurement (first call always returns 0.0)
        if self._proc is not None:
            self._proc.cpu_percent(interval=None)

    # ── Public API ────────────────────────────────────────────────────────────

    def poll(self) -> float:
        """
        Measure current resource usage and return the updated scale factor.
        Call this once per training iteration (non-blocking).
        """
        if not _PSUTIL_AVAILABLE:
            return self._scale  # can't measure → stay at current scale

        try:
            # Process CPU as fraction of one logical core, normalised to [0,1]
            proc_cpu_frac = (self._proc.cpu_percent(interval=None) / 100.0) / self._cpu_count
            sys_cpu_frac  = _psutil.cpu_percent(interval=None) / 100.0

            vm = _psutil.virtual_memory()
            proc_ram_frac = self._proc.memory_info().rss / self._total_ram
            sys_ram_frac  = vm.percent / 100.0

            cpu_budget_ok = sys_cpu_frac  < (1.0 - self._CPU_HEADROOM)
            ram_budget_ok = sys_ram_frac  < (1.0 - self._RAM_HEADROOM)

            if cpu_budget_ok and ram_budget_ok:
                self._scale = min(self._SCALE_MAX, self._scale + self._RAMP_UP_STEP)
            else:
                self._scale = max(self._SCALE_MIN, self._scale - self._RAMP_DOWN_STEP)

        except Exception:
            pass  # psutil hiccup — keep current scale

        return self._scale

    @property
    def scale(self) -> float:
        return self._scale

    def batch_size(self, base: int) -> int:
        """Scale a base batch size, always at least 1."""
        return max(1, int(base * self._scale))

    def game_count(self, base: int) -> int:
        """Scale a base game count, always at least 1."""
        return max(1, int(base * self._scale))

    def sleep_secs(self, base: float) -> float:
        """More idle time when under pressure; no extra sleep when at full scale."""
        return base * max(0.0, 1.0 - self._scale)

    def status_line(self) -> str:
        if not _PSUTIL_AVAILABLE:
            return f"[resource] scale={self._scale:.2f} (psutil unavailable)"
        try:
            vm  = _psutil.virtual_memory()
            cpu = _psutil.cpu_percent(interval=None)
            return (
                f"[resource] scale={self._scale:.2f}  "
                f"sys_cpu={cpu:.0f}%  "
                f"sys_ram={vm.percent:.0f}%  "
                f"proc_ram={self._proc.memory_info().rss / 1024**2:.0f}MB"
            )
        except Exception:
            return f"[resource] scale={self._scale:.2f}"


class NodeBridge:
    """
    Fire-and-forget client for w1z4rd_api and node services.

    • Neuro training (POST /neuro/train): each chess position is sent as an
      EnvironmentSnapshot.  The neuro fabric learns which pieces co-occur in
      good/bad positions and builds Hebbian weight patterns.
    • Neuro activation (POST /neuro/activate): query what the fabric predicts
      given the current position's piece labels — used as an additional move
      prior that strengthens as training deepens.
    • QA ingest (POST /qa/ingest to node): push (context_moves → next_move)
      pairs so the QA runtime builds a retrieval index for move prediction.

    All network I/O runs in a daemon worker thread; the main training thread
    never blocks waiting for HTTP.
    """

    def __init__(
        self,
        service_url: str = SERVICE_URL,
        node_url: str = NODE_URL,
    ) -> None:
        self.service_url = service_url.rstrip("/")
        self.node_url = node_url.rstrip("/")
        self._service_ok: Optional[bool] = None
        self._node_ok: Optional[bool] = None
        self._service_last_probe: float = 0.0   # epoch seconds of last probe
        self._node_last_probe: float = 0.0
        self._RETRY_INTERVAL: float = 30.0       # re-probe failed services every 30 s
        self._queue: "list[tuple[str, dict]]" = []
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # ── probe ─────────────────────────────────────────────────────────────────

    def _probe(self, url: str, path: str) -> bool:
        try:
            req = urllib.request.urlopen(f"{url}/{path}", timeout=2)
            return req.status == 200
        except Exception:
            return False

    @property
    def service_available(self) -> bool:
        now = time.monotonic()
        if self._service_ok is None or (not self._service_ok and now - self._service_last_probe > self._RETRY_INTERVAL):
            self._service_last_probe = now
            result = self._probe(self.service_url, "healthz")
            if result != self._service_ok:
                status = "UP" if result else "unreachable"
                print(f"  [bridge] w1z4rd_api ({self.service_url}): {status}", flush=True)
            self._service_ok = result
        return bool(self._service_ok)

    @property
    def node_available(self) -> bool:
        now = time.monotonic()
        if self._node_ok is None or (not self._node_ok and now - self._node_last_probe > self._RETRY_INTERVAL):
            self._node_last_probe = now
            result = self._probe(self.node_url, "health")
            if result != self._node_ok:
                status = "UP" if result else "unreachable"
                print(f"  [bridge] node ({self.node_url}): {status}", flush=True)
            self._node_ok = result
        return bool(self._node_ok)

    # ── background worker ─────────────────────────────────────────────────────

    def _enqueue(self, target: str, payload: dict) -> None:
        with self._lock:
            if len(self._queue) < _BRIDGE_QUEUE_MAXSIZE:
                self._queue.append((target, payload))

    def _run(self) -> None:
        while True:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item is None:
                time.sleep(0.05)
                continue
            target, payload = item
            try:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    target, data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=10).close()
            except Exception:
                pass  # fire-and-forget: silently discard on network error

    # ── public API ────────────────────────────────────────────────────────────

    def send_neuro_train(self, snapshot: dict) -> None:
        """POST snapshot to /neuro/train on both services.  Non-blocking.

        The w1z4rd_api (8080) and the new node (8090) both receive the same
        chess board snapshot so their neural fabrics learn in parallel.  They
        use the same EnvironmentSnapshot format; only the endpoint differs.
        """
        body = {"snapshot": snapshot}
        if self.service_available:
            self._enqueue(f"{self.service_url}/neuro/train", body)
        if self.node_available:
            self._enqueue(f"{self.node_url}/neuro/train", body)

    def send_qa_ingest(self, question: str, answer: str, context_labels: List[str]) -> None:
        """POST a Q&A chess example to the node /qa/ingest.  Non-blocking."""
        if not self.node_available:
            return
        payload = {
            "question": question,
            "answer": answer,
            "context_labels": context_labels,
            "domain": "chess",
        }
        self._enqueue(f"{self.node_url}/qa/ingest", payload)

    def send_record_episode(
        self,
        context_labels: List[str],
        predicted: str,
        actual: str,
        streams: List[str],
        surprise: float,
    ) -> None:
        """POST a resolved prediction episode to /neuro/record_episode.  Non-blocking.

        Called after each training iteration so the fabric's episodic store
        accumulates chess-specific prediction outcomes.  The fabric's
        ConditionalSufficiencyTracker and Hebbian inhibitory updates fire
        automatically on receipt — no additional wiring needed.

        `surprise` = (1 - p_correct)^2 for correct predictions (small gradient)
                   = p_confidence^2     for wrong predictions  (large gradient)
        """
        if not self.service_available:
            return
        payload = {
            "context_labels": context_labels,
            "predicted": predicted,
            "actual": actual,
            "streams": streams,
            "surprise": float(surprise),
        }
        self._enqueue(f"{self.service_url}/neuro/record_episode", payload)

    def get_neuro_activate(
        self,
        labels: List[str],
        target_stream: str = "chess",
        hops: int = 3,
        timeout_s: float = 1.0,
    ) -> Dict[str, float]:
        """
        Query /neuro/activate synchronously.  Returns label→strength map.
        Falls back to empty dict on timeout or error.
        """
        if not self.service_available:
            return {}
        try:
            data = json.dumps(
                {"input_labels": labels, "target_stream": target_stream, "hops": hops}
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.service_url}/neuro/activate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                result = json.loads(resp.read())
            return {a["label"]: a["strength"] for a in result.get("activations", [])}
        except Exception:
            return {}


def _symbol_states_to_snapshot(
    symbol_states: Dict[str, Dict[str, Any]],
    ts_unix: int,
    result_label: Optional[str] = None,
) -> dict:
    """
    Convert the training loop's per-ply symbol_states dict to an
    EnvironmentSnapshot JSON object understood by /neuro/train.

    Labels are encoded into the symbol's `properties` and also surfaced as
    a "chess" stream in metadata so the neuro fabric can group them.
    """
    symbols = []
    dynamic_symbol_states = {}
    for sid, info in symbol_states.items():
        pos = info.get("position", {})
        internal = info.get("internal", {})
        piece = internal.get("piece", "?").upper()
        color = internal.get("color", "unknown")
        role  = internal.get("role",  piece)
        zone  = internal.get("zone",  0)
        sym = {
            "id": sid,
            "type": "CUSTOM",
            "position": {"x": pos.get("x", 0.0), "y": pos.get("y", 0.0), "z": 0.0},
            "properties": {
                "piece":  piece,
                "color":  color,
                "role":   role,
                "zone":   str(zone),
                "stream": "chess",
            },
        }
        symbols.append(sym)
        dynamic_symbol_states[sid] = {
            "position": {"x": pos.get("x", 0.0), "y": pos.get("y", 0.0), "z": 0.0},
            "internal_state": {
                "piece":  piece,
                "color":  color,
                "role":   role,
                "stream": "chess",
            },
        }
    meta: Dict[str, Any] = {"stream": "chess"}
    if result_label:
        meta["result"] = result_label

    return {
        "timestamp": {"unix": ts_unix},
        "bounds": {"x_min": 0.0, "x_max": 7.0, "y_min": 0.0, "y_max": 7.0},
        "symbols": symbols,
        "metadata": meta,
        "stack_history": [],
    }


def write_live_board(
    iteration: int,
    game: Optional[Any],
    ply: int,
    outcome_acc: Dict[int, float],
    move_acc: Dict[int, float],
    total_games: int,
    total_plies_actual: int = 0,
    outcome_models: Optional[Any] = None,
    move_counters: Optional[Any] = None,
    move_defaults: Optional[Any] = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> None:
    """
    Write a minimal chess_live_board.json so the viz server shows training progress.
    Pieces are read from the game's symbol_states for the current ply.
    When outcome_models and move_counters are provided, real model predictions are shown.
    """
    BOARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    pieces = []
    if game is not None and ply < len(game.moves):
        for sid, info in game.replay_symbol_states(ply).items():
            pos = info.get("position", {})
            internal = info.get("internal", {})
            sq_name = sid.rsplit("_", 1)[-1] if "_" in sid else "a1"
            try:
                sq = chess.parse_square(sq_name)
                file_idx = chess.square_file(sq)
                rank_idx = chess.square_rank(sq)
            except Exception:
                file_idx = int(pos.get("x", 0))
                rank_idx = int(pos.get("y", 0))
            color = internal.get("color", "white")
            piece = internal.get("piece", "P").upper()
            pieces.append({
                "id":    sid,
                "file":  file_idx,
                "rank":  rank_idx,
                "color": color,
                "piece": piece,
            })

    # Compute real outcome probabilities from the model at this ply.
    outcome_probs = {"1-0": 0.33, "1/2-1/2": 0.34, "0-1": 0.33}
    if outcome_models and game is not None and ply > 0:
        try:
            valid_scopes = [s for s in outcome_models.keys() if s <= ply]
            if valid_scopes:
                scope = max(valid_scopes)
                counts = np.bincount(
                    game.hashed_moves[:ply].astype(int), minlength=FEATURE_DIM
                ).astype(np.float32)
                board_vecs = game.replay_board_vectors(ply)
                bvec = (
                    board_vecs[ply - 1]
                    if ply - 1 < len(board_vecs)
                    else np.zeros(BOARD_FEATURE_DIM, dtype=np.float32)
                )
                vec = compose_prefix_vector(counts, ply, game.meta_vector, bvec)
                probs = outcome_models[scope].predict_proba(vec[None, :])[0]
                outcome_probs = {
                    "1-0":     float(probs[0]),
                    "1/2-1/2": float(probs[1]),
                    "0-1":     float(probs[2]),
                }
        except Exception:
            pass

    # Compute top predicted moves from the frequency counters at this context.
    predictions: list = []
    if move_counters and game is not None and ply >= context_window:
        try:
            ctx = tuple(int(b) for b in game.hashed_moves[ply - context_window : ply])
            counter = move_counters.get(1, {}).get(ctx)
            if counter:
                top = counter.most_common(5)
                total_c = sum(c for _, c in top) or 1
                predictions = [
                    {"san": mv, "probability": c / total_c, "correct": False}
                    for mv, c in top
                ]
            elif move_defaults and move_defaults.get(1):
                predictions = [{"san": move_defaults[1], "probability": 1.0, "correct": False}]
        except Exception:
            pass

    # Compute last_move (from/to squares for the move that just reached this ply).
    # The viz draws highlight squares using from_file/rank and to_file/rank (0-7).
    last_move = None
    if game is not None and ply > 0 and ply <= len(game.moves):
        try:
            board = chess.Board()
            for san in game.moves[:ply]:
                move = board.push_san(san)
            last_move = {
                "from_file": chess.square_file(move.from_square),
                "from_rank": chess.square_rank(move.from_square),
                "to_file":   chess.square_file(move.to_square),
                "to_rank":   chess.square_rank(move.to_square),
                "san":       game.moves[ply - 1],
            }
        except Exception:
            pass

    best_outcome_scope = min(outcome_acc.keys(), default=6)
    best_move_horizon  = min(move_acc.keys(), default=1)
    payload = {
        "game_count":   total_games,
        "game_id":      getattr(game, "game_id", "training") if game else "training",
        "result":       LABEL_TO_RESULT.get(getattr(game, "result_label", 0), "?"),
        "ply":          ply,
        "total_plies":  len(game.moves) if game else 0,
        "side_to_move": "white" if ply % 2 == 0 else "black",
        "pieces":       pieces,
        "last_move":    last_move,
        "predictions":  predictions,
        "actual_move":  None,
        "reveal_phase": True,
        "prediction_correct_top1": False,
        "square_heat":  [[0.0] * 8 for _ in range(8)],
        "outcome_probs": outcome_probs,
        "model_breakdown": [],
        "model_ledgers": {
            "training": {
                "recent_acc": outcome_acc.get(best_outcome_scope, 0.0),
                "global_acc": outcome_acc.get(best_outcome_scope, 0.0),
                "weight": 1.0,
                "total": iteration,
            }
        },
        "running": {
            "move_top1":   move_acc.get(best_move_horizon, 0.0),
            "move_top3":   move_acc.get(best_move_horizon, 0.0),
            "outcome_acc": outcome_acc.get(best_outcome_scope, 0.0),
            "total_plies": total_plies_actual or total_games * 40,
            "game_count":  total_games,
        },
        "hourly_checkpoints": [],
        "training_mode": True,
        "iteration": iteration,
    }
    try:
        tmp = BOARD_JSON.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(str(tmp), str(BOARD_JSON))
    except Exception:
        try:
            BOARD_JSON.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────

# ------------------------
# Relational / motif utils
# ------------------------

def bin_position(pos: Dict[str, float], bins: int = RELATION_BINS) -> Tuple[int, int]:
    return (
        int(math.floor(pos.get("x", 0.0) * bins)),
        int(math.floor(pos.get("y", 0.0) * bins)),
    )


def distance_bucket(a: Dict[str, float], b: Dict[str, float]) -> str:
    dx = a.get("x", 0.0) - b.get("x", 0.0)
    dy = a.get("y", 0.0) - b.get("y", 0.0)
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.5:
        return "near"
    if dist < 2.0:
        return "mid"
    return "far"


def direction_bucket(a: Dict[str, float], b: Dict[str, float]) -> str:
    dx = b.get("x", 0.0) - a.get("x", 0.0)
    dy = b.get("y", 0.0) - a.get("y", 0.0)
    if abs(dx) > abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) > 0:
        return "north" if dy > 0 else "south"
    return "same"


def relational_signature(symbol_states: Dict[str, Dict[str, Any]]) -> List[str]:
    sig = []
    ids = sorted(symbol_states.keys())
    # unary role/zone tokens
    for sid in ids:
        info = symbol_states[sid]
        pos = info["position"]
        internal = info.get("internal", {})
        role = internal.get("role") or internal.get("piece", "UNK").upper()
        side = internal.get("color", "u")
        bx, by = bin_position(pos, bins=ZONE_BINS)
        sig.append(f"u:{role}:{side}:{bx}{by}")
    # pairwise relational tokens with roles
    for i, first in enumerate(ids):
        for second in ids[i + 1:]:
            a_info = symbol_states[first]
            b_info = symbol_states[second]
            a = a_info["position"]
            b = b_info["position"]
            role_a = a_info.get("internal", {}).get("role") or a_info.get("internal", {}).get("piece", "UNK").upper()
            role_b = b_info.get("internal", {}).get("role") or b_info.get("internal", {}).get("piece", "UNK").upper()
            dir_bucket = direction_bucket(a, b)
            dist_bucket = distance_bucket(a, b)
            sig.append(f"r:{role_a}->{role_b}:{dir_bucket}:{dist_bucket}")
    return sig


def motif_hash(symbol_states: Dict[str, Dict[str, Any]]) -> str:
    parts = relational_signature(symbol_states)
    return "|".join(parts)


def print_runtime_banner(
    outcome_scopes: Sequence[int],
    move_horizons: Sequence[int],
    context_window: int,
    context_stride: int,
) -> None:
    hw = os.uname() if hasattr(os, "uname") else None
    banner = [
        "=== Chess Training Loop ===",
        f"Python {sys.version.split()[0]} | NumPy {np.__version__}",
        f"CPU cores detected: {os.cpu_count() or 1}",
        f"Thread caps -> OMP:{os.environ.get('OMP_NUM_THREADS')}, "
        f"MKL:{os.environ.get('MKL_NUM_THREADS')}, "
        f"OpenBLAS:{os.environ.get('OPENBLAS_NUM_THREADS')}",
        f"Outcome scopes: {list(outcome_scopes)} | Move horizons: {list(move_horizons)}",
        f"Context window: {context_window} moves (stride {context_stride})",
    ]
    if hw:
        banner.append(f"Platform: {hw.sysname} {hw.release} ({hw.machine})")
    print("\n".join(banner))


def stable_bucket(text: str, mod: int) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % mod


def zone_bin_from_square(square: int, bins: int = ZONE_BINS) -> int:
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    step = max(1, 8 // bins)
    bx = min(bins - 1, file_idx // step)
    by = min(bins - 1, rank_idx // step)
    return by * bins + bx


def deterministic_split(identifier: str) -> bool:
    """Returns True if record goes to train split."""
    digest = hashlib.blake2s(identifier.encode("utf-8"), digest_size=2).digest()
    return digest[0] % 5 != 0  # ~80/20 split


@dataclass(frozen=True)
class GameRecord:
    """
    Lean game record — board_vectors and symbol_states are NOT stored here.
    They are computed on-demand from the raw SAN moves to keep RAM proportional
    to the number of games loaded, not to the number of plies × feature dimensions.

    Use replay_board_vectors(ply_limit) or replay_symbol_states(ply) to access
    position data; the callers are responsible for discarding the results once done.
    """
    game_id: str
    result_label: int
    moves: tuple[str, ...]  # SAN strings; board can be fully replayed from these
    hashed_moves: np.ndarray  # shape (len(moves),)
    is_train: bool
    meta_vector: np.ndarray  # shape (METADATA_DIM,)
    move_factors: tuple["MoveFactor", ...]  # factorized move attributes per ply

    def replay_board_vectors(self, ply_limit: Optional[int] = None) -> List[np.ndarray]:
        """Replay moves up to ply_limit, return per-ply board vectors (local, not stored)."""
        board = chess.Board()
        vecs: List[np.ndarray] = []
        limit = ply_limit if ply_limit is not None else len(self.moves)
        for san in self.moves[:limit]:
            try:
                board.push(board.parse_san(san))
            except ValueError:
                break
            vecs.append(board_to_vector(board))
        return vecs

    def replay_symbol_states(self, ply: int) -> Dict[str, Any]:
        """Replay moves up to ply (inclusive), return board_symbol_states at that ply."""
        board = chess.Board()
        for san in self.moves[:ply + 1]:
            try:
                board.push(board.parse_san(san))
            except ValueError:
                break
        return board_symbol_states(board)


@dataclass(frozen=True)
class MoveFactor:
    role: str
    side: str
    start_bin: int
    end_bin: int
    capture: int
    promo: str


def load_games(limit: int | None) -> list[GameRecord]:
    """
    Load game records without pre-computing board vectors or symbol states.
    Board vectors are ~3 GB for 25k games; keeping them out of GameRecord
    means load_games uses ~200 MB regardless of dataset size, and board
    data is only materialised on demand during featurization.
    """
    if not DATA_PATH.exists():
        raise SystemExit(
            "Missing processed dataset. Run scripts/preprocess_chess_games.py first."
        )
    records: list[GameRecord] = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(records) >= limit:
                break
            payload = json.loads(line)
            result = payload["result"]
            if result not in RESULT_TO_LABEL:
                continue
            moves = tuple(payload["moves"])
            if len(moves) < 6:
                continue
            hashed = np.fromiter(
                (stable_bucket(mv, FEATURE_DIM) for mv in moves),
                dtype=np.int16,
                count=len(moves),
            )
            tokens = metadata_tokens(payload, len(moves))
            meta_vec = metadata_vector(tokens)
            # Compute move_factors (cheap: one board pass, no vector storage)
            board = chess.Board()
            move_factors: list[MoveFactor] = []
            valid = True
            for move_san in moves:
                try:
                    move = board.parse_san(move_san)
                except ValueError:
                    valid = False
                    break
                move_factors.append(factor_from_move(board, move))
                board.push(move)
            if not valid or len(move_factors) != len(moves):
                continue
            records.append(GameRecord(
                game_id=payload["id"],
                result_label=RESULT_TO_LABEL[result],
                moves=moves,
                hashed_moves=hashed,
                is_train=deterministic_split(payload["id"]),
                meta_vector=meta_vec,
                move_factors=tuple(move_factors),
            ))
    if not records:
        raise SystemExit("No valid games loaded. Check dataset integrity.")
    return records


def metadata_tokens(payload: dict, ply_count: int) -> list[str]:
    tokens: list[str] = []
    white = (payload.get("white") or payload.get("White") or "").strip()
    black = (payload.get("black") or payload.get("Black") or "").strip()
    eco = (payload.get("eco") or payload.get("ECO") or "").strip()
    opening = (payload.get("opening") or payload.get("Opening") or "").strip()
    if white:
        tokens.append(f"white:{white.lower()}")
    if black:
        tokens.append(f"black:{black.lower()}")
    if eco:
        tokens.append(f"eco:{eco.lower()}")
    if opening:
        tokens.append(f"opening:{opening.lower()}")
    ply_bucket = min((ply_count // 10) * 10, 80)
    tokens.append(f"ply:{ply_bucket}")
    return tokens


def metadata_vector(tokens: list[str]) -> np.ndarray:
    vec = np.zeros(METADATA_DIM, dtype=np.float32)
    if not tokens:
        return vec
    for token in tokens:
        vec[stable_bucket(token, METADATA_DIM)] += 1.0
    return vec


def board_to_vector(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 64), dtype=np.float32)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        plane_index = (6 if piece.color == chess.BLACK else 0) + (piece.piece_type - 1)
        planes[plane_index, square] = 1.0
    extras = np.array(
        [
            1.0 if board.turn == chess.WHITE else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0,
            1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0,
            1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0,
            board.halfmove_clock / 100.0,
            board.fullmove_number / 200.0,
            1.0 if board.is_check() else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([planes.reshape(-1), extras], axis=0)


def board_symbol_states(board: chess.Board) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    for square, piece in board.piece_map().items():
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        color = "white" if piece.color == chess.WHITE else "black"
        role = piece.symbol().upper()
        zone = zone_bin_from_square(square)
        symbol_id = f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}"
        states[symbol_id] = {
            "position": {"x": float(file_idx), "y": float(rank_idx), "z": 0.0},
            "internal": {
                "piece": piece.symbol(),
                "color": color,
                "role": role,
                "zone": zone,
            },
        }
    return states


def factor_from_move(board: chess.Board, move: chess.Move) -> MoveFactor:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return MoveFactor(role="UNK", side="unknown", start_bin=0, end_bin=0, capture=0, promo="")
    role = piece.symbol().upper()
    side = "white" if piece.color == chess.WHITE else "black"
    start_bin = zone_bin_from_square(move.from_square)
    end_bin = zone_bin_from_square(move.to_square)
    capture = 1 if board.is_capture(move) else 0
    promo = ""
    if move.promotion:
        promo_piece = chess.Piece(move.promotion, piece.color)
        promo = promo_piece.symbol().upper()
    return MoveFactor(role=role, side=side, start_bin=start_bin, end_bin=end_bin, capture=capture, promo=promo)


def compose_prefix_vector(
    counts: np.ndarray,
    length: int,
    meta_vec: np.ndarray,
    board_vec: np.ndarray,
) -> np.ndarray:
    vec = np.empty(TOTAL_FEATURE_DIM, dtype=np.float32)
    vec[:FEATURE_DIM] = counts / np.sqrt(float(length))
    vec[FEATURE_DIM : FEATURE_DIM + METADATA_DIM] = meta_vec
    vec[FEATURE_DIM + METADATA_DIM :] = board_vec
    return vec


def select_window_starts(total_positions: int, limit: int) -> list[int]:
    if total_positions <= 0:
        return []
    if limit <= 1 or total_positions == 1:
        return [0]
    if total_positions <= limit:
        return list(range(total_positions))
    step = (total_positions - 1) / float(limit - 1)
    starts = {0, total_positions - 1}
    for idx in range(1, limit - 1):
        position = int(round(idx * step))
        position = max(0, min(total_positions - 1, position))
        starts.add(position)
    return sorted(starts)


def build_outcome_features(
    games: Iterable[GameRecord],
    scopes: Sequence[int],
    multi_frame_windows: int = 1,
) -> dict[int, dict[str, np.ndarray]]:
    scopes_sorted = sorted(set(int(scope) for scope in scopes if scope > 0))
    if not scopes_sorted:
        return {}
    scope_set = set(scopes_sorted)
    min_scope = scopes_sorted[0]
    max_scope = scopes_sorted[-1]
    datasets: dict[int, dict[str, np.ndarray]] = {}
    for scope in scopes_sorted:
        datasets[scope] = {
            "train_X": [],
            "train_y": [],
            "test_X": [],
            "test_y": [],
        }
    _zero_board_vec = np.zeros(BOARD_FEATURE_DIM, dtype=np.float32)
    for record in tqdm(games, desc="Featurizing outcomes"):
        total_moves = record.hashed_moves.shape[0]
        if total_moves < min_scope:
            continue
        # Replay board vectors once per game (discarded after this record)
        board_vecs = record.replay_board_vectors(min(max_scope, total_moves))
        if multi_frame_windows > 1:
            for scope in scopes_sorted:
                if total_moves < scope:
                    continue
                total_positions = total_moves - scope + 1
                for start in select_window_starts(total_positions, multi_frame_windows):
                    end = start + scope
                    segment = record.hashed_moves[start:end]
                    counts = np.bincount(segment, minlength=FEATURE_DIM).astype(
                        np.float32
                    )
                    board_vec = board_vecs[end - 1] if end - 1 < len(board_vecs) else _zero_board_vec
                    vec = compose_prefix_vector(
                        counts, scope, record.meta_vector, board_vec
                    )
                    key = "train" if record.is_train else "test"
                    datasets[scope][f"{key}_X"].append(vec)
                    datasets[scope][f"{key}_y"].append(record.result_label)
        else:
            limit = min(max_scope, total_moves)
            counts = np.zeros(FEATURE_DIM, dtype=np.float32)
            for idx in range(limit):
                bucket = int(record.hashed_moves[idx])
                counts[bucket] += 1.0
                length = idx + 1
                if length in scope_set:
                    board_vec = board_vecs[length - 1] if length - 1 < len(board_vecs) else _zero_board_vec
                    vec = compose_prefix_vector(
                        counts, length, record.meta_vector, board_vec
                    )
                    key = "train" if record.is_train else "test"
                    datasets[length][f"{key}_X"].append(vec)
                    datasets[length][f"{key}_y"].append(record.result_label)
    # convert to numpy arrays
    for scope, blob in datasets.items():
        for split in ("train", "test"):
            xs = blob[f"{split}_X"]
            ys = blob[f"{split}_y"]
            if xs:
                blob[f"{split}_X"] = np.stack(xs, axis=0)
                blob[f"{split}_y"] = np.array(ys, dtype=np.int64)
            else:
                blob[f"{split}_X"] = np.zeros((0, TOTAL_FEATURE_DIM), dtype=np.float32)
                blob[f"{split}_y"] = np.zeros((0,), dtype=np.int64)
    return datasets


class OutcomeNetwork:
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int, l2: float = 1e-4):
        self.n_classes = n_classes
        self.W1 = np.random.randn(n_features, hidden_dim).astype(np.float32) * 0.01
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.01
        self.b2 = np.zeros((hidden_dim,), dtype=np.float32)
        self.W3 = np.random.randn(hidden_dim, n_classes).astype(np.float32) * 0.01
        self.b3 = np.zeros((n_classes,), dtype=np.float32)
        self.l2 = l2
        self.hidden_dim = hidden_dim

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        h1 = self._relu(z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = self._relu(z2)
        logits = h2 @ self.W3 + self.b3
        return logits, h1, h2, z1

    # γ (gamma) for focal loss: 2.0 is the standard value.
    # High γ → confident wrong predictions get a proportionally larger gradient;
    # easy correct predictions get suppressed.  Mirrors dopaminergic prediction error.
    FOCAL_GAMMA: float = 2.0

    def train_epoch(self, X: np.ndarray, y: np.ndarray, lr: float, batch_size: int) -> float:
        if X.size == 0:
            return 0.0
        num_samples = X.shape[0]
        order = np.random.permutation(num_samples)
        total_loss = 0.0
        batches = max(1, int(np.ceil(num_samples / batch_size)))
        for idx in range(batches):
            batch_idxs = order[idx * batch_size : (idx + 1) * batch_size]
            xb = X[batch_idxs]
            yb = y[batch_idxs]
            logits, h1, h2, z1 = self._forward(xb)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits, dtype=np.float32)
            probs = exp / exp.sum(axis=1, keepdims=True)

            # ── Focal loss ──────────────────────────────────────────────────
            # Standard cross-entropy: diff = probs - y_onehot
            # Focal modulation: scale each sample's gradient by (1 - p_t)^γ
            # where p_t = probability assigned to the correct class.
            # This suppresses easy-correct updates and amplifies confident-wrong ones.
            p_correct = probs[np.arange(yb.shape[0]), yb]          # (B,)
            focal_weight = (1.0 - p_correct[:, None]) ** self.FOCAL_GAMMA  # (B, 1) broadcast

            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(yb.shape[0]), yb] = 1.0
            diff = focal_weight * (probs - y_onehot)               # focal-scaled gradient
            # ────────────────────────────────────────────────────────────────

            grad_W3 = h2.T @ diff / yb.shape[0] + self.l2 * self.W3
            grad_b3 = diff.mean(axis=0)
            dh2 = diff @ self.W3.T
            dz2 = dh2 * (h2 > 0)
            grad_W2 = h1.T @ dz2 / yb.shape[0] + self.l2 * self.W2
            grad_b2 = dz2.mean(axis=0)
            dh1 = dz2 @ self.W2.T
            dz1 = dh1 * (z1 > 0)
            grad_W1 = xb.T @ dz1 / yb.shape[0] + self.l2 * self.W1
            grad_b1 = dz1.mean(axis=0)
            self.W3 -= lr * grad_W3
            self.b3 -= lr * grad_b3
            self.W2 -= lr * grad_W2
            self.b2 -= lr * grad_b2
            self.W1 -= lr * grad_W1
            self.b1 -= lr * grad_b1
            # Focal loss value: -(1 - p_t)^γ * log(p_t)
            batch_loss = float(
                -(((1.0 - p_correct) ** self.FOCAL_GAMMA) * np.log(p_correct + 1e-9)).mean()
            )
            reg = 0.5 * self.l2 * (
                np.sum(self.W1 * self.W1)
                + np.sum(self.W2 * self.W2)
                + np.sum(self.W3 * self.W3)
            )
            total_loss += float(batch_loss + reg)
        return total_loss / batches

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=np.int64)
        logits, _, _, _ = self._forward(X)
        return logits.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if X.size == 0:
            return 0.0
        preds = self.predict(X)
        return float((preds == y).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        logits, _, _, _ = self._forward(X)
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        return exp / exp.sum(axis=1, keepdims=True)


class ResidualRegressor:
    def __init__(self, n_features: int, l2: float = 1e-5):
        self.w = np.zeros((n_features,), dtype=np.float32)
        self.b = 0.0
        self.l2 = l2

    def train_epoch(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        if X.size == 0:
            return 0.0
        logits = X @ self.w + self.b
        preds = 1.0 / (1.0 + np.exp(-logits))
        diff = preds - y
        grad_w = (X.T @ diff) / y.shape[0] + self.l2 * self.w
        grad_b = diff.mean()
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        loss = -np.mean(
            y * np.log(preds + 1e-9) + (1 - y) * np.log(1 - preds + 1e-9)
        )
        loss += 0.5 * self.l2 * float(np.sum(self.w * self.w))
        return float(loss)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=np.float32)
        logits = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-logits))

    def energy(self, X: np.ndarray) -> float:
        if X.size == 0:
            return 0.0
        probs = self.predict_proba(X)
        return float(np.mean(probs))


@dataclass
class ReinforcementStats:
    positive: float = 0.0
    negative: float = 0.0

    def score(self) -> float:
        total = self.positive + self.negative
        if total <= 1e-9:
            return 0.0
        return (self.positive - self.negative) / total


class ReinforcementLedger:
    def __init__(self, scopes: Sequence[int]):
        self.stats: dict[int, ReinforcementStats] = {
            int(scope): ReinforcementStats() for scope in scopes
        }

    def reward(self, scope: int, amount: float) -> None:
        if amount <= 0.0:
            return
        stats = self.stats.setdefault(int(scope), ReinforcementStats())
        stats.positive += float(amount)

    def punish(self, scope: int, amount: float) -> None:
        if amount <= 0.0:
            return
        stats = self.stats.setdefault(int(scope), ReinforcementStats())
        stats.negative += float(amount)

    def score(self, scope: int) -> float:
        stats = self.stats.get(int(scope))
        if stats is None:
            return 0.0
        return stats.score()

    def snapshot(self) -> dict[int, dict[str, float]]:
        return {
            scope: {
                "positive": stats.positive,
                "negative": stats.negative,
                "score": stats.score(),
            }
            for scope, stats in self.stats.items()
        }

    def lr_boosts(self, scale: float) -> dict[int, float]:
        boosts: dict[int, float] = {}
        for scope in self.stats:
            boost = 1.0 + self.score(scope) * scale
            boosts[scope] = max(0.5, min(1.5, boost))
        return boosts

    def decay(self, factor: float) -> None:
        clamped = max(0.0, min(1.0, factor))
        for stats in self.stats.values():
            stats.positive *= clamped
            stats.negative *= clamped


def build_move_datasets(
    games: Iterable[GameRecord],
    horizons: Sequence[int],
    context_window: int,
    context_stride: int,
) -> tuple[
    dict[int, dict[str, list[tuple[tuple[int, ...], str, str, str, MoveFactor]]]],
    dict[int, str],
    dict[int, dict[tuple[int, ...], tuple[str, ...]]],
    dict[str, MoveFactor],
    dict[int, dict[str, Counter[str]]],  # anchor motif counters per horizon
]:
    horizons_sorted = sorted(set(int(h) for h in horizons if h > 0))
    samples: dict[int, dict[str, list[tuple[tuple[int, ...], str, str, str, MoveFactor]]]] = {}
    context_lookup: dict[int, dict[tuple[int, ...], tuple[str, ...]]] = {}
    for horizon in horizons_sorted:
        samples[horizon] = {"train": [], "test": []}
        context_lookup[horizon] = {}
    global_defaults: dict[int, Counter[str]] = {h: Counter() for h in horizons_sorted}
    move_factor_lookup: dict[str, MoveFactor] = {}
    anchor_counters: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    min_horizon = horizons_sorted[0] if horizons_sorted else 1
    for record in tqdm(games, desc="Collecting move windows"):
        moves = record.moves
        hashed = record.hashed_moves
        # Replay all symbol states once per game for motif hashing (discarded after)
        _board = chess.Board()
        _all_sym: list = []
        for _san in moves:
            try:
                _board.push(_board.parse_san(_san))
            except ValueError:
                break
            _all_sym.append(board_symbol_states(_board))
        max_target = len(moves) - min_horizon
        if max_target <= context_window:
            continue
        stop = len(moves) - min_horizon
        for idx in range(context_window, stop, context_stride):
            context_hash = tuple(
                int(bucket) for bucket in hashed[idx - context_window : idx]
            )
            actual_context = moves[idx - context_window : idx]
            for horizon in horizons_sorted:
                target_idx = idx + horizon
                if target_idx >= len(moves):
                    continue
                target = moves[target_idx]
                target_factor = record.move_factors[target_idx]
                move_factor_lookup.setdefault(target, target_factor)
                motif_prev = motif_hash(_all_sym[idx - 1]) if idx - 1 < len(_all_sym) else ""
                motif_next = motif_hash(_all_sym[target_idx - 1]) if target_idx - 1 < len(_all_sym) else ""
                global_defaults[horizon][target] += 1
                key = "train" if record.is_train else "test"
                samples[horizon][key].append((context_hash, target, motif_prev, motif_next, target_factor))
                context_lookup[horizon].setdefault(context_hash, actual_context)
                if motif_prev and (idx % ANCHOR_STRIDE == 0):
                    anchor_counters[horizon][motif_prev][target] += 1
    defaults = {
        horizon: counter.most_common(1)[0][0] if counter else None
        for horizon, counter in global_defaults.items()
    }
    return samples, defaults, context_lookup, move_factor_lookup, anchor_counters


def train_move_models(
    datasets: dict[int, dict[str, list[tuple[tuple[int, ...], str, str, str, MoveFactor]]]],
    defaults: dict[int, str],
    move_factor_lookup: dict[str, MoveFactor],
    anchor_counters: dict[int, dict[str, Counter[str]]],
    prior_topk: int = PRIOR_TOPK_MOTIFS,
) -> tuple[
    dict[int, dict[tuple[int, ...], str]],
    dict[int, dict[tuple[int, ...], Counter[str]]],
    dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]],
    dict[str, Counter[str]],
    dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]],  # factor counters
    dict[str, Counter[str]],  # global factor counters
    dict[int, dict[int, dict[tuple[int, ...], Counter[str]]]],  # cluster context counters
    dict[int, dict[str, Counter[str]]],  # cluster motif globals
    dict[str, MoveFactor],
    dict[int, dict[str, Counter[str]]],  # anchor counters
]:
    models: dict[int, dict[tuple[int, ...], str]] = {}
    counters: dict[int, dict[tuple[int, ...], Counter[str]]] = {}
    motif_counters: dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]] = {}
    motif_global: dict[str, Counter[str]] = defaultdict(Counter)
    factor_counters: dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]] = {}
    factor_global: dict[str, Counter[str]] = defaultdict(Counter)
    cluster_context_counters: dict[int, dict[int, dict[tuple[int, ...], Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    cluster_motif_global: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for horizon, data in datasets.items():
        counter_map: dict[tuple[int, ...], Counter[str]] = defaultdict(Counter)
        motif_map: dict[tuple[int, ...], dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        factor_map: dict[tuple[int, ...], dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for context, target, motif_prev, _motif_next, factor in data["train"]:
            counter_map[context][target] += 1
            if motif_prev:
                motif_map[context][motif_prev][target] += 1
                motif_global[motif_prev][target] += 1
                cluster_id = stable_bucket(motif_prev, PRIOR_CLUSTERS)
                cluster_motif_global[cluster_id][motif_prev][target] += 1
                cluster_context_counters[horizon][cluster_id][context][target] += 1
            # factorized counters
            factor_dict = {
                "role": factor.role,
                "side": factor.side,
                "start": factor.start_bin,
                "end": factor.end_bin,
                "capture": factor.capture,
                "promo": factor.promo,
            }
            for name, val in factor_dict.items():
                factor_map[context][name][val] += 1
                factor_global[name][val] += 1
            move_factor_lookup.setdefault(target, factor)
        counters[horizon] = counter_map
        motif_counters[horizon] = motif_map
        factor_counters[horizon] = factor_map
        models[horizon] = {
            ctx: counts.most_common(1)[0][0]
            for ctx, counts in counter_map.items()
            if counts
        }
        if defaults.get(horizon) is None and data["train"]:
            most_common = Counter(
                target for _, target, _, _ in data["train"]
            ).most_common(1)
            if most_common:
                defaults[horizon] = most_common[0][0]
    # trim motif maps to top-k to reduce noise/compute
    if prior_topk > 0:
        for horizon, context_map in motif_counters.items():
            for context, motif_map in context_map.items():
                for motif_key, cnt in list(motif_map.items()):
                    motif_map[motif_key] = Counter(dict(cnt.most_common(prior_topk)))
        for motif_key, cnt in list(motif_global.items()):
            motif_global[motif_key] = Counter(dict(cnt.most_common(prior_topk)))
        for cluster_id, motif_map in list(cluster_motif_global.items()):
            for motif_key, cnt in list(motif_map.items()):
                cluster_motif_global[cluster_id][motif_key] = Counter(
                    dict(cnt.most_common(prior_topk))
                )
        # trim factor/global
        for name, cnt in list(factor_global.items()):
            factor_global[name] = Counter(dict(cnt.most_common(prior_topk)))
        for horizon, context_map in factor_counters.items():
            for context, factor_map in context_map.items():
                for name, cnt in list(factor_map.items()):
                    factor_map[name] = Counter(dict(cnt.most_common(prior_topk)))

    # trim anchors
    if prior_topk > 0:
        for horizon, motif_map in anchor_counters.items():
            for motif_key, cnt in list(motif_map.items()):
                anchor_counters[horizon][motif_key] = Counter(dict(cnt.most_common(prior_topk)))

    return (
        models,
        counters,
        motif_counters,
        motif_global,
        factor_counters,
        factor_global,
        cluster_context_counters,
        cluster_motif_global,
        move_factor_lookup,
        anchor_counters,
    )


def evaluate_move_models(
    models: dict[int, dict[tuple[int, ...], str]],
    datasets: dict[int, dict[str, list[tuple[tuple[int, ...], str, str, str, MoveFactor]]]],
    defaults: dict[int, str],
    counters: dict[int, dict[tuple[int, ...], Counter[str]]] | None = None,
    motif_counters: dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]] | None = None,
    motif_global: dict[str, Counter[str]] | None = None,
    factor_counters: dict[int, dict[tuple[int, ...], dict[str, Counter[str]]]] | None = None,
    factor_global: dict[str, Counter[str]] | None = None,
    cluster_context: dict[int, dict[int, dict[tuple[int, ...], Counter[str]]]] | None = None,
    cluster_motif: dict[int, dict[str, Counter[str]]] | None = None,
    move_factor_lookup: dict[str, MoveFactor] | None = None,
    anchor_counters: dict[int, dict[str, Counter[str]]] | None = None,
    priors: dict | None = None,
    prior_blend: float = PRIOR_BLEND,
    factor_blend: float = FACTOR_BLEND,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    beam_width: int = BEAM_WIDTH,
    beam_steps: int = BEAM_STEPS,
) -> dict[int, float]:
    accuracies: dict[int, float] = {}
    motif_probs = (priors or {}).get("motif_probs", {})
    transition_probs = (priors or {}).get("transition_probs", {})
    for horizon, data in datasets.items():
        correct = 0
        total = len(data["test"])
        if total == 0:
            accuracies[horizon] = 0.0
            continue
        fallback = defaults.get(horizon)
        motif_map = (motif_counters or {}).get(horizon, {})
        factor_map = (factor_counters or {}).get(horizon, {})
        ctx_counter = (counters or {}).get(horizon, {})
        cluster_ctx = (cluster_context or {}).get(horizon, {})
        anchor_map = (anchor_counters or {}).get(horizon, {})
        for context, target, motif_prev, motif_next, target_factor in data["test"]:
            scores: Counter[str] = Counter()
            candidates: set[str] = set()
            if motif_prev:
                motif_counts = motif_map.get(context, {}).get(motif_prev)
                if motif_counts:
                    for mv, c in motif_counts.items():
                        scores[mv] += c * (1.0 + prior_blend)
                        candidates.add(mv)
                if motif_global:
                    mg = motif_global.get(motif_prev)
                    if mg:
                        for mv, c in mg.items():
                            scores[mv] += c * (0.5 * prior_blend)
                            candidates.add(mv)
                base_p = float(motif_probs.get(motif_prev, 0.0))
                if base_p > PRIOR_EPS:
                    scores[target] += base_p * prior_blend
                if motif_next:
                    trans_key = f"{motif_prev}→{motif_next}"
                    trans_p = float(transition_probs.get(trans_key, 0.0))
                    if trans_p > PRIOR_EPS:
                        scores[target] += trans_p * prior_blend
                cluster_id = stable_bucket(motif_prev, PRIOR_CLUSTERS)
                if cluster_motif:
                    cm = cluster_motif.get(cluster_id, {}).get(motif_prev)
                    if cm:
                        for mv, c in cm.items():
                            scores[mv] += c * (0.5 * prior_blend)
                            candidates.add(mv)
                if cluster_ctx:
                    ctx_counts_cluster = cluster_ctx.get(cluster_id, {}).get(context)
                    if ctx_counts_cluster:
                        for mv, c in ctx_counts_cluster.most_common(3):
                            scores[mv] += c
                            candidates.add(mv)
                if anchor_map:
                    anchor_counts = anchor_map.get(motif_prev)
                    if anchor_counts:
                        for mv, c in anchor_counts.most_common(3):
                            scores[mv] += c * prior_blend
                            candidates.add(mv)
            # context frequency prior
            ctx_counts = ctx_counter.get(context)
            if ctx_counts:
                # Ensure we have a Counter (older checkpoints may store defaultdict)
                if not isinstance(ctx_counts, Counter):
                    ctx_counts = Counter(ctx_counts)
                for mv, c in ctx_counts.most_common(3):
                    scores[mv] += c
                    candidates.add(mv)
            # fallback to context default
            if not scores:
                ctx_best = models.get(horizon, {}).get(context)
                if ctx_best:
                    scores[ctx_best] += 1.0
                    candidates.add(ctx_best)
            if not scores and fallback:
                scores[fallback] += 1.0
                candidates.add(fallback)

            # Factorized priors to re-rank candidates
            if move_factor_lookup and (factor_map or factor_global) and candidates:
                factor_counts_ctx = factor_map.get(context, {}) if factor_map else {}
                for mv in list(candidates):
                    mv_factor = move_factor_lookup.get(mv)
                    if not mv_factor:
                        continue
                    mv_score = 0.0
                    mv_factor_dict = {
                        "role": mv_factor.role,
                        "side": mv_factor.side,
                        "start": mv_factor.start_bin,
                        "end": mv_factor.end_bin,
                        "capture": mv_factor.capture,
                        "promo": mv_factor.promo,
                    }
                    for name, val in mv_factor_dict.items():
                        ctx_factor_counter = factor_counts_ctx.get(name)
                        if ctx_factor_counter:
                            mv_score += math.log1p(ctx_factor_counter.get(val, 0))
                        if factor_global:
                            mv_score += 0.5 * math.log1p(factor_global.get(name, {}).get(val, 0))
                    if mv_score > 0:
                        scores[mv] += mv_score * factor_blend

            # Beam re-rank for longer horizons using horizon-1 counters as a cheap lookahead
            if scores and horizon > 1 and counters:
                base_h = 1 if 1 in counters else min(counters.keys())
                base_counters = counters.get(base_h, {})
                def beam_roll(ctx_hash: tuple[int, ...]) -> Counter[str]:
                    beam: list[tuple[list[int], float, str | None]] = [(list(ctx_hash), 0.0, None)]
                    for _ in range(min(beam_steps, horizon)):
                        new_beam: list[tuple[list[int], float, str | None]] = []
                        for seq, sc, last_mv in beam:
                            ctx_tail = tuple(seq[-context_window:])
                            ccounts = base_counters.get(ctx_tail)
                            if not ccounts:
                                continue
                            for mv, c in ccounts.most_common(beam_width):
                                new_seq = seq + [stable_bucket(mv, FEATURE_DIM)]
                                new_beam.append((new_seq, sc + math.log1p(c), mv))
                        if not new_beam:
                            break
                        new_beam.sort(key=lambda x: x[1], reverse=True)
                        beam = new_beam[:beam_width]
                    leaf_scores: Counter[str] = Counter()
                    for _seq, sc, last_mv in beam:
                        if last_mv:
                            leaf_scores[last_mv] += sc
                    return leaf_scores

                beam_scores = beam_roll(context)
                for mv, sc in beam_scores.items():
                    scores[mv] += sc * 0.5

            pred = scores.most_common(1)[0][0] if scores else fallback
            if pred == target:
                correct += 1
        accuracies[horizon] = correct / total if total else 0.0
    return accuracies


def format_accuracy(name: str, value: float) -> str:
    pct = int(round(value * 100))
    return f"{name}: {pct}%"


def write_run_summary(
    summary_path: Path,
    start_time: float,
    end_time: float,
    iteration: int,
    outcome_acc: dict[int, float],
    move_acc: dict[int, float],
    outcome_energy: dict[int, float],
    log_path: Path,
    priors_path: Path | None,
    max_games: int,
    max_iterations: int,
    max_runtime_minutes: float,
) -> Path:
    def _fmt_ts(ts: float) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))

    lines = [
        f"Run started: {_fmt_ts(start_time)}",
        f"Run finished: {_fmt_ts(end_time)}",
        f"Iterations completed: {iteration}",
        f"Max iterations: {max_iterations}",
        f"Max runtime (minutes): {max_runtime_minutes}",
        f"Games cap: {max_games}",
        f"Log file: {log_path}",
        f"Priors: {priors_path if priors_path else 'none'}",
        "",
        "Outcome accuracy:",
    ]
    for scope, acc in sorted(outcome_acc.items()):
        lines.append(f"  {scope}-ply: {acc:.4f}")
    lines.append("Move accuracy:")
    for horizon, acc in sorted(move_acc.items()):
        lines.append(f"  +{horizon}: {acc:.4f}")
    lines.append("Outcome energy:")
    for scope, energy in sorted(outcome_energy.items()):
        lines.append(f"  {scope}-ply: {energy:.6f}")
    lines.append("")
    lines.append(f"Finished at {lines[1].split(': ',1)[1]}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def try_open_summary(summary_path: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(summary_path)  # type: ignore[attr-defined]
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(summary_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # pragma: no cover - best-effort UX
        print(f"Could not auto-open summary file: {exc}")


class PlyAnimator:
    """
    Background thread that advances through chess game plies at a fixed rate
    (~0.8 s/ply) and writes chess_live_board.json so the viz server shows a
    smoothly animated board regardless of how fast the training loop runs.

    The main training loop updates this object's shared state each iteration;
    the animator thread continuously cycles through plies of the current game.
    """

    PLY_INTERVAL = 0.8  # seconds between ply writes

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._game: Optional[Any] = None
        self._games: List[Any] = []
        self._ply: int = 0
        self._outcome_acc: Dict[int, float] = {}
        self._move_acc: Dict[int, float] = {}
        self._total_games: int = 0
        self._total_plies: int = 0
        self._iteration: int = 0
        self._outcome_models: Optional[Any] = None
        self._move_counters: Optional[Any] = None
        self._move_defaults: Optional[Any] = None
        self._context_window: int = DEFAULT_CONTEXT_WINDOW
        self._stopped: bool = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="PlyAnimator")

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True

    def update(
        self,
        games: List[Any],
        game: Optional[Any],
        outcome_acc: Dict[int, float],
        move_acc: Dict[int, float],
        total_games: int,
        total_plies: int,
        iteration: int,
        outcome_models: Optional[Any] = None,
        move_counters: Optional[Any] = None,
        move_defaults: Optional[Any] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
    ) -> None:
        """Called by the main loop each iteration to push updated state."""
        with self._lock:
            new_game = game is not self._game
            self._games = games
            self._game = game
            if new_game:
                self._ply = 0          # reset ply when game changes
            self._outcome_acc = outcome_acc
            self._move_acc = move_acc
            self._total_games = total_games
            self._total_plies = total_plies
            self._iteration = iteration
            self._outcome_models = outcome_models
            self._move_counters = move_counters
            self._move_defaults = move_defaults
            self._context_window = context_window

    def _run(self) -> None:
        while not self._stopped:
            time.sleep(self.PLY_INTERVAL)
            with self._lock:
                game = self._game
                if game is None:
                    continue
                n_moves = max(1, len(game.moves))
                self._ply = (self._ply + 1) % n_moves
                ply = self._ply
                outcome_acc   = self._outcome_acc
                move_acc      = self._move_acc
                total_games   = self._total_games
                total_plies   = self._total_plies
                iteration     = self._iteration
                outcome_models = self._outcome_models
                move_counters  = self._move_counters
                move_defaults  = self._move_defaults
                context_window = self._context_window
            try:
                write_live_board(
                    iteration, game, ply,
                    outcome_acc, move_acc, total_games,
                    total_plies_actual=total_plies,
                    outcome_models=outcome_models,
                    move_counters=move_counters,
                    move_defaults=move_defaults,
                    context_window=context_window,
                )
            except Exception:
                pass  # never crash the animator; stale viz beats no viz


def _kill_stale_instances() -> None:
    """Kill any other chess_training_loop.py processes before starting."""
    if not _PSUTIL_AVAILABLE:
        return
    my_pid = os.getpid()
    my_script = Path(__file__).name
    for proc in _psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.pid == my_pid:
                continue
            cmdline = " ".join(proc.info.get("cmdline") or [])
            if "python" in proc.info.get("name", "").lower() and my_script in cmdline:
                print(f"  [single-instance] Killing stale {my_script} (PID {proc.pid})", flush=True)
                proc.kill()
        except (_psutil.NoSuchProcess, _psutil.AccessDenied):
            pass


def main() -> None:
    _kill_stale_instances()
    parser = argparse.ArgumentParser(description="Chess outcome + move training loop")
    parser.add_argument("--max-games", type=int, default=8000, help="Game cap")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Number of training iterations (0 = run forever)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.4,
        help="Base learning rate for softmax regressors",
    )
    parser.add_argument(
        "--epochs-per-iteration",
        type=int,
        default=1,
        help="How many gradient epochs per outer iteration",
    )
    parser.add_argument("--seed", type=int, default=7, help="Deterministic seed")
    parser.add_argument(
        "--outcome-scopes",
        type=int,
        nargs="+",
        default=DEFAULT_OUTCOME_SCOPES,
        help="Half-move horizons for outcome models",
    )
    parser.add_argument(
        "--move-horizons",
        type=int,
        nargs="+",
        default=DEFAULT_MOVE_HORIZONS,
        help="Future move horizons to predict",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help="Moves to keep in each context window",
    )
    parser.add_argument(
        "--context-stride",
        type=int,
        default=DEFAULT_CONTEXT_STRIDE,
        help="Stride when sliding move contexts",
    )
    parser.add_argument(
        "--multi-frame-windows",
        type=int,
        default=3,
        help="Temporal windows per scope when building outcome features (>=1)",
    )
    parser.add_argument(
        "--surprise-buffer-size",
        type=int,
        default=2048,
        help="Per-scope buffer size for storing recent mispredictions",
    )
    parser.add_argument(
        "--surprise-sample-size",
        type=int,
        default=256,
        help="Number of train samples to probe for surprises each iteration",
    )
    parser.add_argument(
        "--surprise-batch-size",
        type=int,
        default=256,
        help="Batch size drawn from buffers for focused training",
    )
    parser.add_argument(
        "--residual-sample-size",
        type=int,
        default=512,
        help="Samples per scope for residual (delta) model updates",
    )
    parser.add_argument(
        "--residual-lr-scale",
        type=float,
        default=0.3,
        help="Learning rate multiplier for residual models",
    )
    parser.add_argument(
        "--branch-sample-size",
        type=int,
        default=3,
        help="How many simultaneous futures to log each iteration",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension for the outcome network",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Mini-batch size for training",
    )
    parser.add_argument(
        "--anneal-samples",
        type=int,
        default=48,
        help="Outcome contexts per scope to anneal each iteration",
    )
    parser.add_argument(
        "--anneal-iterations",
        type=int,
        default=40,
        help="Annealing steps when reconciling simultaneous futures",
    )
    parser.add_argument(
        "--anneal-t-start",
        type=float,
        default=3.0,
        help="Starting temperature for annealed futures",
    )
    parser.add_argument(
        "--anneal-t-end",
        type=float,
        default=0.3,
        help="Ending temperature for annealed futures",
    )
    parser.add_argument(
        "--anneal-log-samples",
        type=int,
        default=5,
        help="How many annealed futures to surface in logs",
    )
    parser.add_argument(
        "--reinforcement-scale",
        type=float,
        default=0.4,
        help="Scale factor for positive/negative reinforcement to learning rates",
    )
    parser.add_argument(
        "--reinforcement-decay",
        type=float,
        default=0.85,
        help="Decay factor (0-1) applied to reinforcement ledger each iteration",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to append per-iteration accuracy logs",
    )
    parser.add_argument(
        "--export-annealing",
        type=Path,
        help="Optional path to write a JSON snapshot for the annealing engine after each iteration",
    )
    parser.add_argument(
        "--relational-priors",
        type=Path,
        default=None,
        help="Optional path to relational priors JSON (from build_relational_priors.py)",
    )
    parser.add_argument(
        "--prior-topk",
        type=int,
        default=PRIOR_TOPK_MOTIFS,
        help="Top-k motif counts to retain per context",
    )
    parser.add_argument(
        "--prior-blend",
        type=float,
        default=PRIOR_BLEND,
        help="Blend weight for motif/transition priors in move prediction",
    )
    parser.add_argument(
        "--factor-blend",
        type=float,
        default=FACTOR_BLEND,
        help="Blend weight for factorized role/zone priors in move prediction",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=BEAM_WIDTH,
        help="Beam width for lookahead re-ranking on longer horizons",
    )
    parser.add_argument(
        "--beam-steps",
        type=int,
        default=BEAM_STEPS,
        help="Beam depth (steps) for lookahead re-ranking on longer horizons",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("logs/chess_run_summary.txt"),
        help="Path to write a completion summary (auto-open if possible)",
    )
    parser.add_argument(
        "--max-runtime-minutes",
        type=float,
        default=0.0,
        help="Optional wall-clock runtime cap (minutes). 0 = no cap.",
    )
    args = parser.parse_args()

    outcome_scopes = sorted({int(s) for s in args.outcome_scopes if s > 0})
    move_horizons = sorted({int(h) for h in args.move_horizons if h > 0})
    context_window = max(2, args.context_window)
    context_stride = max(1, args.context_stride)
    multi_frame_windows = max(1, args.multi_frame_windows)
    if not outcome_scopes:
        raise SystemExit("Provide at least one outcome scope > 0")
    if not move_horizons:
        raise SystemExit("Provide at least one move horizon > 0")

    print_runtime_banner(outcome_scopes, move_horizons, context_window, context_stride)

    # ── Node / service bridge (fire-and-forget; never blocks training) ────────
    bridge = NodeBridge(SERVICE_URL, NODE_URL)
    # Probe eagerly so "UP / unreachable" prints before the first iteration.
    _ = bridge.service_available
    _ = bridge.node_available

    all_games = load_games(args.max_games)

    # ── Ply animator (starts immediately so board shows something right away) ──
    animator = PlyAnimator()
    if all_games:
        animator.update(
            all_games, all_games[0], {}, {}, len(all_games), 0, 0
        )
    animator.start()
    print("  [animator] Background ply animator started", flush=True)

    priors = None
    if args.relational_priors:
        if args.relational_priors.exists():
            with args.relational_priors.open("r", encoding="utf-8") as handle:
                priors = json.load(handle)
            print(f"Loaded relational priors from {args.relational_priors}")
        else:
            print(f"Relational priors file not found: {args.relational_priors}, proceeding without priors")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    print(f"Loaded {len(all_games)} games from {DATA_PATH}")
    total_plies_actual = sum(len(g.moves) for g in all_games)
    print(f"Dataset: {len(all_games):,} games | {total_plies_actual:,} total plies")
    # Tracks plies sent to the node in THIS session only (resets to 0 each run).
    # The viz shows this as "Plies seen" so clearing node data shows 0 → growing.
    session_plies_processed = 0

    result_names = {0: "1-0", 1: "1/2-1/2", 2: "0-1"}

    # ── Initial neuro train primer (small, resource-aware) ────────────────────
    # We send only a tiny sample of board positions at startup — enough to give
    # the fabric its first observations before the first training iteration.
    # The per-iteration burst (throttled by ResourceMonitor) handles the bulk.
    # Sending 40,000 snapshots at startup pinned the CPU for 2+ minutes because
    # replay_symbol_states() is called synchronously for every snapshot even
    # when the bridge queue (cap=64) drops >99% of them.
    if bridge.service_available:
        ts_base = int(time.time())
        n_sent = 0
        primer_games = min(5, len(all_games))   # max 5 games × 20 plies = 100 snapshots
        for gi in range(primer_games):
            game = all_games[gi]
            for pi in range(min(NEURO_MAX_PLIES_PER_GAME, len(game.moves))):
                sym_states = game.replay_symbol_states(pi)
                snap = _symbol_states_to_snapshot(
                    sym_states, ts_base + gi * 1000 + pi,
                    result_names.get(game.result_label),
                )
                bridge.send_neuro_train(snap)
                n_sent += 1
                session_plies_processed += 1
        print(f"  [bridge] Queued {n_sent} primer neuro-train snapshots ({primer_games} games)", flush=True)

    # ── Initial QA primer (small, resource-aware) ─────────────────────────────
    # Same reasoning: cap to 10 games / 200 examples at startup; per-iteration
    # bursts in the main loop grow this organically as resources allow.
    if bridge.node_available:
        n_qa = 0
        qa_primer_games = min(10, len(all_games))
        for game in all_games[:qa_primer_games]:
            for pi, move in enumerate(game.moves[:-1]):
                context = list(game.moves[max(0, pi - 5):pi])
                answer  = game.moves[pi]
                question = f"chess_move:{' '.join(context)}"
                bridge.send_qa_ingest(question, answer, context)
                n_qa += 1
        print(f"  [bridge] Queued {n_qa} primer QA examples ({qa_primer_games} games)", flush=True)

    outcome_data = build_outcome_features(all_games, outcome_scopes, multi_frame_windows)
    move_datasets, move_defaults, context_lookup, move_factor_lookup, anchor_counters = build_move_datasets(
        all_games, move_horizons, context_window, context_stride
    )
    move_models, move_counters, move_motif_counters, move_motif_global, factor_counters, factor_global, cluster_context_counters, cluster_motif_global, factor_lookup, anchor_counters = train_move_models(
        move_datasets, move_defaults, move_factor_lookup, anchor_counters, prior_topk=args.prior_topk
    )

    for scope in outcome_scopes:
        train_count = outcome_data[scope]["train_X"].shape[0]
        test_count = outcome_data[scope]["test_X"].shape[0]
        print(
            f"Outcome scope {scope:>2} ply -> train {train_count:,} | test {test_count:,} examples"
        )
    for horizon in move_horizons:
        train_size = len(move_datasets[horizon]["train"])
        test_size = len(move_datasets[horizon]["test"])
        print(
            f"Move horizon +{horizon:<2} -> train {train_size:,} | test {test_size:,} contexts"
        )

    outcome_models = {
        scope: OutcomeNetwork(TOTAL_FEATURE_DIM, len(RESULT_TO_LABEL), args.hidden_dim)
        for scope in outcome_scopes
    }
    residual_models = {
        scope: ResidualRegressor(TOTAL_FEATURE_DIM)
        for scope in outcome_scopes
    }
    surprise_buffers = init_surprise_buffers(outcome_scopes, args.surprise_buffer_size)
    ledger = ReinforcementLedger(outcome_scopes)
    scope_lr_boosts: dict[int, float] = {scope: 1.0 for scope in outcome_scopes}
    prev_outcome_acc: dict[int, float] = {}

    result_names = {0: "1-0", 1: "1/2-1/2", 2: "0-1"}
    monitor = ResourceMonitor()
    run_start = time.time()
    max_runtime_sec = max(0.0, args.max_runtime_minutes * 60.0)
    iteration = 0
    log_path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    last_outcome_acc: dict[int, float] = {}
    last_move_acc: dict[int, float] = {}
    last_outcome_energy: dict[int, float] = {}
    while True:
        iteration += 1
        scale = monitor.poll()
        effective_batch = monitor.batch_size(args.batch_size)
        start = time.time()
        print(f"\n{monitor.status_line()}", flush=True)
        for _ in range(args.epochs_per_iteration):
            base_lr = args.lr / (1.0 + 0.05 * max(iteration - 1, 0))
            for scope in outcome_scopes:
                dataset = outcome_data[scope]
                scope_lr = base_lr * scope_lr_boosts.get(scope, 1.0)
                _ = outcome_models[scope].train_epoch(
                    dataset["train_X"], dataset["train_y"], scope_lr, effective_batch
                )
            harvest_surprises(
                rng,
                outcome_models,
                outcome_data,
                surprise_buffers,
                args.surprise_sample_size,
            )
            for scope, buffer in surprise_buffers.items():
                maybe_batch = sample_buffer(rng, buffer, args.surprise_batch_size)
                if maybe_batch is None:
                    continue
                buf_X, buf_y = maybe_batch
                scope_lr = base_lr * scope_lr_boosts.get(scope, 1.0)
                _ = outcome_models[scope].train_epoch(
                    buf_X,
                    buf_y,
                    scope_lr * 1.25,
                    min(effective_batch, buf_X.shape[0]),
                )
        update_residual_models(
            rng,
            outcome_models,
            residual_models,
            outcome_data,
            args.residual_sample_size,
            args.lr * args.residual_lr_scale,
        )
        outcome_acc = {
            scope: outcome_models[scope].accuracy(
                outcome_data[scope]["test_X"], outcome_data[scope]["test_y"]
            )
            for scope in outcome_scopes
        }
        outcome_energy = {
            scope: residual_models[scope].energy(outcome_data[scope]["test_X"])
            for scope in outcome_scopes
        }
        move_acc = evaluate_move_models(
        move_models,
        move_datasets,
        move_defaults,
        move_counters,
        move_motif_counters,
        move_motif_global,
        factor_counters,
        factor_global,
        cluster_context_counters,
        cluster_motif_global,
        move_factor_lookup=factor_lookup,
        anchor_counters=anchor_counters,
        priors=priors,
        prior_blend=args.prior_blend,
        factor_blend=args.factor_blend,
        context_window=context_window,
        beam_width=args.beam_width,
        beam_steps=args.beam_steps,
    )
        future_samples = sample_future_branches(
            rng,
            move_horizons,
            move_datasets,
            context_lookup,
            move_counters,
            args.branch_sample_size,
        )
        annealed_samples, annealed_stats = anneal_future_predictions(
            rng,
            outcome_models,
            residual_models,
            outcome_data,
            args.anneal_samples,
            args.anneal_iterations,
            args.anneal_t_start,
            args.anneal_t_end,
            args.anneal_log_samples,
        )
        duration = time.time() - start
        for scope, acc in outcome_acc.items():
            prev = prev_outcome_acc.get(scope)
            if prev is not None:
                delta = acc - prev
                if delta > 0:
                    ledger.reward(scope, delta)
                elif delta < 0:
                    ledger.punish(scope, -delta)
            prev_outcome_acc[scope] = acc
        for scope, scope_stats in annealed_stats.items():
            ledger.reward(scope, scope_stats.get("reward", 0.0))
            ledger.punish(scope, scope_stats.get("penalty", 0.0))
        reinforcement_view = ledger.snapshot()
        scope_lr_boosts = ledger.lr_boosts(args.reinforcement_scale)
        ledger.decay(args.reinforcement_decay)
        outcome_summary = ", ".join(
            format_accuracy(f"{scope}-ply", acc) for scope, acc in outcome_acc.items()
        )
        move_summary = ", ".join(
            format_accuracy(f"+{h}", acc) for h, acc in move_acc.items()
        )
        energy_summary = ", ".join(
            f"{scope}-ply:{energy:.3f}" for scope, energy in outcome_energy.items()
        )
        reinforcement_summary = ", ".join(
            f"{scope}-ply:{reinforcement_view.get(scope, {}).get('score', 0.0):+.2f}"
            for scope in outcome_scopes
            if scope in reinforcement_view
        )
        if not reinforcement_summary:
            reinforcement_summary = "n/a"
        print(
            f"[iter {iteration}] "
            f"Outcome scopes -> {outcome_summary} | "
            f"Energy -> {energy_summary} | "
            f"Move horizons -> {move_summary} | "
            f"Reinforcement -> {reinforcement_summary} "
            f"(took {duration:.2f}s)"
        )
        log_record = {
            "iteration": iteration,
            "timestamp": time.time(),
            "duration_sec": duration,
            "outcome_accuracy": outcome_acc,
            "outcome_energy": outcome_energy,
            "move_accuracy": move_acc,
            "future_samples": future_samples,
            "annealed_future_samples": annealed_samples,
            "reinforcement": reinforcement_view,
            "lr_boosts": scope_lr_boosts,
            "max_games": args.max_games,
        }
        log_handle.write(json.dumps(log_record) + "\n")
        log_handle.flush()

        last_outcome_acc = outcome_acc
        last_move_acc = move_acc
        last_outcome_energy = outcome_energy

        # ── Node feedback loop ────────────────────────────────────────────────
        # 1. Update the ply animator with latest stats.  The animator runs in a
        #    background thread and writes chess_live_board.json every 0.8 s,
        #    cycling through plies independently of training speed.
        sample_game = all_games[iteration % len(all_games)] if all_games else None
        animator.update(
            all_games, sample_game,
            outcome_acc, move_acc, len(all_games), session_plies_processed, iteration,
            outcome_models=outcome_models,
            move_counters=move_counters,
            move_defaults=move_defaults,
            context_window=context_window,
        )
        # Also write immediately so stats are visible right after the iteration
        # (the animator will advance the ply from here on its own schedule).
        sample_ply = 0 if sample_game is None else (iteration * 7) % max(1, len(sample_game.moves))
        write_live_board(
            iteration, sample_game, sample_ply, outcome_acc, move_acc, len(all_games),
            total_plies_actual=session_plies_processed,
            outcome_models=outcome_models,
            move_counters=move_counters,
            move_defaults=move_defaults,
            context_window=context_window,
        )

        # 2. Every other iteration send a fresh neuro-train burst using a
        #    random sample of games so the fabric continues accumulating
        #    patterns as accuracy improves.
        #    ResourceMonitor.game_count() already scales this down under pressure;
        #    cap plies per game to 5 (was 20) to halve the replay work.
        if bridge.service_available and iteration % 2 == 0:
            ts_now = int(time.time())
            # game_count(16) = monitor-scaled; at scale=1.0 → 16 games × 5 plies = 80 snapshots
            # (was 32 games × 20 plies = 640 replay_symbol_states calls per burst)
            neuro_sample_n = monitor.game_count(16)
            neuro_plies    = 5  # 5 plies max per game (was NEURO_MAX_PLIES_PER_GAME=20)
            sample_indices = rng.sample(range(len(all_games)), min(neuro_sample_n, len(all_games)))
            for gi in sample_indices:
                game = all_games[gi]
                for pi in range(min(neuro_plies, len(game.moves))):
                    sym_states = game.replay_symbol_states(pi)
                    snap = _symbol_states_to_snapshot(
                        sym_states, ts_now + gi * 1000 + pi,
                        result_names.get(game.result_label),
                    )
                    bridge.send_neuro_train(snap)
                    session_plies_processed += 1

        # ── Virtual-sensor episode recording ─────────────────────────────────
        # Feed prediction outcomes into the fabric's episodic store so the
        # neural fabric learns from chess model right/wrong calls the same way
        # hardware sensors do — no per-domain neuroscience knowledge required.
        #
        # For each outcome scope, sample a small batch of test examples, run
        # the model, and record each resolved episode.  Focal-loss surprise:
        #   correct:  (1 - p_correct)^2  → small gradient for easy wins
        #   wrong:    p_confidence^2     → large gradient for confident misses
        if bridge.service_available and outcome_models and outcome_data:
            episode_batch_size = max(1, monitor.game_count(8))
            for scope in outcome_scopes:
                test_X = outcome_data[scope]["test_X"]
                test_y = outcome_data[scope]["test_y"]
                if test_X.shape[0] == 0:
                    continue
                batch_idx = rng.sample(
                    range(test_X.shape[0]),
                    min(episode_batch_size, test_X.shape[0]),
                )
                bX = test_X[batch_idx]
                by = test_y[batch_idx]
                probs_batch = outcome_models[scope].predict_proba(bX)  # (B, 3)
                preds_batch = probs_batch.argmax(axis=1)
                for i, (pred_label, true_label) in enumerate(zip(preds_batch, by)):
                    p_confidence = float(probs_batch[i, pred_label])
                    p_correct = float(probs_batch[i, true_label])
                    correct = pred_label == true_label
                    surprise = (
                        float((1.0 - p_correct) ** 2) if correct
                        else float(p_confidence ** 2)
                    )
                    predicted_str = result_names.get(int(pred_label), "?")
                    actual_str = result_names.get(int(true_label), "?")
                    context_labels = [
                        f"chess::scope::{scope}",
                        f"chess::outcome::{predicted_str.replace('/', '_')}",
                        f"chess::ply_bucket::{scope // 10 * 10}",
                    ]
                    bridge.send_record_episode(
                        context_labels=context_labels,
                        predicted=f"outcome::{predicted_str}",
                        actual=f"outcome::{actual_str}",
                        streams=["chess"],
                        surprise=surprise,
                    )

        if args.export_annealing:
            export_snapshot(
                args.export_annealing,
                iteration,
                all_games,
                outcome_models,
                outcome_scopes,
                move_models,
                move_horizons,
                rng,
            )

        # Adaptive sleep — longer when under pressure so the OS breathes
        idle = monitor.sleep_secs(0.5)
        if idle > 0.0:
            time.sleep(idle)

        should_stop = 0 < args.max_iterations <= iteration
        if not should_stop and max_runtime_sec > 0:
            should_stop = (time.time() - run_start) >= max_runtime_sec
        if should_stop:
            break

    log_handle.close()
    run_end = time.time()
    summary_path = write_run_summary(
        args.summary_file,
        run_start,
        run_end,
        iteration,
        last_outcome_acc,
        last_move_acc,
        last_outcome_energy,
        log_path,
        args.relational_priors,
        args.max_games,
        args.max_iterations,
        args.max_runtime_minutes,
    )
    print(f"Wrote summary to {summary_path}")
    try_open_summary(summary_path)
@dataclass
class SurpriseSample:
    features: np.ndarray
    label: int


def init_surprise_buffers(scopes: Sequence[int], maxlen: int) -> dict[int, deque[SurpriseSample]]:
    return {scope: deque(maxlen=maxlen) for scope in scopes}


def harvest_surprises(
    rng: random.Random,
    models: dict[int, "OutcomeNetwork"],
    datasets: dict[int, dict[str, np.ndarray]],
    buffers: dict[int, deque[SurpriseSample]],
    per_scope_samples: int,
) -> None:
    for scope, model in models.items():
        buffer = buffers.get(scope)
        if buffer is None:
            continue
        train_X = datasets[scope]["train_X"]
        train_y = datasets[scope]["train_y"]
        if train_X.shape[0] == 0 or per_scope_samples <= 0:
            continue
        idxs = rng.sample(
            range(train_X.shape[0]),
            min(per_scope_samples, train_X.shape[0]),
        )
        subset_X = train_X[idxs]
        subset_y = train_y[idxs]
        preds = model.predict(subset_X)
        mis_mask = preds != subset_y
        if not mis_mask.any():
            continue
        for feats, label in zip(subset_X[mis_mask], subset_y[mis_mask]):
            buffer.append(SurpriseSample(features=feats.copy(), label=int(label)))


def sample_buffer(
    rng: random.Random,
    buffer: deque[SurpriseSample],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not buffer:
        return None
    batch = rng.sample(list(buffer), min(batch_size, len(buffer)))
    X = np.stack([item.features for item in batch], axis=0)
    y = np.array([item.label for item in batch], dtype=np.int64)
    return X, y


def update_residual_models(
    rng: random.Random,
    outcome_models: dict[int, OutcomeNetwork],
    residual_models: dict[int, ResidualRegressor],
    datasets: dict[int, dict[str, np.ndarray]],
    sample_size: int,
    lr_scale: float,
) -> None:
    for scope, model in outcome_models.items():
        residual = residual_models.get(scope)
        if residual is None:
            continue
        train_X = datasets[scope]["train_X"]
        train_y = datasets[scope]["train_y"]
        if train_X.shape[0] == 0:
            continue
        idxs = rng.sample(
            range(train_X.shape[0]),
            min(sample_size, train_X.shape[0]),
        )
        subset_X = train_X[idxs]
        subset_y = train_y[idxs]
        preds = model.predict(subset_X)
        errors = (preds != subset_y).astype(np.float32)
        residual.train_epoch(subset_X, errors, lr_scale)


def sample_future_branches(
    rng: random.Random,
    horizons: Sequence[int],
    datasets: dict[int, dict[str, list[tuple[tuple[int, ...], str, str, str, MoveFactor]]]],
    context_lookup: dict[int, dict[tuple[int, ...], tuple[str, ...]]],
    move_counters: dict[int, dict[tuple[int, ...], Counter[str]]],
    sample_size: int,
) -> list[dict]:
    samples: list[dict] = []
    if sample_size <= 0:
        return samples
    horizon_list = [h for h in horizons if datasets[h]["test"]]
    if not horizon_list:
        return samples
    for _ in range(sample_size):
        horizon = rng.choice(horizon_list)
        context_entry = rng.choice(datasets[horizon]["test"])
        context_hash, target, motif_prev, motif_next, target_factor = context_entry
        context_moves = context_lookup[horizon].get(
            context_hash, tuple(f"#{idx}" for idx in range(len(context_hash)))
        )
        counter = move_counters.get(horizon, {}).get(context_hash, Counter())
        top_preds = [move for move, _ in counter.most_common(3)]
        samples.append(
            {
                "horizon": horizon,
                "context": context_moves,
                "top_predictions": top_preds,
                "actual": target,
                "motif_prev": motif_prev,
                "motif_next": motif_next,
                "factor": {
                    "role": target_factor.role,
                    "side": target_factor.side,
                    "start_bin": target_factor.start_bin,
                    "end_bin": target_factor.end_bin,
                    "capture": target_factor.capture,
                    "promo": target_factor.promo,
                },
            }
        )
    return samples


def _energy_for_label(
    label_idx: int,
    probs: np.ndarray,
    ranks: dict[int, int],
    residual_penalty: float,
) -> float:
    confidence = float(probs[label_idx])
    rank_penalty = 0.05 * ranks.get(label_idx, 0)
    return (1.0 - confidence) + residual_penalty + rank_penalty


def anneal_future_predictions(
    rng: random.Random,
    outcome_models: dict[int, OutcomeNetwork],
    residual_models: dict[int, ResidualRegressor],
    datasets: dict[int, dict[str, np.ndarray]],
    sample_size: int,
    iterations: int,
    t_start: float,
    t_end: float,
    log_limit: int,
) -> tuple[list[dict], dict[int, dict[str, float]]]:
    samples: list[dict] = []
    stats: dict[int, dict[str, float]] = {}
    if sample_size <= 0 or iterations <= 0:
        return samples, stats
    for scope, model in outcome_models.items():
        test_X = datasets.get(scope, {}).get("test_X")
        test_y = datasets.get(scope, {}).get("test_y")
        if test_X is None or test_y is None or test_X.shape[0] == 0:
            continue
        total = test_X.shape[0]
        if total == 0:
            continue
        count = min(sample_size, total)
        if count <= 0:
            continue
        idxs = (
            rng.sample(range(total), count) if count < total else list(range(total))
        )
        particles = []
        residual_model = residual_models.get(scope)
        for idx in idxs:
            x = test_X[idx : idx + 1]
            probs = model.predict_proba(x)[0]
            if residual_model is not None:
                residual_penalty = float(residual_model.predict_proba(x)[0])
            else:
                residual_penalty = 0.0
            order = np.argsort(probs)[::-1]
            ranks = {int(label): int(rank) for rank, label in enumerate(order)}
            n_classes = probs.shape[0]
            candidate = rng.choices(range(n_classes), weights=probs, k=1)[0]
            energy = _energy_for_label(candidate, probs, ranks, residual_penalty)
            particles.append(
                {
                    "idx": idx,
                    "label": candidate,
                    "best_label": candidate,
                    "energy": energy,
                    "best_energy": energy,
                    "probs": probs,
                    "ranks": ranks,
                    "penalty": residual_penalty,
                    "actual": int(test_y[idx]),
                    "n_classes": n_classes,
                }
            )
        if not particles:
            continue
        steps = max(1, iterations)
        for step in range(steps):
            progress = step / max(1, steps - 1)
            temperature = t_start + (t_end - t_start) * progress
            for particle in particles:
                candidate = rng.randrange(particle["n_classes"])
                candidate_energy = _energy_for_label(
                    candidate, particle["probs"], particle["ranks"], particle["penalty"]
                )
                if candidate_energy < particle["energy"]:
                    accept = True
                else:
                    delta = particle["energy"] - candidate_energy
                    accept_prob = math.exp(delta / max(temperature, 1e-3))
                    accept = rng.random() < accept_prob
                if accept:
                    particle["label"] = candidate
                    particle["energy"] = candidate_energy
                if candidate_energy < particle["best_energy"]:
                    particle["best_label"] = candidate
                    particle["best_energy"] = candidate_energy
        scope_stats = stats.setdefault(
            scope, {"reward": 0.0, "penalty": 0.0, "samples": 0.0}
        )
        for particle in particles:
            actual = particle["actual"]
            predicted = particle["best_label"]
            energy = particle["best_energy"]
            confidence = float(particle["probs"][predicted])
            reward = max(0.0, 1.0 - energy)
            if predicted == actual:
                scope_stats["reward"] += reward
            else:
                scope_stats["penalty"] += energy
            scope_stats["samples"] += 1.0
            if len(samples) < log_limit:
                samples.append(
                    {
                        "scope": scope,
                        "actual": LABEL_TO_RESULT.get(actual, str(actual)),
                        "predicted": LABEL_TO_RESULT.get(predicted, str(predicted)),
                        "energy": energy,
                        "confidence": confidence,
                        "residual_penalty": particle["penalty"],
                    }
                )
    return samples, stats


def export_snapshot(
    path: Path,
    iteration: int,
    games: list[GameRecord],
    outcome_models: dict[int, OutcomeNetwork],
    scopes: Sequence[int],
    move_models: dict[int, dict[tuple[int, ...], str]],
    horizons: Sequence[int],
    rng: random.Random,
) -> None:
    sample = rng.choice(games)
    scoped_states = []
    for scope in scopes:
        if sample.hashed_moves.shape[0] < scope:
            continue
        _export_vecs = sample.replay_board_vectors(scope)
        _export_bvec = _export_vecs[scope - 1] if scope - 1 < len(_export_vecs) else np.zeros(BOARD_FEATURE_DIM, dtype=np.float32)
        vec = compose_prefix_vector(
            np.bincount(sample.hashed_moves[:scope], minlength=FEATURE_DIM).astype(np.float32),
            scope,
            sample.meta_vector,
            _export_bvec,
        )
        logits, _, _, _ = outcome_models[scope]._forward(vec[None, :])
        probs = np.exp(logits - logits.max()) / np.exp(logits - logits.max()).sum()
        scoped_states.append(
            {
                "scope": scope,
                "probabilities": probs.flatten().tolist(),
                "label_map": LABEL_TO_RESULT,
            }
        )
    move_examples = []
    for horizon in horizons:
        contexts = move_models.get(horizon)
        if not contexts:
            continue
        context, prediction = rng.choice(list(contexts.items()))
        move_examples.append(
            {
                "horizon": horizon,
                "context_hash": list(context),
                "predicted_move": prediction,
            }
        )
    payload = {
        "iteration": iteration,
        "timestamp": time.time(),
        "sample_game": sample.game_id,
        "outcome_states": scoped_states,
        "move_examples": move_examples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

if __name__ == "__main__":
    main()
