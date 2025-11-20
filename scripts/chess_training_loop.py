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
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "chess" / "processed_games.jsonl"
DEFAULT_LOG_PATH = ROOT / "logs" / "chess_training_metrics.log"
FEATURE_DIM = 512  # hashed bag-of-move features
METADATA_DIM = 128  # hashed player/opening features
TOTAL_FEATURE_DIM = FEATURE_DIM + METADATA_DIM
OUTCOME_SCOPES = [6, 10, 14, 18, 22, 26, 30, 40]  # number of half-moves observed
MOVE_HORIZONS = [1, 5, 10, 15, 20]  # moves to predict ahead
CONTEXT_WINDOW = 6  # use last N moves as context for move prediction
CONTEXT_STRIDE = 2  # step when iterating contexts to keep volume manageable
RESULT_TO_LABEL = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
LABEL_TO_RESULT = {0: "White win", 1: "Draw", 2: "Black win"}


def stable_bucket(text: str, mod: int) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % mod


def deterministic_split(identifier: str) -> bool:
    """Returns True if record goes to train split."""
    digest = hashlib.blake2s(identifier.encode("utf-8"), digest_size=2).digest()
    return digest[0] % 5 != 0  # ~80/20 split


@dataclass(frozen=True)
class GameRecord:
    game_id: str
    result_label: int
    moves: tuple[str, ...]
    hashed_moves: np.ndarray  # shape (len(moves),)
    is_train: bool
    meta_vector: np.ndarray  # shape (METADATA_DIM,)


def load_games(limit: int | None) -> list[GameRecord]:
    if not DATA_PATH.exists():
        raise SystemExit(
            "Missing processed dataset. Run scripts/preprocess_chess_games.py first."
        )
    records: list[GameRecord] = []
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if limit is not None and len(records) >= limit:
                break
            payload = json.loads(line)
            result = payload["result"]
            if result not in RESULT_TO_LABEL:
                continue
            moves = tuple(payload["moves"])
            if len(moves) < 6:  # skip extremely short games
                continue
            hashed = np.fromiter(
                (stable_bucket(mv, FEATURE_DIM) for mv in moves),
                dtype=np.int16,
                count=len(moves),
            )
            tokens = metadata_tokens(payload, len(moves))
            meta_vec = metadata_vector(tokens)
            record = GameRecord(
                game_id=payload["id"],
                result_label=RESULT_TO_LABEL[result],
                moves=moves,
                hashed_moves=hashed,
                is_train=deterministic_split(payload["id"]),
                meta_vector=meta_vec,
            )
            records.append(record)
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
    vec /= len(tokens)
    return vec


def prefix_vector(record: GameRecord, upto: int) -> np.ndarray:
    upto = min(upto, record.hashed_moves.shape[0])
    if upto <= 0:
        move_hist = np.zeros(FEATURE_DIM, dtype=np.float32)
    else:
        move_hist = np.bincount(
            record.hashed_moves[:upto], minlength=FEATURE_DIM
        ).astype(np.float32)
        move_hist /= np.sqrt(float(upto))
    return np.concatenate([move_hist, record.meta_vector], axis=0)


def build_outcome_features(
    games: Iterable[GameRecord],
    scopes: list[int],
) -> dict[int, dict[str, np.ndarray]]:
    datasets: dict[int, dict[str, np.ndarray]] = {}
    for scope in scopes:
        datasets[scope] = {
            "train_X": [],
            "train_y": [],
            "test_X": [],
            "test_y": [],
        }
    for record in tqdm(games, desc="Featurizing outcomes"):
        for scope in scopes:
            if record.hashed_moves.shape[0] < scope:
                continue
            vec = prefix_vector(record, scope)
            key = "train" if record.is_train else "test"
            datasets[scope][f"{key}_X"].append(vec)
            datasets[scope][f"{key}_y"].append(record.result_label)
    # convert to numpy arrays
    for scope, blob in datasets.items():
        for split in ("train", "test"):
            xs = blob[f"{split}_X"]
            ys = blob[f"{split}_y"]
            if xs:
                blob[f"{split}_X"] = np.stack(xs, axis=0)
                blob[f"{split}_y"] = np.array(ys, dtype=np.int64)
            else:
                blob[f"{split}_X"] = np.zeros((0, FEATURE_DIM), dtype=np.float32)
                blob[f"{split}_y"] = np.zeros((0,), dtype=np.int64)
    return datasets


class SoftmaxRegressor:
    def __init__(self, n_features: int, n_classes: int, l2: float = 1e-4):
        self.W = np.zeros((n_features, n_classes), dtype=np.float32)
        self.b = np.zeros((n_classes,), dtype=np.float32)
        self.l2 = l2

    def train_epoch(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        if X.size == 0:
            return 0.0
        logits = X @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits, dtype=np.float32)
        probs = exp / exp.sum(axis=1, keepdims=True)
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(y.shape[0]), y] = 1.0
        diff = probs - y_onehot
        grad_W = (X.T @ diff) / y.shape[0] + self.l2 * self.W
        grad_b = diff.mean(axis=0)
        self.W -= lr * grad_W
        self.b -= lr * grad_b
        loss = -np.log(probs[np.arange(y.shape[0]), y] + 1e-9).mean()
        loss += 0.5 * self.l2 * np.sum(self.W * self.W)
        return float(loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros((0,), dtype=np.int64)
        logits = X @ self.W + self.b
        return logits.argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        if X.size == 0:
            return 0.0
        preds = self.predict(X)
        return float((preds == y).mean())


def build_move_datasets(
    games: Iterable[GameRecord],
    horizons: list[int],
) -> tuple[
    dict[int, dict[str, list[tuple[tuple[str, ...], str]]]],
    dict[int, str],
]:
    samples: dict[int, dict[str, list[tuple[tuple[str, ...], str]]]] = {}
    for horizon in horizons:
        samples[horizon] = {"train": [], "test": []}
    global_defaults: dict[int, Counter[str]] = {h: Counter() for h in horizons}
    for record in tqdm(games, desc="Collecting move windows"):
        moves = record.moves
        max_target = len(moves) - min(horizons)
        if max_target <= CONTEXT_WINDOW:
            continue
        for idx in range(CONTEXT_WINDOW, len(moves) - min(horizons), CONTEXT_STRIDE):
            context = moves[idx - CONTEXT_WINDOW : idx]
            for horizon in horizons:
                target_idx = idx + horizon
                if target_idx >= len(moves):
                    continue
                target = moves[target_idx]
                global_defaults[horizon][target] += 1
                key = "train" if record.is_train else "test"
                samples[horizon][key].append((context, target))
    defaults = {
        horizon: counter.most_common(1)[0][0] if counter else None
        for horizon, counter in global_defaults.items()
    }
    return samples, defaults


def train_move_models(
    datasets: dict[int, dict[str, list[tuple[tuple[str, ...], str]]]],
    defaults: dict[int, str],
) -> dict[int, dict[tuple[str, ...], str]]:
    models: dict[int, dict[tuple[str, ...], str]] = {}
    for horizon, data in datasets.items():
        counter_map: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        for context, target in data["train"]:
            counter_map[context][target] += 1
        models[horizon] = {
            ctx: counts.most_common(1)[0][0]
            for ctx, counts in counter_map.items()
            if counts
        }
        if defaults.get(horizon) is None and data["train"]:
            # fallback to absolute majority from seen contexts
            most_common = Counter(
                target for _, target in data["train"]
            ).most_common(1)
            if most_common:
                defaults[horizon] = most_common[0][0]
    return models


def evaluate_move_models(
    models: dict[int, dict[tuple[str, ...], str]],
    datasets: dict[int, dict[str, list[tuple[tuple[str, ...], str]]]],
    defaults: dict[int, str],
) -> dict[int, float]:
    accuracies: dict[int, float] = {}
    for horizon, data in datasets.items():
        correct = 0
        total = len(data["test"])
        if total == 0:
            accuracies[horizon] = 0.0
            continue
        fallback = defaults.get(horizon)
        for context, target in data["test"]:
            pred = models[horizon].get(context, fallback)
            if pred == target:
                correct += 1
        accuracies[horizon] = correct / total if total else 0.0
    return accuracies


def format_accuracy(name: str, value: float) -> str:
    pct = int(round(value * 100))
    return f"{name}: {pct}%"


def main() -> None:
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
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help="Path to append per-iteration accuracy logs",
    )
    args = parser.parse_args()

    all_games = load_games(args.max_games)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    print(f"Loaded {len(all_games)} games from {DATA_PATH}")

    outcome_data = build_outcome_features(all_games, OUTCOME_SCOPES)
    move_datasets, move_defaults = build_move_datasets(all_games, MOVE_HORIZONS)
    move_models = train_move_models(move_datasets, move_defaults)

    outcome_models = {
        scope: SoftmaxRegressor(TOTAL_FEATURE_DIM, len(RESULT_TO_LABEL))
        for scope in OUTCOME_SCOPES
    }

    iteration = 0
    log_path = args.log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    while True:
        iteration += 1
        start = time.time()
        for _ in range(args.epochs_per_iteration):
            for scope in OUTCOME_SCOPES:
                dataset = outcome_data[scope]
                lr = args.lr / (1.0 + 0.05 * (iteration - 1))
                loss = outcome_models[scope].train_epoch(dataset["train_X"], dataset["train_y"], lr)
        outcome_acc = {
            scope: outcome_models[scope].accuracy(
                outcome_data[scope]["test_X"], outcome_data[scope]["test_y"]
            )
            for scope in OUTCOME_SCOPES
        }
        move_acc = evaluate_move_models(move_models, move_datasets, move_defaults)
        duration = time.time() - start
        outcome_summary = ", ".join(
            format_accuracy(f"{scope}-ply", acc) for scope, acc in outcome_acc.items()
        )
        move_summary = ", ".join(
            format_accuracy(f"+{h}", acc) for h, acc in move_acc.items()
        )
        print(
            f"[iter {iteration}] "
            f"Outcome scopes -> {outcome_summary} | "
            f"Move horizons -> {move_summary} "
            f"(took {duration:.2f}s)"
        )
        log_record = {
            "iteration": iteration,
            "timestamp": time.time(),
            "duration_sec": duration,
            "outcome_accuracy": outcome_acc,
            "move_accuracy": move_acc,
            "max_games": args.max_games,
        }
        log_handle.write(json.dumps(log_record) + "\n")
        log_handle.flush()

        if 0 < args.max_iterations <= iteration:
            break


if __name__ == "__main__":
    main()
