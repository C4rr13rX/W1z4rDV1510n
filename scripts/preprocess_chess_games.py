#!/usr/bin/env python3
"""
Parse downloaded PGN files (from data/chess/*/*.pgn) and emit a compact JSONL
dataset with per-game move lists + final results. The JSONL file is stored at
data/chess/processed_games.jsonl and is deterministic across runs so downstream
training scripts can consume it repeatedly.
"""

import argparse
import json
from pathlib import Path
from typing import Iterator

import chess.pgn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "chess"
OUTPUT_PATH = DATA_DIR / "processed_games.jsonl"


def iter_pgn_files() -> Iterator[Path]:
    for path in sorted(DATA_DIR.rglob("*.pgn")):
        yield path


def parse_games(max_games: int | None = None) -> Iterator[dict]:
    count = 0
    for path in iter_pgn_files():
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            game_idx = 0
            while True:
                if max_games is not None and count >= max_games:
                    return
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                result = (game.headers.get("Result") or "").strip()
                if result not in {"1-0", "0-1", "1/2-1/2"}:
                    continue
                board = game.board()
                moves: list[str] = []
                for move in game.mainline_moves():
                    moves.append(board.san(move))
                    board.push(move)
                if not moves:
                    continue
                yield {
                    "id": f"{path.stem}-{game_idx}",
                    "result": result,
                    "moves": moves,
                    "white": game.headers.get("White", "").strip(),
                    "black": game.headers.get("Black", "").strip(),
                    "eco": game.headers.get("ECO", "").strip(),
                    "opening": game.headers.get("Opening", "").strip(),
                }
                game_idx += 1
                count += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PGNs to JSONL dataset")
    parser.add_argument(
        "--max-games",
        type=int,
        default=20000,
        help="Limit the number of games parsed (default: 20000)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = sum(1 for _ in iter_pgn_files())
    if total == 0:
        raise SystemExit(
            "No PGN files found under data/chess. Download players from pgnmentor first."
        )
    with OUTPUT_PATH.open("w", encoding="utf-8") as out_f:
        for record in tqdm(
            parse_games(args.max_games), total=args.max_games, desc="Parsing games"
        ):
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
