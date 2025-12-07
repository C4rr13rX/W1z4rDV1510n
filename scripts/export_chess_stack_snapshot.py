#!/usr/bin/env python3
"""
Convert a processed chess game (from processed_games.jsonl) into an EnvironmentSnapshot
with a stack_history suitable for the quantum/stack-aware annealer.

You can optionally sparsify the stack (keep every Nth ply) to simulate partial /
staggered observations when testing gap-filling + forecasting.
"""

import argparse
import json
from pathlib import Path

import chess


def load_game(path: Path, index: int) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx == index:
                return json.loads(line)
    raise IndexError(f"Game index {index} out of range")


def board_state(board: chess.Board) -> dict:
    states: dict[str, dict] = {}
    for square, piece in board.piece_map().items():
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        color = "white" if piece.color == chess.WHITE else "black"
        symbol_id = f"{color}_{piece.symbol().upper()}_{chess.square_name(square)}"
        states[symbol_id] = {
            "position": {"x": float(file_idx), "y": float(rank_idx), "z": 0.0},
            "velocity": None,
            "internal_state": {
                "piece": piece.symbol(),
                "color": color,
                "square": chess.square_name(square),
            },
        }
    return states


def build_snapshot(
    record: dict, plies: int, stride: int, stride_offset: int, include_terminal: bool
) -> dict:
    board = chess.Board()
    stack_history = []
    moves = record.get("moves", [])
    terminal_idx = min(plies, len(moves)) - 1
    for idx, move_san in enumerate(moves[:plies]):
        if stride > 1 and ((idx - stride_offset) % stride != 0):
            if include_terminal and idx == terminal_idx:
                pass
            else:
                continue
        try:
            move = board.parse_san(move_san)
        except ValueError:
            break
        board.push(move)
        stack_history.append(
            {
                "timestamp": {"unix": idx},
                "symbol_states": board_state(board),
            }
        )
    symbols = [
        {
            "id": symbol_id,
            "type": "CUSTOM",
            "position": state["position"],
            "properties": {
                "piece": state["internal_state"]["piece"],
                "color": state["internal_state"]["color"],
                "radius": 0.45,
            },
        }
        for symbol_id, state in (stack_history[-1]["symbol_states"] if stack_history else {}).items()
    ]
    snapshot = {
        "timestamp": {"unix": 0},
        "bounds": {"width": 8.0, "height": 8.0, "depth": 1.0},
        "symbols": symbols,
        "metadata": {
            "game_id": record.get("id"),
            "result": record.get("result"),
            "plies_sampled": len(stack_history),
            "stride": stride,
            "stride_offset": stride_offset,
            "terminal_included": include_terminal,
        },
        "stack_history": stack_history,
    }
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--games-file",
        type=Path,
        default=Path("data/chess/processed_games.jsonl"),
        help="Path to processed_games.jsonl",
    )
    parser.add_argument(
        "--game-index",
        type=int,
        default=0,
        help="Zero-based index of the game to export",
    )
    parser.add_argument(
        "--plies",
        type=int,
        default=12,
        help="How many half-moves to include in the stack_history",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every Nth ply (stride) to create sparse/partial histories",
    )
    parser.add_argument(
        "--stride-offset",
        type=int,
        default=0,
        help="Offset to apply when striding plies (e.g., 1 keeps plies 1, 1+stride, ...)",
    )
    parser.add_argument(
        "--skip-terminal",
        action="store_true",
        help="Do not force-include the terminal ply when it is skipped by the stride",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chess/stack_snapshot.json"),
        help="Where to write the EnvironmentSnapshot JSON",
    )
    args = parser.parse_args()

    record = load_game(args.games_file, args.game_index)
    snapshot = build_snapshot(
        record,
        args.plies,
        max(1, args.stride),
        max(0, args.stride_offset),
        not args.skip_terminal,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    print(f"Wrote snapshot with {len(snapshot['symbols'])} symbols and {len(snapshot['stack_history'])} frames to {args.output}")


if __name__ == "__main__":
    main()
