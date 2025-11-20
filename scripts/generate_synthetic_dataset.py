#!/usr/bin/env python3
"""
Synthetic dataset generator for W1z4rDV1510n.

Creates:
  - Environment snapshot with hundreds of symbols (people, objects, walls, exits)
  - Timeline dataset (JSONL) describing multi-label thought/intent annotations
  - Pre-baked run config that references the generated snapshot
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

Position = Tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data", type=Path)
    parser.add_argument("--n-people", default=180, type=int)
    parser.add_argument("--n-vehicles", default=25, type=int)
    parser.add_argument("--n-objects", default=45, type=int)
    parser.add_argument("--n-walls", default=18, type=int)
    parser.add_argument("--n-steps", default=240, type=int, help="timeline steps")
    parser.add_argument("--cell-size", default=1.0, type=float, help="search cell size to feed config")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--train-ratio",
        default=0.8,
        type=float,
        help="fraction of timeline rows written to *_train.jsonl",
    )
    return parser.parse_args()


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def random_position(width: float, height: float, depth: float = 0.0) -> Position:
    return (
        random.uniform(0, width),
        random.uniform(0, height),
        random.uniform(0, depth),
    )


def random_thoughts(max_labels: int = 4) -> List[str]:
    pool = [
        "observe_crowd",
        "plan_exit",
        "fearful",
        "calm",
        "seeking_friend",
        "monitor_obstacle",
        "curious",
        "distracted",
        "focused_task",
        "calling_out",
        "alert_security",
        "navigate_corridor",
        "avoid_vehicle",
        "guarding_area",
        "waiting_for_signal",
        "evaluate_risk",
        "multi_tasking",
    ]
    n_labels = random.randint(2, max_labels)
    return random.sample(pool, n_labels)


def random_properties(symbol_type: str) -> Dict[str, object]:
    if symbol_type == "Person":
        return {
            "role": random.choice(
                ["bystander", "security", "maintenance", "vip", "staff", "resident"]
            ),
            "group_id": f"group_{random.randint(0, 8)}",
            "thought_labels": random_thoughts(),
            "goal_id": random.choice(["exit_west", "exit_east", "exit_roof"]),
        }
    if symbol_type == "Wall":
        return {
            "width": round(random.uniform(3.0, 12.0), 2),
            "height": round(random.uniform(2.5, 4.0), 2),
            "thickness": round(random.uniform(0.2, 0.8), 2),
            "blocking": True,
        }
    if symbol_type == "Object":
        return {
            "category": random.choice(
                ["desk", "sofa", "kiosk", "tree", "fountain", "statue", "vehicle"]
            ),
            "walkable": random.choice([True, False]),
            "importance": random.random(),
        }
    if symbol_type == "Exit":
        return {"priority": random.choice(["primary", "secondary"]), "walkable": True}
    return {}


@dataclass
class SymbolSpec:
    symbol_id: str
    symbol_type: str
    position: Position
    properties: Dict[str, object]

    def to_json(self) -> Dict[str, object]:
        x, y, z = self.position
        return {
            "id": self.symbol_id,
            "type": self.symbol_type,
            "position": {"x": x, "y": y, "z": z},
            "properties": self.properties,
        }


def build_symbols(args: argparse.Namespace, bounds: Dict[str, float]) -> List[SymbolSpec]:
    width = bounds["width"]
    height = bounds["height"]
    symbols: List[SymbolSpec] = []

    for idx in range(args.n_people):
        position = random_position(width, height, bounds["depth"] * 0.1)
        symbols.append(
            SymbolSpec(
                symbol_id=f"person_{idx:04d}",
                symbol_type="Person",
                position=position,
                properties=random_properties("Person"),
            )
        )

    for idx in range(args.n_vehicles):
        position = random_position(width, height)
        props = {
            "category": random.choice(["car", "truck", "drone"]),
            "velocity_hint": random.uniform(2.0, 6.5),
            "walkable": False,
            "blocking": True,
        }
        symbols.append(
            SymbolSpec(
                symbol_id=f"vehicle_{idx:03d}",
                symbol_type="Object",
                position=position,
                properties=props,
            )
        )

    for idx in range(args.n_objects):
        position = random_position(width, height)
        symbols.append(
            SymbolSpec(
                symbol_id=f"object_{idx:03d}",
                symbol_type="Object",
                position=position,
                properties=random_properties("Object"),
            )
        )

    for idx in range(args.n_walls):
        x = random.uniform(0.0, width)
        y = random.uniform(0.0, height)
        position = (x, y, 0.0)
        prop = random_properties("Wall")
        prop.setdefault("rotation_degrees", random.choice([0, 90, 45]))
        symbols.append(
            SymbolSpec(
                symbol_id=f"wall_{idx:02d}",
                symbol_type="Wall",
                position=position,
                properties=prop,
            )
        )

    exits = [
        SymbolSpec(
            symbol_id="exit_west",
            symbol_type="Exit",
            position=(width * 0.05, height / 2.0, 0.0),
            properties=random_properties("Exit"),
        ),
        SymbolSpec(
            symbol_id="exit_east",
            symbol_type="Exit",
            position=(width * 0.95, height / 2.0, 0.0),
            properties=random_properties("Exit"),
        ),
        SymbolSpec(
            symbol_id="exit_roof",
            symbol_type="Exit",
            position=(width / 2.0, height / 2.0, bounds["depth"] * 0.9),
            properties=random_properties("Exit"),
        ),
    ]
    symbols.extend(exits)
    return symbols


def generate_snapshot(args: argparse.Namespace) -> Tuple[Dict[str, object], List[SymbolSpec]]:
    base_time = datetime(2025, 11, 18, 12, 0, tzinfo=timezone.utc)
    bounds = {"width": 120.0, "height": 90.0, "depth": 15.0}
    symbols = build_symbols(args, bounds)
    snapshot = {
        "timestamp": {"unix": int(base_time.timestamp())},
        "bounds": bounds,
        "symbols": [symbol.to_json() for symbol in symbols],
        "metadata": {
            "description": "synthetic multi-entity snapshot generated locally",
            "generator": "scripts/generate_synthetic_dataset.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol_counts": {
                "people": args.n_people,
                "vehicles": args.n_vehicles,
                "objects": args.n_objects,
                "walls": args.n_walls,
                "exits": 3,
            },
            "spatial_resolution": args.cell_size,
        },
    }
    return snapshot, symbols


def random_walk(
    symbols: List[SymbolSpec],
    bounds: Dict[str, float],
    steps: int,
    step_duration: float = 1.0,
) -> List[Dict[str, object]]:
    timeline: List[Dict[str, object]] = []
    positions: Dict[str, Position] = {symbol.symbol_id: symbol.position for symbol in symbols}
    width = bounds["width"]
    height = bounds["height"]
    base_timestamp = int(datetime(2025, 11, 18, 12, 0, tzinfo=timezone.utc).timestamp())

    for step in range(steps):
        timestamp = base_timestamp + int(step * step_duration)
        for symbol in symbols:
            if not symbol.symbol_id.startswith("person_"):
                continue
            last_pos = positions[symbol.symbol_id]
            step_size = random.gauss(mu=0.0, sigma=0.8)
            heading = random.uniform(0, 2 * math.pi)
            delta_x = math.cos(heading) * step_size
            delta_y = math.sin(heading) * step_size
            new_x = clamp(last_pos[0] + delta_x, 0.0, width)
            new_y = clamp(last_pos[1] + delta_y, 0.0, height)
            new_z = clamp(last_pos[2] + random.gauss(0.0, 0.15), 0.0, bounds["depth"])
            new_pos = (new_x, new_y, new_z)
            velocity = (new_x - last_pos[0], new_y - last_pos[1], new_z - last_pos[2])
            positions[symbol.symbol_id] = new_pos
            timeline.append(
                {
                    "timestamp": timestamp,
                    "symbol_id": symbol.symbol_id,
                    "position": {"x": round(new_x, 3), "y": round(new_y, 3), "z": round(new_z, 3)},
                    "velocity": {"x": round(velocity[0], 3), "y": round(velocity[1], 3), "z": round(velocity[2], 3)},
                    "thoughts": random_thoughts(),
                    "attention_targets": random.sample(
                        ["crowd_flow", "obstacle", "vehicle", "exit", "private_convo", "signal_panel"], 3
                    ),
                    "sensor_confidence": round(random.uniform(0.6, 0.99), 3),
                }
            )
    return timeline


def split_and_write_timeline(entries: List[Dict[str, object]], output_dir: Path, train_ratio: float) -> Dict[str, object]:
    total = len(entries)
    split_idx = max(1, min(total - 1, int(total * train_ratio)))
    train_entries = entries[:split_idx]
    test_entries = entries[split_idx:]
    train_path = output_dir / "timeline_train.jsonl"
    test_path = output_dir / "timeline_test.jsonl"
    with train_path.open("w", encoding="utf-8") as fp:
        for record in train_entries:
            fp.write(json.dumps(record))
            fp.write("\n")
    with test_path.open("w", encoding="utf-8") as fp:
        for record in test_entries:
            fp.write(json.dumps(record))
            fp.write("\n")
    return {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "train_rows": len(train_entries),
        "test_rows": len(test_entries),
    }


def write_snapshot(snapshot: Dict[str, object], output_dir: Path) -> Path:
    path = output_dir / "synthetic_snapshot.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(snapshot, fp, indent=2)
    return path


def write_run_config(snapshot_path: Path, args: argparse.Namespace) -> Path:
    config = {
        "snapshot_file": str(snapshot_path.as_posix()),
        "t_end": {"unix": snapshot_path.stat().st_mtime_ns // 1_000_000_000 + 600},
        "n_particles": 384,
        "schedule": {"t_start": 5.0, "t_end": 0.2, "n_iterations": 120},
        "energy": {
            "w_motion": 1.0,
            "w_collision": 6.5,
            "w_goal": 1.2,
            "w_group_cohesion": 0.9,
            "w_ml_prior": 0.5,
            "w_path_feasibility": 2.5,
            "w_env_constraints": 3.2,
        },
        "proposal": {"local_move_prob": 0.85, "max_step_size": 1.5},
        "init_strategy": {
            "use_ml_priors": True,
            "noise_level": 0.65,
            "random_fraction": 0.25,
            "copy_snapshot_fraction": 0.1,
        },
        "resample": {"enabled": True, "effective_sample_size_threshold": 0.35, "mutation_rate": 0.08},
        "search": {"cell_size": args.cell_size},
        "ml_backend": "SIMPLE_RULES",
        "hardware_backend": "Cpu",
        "random_seed": args.seed,
        "output": {"save_best_state": True, "save_population_summary": True, "output_path": "logs/latest_results.json"},
    }
    path = snapshot_path.parent / "synthetic_run_config.json"
    with path.open("w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)
    return path


def summarize_timeline(entries: List[Dict[str, object]]) -> Dict[str, float]:
    speeds = []
    for record in entries:
        vx = record["velocity"]["x"]
        vy = record["velocity"]["y"]
        vz = record["velocity"]["z"]
        speeds.append(math.sqrt(vx * vx + vy * vy + vz * vz))
    return {
        "mean_speed": statistics.fmean(speeds),
        "max_speed": max(speeds),
        "min_speed": min(speeds),
    }


def write_metadata(snapshot_path: Path, config_path: Path, timeline_stats: Dict[str, object], output_dir: Path) -> None:
    meta = {
        "snapshot_path": str(snapshot_path),
        "run_config_path": str(config_path),
        "timeline": timeline_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "synthetic_dataset_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot, symbol_specs = generate_snapshot(args)
    snapshot_path = write_snapshot(snapshot, output_dir)
    timeline_entries = random_walk(symbol_specs, snapshot["bounds"], args.n_steps)
    split_info = split_and_write_timeline(timeline_entries, output_dir, args.train_ratio)
    timeline_stats = summarize_timeline(timeline_entries)
    timeline_stats.update(split_info)
    config_path = write_run_config(snapshot_path, args)
    write_metadata(snapshot_path, config_path, timeline_stats, output_dir)
    print(
        json.dumps(
            {
                "snapshot": str(snapshot_path),
                "run_config": str(config_path),
                "timeline_train": split_info["train_path"],
                "timeline_test": split_info["test_path"],
                "rows": {
                    "train": split_info["train_rows"],
                    "test": split_info["test_rows"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
