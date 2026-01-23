#!/usr/bin/env python3
"""
Convert traffic sensor streams into StreamEnvelope JSONL for CrowdTraffic.

Supports JSON lines or CSV input with flexible field mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic sensor -> CrowdTraffic JSONL bridge")
    parser.add_argument("--input", default="-", help="Input file path or '-' for stdin")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    parser.add_argument("--columns", help="CSV columns (comma-separated) if no header")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter")
    parser.add_argument("--timestamp-field", default="timestamp", help="Timestamp field name")
    parser.add_argument("--sensor-id-field", default="sensor_id", help="Sensor id field name")
    parser.add_argument("--sensor-id", help="Static sensor id override")
    parser.add_argument("--lat-field", default="lat", help="Latitude field name")
    parser.add_argument("--lon-field", default="lon", help="Longitude field name")
    parser.add_argument("--lat", type=float, help="Static latitude override")
    parser.add_argument("--lon", type=float, help="Static longitude override")
    parser.add_argument("--source-type", default="loop", help="Sensor source type label")
    parser.add_argument("--segment-length-m", type=float, default=0.0, help="Segment length for density")
    parser.add_argument("--window-secs", type=float, default=1.0, help="Aggregation window seconds")
    parser.add_argument("--speed-unit", choices=["mps", "kph", "mph"], default="mps")
    parser.add_argument("--expected-interval", type=float, default=1.0, help="Expected sample interval seconds")
    parser.add_argument("--drop-on-missing", action="store_true", help="Drop samples with missing fields")
    return parser.parse_args()


def now_unix() -> int:
    return int(time.time())


def parse_timestamp(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "unix" in value:
            try:
                return int(value["unix"])
            except Exception:
                return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return int(float(value))
        except Exception:
            pass
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return None
    return None


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except Exception:
            return None
    return None


def convert_speed(value: Optional[float], unit: str) -> Optional[float]:
    if value is None:
        return None
    if unit == "mps":
        return value
    if unit == "kph":
        return value / 3.6
    if unit == "mph":
        return value * 0.44704
    return value


def compute_density(count: Optional[float], segment_length: float) -> Optional[float]:
    if count is None or segment_length <= 0:
        return None
    return max(0.0, count / segment_length)


def compute_quality(missing_ratio: float, jitter_ratio: float, computed_fields: int) -> float:
    missing_penalty = min(missing_ratio * 0.7, 0.7)
    jitter_penalty = min(jitter_ratio * 0.3, 0.3)
    computed_penalty = min(computed_fields * 0.1, 0.2)
    quality = 1.0 - missing_penalty - jitter_penalty - computed_penalty
    return max(0.0, min(1.0, quality))


def build_envelope(timestamp: int, payload: Dict[str, float], metadata: Dict[str, object]) -> Dict[str, object]:
    return {
        "source": "CROWD_TRAFFIC",
        "timestamp": {"unix": timestamp},
        "payload": {"kind": "JSON", "value": payload},
        "metadata": metadata,
    }


def iter_json_lines(handle: Iterable[str]) -> Iterable[Dict[str, object]]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                yield value
        except Exception:
            continue


def iter_csv_lines(handle: Iterable[str], args: argparse.Namespace) -> Iterable[Dict[str, object]]:
    reader = None
    if args.columns:
        fieldnames = [name.strip() for name in args.columns.split(",") if name.strip()]
        reader = csv.DictReader(handle, fieldnames=fieldnames, delimiter=args.delimiter)
    else:
        reader = csv.DictReader(handle, delimiter=args.delimiter)
    for row in reader:
        yield row


def main() -> None:
    args = parse_args()
    last_ts_by_sensor: Dict[str, int] = defaultdict(int)
    if args.input == "-":
        handle = sys.stdin
    else:
        handle = open(args.input, "r", encoding="utf-8")
    iterator = iter_json_lines(handle) if args.format == "json" else iter_csv_lines(handle, args)
    for row in iterator:
        sensor_id = args.sensor_id or str(row.get(args.sensor_id_field) or "sensor-0")
        ts_value = row.get(args.timestamp_field) or row.get("time") or row.get("unix")
        timestamp = parse_timestamp(ts_value) or now_unix()
        density = parse_float(row.get("density"))
        velocity = parse_float(row.get("velocity") or row.get("speed"))
        direction = parse_float(row.get("direction_deg") or row.get("heading"))
        lat = args.lat if args.lat is not None else parse_float(row.get(args.lat_field))
        lon = args.lon if args.lon is not None else parse_float(row.get(args.lon_field))
        count = parse_float(row.get("count") or row.get("volume"))
        segment_length = args.segment_length_m
        computed_fields = 0
        if density is None:
            density = compute_density(count, segment_length)
            if density is not None:
                computed_fields += 1
        velocity = convert_speed(velocity, args.speed_unit)
        if velocity is None and parse_float(row.get("speed")) is not None:
            computed_fields += 1
        if direction is None:
            direction = 0.0
            computed_fields += 1
        missing_fields = [name for name, value in [("density", density), ("velocity", velocity)] if value is None]
        if missing_fields and args.drop_on_missing:
            continue
        missing_ratio = len(missing_fields) / 2.0
        last_ts = last_ts_by_sensor.get(sensor_id, 0)
        dt = max(0.0, timestamp - last_ts) if last_ts else 0.0
        last_ts_by_sensor[sensor_id] = timestamp
        jitter_ratio = 0.0
        if last_ts:
            expected = max(args.expected_interval, 1e-6)
            jitter_ratio = abs(dt - expected) / expected
        quality = compute_quality(missing_ratio, jitter_ratio, computed_fields)
        payload = {
            "density": float(density or 0.0),
            "velocity": float(velocity or 0.0),
            "direction_deg": float(direction or 0.0),
        }
        metadata = {
            "sensor_id": sensor_id,
            "source_type": args.source_type,
            "segment_length_m": segment_length,
            "window_secs": args.window_secs,
            "lat": lat,
            "lon": lon,
            "quality": quality,
            "confidence": quality,
            "missing_ratio": missing_ratio,
            "jitter_ratio": jitter_ratio,
            "computed_fields": computed_fields,
            "ingestor": "traffic_sensor_bridge",
        }
        envelope = build_envelope(timestamp, payload, metadata)
        sys.stdout.write(json.dumps(envelope) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
