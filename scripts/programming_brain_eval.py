#!/usr/bin/env python3
"""Stage-gate evaluator for the persistent programming brain."""
from __future__ import annotations

import argparse
import http.client
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
TODDLER = [
    ("dog", "animal"), ("cat", "animal"), ("cow", "animal"),
    ("horse", "animal"), ("bird", "animal"), ("fish", "animal"),
    ("apple", "food"), ("banana", "food"), ("bread", "food"),
    ("cake", "food"), ("milk", "food"), ("car", "vehicle"),
    ("truck", "vehicle"), ("bike", "vehicle"), ("plane", "vehicle"),
    ("boat", "vehicle"), ("red", "color"), ("blue", "color"),
    ("green", "color"), ("yellow", "color"), ("ball", "toy"),
    ("doll", "toy"), ("kite", "toy"), ("drum", "toy"),
    ("tree", "nature"), ("flower", "nature"), ("river", "nature"),
    ("mountain", "nature"), ("hand", "body"), ("foot", "body"),
    ("eye", "body"), ("mouth", "body"),
]
K12 = [
    "piano", "guitar", "triangle", "square", "seven", "nine", "sad",
    "happy", "doctor", "school", "rose", "oak", "hammer", "saw",
    "football", "tennis",
]
OOV = ["quasarithmetic", "purple elephant theorem", "zxqv compiler"]


def accepted_answers(path: Path) -> dict[str, set[str]]:
    accepted: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompt = str(row.get("prompt") or row.get("question") or "").strip()
            answer = str(row.get("response") or row.get("answer") or "").strip()
            if prompt and answer:
                accepted.setdefault(prompt, set()).add(answer)
    return accepted


class BrainClient:
    def __init__(self, endpoint: str) -> None:
        url = urlparse(endpoint)
        self.prefix = url.path.rstrip("/")
        self.conn = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=60)

    def chat(self, text: str) -> dict:
        self.conn.request("POST", f"{self.prefix}/brain/chat", json.dumps({"text": text}),
                          {"Content-Type": "application/json"})
        response = self.conn.getresponse()
        payload = response.read()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {payload[:300]!r}")
        return json.loads(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8291")
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()
    accepted = accepted_answers(ROOT / "data/training/categorical_unified_001.jsonl")
    brain = BrainClient(args.endpoint)

    toddler_rows = [(p, expected, brain.chat(p).get("reply", "")) for p, expected in TODDLER]
    k12_rows = [(p, (reply := brain.chat(p).get("reply", "")), reply in accepted.get(p, set()))
                for p in K12]
    oov_rows = []
    for prompt in OOV:
        response = brain.chat(prompt)
        honest = (not response.get("reply")
                  and bool((response.get("grounding") or {}).get("outside_grounding")))
        oov_rows.append((prompt, response.get("reply", ""), honest))

    report = {
        "toddler_exact": sum(expected == reply for _, expected, reply in toddler_rows),
        "toddler_total": len(toddler_rows),
        "k12_trained_answer": sum(hit for _, _, hit in k12_rows),
        "k12_total": len(k12_rows),
        "oov_honest": sum(hit for _, _, hit in oov_rows),
        "oov_total": len(oov_rows),
    }
    if args.details:
        report["toddler_failures"] = [row for row in toddler_rows if row[1] != row[2]]
        report["k12_failures"] = [row for row in k12_rows if not row[2]]
        report["oov_failures"] = [row for row in oov_rows if not row[2]]
    print(json.dumps(report, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
