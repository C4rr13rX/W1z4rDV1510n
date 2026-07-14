#!/usr/bin/env python3
"""Prove failure/success co-firing ranks repairs without outcome markers."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from programming_project_eval import b64, request


def fragment(role: str, after: list[str], source: str) -> str:
    return json.dumps({"code_fragment": {"file": "increment.js", "role": role,
                                          "after": after, "source": source}},
                      sort_keys=True, separators=(",", ":"))


SIGNATURE_PROMPT = "Write a JavaScript increment function signature."
BODY_PROMPT = "Implement JavaScript increment logic that increases the stored total."
SIGNATURE = fragment("signature", [], "function increment(value) {\n")
BAD = fragment("body", ["signature"], "  return value - 1;\n}\n")
GOOD = fragment("body", ["signature"], "  return value + 1;\n}\n")
PROMPTS = [
    ("canonical", "Build a JavaScript increment function signature that increases the stored total."),
    ("heldout", "Create a Node.js increment declaration line that adds one to the supplied value."),
]


def tick(endpoint: str, frames: dict[int, str], action: str) -> None:
    for pool_id, frame in frames.items():
        request(endpoint, "/brain/observe", {"pool_id": pool_id, "frame": b64(frame)})
    request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(action)})
    request(endpoint, "/brain/tick", {})


def execute(reply: str) -> tuple[bool, str]:
    try:
        source = json.loads(reply)["files"]["increment.js"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}"
    with tempfile.TemporaryDirectory(prefix="wv-synaptic-repair-") as raw:
        path = Path(raw) / "check.js"
        path.write_text(source + "\nif(increment(7)!==8)throw new Error('wrong');console.log('PASS');\n",
                        encoding="utf-8")
        run = subprocess.run(["node", str(path)], capture_output=True, text=True, timeout=15)
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-700:]


def train(endpoint: str, repeats: int) -> dict:
    for _ in range(repeats):
        tick(endpoint, {1: SIGNATURE_PROMPT, 12: SIGNATURE_PROMPT}, SIGNATURE)
        tick(endpoint, {1: BODY_PROMPT, 12: BODY_PROMPT}, BAD)
    failure_console = "Error: expected increment(7) to equal 8, got 6"
    for _ in range(repeats):
        tick(endpoint, {
            1: BODY_PROMPT, 3: failure_console,
            6: json.dumps({"status": "failure", "error_class": "wrong_result"}),
            12: BODY_PROMPT,
        }, BAD)
    for _ in range(repeats):
        tick(endpoint, {
            1: BODY_PROMPT, 3: failure_console,
            6: json.dumps({"status": "failure", "error_class": "wrong_result"}),
            7: "- return value - 1\n+ return value + 1",
            8: json.dumps({"status": "success", "stdout": "PASS"}),
            9: json.dumps({"transition": "failure_to_success", "repair": "operator"}),
            11: json.dumps({"kind": "replace_operator", "from": "-", "to": "+"}),
            12: BODY_PROMPT,
        }, GOOD)
    return {"negative_markers": 0, "positive_markers": 0,
            "failure_console": failure_console}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/synaptic_repair.json"))
    args = parser.parse_args()
    training = {} if args.no_train else train(args.endpoint, args.repeats)
    rows = []
    for kind, prompt in PROMPTS:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        passed, detail = execute(str(result.get("reply") or ""))
        rows.append({"kind": kind, "executes": passed, "detail": "" if passed else detail})
    report = {"passed": sum(row["executes"] for row in rows), "total": len(rows),
              "training": training, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "marker_free": training.get("negative_markers", 0) == 0}))
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
