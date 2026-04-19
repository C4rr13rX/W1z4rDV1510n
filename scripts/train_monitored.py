#!/usr/bin/env python3
"""
Monitored training orchestrator.

Runs all training corpora in ordered batches and tests after each phase.
Goal: verify that new training does NOT overwrite previously learned answers.
Each test checks a probe question through /neuro/pipeline and scores pass/fail.

Usage:
    python scripts/train_monitored.py --node localhost:8090
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional

PYTHON = sys.executable
NODE   = "localhost:8090"
DATA   = "D:/w1z4rdv1510n-data"
SCRIPTS = "D:/Projects/W1z4rDV1510n/scripts"

# ── Probe definitions ──────────────────────────────────────────────────────────
# Each probe specifies a question and what the answer must contain.
# After each training phase we run ALL probes and report pass/fail.
# A probe "fails" if:
#   - hypothesis=True (no answer)
#   - answer doesn't contain any of the required keywords

@dataclass
class Probe:
    question: str
    require_any: list[str]          # answer must contain at least one of these
    pool: str = "knowledge"         # "knowledge" or "correction"
    scope_prefix: str = ""          # prefix to force scope detection
    phase_introduced: int = 0       # first phase where this should pass

PROBES: list[Probe] = [
    # ── Foundation / basic facts ──────────────────────────────────────────────
    Probe("What is the sun?",
          ["star", "solar", "light", "energy", "nuclear"], phase_introduced=1),
    Probe("What is water?",
          ["hydrogen", "oxygen", "molecule", "H2O", "liquid"], phase_introduced=1),
    Probe("What is gravity?",
          ["force", "attract", "mass", "weight", "pull"], phase_introduced=1),
    Probe("What is photosynthesis?",
          ["plant", "sunlight", "chlorophyll", "glucose", "carbon"], phase_introduced=1),
    Probe("What is the speed of light?",
          ["299", "300", "kilometer", "metre", "second", "fast"], phase_introduced=1),
    # ── K-12 geography ────────────────────────────────────────────────────────
    Probe("What is the capital of France?",
          ["paris", "Paris"], phase_introduced=2),
    Probe("What is the capital of the United States?",
          ["washington", "Washington", "D.C", "D.C."], phase_introduced=2),
    # ── Science ───────────────────────────────────────────────────────────────
    Probe("What is DNA?",
          ["genetic", "gene", "nucleotide", "chromosome", "deoxyribonucleic"], phase_introduced=2),
    Probe("What is an atom?",
          ["proton", "neutron", "electron", "nucleus", "element"], phase_introduced=2),
    # ── Math ──────────────────────────────────────────────────────────────────
    Probe("What is pi?",
          ["3.14", "circle", "circumference", "ratio", "diameter"], phase_introduced=3),
    Probe("What is a prime number?",
          ["divisible", "factor", "only", "itself", "one"], phase_introduced=3),
    # ── Corrections (correction pool) ─────────────────────────────────────────
    Probe("Correct the grammar: Me and him went to the store.",
          ["I", "He", "went"], pool="correction", scope_prefix="", phase_introduced=4),
    Probe("Correct the grammar: She don't know the answer.",
          ["doesn't", "does not"], pool="correction", phase_introduced=4),
    Probe("What is the correct spelling of 'recieve'?",
          ["receive", "ei", "i before e"], pool="correction", phase_introduced=4),
]

# ── Training phases ────────────────────────────────────────────────────────────
# Each phase runs one or more scripts, then probes run.
# Batches per phase controls how much training before the next check.

@dataclass
class Phase:
    name: str
    index: int
    commands: list[list[str]]

def phases(node: str) -> list[Phase]:
    return [
        Phase("Foundation (Stage 0 — K-12 books)", 1, [
            [PYTHON, f"{SCRIPTS}/train_foundation.py", "--node", f"http://{node}"],
        ]),
        Phase("K-12 curriculum (stages 0-2)", 2, [
            [PYTHON, f"{SCRIPTS}/train_k12.py", "--node", f"http://{node}",
             "--stages", "0,1,2", "--resume"],
        ]),
        Phase("Math (Stage 34)", 3, [
            [PYTHON, f"{SCRIPTS}/build_math_corpus.py",
             "--stages", "34", "--node", node, "--data-dir", DATA],
        ]),
        Phase("Spelling corrections (Stage 41)", 4, [
            [PYTHON, f"{SCRIPTS}/build_misspellings_corpus.py",
             "--node", node, "--repeats", "20"],
        ]),
        Phase("Grammar/communication corrections (Stage 42)", 4, [
            [PYTHON, f"{SCRIPTS}/build_communication_corpus.py",
             "--node", node, "--repeats", "20"],
        ]),
        Phase("Foreign languages (Stages 36-40)", 5, [
            [PYTHON, f"{SCRIPTS}/build_foreign_language_corpus.py",
             "--stages", "36,37,38,39,40", "--node", node, "--data-dir", DATA],
        ]),
        Phase("JSON scoped responses", 5, [
            [PYTHON, f"{SCRIPTS}/train_json_responses.py",
             "--node", node, "--batches", "50"],
        ]),
        # Repeat corrections to reinforce after the larger knowledge training
        Phase("Correction reinforcement (Stage 41+42 x50)", 5, [
            [PYTHON, f"{SCRIPTS}/build_misspellings_corpus.py",
             "--node", node, "--repeats", "50"],
            [PYTHON, f"{SCRIPTS}/build_communication_corpus.py",
             "--node", node, "--repeats", "50"],
        ]),
    ]


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def post(node: str, path: str, body: dict, timeout: int = 30) -> dict:
    data = json.dumps(body).encode()
    url  = f"http://{node}{path}"
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc), "hypothesis": True, "answer": ""}


def probe_pipeline(node: str, probe: Probe) -> tuple[bool, str]:
    """Run a probe through /neuro/pipeline and return (passed, answer)."""
    question = probe.scope_prefix + probe.question if probe.scope_prefix else probe.question
    resp = post(node, "/neuro/pipeline", {"text": question, "hops": 3})

    if resp.get("error"):
        return False, f"ERROR: {resp['error']}"

    if resp.get("hypothesis") or not resp.get("answer"):
        return False, "(no answer — hypothesis)"

    answer = (resp.get("answer") or "").lower()
    passed = any(kw.lower() in answer for kw in probe.require_any)
    return passed, resp.get("answer", "")[:120]


# ── Report ─────────────────────────────────────────────────────────────────────

def run_probes(node: str, phase_index: int) -> dict:
    results = []
    passed = 0
    failed = 0
    for probe in PROBES:
        if probe.phase_introduced > phase_index:
            continue  # not expected yet
        ok, answer = probe_pipeline(node, probe)
        results.append({
            "question": probe.question[:70],
            "passed": ok,
            "answer": answer,
        })
        if ok: passed += 1
        else: failed += 1
    return {"passed": passed, "failed": failed, "results": results}


def print_report(phase_name: str, report: dict) -> None:
    p, f = report["passed"], report["failed"]
    total = p + f
    pct = 100 * p // total if total else 0
    print(f"\n{'='*70}")
    print(f"  PROBE RESULTS after: {phase_name}")
    print(f"  Score: {p}/{total} ({pct}%)")
    print(f"{'='*70}")
    for r in report["results"]:
        icon = "✓" if r["passed"] else "✗"
        print(f"  {icon}  {r['question']}")
        if not r["passed"]:
            print(f"       → {r['answer']}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=NODE)
    parser.add_argument("--phase-start", type=int, default=1,
                        help="Skip phases before this index (1-based, for resuming)")
    args = parser.parse_args()

    print(f"\nMonitored training — node: {args.node}")
    print(f"Probes defined: {len(PROBES)}")
    print("Pools cleared, starting fresh.\n")

    env = {"PYTHONUTF8": "1"}
    import os
    env.update(os.environ)

    for phase in phases(args.node):
        if phase.index < args.phase_start:
            print(f"  [skip] Phase {phase.index}: {phase.name}")
            continue

        print(f"\n{'─'*70}")
        print(f"  Phase {phase.index}: {phase.name}")
        print(f"{'─'*70}")

        for cmd in phase.commands:
            print(f"\n  Running: {' '.join(cmd[1:3])}...")
            t0 = time.time()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"  [WARNING] Script exited {result.returncode}")
                if result.stderr:
                    print("  STDERR:", result.stderr[-500:])
            else:
                print(f"  Completed in {elapsed:.0f}s")
            # Show last 3 lines of stdout for progress
            lines = (result.stdout or "").strip().splitlines()
            for line in lines[-3:]:
                if line.strip():
                    print(f"  | {line}")

        # Checkpoint both pools
        post(args.node, "/qa/checkpoint", {})
        print("  Pools checkpointed.")

        # Run probes
        report = run_probes(args.node, phase.index)
        print_report(phase.name, report)

        # Alert on regression
        if report["failed"] > 0:
            print(f"  ⚠  {report['failed']} probe(s) failing — check for interference.")
        else:
            print(f"  All probes passing ✓")

    print("\n" + "="*70)
    print("  TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
