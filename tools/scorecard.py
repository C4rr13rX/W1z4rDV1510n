#!/usr/bin/env python3
"""
The brain's scorecard and gate: stays right, stays small, never hurts the PC.

    python tools/scorecard.py                  build, run every scale, compare to the baseline
    python tools/scorecard.py --save-baseline  ...and make this run the new baseline
    python tools/scorecard.py --scales 1,4

Runs crates/brain/examples/scorecard.rs once per scale (a scene world of
rooms, objects and properties), each run inside tools/capped.py so it is
killed at --cap-mb instead of paging the machine to a standstill.

FAILS (exit 1) when, against docs/scorecard-baseline.json:
  - recall_pct drops at any scale              (it must know what it was taught)
  - integration_pct drops at any scale         (it must not get worse at deriving)
  - peak_mb exceeds --budget-mb at any scale   (the RAM promise: 2 GB, whatever the corpus)
  - peak_mb rises more than 15% over the baseline at any scale
  - a run crashed, hit the cap or timed out
--stress adds scale 64 (~10K facts, ~3.5 min). It is REPORTED, not gated,
until it fits the budget: measured 2026-09-30 it peaked at 2,471 MB for
~24 MB of neuron data -- the working set is not the neurons. Getting scale
64 under the budget, then flat, is the goal.
Improvements are printed; lock them in with --save-baseline.

The number to push DOWN is growth: peak_mb at the largest scale over peak_mb
at the smallest. A brain whose working set is its symbols, not its corpus,
keeps that near 1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "docs" / "scorecard-baseline.json"
LATEST = ROOT / "logs" / "scorecard-latest.json"      # last --stress run: what --metrics reports
GATE_RUN = ROOT / "logs" / "scorecard-gate.json"      # last plain run (the gate's)
sys.path.insert(0, str(ROOT / "tools"))
from capped import target_dir  # noqa: E402

EXE = target_dir(ROOT) / "release" / "examples" / "scorecard.exe"
BUILD_CAP_MB = 6000


class Scorecard:
    def __init__(self, scales: list[int], cap_mb: int, budget_mb: int, timeout: float,
                 report_only: frozenset[int] = frozenset()):
        self.scales, self.cap_mb, self.budget_mb, self.timeout = scales, cap_mb, budget_mb, timeout
        self.report_only = report_only

    def build(self) -> None:
        cmd = [sys.executable, str(ROOT / "tools" / "capped.py"), "--mb", str(BUILD_CAP_MB), "--",
               "cargo", "build", "--release", "-j", "2", "--example", "scorecard", "-p", "w1z4rd-brain"]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit("scorecard: build failed")

    def run_scale(self, scale: int) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            peak_file = Path(tmp) / "peak.json"
            cmd = [sys.executable, str(ROOT / "tools" / "capped.py"), "--mb", str(self.cap_mb),
                   "--timeout", str(self.timeout), "--json", str(peak_file), "--",
                   str(EXE), "--scale", str(scale)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            job = json.loads(peak_file.read_text()) if peak_file.exists() else {}
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        row = json.loads(lines[-1]) if lines else {"scale": scale}
        row.update(peak_mb=job.get("peak_mb"), wall_s=job.get("secs"), exit=job.get("exit"),
                   hit_cap=job.get("hit_cap"), timed_out=job.get("timed_out"))
        if proc.returncode != 0 and not lines:
            row["error"] = (proc.stderr or "").strip().splitlines()[-3:]
        return row

    def run(self) -> list[dict]:
        rows = []
        for s in self.scales:
            rows.append(self.run_scale(s))
            print(self.format_row(rows[-1]), flush=True)
        return rows

    @staticmethod
    def header() -> str:
        return (f"{'scale':>5} {'facts':>6} {'recall%':>8} {'integr%':>8} {'peak_mb':>8} "
                f"{'est_mb':>7} {'hub':>8} {'terminals':>10} {'train_s':>8} {'wall_s':>7}")

    @staticmethod
    def format_row(r: dict) -> str:
        def f(k, fmt):
            v = r.get(k)
            return format(v, fmt) if isinstance(v, (int, float)) else format("-", fmt.split(".")[0].rstrip("fd"))
        line = (f"{f('scale','>5d')} {f('facts','>6d')} {f('recall_pct','>8.1f')} "
                f"{f('integration_pct','>8.1f')} {f('peak_mb','>8.1f')} {f('est_resident_mb','>7.1f')} "
                f"{f('hub_fanout','>8d')} {f('terminals','>10d')} {f('train_s','>8.1f')} {f('wall_s','>7.1f')}")
        if r.get("hit_cap"):
            line += "  KILLED AT MEMORY CAP"
        if r.get("timed_out"):
            line += "  TIMED OUT"
        if r.get("error"):
            line += "  ERROR " + " | ".join(r["error"])
        # Per-family integration, indented so it prints under its scale's row
        # (and so gate.py's digit-leading filter still shows it). The total alone
        # cannot distinguish four families at 22% from one at 90% and three at 0,
        # which is the whole difficulty of M3.
        fams = r.get("integration_families") or []
        if fams:
            line += (f"\n{r.get('scale', '-'):>5}   integration by family: " + "  ".join(
                f"{f['name']}({f['hops']}h) {f['hits']}/{f['probes']} {f['pct']:.1f}%"
                + ("[" + ",".join(f"{k}:{v}" for k, v in (f.get("miss_kinds") or {}).items()) + "]"
                   if f.get("miss_kinds") else "")
                for f in fams))
        return line

    def verdict(self, rows: list[dict], base: list[dict] | None) -> list[str]:
        problems = []
        by_scale = {b["scale"]: b for b in (base or [])}
        for r in rows:
            s = r["scale"]
            if s in self.report_only and (r.get("hit_cap") or r.get("timed_out")):
                continue
            if r.get("hit_cap") or r.get("timed_out") or r.get("exit") not in (0, None) or "recall_pct" not in r:
                problems.append(f"scale {s}: run did not complete cleanly")
                continue
            if (r.get("peak_mb") or 0) > self.budget_mb and s not in self.report_only:
                problems.append(f"scale {s}: peak {r['peak_mb']:.0f} MB over the {self.budget_mb} MB budget")
            b = by_scale.get(s)
            if b and b.get("peak_mb") and (r.get("peak_mb") or 0) > b["peak_mb"] * 1.15:
                problems.append(f"scale {s}: peak {r['peak_mb']:.0f} MB, baseline {b['peak_mb']:.0f} MB (+15% allowed)")
            for k in ("recall_pct", "integration_pct"):
                if b and k in b and r[k] < b[k]:
                    problems.append(f"scale {s}: {k} fell {b[k]:.1f} -> {r[k]:.1f}")
        return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", default="1,4,16")
    ap.add_argument("--cap-mb", type=int, default=3000, help="hard kill per run")
    ap.add_argument("--budget-mb", type=int, default=2048, help="gate: peak RAM allowed per run")
    ap.add_argument("--timeout", type=float, default=300, help="seconds per run")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--stress", action="store_true", help="add scale 64, reported but not budget-gated")
    ap.add_argument("--metrics", action="store_true",
                    help="print the last run (logs/scorecard-latest.json) as one JSON line and exit")
    args = ap.parse_args()
    if args.metrics:
        rows = json.loads(LATEST.read_text()) if LATEST.exists() else []
        keys = ("recall_pct", "integration_pct", "peak_mb", "hub_fanout")
        print(json.dumps({f"s{r['scale']}_{k}": r.get(k) for r in rows for k in keys}))
        return 0

    scales = [int(s) for s in args.scales.split(",")]
    report_only = frozenset({64}) if args.stress else frozenset()
    if args.stress and 64 not in scales:
        scales.append(64)
    card = Scorecard(scales, args.cap_mb, args.budget_mb, args.timeout, report_only)
    if not args.no_build:
        card.build()
    print(card.header())
    rows = card.run()
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    problems = card.verdict(rows, base)

    ok = [r for r in rows if isinstance(r.get("peak_mb"), (int, float)) and r.get("peak_mb")]
    if len(ok) >= 2:
        print(f"\nRAM growth x{ok[-1]['peak_mb'] / ok[0]['peak_mb']:.2f} from scale {ok[0]['scale']} "
              f"to {ok[-1]['scale']} (push toward 1.0); hub fan-out {ok[0].get('hub_fanout')} -> "
              f"{ok[-1].get('hub_fanout')}")
    out = LATEST if args.stress else GATE_RUN
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    if args.save_baseline and not problems:
        BASELINE.write_text(json.dumps(rows, indent=1))
        print(f"baseline saved: {BASELINE.relative_to(ROOT)}")
    if base is None and not args.save_baseline:
        print("no baseline yet: run with --save-baseline to set one")
    print("\nscorecard: " + ("FAILED -- " + "; ".join(problems) if problems else "OK"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
