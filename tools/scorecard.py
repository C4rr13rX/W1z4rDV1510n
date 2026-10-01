#!/usr/bin/env python3
"""
The brain's scorecard and gate: stays right, stays small, never hurts the PC.

    python tools/scorecard.py                  build, run every scale, compare to the baseline
    python tools/scorecard.py --save-baseline  ...and make this run the new baseline
    python tools/scorecard.py --scales 1,4
    python tools/scorecard.py --stress --with-store   M4: the same scales again with
                                                      a .wbrain store attached

Runs crates/brain/examples/scorecard.rs once per scale (a scene world of
rooms, objects and properties), each run inside tools/capped.py so it is
killed at --cap-mb instead of paging the machine to a standstill.

FAILS (exit 1) when, against docs/scorecard-baseline.json:
  - recall_pct drops at any scale              (it must know what it was taught)
  - integration_pct drops at any scale         (it must not get worse at deriving)
  - integration_wrong_pct is not strictly BELOW integration_pct at any scale
                                               (invention is worse than silence)
  - integration_wrong_pct rises more than 2 points over the baseline at any scale
                                               (precision may not be bought by inventing more)
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

--with-store runs every scale a SECOND time with a `.wbrain` container attached
to a temp dir and the whole brain slept into it after training, and prints the
paging readout under each no-store row. Until 2026-10-01 the scorecard attached
NO store at any scale, so M4's criterion -- peak RAM independent of the corpus
WITH THE STORE ATTACHED -- had never been measured by the thing that gates it.
A store-attached run must keep recall_pct at the no-store level (paging to SSD
may not lose a taught fact) and must actually evict, or it is measuring the same
resident brain twice.

First measurement, 2026-10-01, `--stress --with-store` (scale 64 needed
--store-timeout; it times out at the shared 300 s):

  scale  no-store peak  store peak        recall  slept/page_outs/page_ins  container
      1       16.4 MB    23.1 MB (x1.41)   100.0       241 /    241 /   241    0.6 MB
      4       18.8 MB    24.5 MB (x1.30)   100.0       803 /    803 /   803    2.4 MB
     16       24.9 MB    31.0 MB (x1.24)   100.0      3035 /   3035 /  3035    8.9 MB
     64       40.5 MB    45.8 MB (x1.13)   100.0      5496 /  11963 / 11963   29.8 MB

Two things that reading settles. Recall is 100.0 at every scale with the store
attached, so paging to SSD loses no taught fact. And peak RAM is HIGHER with the
store, not lower, at every scale -- because page_ins equals page_outs exactly:
the answer phases pull every body back, and the run ends with 0 neurons evicted
and every terminal resident (129,639 of them at scale 64). M4's "peak RAM
independent of the corpus" is therefore not merely unachieved, it is not yet
approached; nothing bounds the working set, and the container is pure overhead
on top of a fully resident brain. The zoom -- paging in only what the goal needs
-- is the missing mechanism, and this mode is how it gets measured.

`cold_offsets` and `evicted_set` read 0 at all four scales. That is the normal
reading for a `.wbrain` brain, not an absence: see `classify_zero_buckets` in
crates/brain/examples/scorecard.rs.
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
# A --with-store run writes HERE as well, because a store run costs ~10x the
# wall time of its no-store twin and the next plain gate run would otherwise
# destroy it: measured 2026-10-01, the gate's `--stress` overwrote
# scorecard-latest.json and the four-scale store twins were gone minutes after
# they were measured, leaving the table only in terminal scrollback.
STORE_RUN = ROOT / "logs" / "scorecard-store.json"
sys.path.insert(0, str(ROOT / "tools"))
from capped import target_dir  # noqa: E402

EXE = target_dir(ROOT) / "release" / "examples" / "scorecard.exe"
BUILD_CAP_MB = 6000


class Scorecard:
    def __init__(self, scales: list[int], cap_mb: int, budget_mb: int, timeout: float,
                 report_only: frozenset[int] = frozenset(), with_store: bool = False,
                 store_timeout: float | None = None):
        self.scales, self.cap_mb, self.budget_mb, self.timeout = scales, cap_mb, budget_mb, timeout
        self.report_only = report_only
        self.with_store = with_store
        self.store_timeout = store_timeout if store_timeout else timeout * 10

    def build(self) -> None:
        cmd = [sys.executable, str(ROOT / "tools" / "capped.py"), "--mb", str(BUILD_CAP_MB), "--",
               "cargo", "build", "--release", "-j", "2", "--example", "scorecard", "-p", "w1z4rd-brain"]
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            raise SystemExit("scorecard: build failed")

    def run_scale(self, scale: int, store: bool = False) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            peak_file = Path(tmp) / "peak.json"
            # Store mode needs its own timeout. Measured 2026-10-01, store wall
            # against no-store wall: 3.4s/0.1, 30.3/1.4, 126.3/11.8 -- roughly
            # 10x, because every answered question pages its neurons back from
            # the container (recall 25.2 ms per question at scale 16). At the
            # shared 300 s the scale-64 store run timed out and the table
            # reported the scale M4's criterion is actually about as a blank.
            timeout = self.store_timeout if store else self.timeout
            cmd = [sys.executable, str(ROOT / "tools" / "capped.py"), "--mb", str(self.cap_mb),
                   "--timeout", str(timeout), "--json", str(peak_file), "--",
                   str(EXE), "--scale", str(scale)]
            if store:
                # A temp .wbrain container, deleted with the temp dir. The
                # brain writes neuron bodies here, so it must be real disk, not
                # a path the exe only names.
                store_dir = Path(tmp) / "wbrain"
                cmd += ["--store", str(store_dir)]
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            job = json.loads(peak_file.read_text()) if peak_file.exists() else {}
            if store and job.get("exit") == 0:
                container = store_dir / "brain.wbrain"
                job["store_bytes"] = container.stat().st_size if container.exists() else 0
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        row = json.loads(lines[-1]) if lines else {"scale": scale}
        row.update(peak_mb=job.get("peak_mb"), wall_s=job.get("secs"), exit=job.get("exit"),
                   hit_cap=job.get("hit_cap"), timed_out=job.get("timed_out"),
                   store_bytes=job.get("store_bytes"))
        if proc.returncode != 0 and not lines:
            row["error"] = (proc.stderr or "").strip().splitlines()[-3:]
        return row

    def run(self) -> list[dict]:
        """The no-store rows, each optionally followed by its store-attached
        twin. The store run is a SECOND run of the same scale, printed under it,
        so the two are side by side in one table and neither replaces the
        other -- the gate's baseline keeps comparing the no-store rows it
        always compared."""
        rows = []
        for s in self.scales:
            rows.append(self.run_scale(s))
            print(self.format_row(rows[-1]), flush=True)
            if self.with_store:
                twin = self.run_scale(s, store=True)
                rows[-1]["store_run"] = twin
                print(self.format_store_line(twin, rows[-1]), flush=True)
        return rows

    @staticmethod
    def format_store_line(t: dict, base: dict) -> str:
        """The paging readout for one scale, under its no-store row.

        `cold_offsets` and `evicted_set` are deliberately NOT the evidence here.
        Both are written only on the legacy ColdTier path (crates/brain/src/pool.rs
        1974 and 2001, each guarded on `wbrain_store.is_none()`), so a .wbrain
        brain reports 0 for them however much it paged out. `evicted_neurons`,
        `page_outs` and `page_ins` are what move."""
        if t.get("hit_cap") or t.get("timed_out") or t.get("exit") not in (0, None):
            return f"{t.get('scale', '-'):>5}   with store: RUN DID NOT COMPLETE {t.get('error') or ''}"
        def num(key, fmt=".1f"):
            v = t.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) else "-"

        peak, bpeak = t.get("peak_mb"), base.get("peak_mb")
        delta = (f" (no-store {bpeak:.1f}, x{peak / bpeak:.2f})"
                 if isinstance(peak, (int, float)) and isinstance(bpeak, (int, float)) and bpeak
                 else "")
        zeros = t.get("zero_buckets") or {}
        unused = ",".join(k for k, v in zeros.items() if v == "unused_when_wbrain_attached")
        never = ",".join(k for k, v in zeros.items() if v == "never_allocated")
        s = t.get("scale", "-")
        return (
            f"{s:>5}   with store: recall {num('recall_pct')} integr {num('integration_pct')} "
            f"peak_mb {num('peak_mb')}{delta} slept {t.get('sleep_serialized')} "
            f"evicted {t.get('evicted_after_sleep')}/{t.get('neurons')} after sleep, "
            f"{t.get('evicted_neurons')} after answering "
            f"page_outs {t.get('page_outs')} page_ins {t.get('page_ins')} "
            f"clean_skips {t.get('clean_skips')} "
            f"resident_terminals {t.get('resident_terminals_after_sleep')} after sleep, "
            f"{t.get('resident_terminals')} after answering "
            f"container {(t.get('store_bytes') or 0) / 1_048_576.0:.1f} MB"
            f"\n{s:>5}   zero buckets: unused_when_wbrain_attached={unused or '-'}"
            f"; never_allocated={never or '-'}"
        )

    @staticmethod
    def header() -> str:
        # `wrong%` is non-empty answers that are not true in the world, over the
        # same probe count as `integr%`. The two do not sum to 100: the rest is
        # silence. It is here and not only in the JSON because until the
        # derivation was switched on it was 0 by construction, every family read
        # `[empty:N]`, and `integr%` alone was a sufficient summary. It stopped
        # being one the first run the derivation fired: a family went from
        # 0 correct / 0 wrong to 2 correct / 19 wrong and that reads as a pure
        # gain in `integr%`. A brain that invents an answer is worse in the
        # product than one that says nothing, so the cost of a gain prints
        # beside the gain.
        return (f"{'scale':>5} {'facts':>6} {'recall%':>8} {'integr%':>8} {'wrong%':>7} "
                f"{'peak_mb':>8} "
                f"{'est_mb':>7} {'hub':>8} {'terminals':>10} {'train_s':>8} {'wall_s':>7}")

    @staticmethod
    def format_row(r: dict) -> str:
        def f(k, fmt):
            v = r.get(k)
            return format(v, fmt) if isinstance(v, (int, float)) else format("-", fmt.split(".")[0].rstrip("fd"))
        line = (f"{f('scale','>5d')} {f('facts','>6d')} {f('recall_pct','>8.1f')} "
                f"{f('integration_pct','>8.1f')} {f('integration_wrong_pct','>7.1f')} "
                f"{f('peak_mb','>8.1f')} {f('est_resident_mb','>7.1f')} "
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
                + (f" lit {f['lit_mean']:.1f}/{f['lit_zero']}z" if "lit_mean" in f else "")
                for f in fams))
            if isinstance(r.get("trained_lit_mean"), float):
                line += (f"\n{r.get('scale', '-'):>5}   query-pool neurons lit per question: "
                         f"trained {r['trained_lit_mean']:.1f} ({r.get('trained_lit_zero')} of "
                         f"{r.get('facts')} lit nothing)")
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
            # Owner, 2026-10-01: recall of trained material is ALWAYS perfect.
            # An absolute rule, not "no worse than the baseline" -- a baseline
            # saved below 100% must not make a miss look acceptable.
            if r.get("recall_pct", 0) < 100.0:
                problems.append(f"scale {s}: recall {r.get('recall_pct', 0):.2f}% -- trained material must recall 100%")
            for k in ("recall_pct", "integration_pct"):
                if b and k in b and r[k] < b[k]:
                    problems.append(f"scale {s}: {k} fell {b[k]:.1f} -> {r[k]:.1f}")
            problems += self.invention_problems(r, b)
            problems += self.store_problems(r)
        return problems

    # How far integration_wrong_pct may rise over the baseline before the gate
    # reds. Two points, not zero: integration_pct itself is not reproducible run
    # to run (item 8accd975 -- two same-parameter draws measured 0.13 apart at
    # scale 64), so a zero-tolerance rule would red innocent passes on noise.
    WRONG_RISE_ALLOWED = 2.0

    def invention_problems(self, r: dict, b: dict | None) -> list[str]:
        """Invention is worse than silence, so it is gated in its own right.

        `integration_wrong_pct` counts non-empty integration answers that are
        NOT true in the world, over the same probe count as `integration_pct`.
        Two rules, and they do different jobs:

        * The ABSOLUTE rule. A derivation that is wrong more often than right is
          not a mechanism, it is a guess with a score attached, and no
          baseline-relative rule can see that -- measured 2026-10-01 at scale 64,
          20.5 % correct against 31.9 % wrong, with every gate rule green.
        * The RELATIVE rule. Precision must come from rejecting wrong
          derivations and never from deriving less, so `integration_pct` already
          may not fall. Its mirror is that the wrong count may not RISE: without
          it a change can buy integration by inventing more, which is the
          direction this scorecard was blind to for five passes.
        """
        bad = []
        w = r.get("integration_wrong_pct")
        if w is None:
            return bad
        s, i = r["scale"], r.get("integration_pct", 0.0)
        if w >= i:
            bad.append(f"scale {s}: integration_wrong_pct {w:.1f} is not below "
                       f"integration_pct {i:.1f} -- the derivation invents more than it derives")
        if b and (bw := b.get("integration_wrong_pct")) is not None:
            if w > bw + self.WRONG_RISE_ALLOWED:
                bad.append(f"scale {s}: integration_wrong_pct rose {bw:.1f} -> {w:.1f} "
                           f"(+{self.WRONG_RISE_ALLOWED:.0f} allowed) -- invention may not grow")
        return bad

    def store_problems(self, r: dict) -> list[str]:
        """What a store-attached run must hold, checked against the no-store run
        of the same scale rather than against a baseline -- there is no stored
        baseline for this mode yet, and these are absolute properties anyway.

        Paging to SSD must not lose a taught fact, and it must actually page:
        a store-attached run that evicted nothing is measuring the same brain
        the no-store run already measured, which is the defect this mode exists
        to end."""
        t = r.get("store_run")
        if not t:
            return []
        s, bad = t.get("scale"), []
        if t.get("hit_cap") or t.get("timed_out") or t.get("exit") not in (0, None):
            return [f"scale {s} with store: run did not complete cleanly"]
        if (t.get("recall_pct") or 0.0) < (r.get("recall_pct") or 0.0):
            bad.append(f"scale {s} with store: recall {t.get('recall_pct'):.1f} "
                       f"below no-store {r.get('recall_pct'):.1f} -- paging lost a taught fact")
        # At the SLEEP BOUNDARY, not at the end: the answer phases page bodies
        # back in, so the end-state count says how big the working set became,
        # not whether the store was ever used. Checking the end state read
        # `evicted 0/241` on a brain that had written all 241 bodies.
        if not (t.get("evicted_after_sleep") or 0) > 0:
            bad.append(f"scale {s} with store: evicted_after_sleep 0 -- the sleep paged nothing out, "
                       f"so this is not a measurement of the store")
        if not (t.get("page_outs") or 0) > 0:
            bad.append(f"scale {s} with store: page_outs 0 -- no body reached the container")
        return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", default="1,4,16")
    ap.add_argument("--cap-mb", type=int, default=3000, help="hard kill per run")
    ap.add_argument("--budget-mb", type=int, default=2048, help="gate: peak RAM allowed per run")
    ap.add_argument("--timeout", type=float, default=300, help="seconds per run")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--stress", action="store_true", help="add scale 64, reported but not budget-gated")
    ap.add_argument("--store-timeout", type=float, default=0,
                    help="seconds per store-attached run (default 10x --timeout: paging makes a "
                         "store run ~10x the wall time of its no-store twin)")
    ap.add_argument("--with-store", action="store_true",
                    help="run every scale a SECOND time with a .wbrain store attached to a temp "
                         "dir and the whole brain slept into it after training, and print the "
                         "paging readout under each no-store row (M4). Doubles the runtime, so it "
                         "is opt-in and the gate's own run stays the no-store one.")
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
    card = Scorecard(scales, args.cap_mb, args.budget_mb, args.timeout, report_only,
                     args.with_store, args.store_timeout)
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
    if args.with_store:
        STORE_RUN.write_text(json.dumps(rows, indent=1))
        print(f"store run saved: {STORE_RUN.relative_to(ROOT)}")
    # Only COMPLETED store runs. A run killed at the cap or the timeout still
    # carries the peak it had reached, and folding that in printed "RAM growth
    # with the store attached x2.12" off a scale-64 run that timed out at 300 s
    # -- a growth figure quoted against a scale that never finished.
    store_ok = [r["store_run"] for r in rows
                if isinstance((r.get("store_run") or {}).get("peak_mb"), (int, float))
                and not r["store_run"].get("timed_out")
                and not r["store_run"].get("hit_cap")
                and r["store_run"].get("exit") in (0, None)]
    if len(store_ok) >= 2:
        print(f"RAM growth with the store attached x{store_ok[-1]['peak_mb'] / store_ok[0]['peak_mb']:.2f} "
              f"from scale {store_ok[0]['scale']} to {store_ok[-1]['scale']} "
              f"(M4: independent of the corpus, so push toward 1.0)")
    if args.save_baseline and not problems:
        # The store twin is a second run of the same scale, so it must not enter
        # the baseline the no-store rows are compared against: by_scale would
        # then hold two rows per scale.
        BASELINE.write_text(json.dumps([{k: v for k, v in r.items() if k != "store_run"} for r in rows],
                                       indent=1))
        print(f"baseline saved: {BASELINE.relative_to(ROOT)}")
    if base is None and not args.save_baseline:
        print("no baseline yet: run with --save-baseline to set one")
    print("\nscorecard: " + ("FAILED -- " + "; ".join(problems) if problems else "OK"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
