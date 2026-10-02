"""Assert the README's numbers against docs/scorecard-baseline.json.

    python tools/check_doc_numbers.py          # exit 0 = every claim matches
    python tools/check_doc_numbers.py -v       # print every claim, not just failures

# Why this exists, and why it is not the rule you already follow

The standing rule is "read the number back off the artifact". Printing a value
before pasting it into a doc obeys the letter of that and misses the thing it
is for: what ends up in the doc is the PASTED text, and nothing had ever
compared the pasted text to the file. This script does that direction --
regex the rendered markdown, parse the baseline JSON, compare.

It found a stale claim on its first real run, and not the one it was written
for. `### Atom fan-out is bounded, and what that bound costs` carried an A/B
table whose left-hand column was the pre-`dea50bd` baseline: 44.3 % / 35.8 %
wrong at scale 16 and 20.5 % / 31.9 % at scale 64, against a current 77.55 %
and 77.29 % at 0.0 % wrong. A reader of the README would conclude the brain
answers wrong a third of the time. The number had not changed in the doc
because nobody re-reads a table they did not write.

# How to add a claim

Append to `CLAIMS`: a `pattern` with one or more capture groups, and a `value`
that takes the loaded baseline (a dict keyed by scale) and returns the tuple
the groups should equal. Floats compare at `TOL`, which is 0.051 -- one unit
of the last digit a README ever quotes. A pattern that matches NOTHING is a
failure, not a pass: a claim nobody can find is how a doc edit silently drops
a check (`tests/vacuous_zero` is this lab's name for the general case).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BASELINE = ROOT / "docs" / "scorecard-baseline.json"
TOL = 0.051


def load() -> dict:
    rows = json.loads(BASELINE.read_text(encoding="utf-8"))
    by_scale = {}
    for row in rows:
        fam = {f["name"]: f for f in row.get("integration_families", [])}
        by_scale[row["scale"]] = (row, fam)
    return by_scale


def fam(b: dict, scale: int, name: str) -> dict:
    return b[scale][1][name]


def row(b: dict, scale: int) -> dict:
    return b[scale][0]


# Each entry: (label, regex over README.md, fn(baseline) -> tuple of expected
# group values). Strings compare exactly; numbers at TOL.
CLAIMS = [
    (
        "latency table",
        re.compile(
            r"^\| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)\S \| ([\d.]+) \| ([\d.]+) \|$",
            re.M,
        ),
        lambda b: [
            (
                s,
                round(row(b, s)["recall_ms"], 3),
                round(row(b, s)["infer_ms"], 3),
                round(row(b, s)["infer_ms"] / row(b, s)["recall_ms"], 1),
                round(row(b, s)["derivation_probes_per_attempt"], 2),
                round(row(b, s)["infer_ms"] / row(b, s)["derivation_probes_per_attempt"], 3),
            )
            for s in sorted(b)
        ],
    ),
    (
        "3-hop hits at scale 64",
        re.compile(r"answers \*\*(\d+) of (\d+) at scale 64\*\*"),
        lambda b: [(fam(b, 64, "next_on_material")["hits"], fam(b, 64, "next_on_material")["probes"])],
    ),
    (
        "starvation totals at scale 64",
        re.compile(r"of the \*\*(\d+)\*\* attempts that\nend starved at scale 64, \*\*(\d+)\*\* are"),
        lambda b: [
            (
                sum(f["derivation_starved"] for f in row(b, 64)["integration_families"]),
                fam(b, 64, "next_on_material")["derivation_starved"]
                + fam(b, 64, "beside_next")["derivation_starved"],
            )
        ],
    ),
    (
        "the working families' probe cost",
        re.compile(
            r"`next_color` is (\d+)/(\d+) at \*\*([\d.]+)\*\*\nprobes per attempt and "
            r"`on_material` is (\d+)/(\d+) at \*\*([\d.]+)\*\*"
        ),
        lambda b: [
            (
                fam(b, 64, "next_color")["hits"],
                fam(b, 64, "next_color")["probes"],
                round(fam(b, 64, "next_color")["probes_per_attempt"], 2),
                fam(b, 64, "on_material")["hits"],
                fam(b, 64, "on_material")["probes"],
                round(fam(b, 64, "on_material")["probes_per_attempt"], 2),
            )
        ],
    ),
    (
        "the stale-ablation correction",
        re.compile(
            r"scale 16 is \*\*([\d.]+) %\*\* integration at \*\*([\d.]+) %\*\*\nwrong and scale 64 is \*\*([\d.]+) %\*\* at \*\*([\d.]+) %\*\*"
        ),
        lambda b: [
            (
                round(row(b, 16)["integration_pct"], 2),
                round(row(b, 16)["integration_wrong_pct"], 1),
                round(row(b, 64)["integration_pct"], 2),
                round(row(b, 64)["integration_wrong_pct"], 1),
            )
        ],
    ),
]


def same(got: str, want) -> bool:
    if isinstance(want, str):
        return got == want
    try:
        return abs(float(got) - float(want)) <= TOL
    except ValueError:
        return False


def main() -> int:
    verbose = "-v" in sys.argv
    text = README.read_text(encoding="utf-8")
    b = load()
    problems = []
    for label, pattern, expected in CLAIMS:
        found = [m.groups() for m in pattern.finditer(text)]
        want = [tuple(t) for t in expected(b)]
        if not found:
            # A claim nobody can find is a check that silently stopped running.
            problems.append(f"{label}: pattern matched NOTHING in README.md")
            continue
        if len(found) != len(want):
            problems.append(f"{label}: README has {len(found)} row(s), baseline has {len(want)}")
            continue
        for i, (got_row, want_row) in enumerate(zip(found, want)):
            for got, w in zip(got_row, want_row):
                if not same(got, w):
                    problems.append(f"{label} row {i}: README {got!r} vs baseline {w!r}")
        if verbose:
            print(f"OK  {label}: {len(found)} row(s)")
    if problems:
        print("README numbers vs baseline artifact: MISMATCH")
        for p in problems:
            print("  " + p)
        return 1
    print(f"README numbers vs baseline artifact: ALL MATCH ({len(CLAIMS)} claim groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
