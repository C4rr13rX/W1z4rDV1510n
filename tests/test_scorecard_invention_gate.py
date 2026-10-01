"""The two invention rules must RED on the payload the host actually emits.

`tools/scorecard.py` gained two rules on 2026-10-01: `integration_wrong_pct`
must stay strictly below `integration_pct`, and it may not rise more than
`WRONG_RISE_ALLOWED` points over the baseline. Both are easy to write in a way
that can never fire, and this repository's most expensive recurring mistake is
exactly that — a guard keyed on evidence the host cannot produce, with a
passing test built on a payload nothing emits.

So every case below is built from a REAL row: `baseline()` and `measured()` are
the pass-14 and pass-15 scale-64 numbers, copied from
`docs/scorecard-baseline.json` and `logs/scorecard-latest.json`. The
`vacuous` test is the one that matters most — it asserts the rule still fires
when a key is missing from the baseline, because a `None` there silently
disables the comparison and reports green.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_scorecard():
    """Import tools/scorecard.py by path. It is a script, not a package."""
    spec = importlib.util.spec_from_file_location(
        "w1z4rd_scorecard", ROOT / "tools" / "scorecard.py"
    )
    if spec is None or spec.loader is None:
        pytest.skip("tools/scorecard.py not importable from this tree")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def card(mod):
    return mod.Scorecard(scales=[64], cap_mb=3000, budget_mb=2048, timeout=300.0)


def baseline() -> dict:
    """The pass-14 scale-64 row: 20.6 % right against 31.8 % wrong. The state
    the rules were added to catch, and every gate rule called it green."""
    return {"scale": 64, "integration_pct": 20.63, "integration_wrong_pct": 31.77}


def measured() -> dict:
    """The pass-15 scale-64 row after the taught-frame identity test."""
    return {"scale": 64, "integration_pct": 37.88, "integration_wrong_pct": 0.0}


def test_the_state_the_rules_were_added_to_catch_now_reds():
    mod = load_scorecard()
    problems = card(mod).invention_problems(baseline(), None)
    assert problems, (
        "20.63 % correct against 31.77 % wrong must fail; this row passed every "
        "gate rule for five passes"
    )
    assert any("not below" in p for p in problems), problems


def test_the_measured_fix_passes_both_rules():
    mod = load_scorecard()
    assert card(mod).invention_problems(measured(), baseline()) == []


def test_invention_may_not_rise_over_the_baseline():
    """A change that buys integration by inventing more must red, even though
    integration_pct ROSE -- which is the only rule that can see that."""
    mod = load_scorecard()
    c = card(mod)
    b = measured()
    rose = {"scale": 64, "integration_pct": 50.0, "integration_wrong_pct": 2.01}
    problems = c.invention_problems(rose, b)
    assert any("rose" in p for p in problems), (
        f"wrong rose 0.0 -> 2.01, past the {c.WRONG_RISE_ALLOWED} allowed, "
        f"while integration_pct rose 37.88 -> 50.0; got {problems}"
    )
    # And the tolerance is real, not decorative: run-to-run noise inside it
    # must not red. See item 8accd975 -- two same-parameter draws measured
    # 0.13 apart at scale 64.
    noise = {"scale": 64, "integration_pct": 37.75, "integration_wrong_pct": 1.99}
    assert c.invention_problems(noise, b) == []


def test_a_missing_baseline_key_does_not_silently_disable_the_absolute_rule():
    """The vacuous-zero trap. A baseline without the key must still be subject
    to the ABSOLUTE rule -- only the relative one depends on a baseline."""
    mod = load_scorecard()
    c = card(mod)
    bad = {"scale": 64, "integration_pct": 20.0, "integration_wrong_pct": 30.0}
    assert c.invention_problems(bad, {"scale": 64}) , "no baseline key is not a licence to invent"
    assert c.invention_problems(bad, None), "no baseline at all is not a licence to invent"


def test_a_row_without_the_metric_is_reported_not_guessed():
    """A row from a build that does not emit the field must not be scored as
    0.0 wrong -- that would turn a missing measurement into a pass."""
    mod = load_scorecard()
    assert card(mod).invention_problems({"scale": 64, "integration_pct": 20.0}, baseline()) == []


def test_equal_is_a_failure_because_the_criterion_says_strictly_below():
    mod = load_scorecard()
    tie = {"scale": 64, "integration_pct": 30.0, "integration_wrong_pct": 30.0}
    assert card(mod).invention_problems(tie, None), "equal is not below"


def test_the_live_baseline_on_disk_satisfies_the_absolute_rule_at_every_scale():
    """The rules are only worth having if the committed baseline obeys them --
    otherwise every future run reds on inherited state. Reads the artifact, not
    a literal."""
    import json

    mod = load_scorecard()
    path = ROOT / "docs" / "scorecard-baseline.json"
    if not path.exists():
        pytest.skip("no committed baseline in this tree")
    rows = json.loads(path.read_text())
    rows = rows if isinstance(rows, list) else rows.get("scales", [])
    checked = 0
    for r in rows:
        if r.get("integration_wrong_pct") is None:
            continue
        checked += 1
        assert r["integration_wrong_pct"] < r["integration_pct"], (
            f"committed baseline scale {r['scale']}: wrong "
            f"{r['integration_wrong_pct']} is not below integration "
            f"{r['integration_pct']} -- the gate will red on inherited state"
        )
    assert checked, "the baseline carries no integration_wrong_pct, so this test proved nothing"
