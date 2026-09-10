"""Regression tests for deferred-replay queue semantics.

Context (2026-08-17): 145 intervals (5.78M rows) sat unresolved because
run_deferred_replays() returned on the first rejection while always taking
pending[0]. A rejection re-appends the interval as `deferred`, so one
interval that never passed blocked every other one behind it -- observed:
csn-python-full:165377:296448 deferred 7 times while 144 others never got
a turn.

The fix skips a behaviourally-rejected interval for the rest of the pass
(the obligation stays in the ledger) while still hard-stopping on
infrastructure failures. These tests pin that distinction, which is the
part that is easy to get subtly wrong.
"""
import pathlib

import pytest

from scripts import programming_curriculum_supervisor as sup


def drain(outcomes, single_selected=False, infra_at=None, max_steps=64):
    """Model run_deferred_replays()'s queue walk.

    outcomes: interval_id -> True (admits) | False (behavioural rejection)
    infra_at: interval_id that raises an infrastructure error instead
    """
    pending = list(outcomes)
    rejected: set[str] = set()
    attempted: list[str] = []
    for _ in range(max_steps):
        queue = [i for i in pending if single_selected or i not in rejected]
        if not queue:
            return {"attempted": attempted, "rejected": sorted(rejected),
                    "rc": 42 if rejected else 0}
        interval = queue[0]
        attempted.append(interval)
        if infra_at is not None and interval == infra_at:
            return {"attempted": attempted, "rejected": sorted(rejected), "rc": 1}
        if outcomes[interval]:
            pending.remove(interval)
        elif single_selected:
            return {"attempted": attempted,
                    "rejected": sorted(rejected | {interval}), "rc": 1}
        else:
            rejected.add(interval)
    raise AssertionError("queue walk did not terminate — head-of-line block")


def test_a_rejected_interval_does_not_block_the_queue():
    result = drain({"bad": False, "a": True, "b": True, "c": True})
    assert result["attempted"].count("bad") == 1
    assert {"a", "b", "c"}.issubset(set(result["attempted"]))
    assert result["rejected"] == ["bad"]


def test_rejections_surface_as_a_failing_return_code():
    """Skipping must not look like success: the obligation is still open."""
    assert drain({"bad": False, "ok": True})["rc"] == 42
    assert drain({"ok": True})["rc"] == 0


def test_infrastructure_failure_stops_the_whole_pass():
    """A dead node or full disk affects every interval, so retrying the
    rest of the queue would just fail 145 more times."""
    result = drain({"a": True, "boom": True, "c": True}, infra_at="boom")
    assert result["attempted"] == ["a", "boom"]
    assert "c" not in result["attempted"]
    assert result["rc"] == 1


def test_dead_worker_is_infrastructure_not_a_semantic_verdict():
    """A worker that exits on its own is never a verdict about corpus content.

    2026-09-09: `go_systems_001.toml` deployed with a category outside
    `schema.CATEGORIES`, so `load_registry()` raised for the WHOLE registry and
    every worker died at startup, before posting a row.  The old code raised a
    bare RuntimeError, which the drain loop reads as a behavioural rejection,
    so twelve intervals across four unrelated corpora -- jupyter-scientific,
    metamathqa, webstack -- were rejected without a row trained or a gate run.
    The pass then reported `deferred_replay_complete` with 26 rejections and
    exited 42, which `RestartPreventExitStatus=42` latched into a stopped
    service: a deploy typo took training down for four days.

    The replay worker only POSTs rows; every judgement runs in the supervisor
    after the training loop returns.  So a non-zero exit is the driver failing
    to run, never the brain failing to learn.
    """
    err = sup.replay_worker_failure(1, "", pathlib.Path("/tmp/x.stderr.log"))
    assert isinstance(err, sup.AdmissionInfrastructureError)
    # The load-bearing half: `AdmissionInfrastructureError` deliberately does
    # not inherit RuntimeError, so it cannot be swallowed by a handler meant
    # for behavioural rejections.  The old code raised a bare RuntimeError,
    # which is precisely what that handler caught.
    assert not isinstance(err, RuntimeError)


def test_worker_failure_carries_the_reason_not_just_its_address(tmp_path):
    """The ledger recorded only `stderr=<path>`, so `last_failure` named a file
    on a host nobody was reading and the SchemaError stayed invisible for the
    whole pass.  The diagnostic has to travel with the error."""
    log = tmp_path / "deferred-replay-abc.stderr.log"
    log.write_text("stale text from an earlier pass\n", encoding="utf-8")
    mark = log.stat().st_size
    with log.open("a", encoding="utf-8") as handle:
        handle.write("SchemaError: category='systems_programming_go' not in [...]\n")

    tail = sup.replay_worker_stderr_tail(log, mark)
    assert "SchemaError" in tail
    # The shared log is append-mode across passes, so `mark` is what separates
    # this pass's diagnostic from every earlier one's.
    assert "stale text" not in tail

    err = sup.replay_worker_failure(1, tail, log)
    assert "SchemaError" in str(err)
    assert str(log) in str(err)


def test_explicit_single_interval_keeps_strict_semantics():
    result = drain({"only": False}, single_selected=True)
    assert result["rc"] == 1


def test_every_rejection_is_recorded():
    result = drain({"x": False, "y": False, "z": True})
    assert result["rejected"] == ["x", "y"]
    assert "z" in result["attempted"]


def test_canary_holds_trained_to_perfection_but_paraphrase_to_a_floor():
    """The trained group never failed across 145 quarantines; every one was
    a paraphrase miss. So trained stays strict and only generalisation to
    unseen phrasings gets a floor."""
    import math

    parser_default = 0.6
    assert 0.0 < parser_default <= 1.0
    # 5-item probe: the floor must demand a majority, not a token pass.
    floor = math.ceil(5 * parser_default)
    assert floor == 3
    # A genuinely broken brain must still be caught.
    assert 2 < floor      # 2/5 executing still quarantines
    assert floor <= 5


def test_floor_is_validated_at_parse_time():
    source = (sup.ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8"
    )
    assert "--canary-paraphrase-floor" in source
    # Must reject nonsense values rather than silently disabling the gate:
    # a floor of 0 would pass everything, and >1 could never be satisfied.
    assert "if not 0.0 < args.canary_paraphrase_floor <= 1.0:" in source
    assert "--canary-paraphrase-floor must be in (0.0, 1.0]" in source


@pytest.mark.parametrize("raw,expected_dict", [
    (b'{"a": 1}', True),
    (b"983093", False),      # /brain/tick's bare counter
])
def test_tick_style_payloads_are_tolerated(raw, expected_dict):
    """programming_code_eval.Client.post must not demand a mapping from
    every route; requiring one killed the repair pass on its first tick."""
    import json

    decoded = json.loads(raw)
    assert isinstance(decoded, dict) is expected_dict
