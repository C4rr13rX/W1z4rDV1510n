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


# ---------------------------------------------------------------------------
# A verdict-less interval consumes every generation (2026-09-11).
#
# The tests above pin the WITHIN-pass fix: an interval that reaches its gate
# and loses joins `rejected_this_pass` and the queue moves on. The failure
# below is the one that survives that fix, because it never reaches a gate at
# all -- it kills the generation, so the set it would have joined dies with
# the process and `unresolved_deferred_intervals`' `(phase, start_row)` order
# hands the restart exactly the same `pending[0]`.
# ---------------------------------------------------------------------------

def _interval(interval_id, phase, start, end):
    return {"interval_id": interval_id, "phase": phase,
            "start_row": start, "end_row": end, "status": "deferred"}


def test_a_stall_is_remembered_across_restarts(tmp_path):
    """The in-memory set cannot carry this fact; the ledger must."""
    runtime = tmp_path
    assert sup.replay_stall_counts(runtime) == {}

    sup.record_replay_stall(runtime, "jupyter:201344:262144",
                            "jupyter", "disk halt")
    sup.record_replay_stall(runtime, "jupyter:201344:262144",
                            "jupyter", "disk halt")
    sup.record_replay_stall(runtime, "other:0:1024", "other", "reboot")

    # Read back by a DIFFERENT call, standing in for the next generation.
    counts = sup.replay_stall_counts(runtime)
    assert counts == {"jupyter:201344:262144": 2, "other:0:1024": 1}

    ledger = (runtime / "curriculum-health.jsonl").read_text(encoding="utf-8")
    assert sup.REPLAY_STALL_KIND in ledger
    # Append-only: recording must not rewrite or drop earlier history.
    assert ledger.count(sup.REPLAY_STALL_KIND) == 3


def test_the_queue_tries_something_else_after_a_generation_is_consumed():
    """The measured host: one 60,800-row span re-selected forever.

    `jupyter-scientific-full:201344:262144` took 361 resource yields and 14
    gate failures over 437 h -- every one of those 14 at-gate, the newest 86 h
    old, so its recorded cause (`polyglot` 11/12) predates the Go corpus that
    closed it and the freshest gate artifact passes 12/12. What blocks it now
    is arithmetic: ~5.9 h of training against a ~1.26 h disk window at the
    measured 237.65 GB/h. It cannot produce a verdict, so it can never be
    skipped, while smaller intervals behind it never get a turn.
    """
    pending = [
        _interval("jupyter-scientific-full:201344:262144",
                  "jupyter-scientific-full", 201344, 262144),
        _interval("jupyter-scientific-full:262144:393216",
                  "jupyter-scientific-full", 262144, 393216),
        _interval("zz-small:0:14336", "zz-small", 0, 14336),
    ]
    # Generation 1 picks pending[0] and is killed by the disk floor.
    assert sup.order_replay_candidates(pending, {})[0]["interval_id"] == \
        "jupyter-scientific-full:201344:262144"

    # Every restart repeats that with no memory of it.
    stalls = {}
    for _ in range(5):
        chosen = sup.order_replay_candidates(pending, stalls)[0]
        assert chosen["interval_id"] == "jupyter-scientific-full:201344:262144"

    # With the stall recorded, the next generation tries something else.
    stalls = {"jupyter-scientific-full:201344:262144": 1}
    assert sup.order_replay_candidates(pending, stalls)[0]["interval_id"] == \
        "jupyter-scientific-full:262144:393216"

    # And the obligation is only DEPRIORITISED, never dropped: it is still in
    # the queue, so it is retried once the rest have had their turn.
    ordered = sup.order_replay_candidates(pending, stalls)
    assert len(ordered) == len(pending)
    assert {event["interval_id"] for event in ordered} == \
        {event["interval_id"] for event in pending}


def test_a_host_with_no_stalls_selects_exactly_what_it_does_now():
    """The reorder must be inert until something actually stalls."""
    pending = [
        _interval("a:0:1024", "a", 0, 1024),
        _interval("b:0:1024", "b", 0, 1024),
        _interval("b:1024:2048", "b", 1024, 2048),
    ]
    assert sup.order_replay_candidates(pending, {}) == pending
    # Stable: equal stall counts preserve the (phase, start_row) tiebreak the
    # ledger fold already applied.
    equal = {event["interval_id"]: 3 for event in pending}
    assert sup.order_replay_candidates(pending, equal) == pending


def test_the_stall_record_is_written_before_the_rollback_discards_it():
    """Ordering matters: the rollback clears the interval's resume state.

    `restore_rejected_deferred_replay` deliberately unlinks the resume record
    and the progress file so the retry replays from the first row. If the
    stall were recorded after that call, a crash between the two would lose
    the only evidence that this interval consumed a generation.
    """
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    recover = source.split("def recover_interrupted_deferred_replay")[1]
    recover = recover.split("\ndef ")[0]
    assert "record_replay_stall" in recover
    assert recover.index("record_replay_stall") < \
        recover.index("restore_rejected_deferred_replay(")


def test_the_reorder_is_actually_wired_into_the_selection_loop():
    """A helper nothing calls is the inert-fix trap this repo keeps hitting.

    `order_replay_candidates` only matters if `run_deferred_replays` applies
    it to the queue it actually selects from, AFTER the `rejected_this_pass`
    filter and BEFORE `pending[0]` is taken.
    """
    source = (sup.ROOT / "scripts"
              / "programming_curriculum_supervisor.py").read_text(
        encoding="utf-8")
    body = source.split("def run_deferred_replays")[1].split("\ndef ")[0]
    assert "order_replay_candidates(" in body, \
        "run_deferred_replays never reorders its queue"
    assert "replay_stall_counts(" in body
    # Anchor on the STATEMENTS, not on prose: this function's comments discuss
    # `pending[0]` and `rejected_this_pass` several lines before either is
    # executed, and a naive substring search scores the comment.
    selection = body.index("event = pending[0]")
    reorder = body.index("order_replay_candidates(")
    skip_filter = body.index("not in rejected_this_pass")
    assert reorder < selection, \
        "the reorder must happen before the interval is chosen"
    assert skip_filter < reorder, \
        "the reorder must apply to the already-filtered queue"
