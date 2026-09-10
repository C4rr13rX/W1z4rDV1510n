python3 - <<'PY'
"""Wait for `go-systems:0:131072` to clear its gate, and report the verdict.

The interval was quarantined 8 h ago by the defect c343443 repairs: a child
evaluator crash arrived as "returned non-zero exit status 1", a string with
the classifier's marker deleted, so 131,072 rows were scored a behavioural
regression. The repaired `programming_integrated_retention.py` is deployed
(`EvaluatorUnavailable` present, 0.8 h old), and the enterprise gate passed
12/12 an hour before this pass began.

So this waits on the ONE thing neither of those facts settles: what the gate
says this time. Watch the health ledger for a new terminal event rather than
the progress file, because the row freezes by design during settlement, the
admission gate and the continuous canary -- a frozen row here is expected and
is not the verdict.

Read-only: tails a ledger and re-reads state files.
"""
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
HEALTH = f"{R}/curriculum-health.jsonl"
DEADLINE = time.time() + 1500.0

TERMINAL = {
    "deferred_replay_admitted", "deferred_replay_failed",
    "midphase_gate_failed", "completion_gate_failed",
    "midphase_failure_scope_corrected", "quarantine_retest_admitted",
    "enterprise_gate_confirmation", "enterprise_gate_unconfirmed",
    "completion_gate_infrastructure_retry", "unrestorable_quarantine_retired",
    "automatic_quarantine_recovery", "fully_deferred_block_advanced",
}

start_size = os.path.getsize(HEALTH)
out = {"start_size": start_size, "events": []}


def row_now():
    try:
        body = json.load(open(f"{R}/deferred-replay-b01232b593532da2.resume.json"))
        return int(body.get("durable_next_row") or -1)
    except Exception:
        return -1


out["row_at_start"] = row_now()

while time.time() < DEADLINE:
    time.sleep(20.0)
    try:
        size = os.path.getsize(HEALTH)
    except OSError:
        continue
    if size <= start_size:
        continue
    with open(HEALTH, encoding="utf-8", errors="replace") as handle:
        handle.seek(start_size)
        fresh = handle.read()
    start_size = size
    for line in fresh.splitlines():
        try:
            event = json.loads(line)
        except Exception:
            continue
        kind = str(event.get("kind") or "")
        error = str(event.get("error") or "")
        out["events"].append({
            "kind": kind,
            "phase": event.get("phase"),
            "interval_id": event.get("interval_id"),
            "passed": event.get("passed"),
            # Head AND tail: a traceback's cause is on its last line, and the
            # first 180 characters of a gate report are tick counters.
            "error": (error if len(error) <= 700
                      else error[:200] + " ...[cut]... " + error[-450:]),
        })
    if any(e["kind"] in TERMINAL for e in out["events"]):
        break

out["row_at_end"] = row_now()
out["waited_seconds"] = round(1500.0 - max(0.0, DEADLINE - time.time()), 1)
try:
    out["status"] = json.load(open(f"{R}/curriculum-supervisor.status.json"))
except Exception as error:
    out["status_error"] = str(error)[:120]
try:
    out["active"] = json.load(open(f"{R}/deferred-replay-active.json"))
except Exception:
    out["active"] = None

print("PROBEJSON " + json.dumps(out))
PY
