python3 - <<'PY'
"""Wait, on the host, for the go-systems block to reach its gate and report.

One SSM call that blocks remotely beats polling from the laptop: each poll is
a command invocation, and sixteen of them over eighty minutes add load to a
host already holding a 3 GB free-memory floor.

The closing measurement is NOT "a process is alive" and NOT "durable_next_row
reached the target". It is a new resolution event in curriculum-health.jsonl
and `hours_since_admission` falling. `phase_forward_harvested` is the event
for a forward block; `deferred_replay_admitted` is the replay equivalent.

Failure states are watched as deliberately as success. A gate that rejects
writes `completion_gate_failed`; a worker that dies writes nothing at all, so
the loop also reports the supervisor going away and the row going stale. A
watcher that only greps for the happy path is silent through a crash, and
silence reads identical to "still running".
"""
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
LEDGER = os.path.join(R, "curriculum-health.jsonl")
STATUS = os.path.join(R, "curriculum-supervisor.status.json")
DEADLINE = time.time() + 75 * 60
TERMINAL = {
    "phase_forward_harvested", "deferred_replay_admitted",
    "quarantine_retest_admitted", "fully_deferred_block_advanced",
    "completion_gate_failed", "midphase_gate_failed",
}


def ledger_size():
    try:
        return os.path.getsize(LEDGER)
    except OSError:
        return 0


def read_new(offset):
    events = []
    try:
        with open(LEDGER, encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
            return events, handle.tell()
    except OSError:
        return events, offset


def status():
    try:
        with open(STATUS, encoding="utf-8") as handle:
            data = json.load(handle)
        return (data.get("durable_next_row"), data.get("state"),
                data.get("phase"), round(time.time() - os.path.getmtime(STATUS), 1))
    except Exception:  # noqa: BLE001
        return None, None, None, None


def avail_gb():
    for line in open("/proc/meminfo", encoding="utf-8"):
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
    return 0.0


offset = ledger_size()
last_row, _, _, _ = status()
seen = []
while time.time() < DEADLINE:
    time.sleep(45)
    events, offset = read_new(offset)
    row, state, phase, age = status()
    for event in events:
        kind = str(event.get("kind") or "")
        if kind in TERMINAL or "gate" in kind or "admit" in kind:
            seen.append({
                "kind": kind,
                "phase": event.get("phase"),
                "passed": event.get("passed"),
                "reason": str(event.get("reason") or "")[:300],
                "first": event.get("first_passed_suites"),
                "confirm": event.get("confirm_passed_suites"),
                "total": event.get("total_suites"),
            })
        if kind in TERMINAL:
            print("PROBE_JSON " + json.dumps({
                "outcome": kind, "row": row, "state": state, "phase": phase,
                "avail_gb": avail_gb(), "events": seen[-12:],
                "waited_s": round(time.time() - (DEADLINE - 75 * 60)),
            }, default=str))
            raise SystemExit(0)
    # The block finished but no verdict yet: the gate itself is running.
    if state and state != "running" and not seen:
        seen.append({"kind": "state_change", "state": state, "row": row})
    # Nothing writing the status file is a fault, not quiet progress.
    if age is not None and age > 900:
        print("PROBE_JSON " + json.dumps({
            "outcome": "status_stale", "status_age_s": age, "row": row,
            "state": state, "avail_gb": avail_gb(), "events": seen[-12:],
        }, default=str))
        raise SystemExit(0)
    last_row = row if row is not None else last_row

row, state, phase, age = status()
print("PROBE_JSON " + json.dumps({
    "outcome": "deadline_no_verdict", "row": row, "state": state,
    "phase": phase, "status_age_s": age, "avail_gb": avail_gb(),
    "events": seen[-12:],
}, default=str))
PY
