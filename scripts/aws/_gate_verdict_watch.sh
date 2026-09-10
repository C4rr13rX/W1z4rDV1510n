python3 - <<'PY'
"""Block on the host until the go-systems block reaches its gate, then report
WHICH suites passed -- not merely whether the gate did.

The block was at row 59648 of 131072 advancing ~21 rows/s, so the gate is due
in roughly an hour. The question it settles is narrow: `polyglot` has failed
127 consecutive enterprise gates on one row, that row now composes correctly
against the running image, and the gate runs the suite TWICE (first, then
confirm) and rejects on either.

Two deliberate differences from the earlier gate watch. It does not exit on a
stale status file: settlement plus twenty-four suite executions freeze the
status file for well over the 900 s that watch treated as a fault, so that
rule would fire on the very event being waited for. And on any gate verdict it
reads `<phase>.enterprise-gate.json` and prints the per-suite names, because
the ledger record carries only counts -- an 11/12 that names no suite is what
made this drought look causeless for four days.

Failure states are watched as deliberately as success: a rejection, a
supervisor that goes away, and a row that stops advancing all print. A watcher
that greps only for the happy path is silent through a crash, and silence
reads exactly like "still running".
"""
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
LEDGER = os.path.join(R, "curriculum-health.jsonl")
STATUS = os.path.join(R, "curriculum-supervisor.status.json")
DEADLINE = time.time() + 82 * 60
TERMINAL = {
    "phase_forward_harvested", "deferred_replay_admitted",
    "quarantine_retest_admitted", "fully_deferred_block_advanced",
    "completion_gate_failed", "midphase_gate_failed",
    "enterprise_gate_confirmation",
}


def size():
    try:
        return os.path.getsize(LEDGER)
    except OSError:
        return 0


def read_new(offset):
    events = []
    try:
        with open(LEDGER, encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            for line in fh:
                line = line.strip()
                if line.startswith("{"):
                    try:
                        events.append(json.loads(line))
                    except Exception:  # noqa: BLE001
                        pass
            return events, fh.tell()
    except OSError:
        return events, offset


def status():
    try:
        with open(STATUS, encoding="utf-8") as fh:
            data = json.load(fh)
        return data, round(time.time() - os.path.getmtime(STATUS), 1)
    except Exception:  # noqa: BLE001
        return {}, None


def suite_verdict(phase):
    """Per-suite names from the gate artifact; the ledger has only counts."""
    path = os.path.join(R, f"{phase}.enterprise-gate.json")
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "path": path}
    return {
        "age_s": round(time.time() - os.path.getmtime(path), 1),
        "passed": data.get("passed"),
        "passed_suites": data.get("passed_suites"),
        "total_suites": data.get("total_suites"),
        "infrastructure_only_failure": data.get("infrastructure_only_failure"),
        "failing": [r.get("name") for r in data.get("results", [])
                    if not r.get("passed")],
        "polyglot": next((r.get("summary") for r in data.get("results", [])
                          if r.get("name") == "polyglot"), None),
    }


offset = size()
seen, rows = [], []
while time.time() < DEADLINE:
    time.sleep(40)
    events, offset = read_new(offset)
    data, age = status()
    row, state, phase = (data.get("durable_next_row"), data.get("state"),
                         data.get("phase"))
    rows.append((round(time.time()), row))
    for ev in events:
        kind = str(ev.get("kind") or "")
        if "gate" in kind or "admit" in kind or "harvest" in kind:
            seen.append({
                "kind": kind, "phase": ev.get("phase"),
                "passed": ev.get("passed"),
                "first": ev.get("first_passed_suites"),
                "confirm": ev.get("confirm_passed_suites"),
                "passed_suites": ev.get("passed_suites"),
                "total": ev.get("total_suites"),
            })
        if kind in TERMINAL:
            print("PROBE_JSON " + json.dumps({
                "outcome": kind, "row": row, "state": state, "phase": phase,
                "status_age_s": age, "events": seen[-14:],
                "suite_verdict": suite_verdict(phase or "go-systems"),
                "waited_min": round((time.time() - (DEADLINE - 82 * 60)) / 60, 1),
            }, default=str))
            raise SystemExit(0)
    # The supervisor going away is a fault; the status file freezing is NOT --
    # settlement and 24 suite executions legitimately stop it for many minutes.
    if not os.path.exists(STATUS):
        print("PROBE_JSON " + json.dumps({"outcome": "status_missing"}))
        raise SystemExit(0)

data, age = status()
recent = [r for _, r in rows[-6:] if r is not None]
print("PROBE_JSON " + json.dumps({
    "outcome": "deadline_no_verdict",
    "row": data.get("durable_next_row"), "state": data.get("state"),
    "phase": data.get("phase"), "status_age_s": age,
    "row_advanced": (len(set(recent)) > 1) if recent else None,
    "events": seen[-14:],
}, default=str))
PY
