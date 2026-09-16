python3 - <<'PY'
"""Is the halt STABLE, and is its arithmetic reproducible from the host?

`auto-restart` is a crash loop, not an absence -- this repository has already
read a 115x ENOSPC restart loop as "the stage ended" because a process census
landed between restarts. So read `ActiveState`/`SubState`/`NRestarts` rather
than inferring from a PID, and confirm the refusal is terminal rather than
re-firing every ten seconds.

Then re-derive the capacity claim independently of the supervisor: free space
by `df`, burn from the ledger, and the queue's cost at the fastest rate the
host has ever measured. A halt this consequential should not rest on one
emitter's own arithmetic.

Read-only.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
sys.path.insert(0, "/srv/wizard/project")
out = {"now": time.time()}


def sh(cmd, timeout=60):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "").strip()[-1500:]
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)


out["unit"] = sh("systemctl show wizard-curriculum-supervisor.service "
                 "-p ActiveState -p SubState -p Result -p ExecMainStatus "
                 "-p NRestarts -p MainPID -p RestartPreventExitStatus "
                 "--no-pager")
out["since"] = sh("systemctl show wizard-curriculum-supervisor.service "
                  "-p ActiveEnterTimestamp -p InactiveEnterTimestamp "
                  "--no-pager")

# A supervisor that exited must leave no wrapper or worker behind training.
census = {"wrapper": 0, "supervisor": 0, "worker": 0, "brain": 0}
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        continue
    parts = [p for p in cmd.split("\0") if p]
    joined = " ".join(parts)
    if "run_programming_curriculum_service.sh" in joined:
        census["wrapper"] += 1
    if "programming_curriculum_supervisor" in joined:
        census["supervisor"] += 1
    if "drive_corpora_brain" in joined:
        census["worker"] += 1
    if "w1z4rd_brain_server" in joined:
        census["brain"] += 1
out["process_census"] = census

usage = shutil.disk_usage(R)
out["free_gb"] = round(usage.free / 2 ** 30, 2)
out["total_gb"] = round(usage.total / 2 ** 30, 2)

# Independent re-derivation of the capacity claim.
from scripts.programming_curriculum_supervisor import (  # noqa: E402
    measure_disk_burn_gb_per_hour,
    measure_phase_rows_per_hour,
    unresolved_deferred_intervals,
)
burn = measure_disk_burn_gb_per_hour(R)
rates = measure_phase_rows_per_hour(R)
pending = unresolved_deferred_intervals(R)
rows = sum(int(e["end_row"]) - int(e["start_row"]) for e in pending)
best = max((r["rows_per_hour"] for r in rates.values()
            if r.get("rows_per_hour")), default=None)
window_gb = usage.free / 2 ** 30 - 150.0
out["independent"] = {
    "burn_gb_per_hour": burn.get("gb_per_hour"),
    "burn_samples": burn.get("samples"),
    "pending_intervals": len(pending),
    "pending_rows": rows,
    "window_gb_above_floor": round(window_gb, 2),
    "window_hours": (round(window_gb / burn["gb_per_hour"], 2)
                     if burn.get("gb_per_hour") else None),
    "fastest_rows_per_hour": best,
    "hours_needed_at_fastest": round(rows / best, 1) if best else None,
}
out["phase_rates"] = rates

# The volume is not full -- that is the whole point of 91 being distinct from
# 90. Record it so nobody resizes a disk that has 400+ GB free.
out["is_volume_full"] = usage.free < 150 * 2 ** 30

try:
    out["status"] = json.loads(
        (R / "curriculum-supervisor.status.json").read_text())
except Exception as exc:  # noqa: BLE001
    out["status"] = str(exc)

out["marker_present"] = (R / "deferred-replay-active.json").is_file()

# The obligations must all still be there. A refusal is not a retirement.
deferred = resolved = 0
try:
    with (R / "curriculum-deferred-intervals.jsonl").open(
            encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("status") == "deferred":
                deferred += 1
            elif event.get("status") == "resolved":
                resolved += 1
except OSError as exc:
    out["ledger_error"] = str(exc)
out["ledger"] = {"deferred_records": deferred, "resolved_records": resolved,
                 "unresolved_now": len(pending)}

print("PROBE_JSON " + json.dumps(out, default=str))
PY
