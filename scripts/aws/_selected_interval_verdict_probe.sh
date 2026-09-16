python3 - <<'PY'
"""Did the census MEASURE the interval it then selected as too large?

`replay_window_census` computes a per-interval verdict and
`replay_queue_is_hopeless` collapses it to one boolean over the WHOLE queue:
true only when nothing fits and nothing is unknown. Selection, though, is
`pending[0]` from `order_replay_candidates`, which sorts on `(stalls, span)`
and never reads a verdict. So if any interval anywhere in the queue is
`unknown`, the queue is not hopeless and the head trains -- even when the head
is the one interval the census just measured as `exceeds`.

The live event this probe answers: `jupyter-scientific-full:393216:524288` is
training at 0.4 rows/s and needs ~88 h to reach its gate, against a window the
same census computes in single-digit hours.

Read-only. Nothing is started, stopped or written.
"""
import json
import pathlib
import subprocess
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
P = "/srv/wizard/project"
sys.path.insert(0, P)
sys.path.insert(0, P + "/scripts")
out = {"now": time.time()}

# DEPLOY IS NOT LOAD. A census that exists in the tree but not in the process
# decides nothing, so record what is actually on disk AND what argv the running
# supervisor was launched with.
sup_path = pathlib.Path(P) / "scripts" / "programming_curriculum_supervisor.py"
try:
    import hashlib
    raw = sup_path.read_bytes()
    out["deployed_supervisor"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "mtime": sup_path.stat().st_mtime,
        "has_window_census": b"def replay_window_census" in raw,
        "has_queue_hopeless": b"def replay_queue_is_hopeless" in raw,
        "has_halt_rollback": b"reclaim_disk_for_floor" in raw,
        "selects_on_verdict": b"census_selectable" in raw,
    }
except OSError as exc:
    out["deployed_supervisor"] = {"error": str(exc)}

# argv of the live supervisor: the floor it is actually enforcing.
argv = []
for proc in pathlib.Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        cmd = (proc / "cmdline").read_bytes().decode("utf-8", "replace")
    except OSError:
        continue
    parts = [part for part in cmd.split("\0") if part]
    if any("programming_curriculum_supervisor" in part for part in parts):
        argv.append({"pid": int(proc.name), "argv": parts,
                     "started": (proc / "stat").stat().st_mtime})
out["supervisor_argv"] = argv

from scripts.programming_curriculum_supervisor import (  # noqa: E402
    measure_disk_burn_gb_per_hour,
    measure_phase_rows_per_hour,
    order_replay_candidates,
    replay_queue_is_hopeless,
    replay_stall_counts,
    replay_stall_observations,
    replay_window_census,
    rollback_reclaim_bytes,
    unresolved_deferred_intervals,
)

out["burn"] = measure_disk_burn_gb_per_hour(R)
out["rollback_reclaim_bytes"] = rollback_reclaim_bytes(R)
out["phase_rates"] = measure_phase_rows_per_hour(R)
stalls = replay_stall_counts(R)
observations = replay_stall_observations(R)
out["stall_counts"] = stalls
out["stall_observation_intervals"] = {
    key: [round(row["rows_per_hour"], 1) for row in value]
    for key, value in observations.items()
}

pending = unresolved_deferred_intervals(R)
ordered = order_replay_candidates(pending, stalls)
floor = 150.0
for entry in argv:
    for index, part in enumerate(entry["argv"]):
        if part == "--min-free-disk-gb" and index + 1 < len(entry["argv"]):
            try:
                floor = float(entry["argv"][index + 1])
            except ValueError:
                pass
out["floor_gb_from_argv"] = floor

census = replay_window_census(R, ordered, floor)
out["pending"] = len(pending)
out["census"] = {
    "window_hours": census["window_hours"],
    "free_gb": round(census["free_bytes"] / 2 ** 30, 2),
    "window_gb": round(census["window_bytes"] / 2 ** 30, 2),
    "fits": census["fits"],
    "unknown_count": len(census["unknown"]),
    "unknown_head": census["unknown"][:8],
    "exceeds_count": len(census["exceeds"]),
    "hopeless": replay_queue_is_hopeless(census),
    "head": census["intervals"][:8],
}

# THE QUESTION. What is the verdict on the interval that is training right now,
# and is it the head of the ordered queue?
try:
    marker = json.loads((R / "deferred-replay-active.json").read_text())
except (OSError, ValueError) as exc:
    marker = {}
    out["marker_error"] = str(exc)
running_id = str(marker.get("interval_id") or "")
out["running"] = {
    "interval_id": running_id,
    "state": marker.get("state"),
    "created_unix": marker.get("created_unix"),
    "age_hours": (round((time.time() - marker["created_unix"]) / 3600.0, 2)
                  if isinstance(marker.get("created_unix"), (int, float))
                  else None),
}
out["head_of_queue"] = (str(ordered[0].get("interval_id"))
                        if ordered else None)
out["running_is_head"] = bool(running_id) and running_id == out["head_of_queue"]
for row in census["intervals"]:
    if row["interval_id"] == running_id:
        out["running_verdict"] = row
        break
else:
    out["running_verdict"] = None

# Has the refusal ever fired, and has a stall ever been recorded?
kinds = {}
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = str(event.get("kind") or "")
            kinds[kind] = kinds.get(kind, 0) + 1
except OSError as exc:
    out["ledger_error"] = str(exc)
out["ledger_kinds"] = {k: v for k, v in sorted(kinds.items())
                       if "stall" in k or "fits" in k or "rollback" in k
                       or "disk" in k}

free = subprocess.run(["df", "-B1", "--output=avail,used,size", str(R)],
                      capture_output=True, text=True, check=False)
out["df"] = free.stdout.strip().splitlines()[-1:]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
