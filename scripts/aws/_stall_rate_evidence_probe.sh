python3 - <<'PY'
"""Why does a stalled interval leave a stall COUNT but no stall RATE?

`replay_stall_counts` counts every `deferred_replay_interrupted_before_gate`
record; `replay_stall_observations` keeps only those carrying a positive
`rows_per_hour`, because `record_replay_stall` publishes a rate only when it
measured `rows_trained > 0` over `hours > 0`. Live: 3 stall records, 3 counted,
exactly 1 with a rate -- and the two without are both `jupyter-scientific-full`,
the phase whose intervals the census therefore scores `unknown` and selects.

Dump the records verbatim and check the inputs that would have produced a rate.

Read-only.
"""
import hashlib
import json
import pathlib
import shutil
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
sys.path.insert(0, "/srv/wizard/project")
out = {"now": time.time()}

records = []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("kind") == "deferred_replay_interrupted_before_gate":
                records.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)
out["stall_records"] = records

# For each stalled interval, does its progress file still exist? The rollback
# unlinks it, so a second generation finds nothing to measure.
probe = {}
for event in records:
    interval_id = str(event.get("interval_id") or "")
    digest = hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]
    path = R / f"deferred-replay-{digest}.progress.json"
    entry = {"digest": digest, "progress_exists": path.is_file()}
    if entry["progress_exists"]:
        try:
            entry["progress"] = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            entry["progress_error"] = str(exc)
        try:
            entry["progress_mtime_age_h"] = round(
                (time.time() - path.stat().st_mtime) / 3600.0, 3)
        except OSError:
            pass
    probe[interval_id] = entry
out["stalled_interval_progress"] = probe

# Every progress file present, so a reader can see which interval owns which.
out["progress_files"] = sorted(
    p.name for p in R.glob("deferred-replay-*.progress.json"))

# The live marker, and how close the running generation is to the floor.
try:
    marker = json.loads((R / "deferred-replay-active.json").read_text())
except (OSError, ValueError) as exc:
    marker = {"error": str(exc)}
out["marker"] = {k: v for k, v in marker.items() if k != "interval"}
interval = marker.get("interval") or {}
out["marker_interval"] = {k: interval.get(k) for k in
                          ("interval_id", "phase", "start_row", "end_row")}

usage = shutil.disk_usage(R)
out["free_gb"] = round(usage.free / 2 ** 30, 2)
out["floor_gb"] = 150.0
out["headroom_gb"] = round(usage.free / 2 ** 30 - 150.0, 2)

# What the running generation has banked, if anything.
running_id = str(marker.get("interval_id") or "")
if running_id:
    digest = hashlib.sha256(running_id.encode("utf-8")).hexdigest()[:16]
    path = R / f"deferred-replay-{digest}.progress.json"
    entry = {"digest": digest, "exists": path.is_file()}
    if path.is_file():
        try:
            entry["progress"] = json.loads(path.read_text())
            entry["age_s"] = round(time.time() - path.stat().st_mtime, 1)
        except (OSError, ValueError) as exc:
            entry["error"] = str(exc)
    out["running_progress"] = entry

print("PROBE_JSON " + json.dumps(out, default=str))
PY
