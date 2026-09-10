python3 - <<'PY'
"""How long until the append-only checkpoint refills the volume?

Two readings taken from different probes minutes apart suggested ~20 GB/h,
which would be a ~30 hour runway rather than the weeks a "stale bases" story
implies. That number decides whether this is a scheduling matter or an
operational deadline, so it gets measured over a defined window inside one
process rather than differenced across two round trips.

Samples the live checkpoint and the volume together: `brain.wbrain` is the
consumer, but reflink sharing means the file's growth and the volume's loss are
not the same number, and it is the VOLUME that stops training.
"""
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = os.path.join(R, "brain", "brain.wbrain")
WINDOW = 420.0

out = {}


def sample():
    stat = os.statvfs(R)
    try:
        size = os.stat(BRAIN).st_blocks * 512
    except OSError:
        size = None
    return {
        "unix": time.time(),
        "free_bytes": stat.f_bavail * stat.f_frsize,
        "brain_bytes": size,
    }


def row():
    import glob
    best = None
    for path in glob.glob(os.path.join(R, "*.progress.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if data.get("durable_next_row") is None:
            continue
        mtime = os.path.getmtime(path)
        if best is None or mtime > best[0]:
            best = (mtime, os.path.basename(path), data)
    return best


first = sample()
row_first = row()
time.sleep(WINDOW)
last = sample()
row_last = row()

elapsed = last["unix"] - first["unix"]
out["window_seconds"] = round(elapsed, 1)
out["free_gb_start"] = round(first["free_bytes"] / 1e9, 2)
out["free_gb_end"] = round(last["free_bytes"] / 1e9, 2)
out["brain_gb_start"] = round((first["brain_bytes"] or 0) / 1e9, 2)
out["brain_gb_end"] = round((last["brain_bytes"] or 0) / 1e9, 2)

volume_loss = first["free_bytes"] - last["free_bytes"]
out["volume_loss_gb_per_hour"] = round(volume_loss / elapsed * 3600 / 1e9, 2)
if first["brain_bytes"] and last["brain_bytes"]:
    out["brain_growth_gb_per_hour"] = round(
        (last["brain_bytes"] - first["brain_bytes"]) / elapsed * 3600 / 1e9, 2
    )

# A rate of zero or below is a settle/gate/canary window, not a safe host --
# say which, rather than publishing an infinite runway.
if volume_loss > 0:
    out["hours_until_full"] = round(last["free_bytes"] / (volume_loss / elapsed) / 3600.0, 1)
    out["days_until_full"] = round(
        last["free_bytes"] / (volume_loss / elapsed) / 86400.0, 2
    )
else:
    out["hours_until_full"] = None
    out["note"] = "volume did not shrink in this window (settle, gate or canary)"

# Liveness over the same window, so a flat disk is not misread as a safe host
# when it is really a stopped one.
if row_first and row_last:
    out["row_file"] = row_last[1]
    out["row_start"] = row_first[2].get("durable_next_row")
    out["row_end"] = row_last[2].get("durable_next_row")
    out["accepted_start"] = row_first[2].get("accepted_episodes")
    out["accepted_end"] = row_last[2].get("accepted_episodes")
    if out["row_end"] is not None and out["row_start"] is not None:
        if out["row_end"] < out["row_start"]:
            out["counter_reset"] = True
            out["rows_per_second"] = None
        else:
            out["rows_per_second"] = round(
                (out["row_end"] - out["row_start"]) / elapsed, 3
            )

try:
    with open(os.path.join(R, "curriculum-supervisor.status.json"), encoding="utf-8") as fh:
        status = json.load(fh)
    out["status_state"] = status.get("state")
    out["status_age_seconds"] = round(time.time() - status.get("updated_unix", 0), 1)
    out["status_interval"] = status.get("interval_id")
except Exception:
    pass

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
