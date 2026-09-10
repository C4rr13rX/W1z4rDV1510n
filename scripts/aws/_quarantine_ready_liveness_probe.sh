python3 - <<'PY'
"""Confirm the quarantine replay named in the wake-up is actually advancing.

The wake-up payload caught a trough: `worker_count: 0`, `counter_reset: true`,
brain 102.8 s old, `rows_per_second: 0.14`. CLAUDE.md says each of those three
is the NORMAL reading across a `deferred_replay_resource_yield` -- the worker
is stopped and respawned once per cooperative memory yield, so an instantaneous
census lands in a trough most of the time, and the counters restart with it.
Liveness is the ROW DELTA, so this samples the row adaptively past the commit
period rather than trusting one instant.

It also reads the block's target from durable interval state rather than from
`curriculum-supervisor.status.json`, whose schema is whichever lifecycle event
wrote last, and dates every ledger field it reports against the supervisor that
is currently running.
"""
import json, os, time, glob, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
now = time.time()
out = {"now": now}


def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


# ---- Which progress file exposes a row, and how fresh is it?
def row_files():
    found = []
    for path in glob.glob(os.path.join(R, "*.progress.json")):
        data = load(path)
        if isinstance(data, dict) and data.get("durable_next_row") is not None:
            found.append((os.path.getmtime(path), path, data))
    found.sort(reverse=True)
    return found


files = row_files()
out["row_writers"] = [
    {
        "file": os.path.basename(p),
        "age_seconds": round(now - m, 1),
        "durable_next_row": d.get("durable_next_row"),
        "accepted_episodes": d.get("accepted_episodes"),
    }
    for m, p, d in files[:5]
]

if not files:
    out["no_row_writer"] = True
else:
    mtime, path, first = files[0]
    out["row_source"] = os.path.basename(path)
    out["row_source_lag_seconds"] = round(now - mtime, 1)

    # ---- Adaptive sample: exit as soon as the row moves, bound at 120 s.
    # A fixed short sample reads 0 rows/s on a healthy block, because the row
    # moves once per COMMITTED BATCH and the file is byte-identical between
    # commits.
    start_row = first.get("durable_next_row")
    start_acc = first.get("accepted_episodes")
    started = time.time()
    end_row, end_acc = start_row, start_acc
    while time.time() - started < 120.0:
        time.sleep(2.0)
        later = load(path)
        end_row = later.get("durable_next_row")
        end_acc = later.get("accepted_episodes")
        if end_row is not None and start_row is not None and end_row != start_row:
            break
    elapsed = time.time() - started
    out["sample_seconds"] = round(elapsed, 1)
    out["row_start"] = start_row
    out["row_end"] = end_row
    out["accepted_start"] = start_acc
    out["accepted_end"] = end_acc

    # A counter that went BACKWARDS is a worker restart, not a negative rate.
    if start_row is not None and end_row is not None:
        if end_row < start_row:
            out["counter_reset"] = True
            out["rows_per_second"] = None
        else:
            out["counter_reset"] = False
            out["rows_per_second"] = round((end_row - start_row) / max(elapsed, 1e-9), 3)

# ---- The block's target, from durable interval state (never status.json).
active = load(os.path.join(R, "deferred-replay-active.json"))
interval = active.get("interval") if isinstance(active, dict) else None
interval_id = None
if isinstance(interval, dict):
    interval_id = interval.get("interval_id")
out["active_state"] = active.get("state") if isinstance(active, dict) else None
out["active_created_unix"] = active.get("created_unix") if isinstance(active, dict) else None
if out.get("active_created_unix"):
    out["active_created_age_hours"] = round((now - out["active_created_unix"]) / 3600.0, 2)
out["active_interval_id"] = interval_id
if interval_id:
    parts = str(interval_id).split(":")
    if len(parts) >= 3:
        try:
            out["block_start_row"] = int(parts[-2])
            out["block_end_row"] = int(parts[-1])
        except Exception:
            pass
# The rejection that SENT this interval to quarantine is history; date it.
if isinstance(interval, dict) and interval.get("updated_unix"):
    out["interval_error_age_hours"] = round((now - interval["updated_unix"]) / 3600.0, 2)

if out.get("block_end_row") and out.get("row_end") is not None:
    remaining = out["block_end_row"] - out["row_end"]
    out["rows_remaining_in_block"] = remaining
    rate = out.get("rows_per_second")
    if rate:
        out["block_gate_eta_hours"] = round(remaining / rate / 3600.0, 2)

# ---- Outstanding quarantine work, from the interval ledger.
state = {}
ledger = os.path.join(R, "curriculum-deferred-intervals.jsonl")
try:
    with open(ledger, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("interval_id"):
                state[row["interval_id"]] = row
except Exception as exc:
    out["ledger_error"] = f"{type(exc).__name__}: {exc}"

counts = collections.Counter()
rows = collections.Counter()
for key, row in state.items():
    status = str(row.get("status") or "?")
    counts[status] += 1
    try:
        parts = key.split(":")
        rows[status] += int(parts[-1]) - int(parts[-2])
    except Exception:
        pass
out["interval_status_counts"] = dict(counts)
out["rows_by_status"] = dict(rows)

# ---- Process census. Gate on the WRAPPER, not the worker: the worker is
# stopped and respawned once per yield, so its census is a duty cycle.
def census(needle):
    hits = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().decode("utf-8", "replace").replace("\0", " ")
        except Exception:
            continue
        if needle in cmd:
            hits += 1
    return hits


out["wrapper_count"] = census("run_programming_curriculum_service.sh")
out["worker_count"] = census("tools.training_standard.drive_corpora_brain")
out["supervisor_count"] = census("curriculum_supervisor")

print("PROBE_JSON " + json.dumps(out, sort_keys=True))
PY
