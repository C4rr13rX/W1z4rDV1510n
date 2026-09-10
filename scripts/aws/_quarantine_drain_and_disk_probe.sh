python3 - <<'PY'
"""Measure whether the quarantine replay can FINISH, not just whether it moves.

The `quarantine_ready` wake-up reports `forward_remaining_rows: 0`, so every
row left in the curriculum is now deferred: 2,782,556 of them. From here the
only question that matters is the drain -- how many intervals are outstanding,
how fast they resolve, and whether the host survives long enough to do it.

Three things are sampled over ONE window so they are directly comparable:

* the row delta, adaptively, because the row moves once per committed batch and
  any sample shorter than the commit period reads 0 rows/s on a healthy block;
* `df` on the runtime volume across that same window, because the `.wbrain`
  neuron store is append-only with no compactor and converts memory pressure
  into permanent disk growth -- a runway measured against the SUPERVISOR's 8 GB
  yield guard is under four minutes of warning, so it must be measured against
  the burn rate instead;
* the interval ledger, so the outstanding quarantine work is counted from
  durable state rather than inferred from the newest lifecycle event.

`du` is deliberately not used: the volume is XFS with reflink=1 and the causal
bases are hardlinks plus reflink clones, so summing file sizes reported 2.48 TB
inside a 1.0 TB volume and predicted a 560 GB reclaim that returned 0.00 GB.
Only `df` measures a reclaim.
"""
import collections
import glob
import json
import os
import shutil
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
now = time.time()
out = {"now": now}


def load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def free_gb(path):
    usage = shutil.disk_usage(path)
    return round(usage.free / 1e9, 3), round(usage.total / 1e9, 3)


# ---- Row writers. Require a row: selecting on mtime alone picks the
# supervisor status file, which during a replay never carries one.
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

disk_start, disk_total = free_gb(R)
out["disk_total_gb"] = disk_total
out["disk_free_start_gb"] = disk_start
sample_started = time.time()

if not files:
    out["no_row_writer"] = True
    time.sleep(60.0)
else:
    mtime, path, first = files[0]
    out["row_source"] = os.path.basename(path)
    out["row_source_lag_seconds"] = round(now - mtime, 1)
    start_row = first.get("durable_next_row")
    start_acc = first.get("accepted_episodes")
    end_row, end_acc = start_row, start_acc
    moved_at = None
    # Sample the FULL window even after the row moves: the disk burn needs a
    # comparable interval, and one commit is not a rate.
    while time.time() - sample_started < 150.0:
        time.sleep(3.0)
        later = load(path)
        end_row = later.get("durable_next_row")
        end_acc = later.get("accepted_episodes")
        if moved_at is None and end_row is not None and start_row is not None and end_row != start_row:
            moved_at = time.time() - sample_started
    elapsed = time.time() - sample_started
    out["sample_seconds"] = round(elapsed, 1)
    out["seconds_to_first_commit"] = round(moved_at, 1) if moved_at is not None else None
    out["row_start"] = start_row
    out["row_end"] = end_row
    out["accepted_start"] = start_acc
    out["accepted_end"] = end_acc
    if start_row is not None and end_row is not None:
        if end_row < start_row:
            out["counter_reset"] = True
            out["rows_per_second"] = None
        else:
            out["counter_reset"] = False
            out["rows_per_second"] = round((end_row - start_row) / max(elapsed, 1e-9), 3)

elapsed = time.time() - sample_started
disk_end, _ = free_gb(R)
out["disk_free_end_gb"] = disk_end
delta = disk_start - disk_end
out["disk_consumed_gb"] = round(delta, 3)
# A NEGATIVE consumption is a reclaim (rollback, prune) inside the window, not
# a burn rate; reporting it as a rate would predict infinite runway.
if delta > 0:
    burn = delta / max(elapsed, 1e-9) * 3600.0
    out["disk_burn_gb_per_hour"] = round(burn, 2)
    out["disk_runway_hours"] = round(disk_end / burn, 2) if burn > 0 else None
else:
    out["disk_burn_gb_per_hour"] = None
    out["disk_reclaimed_in_window_gb"] = round(-delta, 3)

# ---- The append-only store itself, and whether a compactor exists yet.
for name in ("brain/brain.wbrain", "brain/brain.wal"):
    p = os.path.join(R, name)
    try:
        st = os.stat(p)
        out.setdefault("store", {})[os.path.basename(name)] = {
            "gb": round(st.st_size / 1e9, 3),
            "age_seconds": round(now - st.st_mtime, 1),
            "nlink": st.st_nlink,
        }
    except Exception as exc:
        out.setdefault("store", {})[os.path.basename(name)] = f"{type(exc).__name__}"

# ---- Durable interval state: what is left to drain.
active = load(os.path.join(R, "deferred-replay-active.json"))
interval = active.get("interval") if isinstance(active, dict) else None
out["active_state"] = active.get("state") if isinstance(active, dict) else None
created = active.get("created_unix") if isinstance(active, dict) else None
out["active_created_age_hours"] = round((now - created) / 3600.0, 2) if created else None
interval_id = interval.get("interval_id") if isinstance(interval, dict) else None
out["active_interval_id"] = interval_id
if interval_id:
    parts = str(interval_id).split(":")
    if len(parts) >= 3:
        try:
            out["block_start_row"] = int(parts[-2])
            out["block_end_row"] = int(parts[-1])
        except Exception:
            pass
if isinstance(interval, dict) and interval.get("updated_unix"):
    out["interval_error_age_hours"] = round((now - interval["updated_unix"]) / 3600.0, 2)

if out.get("block_end_row") and out.get("row_end") is not None:
    remaining = out["block_end_row"] - out["row_end"]
    out["rows_remaining_in_block"] = remaining
    rate = out.get("rows_per_second")
    if rate:
        out["block_gate_eta_hours"] = round(remaining / rate / 3600.0, 2)

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
newest = {}
for key, row in state.items():
    status = str(row.get("status") or "?")
    counts[status] += 1
    try:
        parts = key.split(":")
        rows[status] += int(parts[-1]) - int(parts[-2])
    except Exception:
        pass
    stamp = row.get("updated_unix") or row.get("created_unix")
    if stamp and (status not in newest or stamp > newest[status]):
        newest[status] = stamp
out["interval_status_counts"] = dict(counts)
out["rows_by_status"] = dict(rows)
out["status_newest_age_hours"] = {
    k: round((now - v) / 3600.0, 2) for k, v in sorted(newest.items())
}

# ---- Drain rate: resolutions per day from the health ledger, so the
# outstanding row count can be turned into a completion estimate.
resolved = []
health = os.path.join(R, "curriculum-health.jsonl")
try:
    with open(health, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") in ("deferred_replay_admitted", "quarantine_retest_admitted"):
                stamp = row.get("unix") or row.get("observed_unix")
                if stamp:
                    resolved.append(float(stamp))
except Exception as exc:
    out["health_error"] = f"{type(exc).__name__}: {exc}"
resolved.sort()
out["admission_events"] = len(resolved)
if resolved:
    out["admissions_last_24h"] = sum(1 for s in resolved if now - s <= 86400)
    out["admissions_last_72h"] = sum(1 for s in resolved if now - s <= 259200)
    out["first_admission_age_hours"] = round((now - resolved[0]) / 3600.0, 2)
    out["last_admission_age_hours"] = round((now - resolved[-1]) / 3600.0, 2)

# ---- Census. Gate on the WRAPPER: the worker is respawned once per yield.
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
