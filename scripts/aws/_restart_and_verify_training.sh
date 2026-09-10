set -u
R=/srv/wizard/runtime/programming-integrated-20260713

systemctl reset-failed wizard-curriculum-supervisor.service 2>/dev/null || true
systemctl start wizard-curriculum-supervisor.service
echo "START_ISSUED"
sleep 45
systemctl show wizard-curriculum-supervisor.service \
  -p ActiveState -p SubState -p Result -p NRestarts --no-pager

python3 - <<'PY'
"""Prove training RESUMED, by the row delta -- never by a process census.

CLAUDE.md: a worker census of 0 is the normal reading mid-yield, so liveness is
the ROW DELTA sampled past the commit period. The row moves once per committed
batch (measured ~32 rows per commit), so this samples adaptively to 150 s with
early exit rather than reading 0 rows/s off a healthy block.
"""
import glob
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def freshest_row_file():
    best = None
    for path in glob.glob(os.path.join(R, "*.progress.json")):
        data = load(path)
        if data.get("durable_next_row") is None:
            continue
        mtime = os.path.getmtime(path)
        if best is None or mtime > best[0]:
            best = (mtime, path, data)
    return best


best = freshest_row_file()
if not best:
    print("NO_ROW_WRITER")
    raise SystemExit(0)

mtime, path, first = best
print(f"ROW_SOURCE {os.path.basename(path)}")
print(f"ROW_SOURCE_LAG_S {time.time() - mtime:.1f}")
start = first.get("durable_next_row")
start_acc = first.get("accepted_episodes")
began = time.time()
end, end_acc = start, start_acc
while time.time() - began < 150.0:
    time.sleep(3.0)
    later = load(path)
    end = later.get("durable_next_row")
    end_acc = later.get("accepted_episodes")
    if end is not None and start is not None and end != start:
        break
elapsed = time.time() - began
print(f"SAMPLE_S {elapsed:.1f}")
print(f"ROW {start} -> {end}")
print(f"ACCEPTED {start_acc} -> {end_acc}")
if start is not None and end is not None:
    if end < start:
        print("COUNTER_RESET true (worker restarted mid-sample)")
    else:
        print(f"ROWS_PER_SECOND {(end - start) / max(elapsed, 1e-9):.3f}")

status = load(os.path.join(R, "curriculum-supervisor.status.json"))
print(f"STATUS_STATE {status.get('state')}")
print(f"STATUS_AGE_S {time.time() - status.get('updated_unix', 0):.1f}")
print(f"STATUS_INTERVAL {status.get('interval_id')}")

active = load(os.path.join(R, "deferred-replay-active.json"))
print(f"ACTIVE_STATE {active.get('state')}")
interval = active.get("interval") or {}
print(f"ACTIVE_INTERVAL {interval.get('interval_id')}")
PY

echo "=== process census ==="
pgrep -af "run_programming_curriculum_service.sh" | head -3
pgrep -af "programming_curriculum_supervisor.py" | head -3
pgrep -af "drive_corpora_brain" | head -3

echo "=== disk + recent errors ==="
df -h /srv/wizard
tail -c 1200 "${R}/curriculum-service.stderr.log" 2>/dev/null || true
