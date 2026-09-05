python3 - <<'PY'
import json, os, re, time, collections, subprocess

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}
TARGET = "jupyter-scientific-full:262144:393216"

# How many times has THIS interval been attempted, and with what verdict?
attempts = []
for name in ("curriculum-health.jsonl", "curriculum-deferred-intervals.jsonl"):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        continue
    for line in open(path, encoding="utf-8"):
        if TARGET not in line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        attempts.append({
            "src": name,
            "event": event.get("event") or event.get("status"),
            "age_h": round((time.time() - (event.get("updated_unix") or 0)) / 3600.0, 2),
            "err": str(event.get("error") or "")[:160],
        })
out["target_attempts"] = attempts[-25:]
out["target_attempt_count"] = len(attempts)

# Which phase does each deferred interval belong to, and how many attempts has
# each accumulated? A phase looping is invisible in a single status read.
per_interval = collections.Counter()
for line in open(R + "/curriculum-health.jsonl", encoding="utf-8"):
    if '"deferred_replay_failed"' not in line:
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    key = event.get("interval_id") or event.get("phase") or "?"
    per_interval[key] += 1
out["failures_per_interval_top"] = dict(per_interval.most_common(10))

# Outstanding intervals by phase, so the remaining work is countable.
state = {}
for line in open(R + "/curriculum-deferred-intervals.jsonl", encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        state[row["interval_id"]] = row
by_phase = collections.Counter()
rows_left = 0
for interval_id, row in state.items():
    if row.get("status") == "deferred":
        by_phase[interval_id.split(":")[0]] += 1
        try:
            rows_left += int(interval_id.split(":")[2]) - int(interval_id.split(":")[1])
        except Exception:
            pass
out["deferred_by_phase"] = dict(by_phase)
out["deferred_rows_left"] = rows_left

# Where is the brain actually listening?
try:
    listen = subprocess.run(["ss", "-lntp"], capture_output=True, text=True,
                            timeout=30).stdout
    out["listening"] = [l.split()[3] for l in listen.splitlines()[1:] if l.split()]
except Exception as error:
    out["listening"] = f"unknown: {error}"

import urllib.request
for port in (8095, 18095):
    try:
        def stat(p=port):
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{p}/stats", timeout=20) as handle:
                return json.loads(handle.read().decode("utf-8"))
        first = stat()
        time.sleep(6)
        second = stat()
        tick_a = first.get("tick") or first.get("total_ticks") or 0
        tick_b = second.get("tick") or second.get("total_ticks") or 0
        out[f"port_{port}"] = {
            "tick": tick_b,
            "tick_delta_6s": tick_b - tick_a,
            "resident_terminals": second.get("resident_terminals"),
            "total_neurons": second.get("total_neurons"),
            "accepted_episodes": second.get("accepted_episodes"),
        }
    except Exception as error:
        out[f"port_{port}"] = f"error: {str(error)[:100]}"

print("PROBEJSON " + json.dumps(out))
PY
