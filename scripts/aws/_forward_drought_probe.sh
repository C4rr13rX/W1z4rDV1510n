#!/bin/bash
# The watchdog fired on "no interval admitted for 102.7h" while the service
# stage is `forward`, not `deferred_replay`. Forward blocks harvest rows; they
# do not admit intervals, so a long forward pass can look like a drought that
# is really just a stage the drought metric does not measure. Distinguish:
#
#   (a) forward is genuinely advancing  -> drought is expected, not a fault
#   (b) forward is looping on a canary  -> real stall, needs intervention
#
# Sample the row twice with a real gap, because a continuous canary freezes the
# row by design (CLAUDE.md) and one sample cannot tell a freeze from a hang.
set -uo pipefail

python3 - <<'PY'
import glob
import json
import pathlib
import re
import subprocess
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return (r.stdout or "").strip()
    except Exception as exc:
        return f"ERR {exc}"


def load(p):
    try:
        return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def snapshot():
    """Every runtime json fresher than an hour, with its row and mtime."""
    seen = {}
    for path in glob.glob(str(runtime / "*.json")):
        p = pathlib.Path(path)
        try:
            st = p.stat()
        except Exception:
            continue
        if time.time() - st.st_mtime > 3600:
            continue
        body = load(p)
        seen[p.name] = {
            "age": round(time.time() - st.st_mtime, 1),
            "state": body.get("state"),
            "phase": body.get("phase"),
            "row": body.get("row"),
            "durable_next_row": body.get("durable_next_row"),
            "canary_row": body.get("canary_row"),
            "accepted_episodes": body.get("accepted_episodes"),
        }
    return seen


first = snapshot()
t0 = time.time()
time.sleep(90)
second = snapshot()
elapsed = time.time() - t0

moved = {}
for name, b in second.items():
    a = first.get(name) or {}
    for key in ("row", "durable_next_row", "canary_row", "accepted_episodes"):
        av, bv = a.get(key), b.get(key)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and bv != av:
            moved.setdefault(name, {})[key] = {
                "from": av, "to": bv, "per_second": round((bv - av) / elapsed, 3)}
out["sample_seconds"] = round(elapsed, 1)
out["fresh_files_before"] = first
out["fresh_files_after"] = second
out["moved"] = moved

# Is the brain itself ticking? A frozen row with a rising tick means the canary
# is working; a frozen row with a frozen tick means nothing is happening.
ticks = []
for _ in range(2):
    body = sh("curl", "-s", "--max-time", "20", "http://127.0.0.1:18095/stats")
    try:
        j = json.loads(body)
        ticks.append({k: j.get(k) for k in (
            "tick", "total_neurons", "total_concepts", "resident_terminals",
            "accepted_episodes")})
    except Exception:
        ticks.append({"raw": body[:200]})
    time.sleep(20)
out["brain_stats_samples"] = ticks

# What is the supervisor's own log saying right now? The status file freezes
# between batches, the log does not.
logs = sorted(glob.glob(str(runtime / "*.log")),
              key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)[:6]
tails = {}
for path in logs:
    p = pathlib.Path(path)
    age = round(time.time() - p.stat().st_mtime, 1)
    try:
        txt = p.read_text(encoding="utf-8", errors="replace")
        tails[p.name] = {"age": age, "tail": txt[-1600:]}
    except Exception as exc:
        tails[p.name] = {"age": age, "error": str(exc)}
out["log_tails"] = tails

# Ledger: what did the last admissions and rejections actually decide?
health = runtime / "curriculum-health.jsonl"
recent = []
if health.exists():
    lines = health.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-400:]:
        try:
            recent.append(json.loads(line))
        except Exception:
            continue
out["health_tail_events"] = [
    {"unix": r.get("unix") or r.get("timestamp"), "event": r.get("event"),
     "phase": r.get("phase"), "reason": (r.get("reason") or "")[:200]}
    for r in recent[-30:]]

buckets = {}
for r in recent:
    buckets[r.get("event")] = buckets.get(r.get("event"), 0) + 1
out["health_tail_bucket_counts"] = buckets

# The enterprise gate verdict per suite -- counts alone hide which suite fails.
gates = sorted(glob.glob(str(runtime / "*.enterprise-gate.json")),
               key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)[:3]
gate_out = []
for path in gates:
    body = load(path)
    gate_out.append({
        "file": pathlib.Path(path).name,
        "age": round(time.time() - pathlib.Path(path).stat().st_mtime, 1),
        "results": [{"name": r.get("name"), "passed": r.get("passed"),
                     "detail": (str(r.get("detail") or ""))[:160]}
                    for r in (body.get("results") or [])],
    })
out["enterprise_gates"] = gate_out

out["memory"] = sh("free", "-g")
out["top"] = sh("bash", "-c",
                "ps -eo pid,rss,etimes,comm --sort=-rss | head -12")

print("PROBE_JSON " + json.dumps(out))
PY
