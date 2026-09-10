python3 - <<'PY'
# Why did the deferred-replay pass end with 26 rejected intervals, and why is
# nothing running afterwards?
#
# `deferred_replay_complete` with a non-empty `rejected_intervals` is a FAILURE
# report (the supervisor returns 42 for it), not a finish line. This probe
# separates the two questions that decide what to do next:
#
#   1. Did the service stop because the unit gave up, or because it is between
#      stages?  `service_stage` plus the unit's own exit history answer that.
#   2. Are the rejections semantic verdicts, or are they resource yields and
#      worker signals wearing a semantic label?  `replay_yield_counted_as_
#      failure` is the precedent: 19 of 19 memory yields once SIGTERMed the
#      worker and every one was scored as a semantic failure.
import collections
import json
import os
import pathlib
import re
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {}


def sh(cmd, timeout=60):
    try:
        done = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except Exception as error:  # noqa: BLE001
        return f"<error: {error}>"


# ---- 1. What is meant to be running, and what stopped it? -----------------
units = sh("systemctl list-units --all --no-legend --no-pager "
           "'wizard*' 2>/dev/null | head -20")
out["units"] = units.splitlines()
out["unit_show"] = {}
for unit in re.findall(r"^\s*\S*?(wizard[\w.-]*\.service)", units, re.M):
    out["unit_show"][unit] = sh(
        "systemctl show %s -p ActiveState -p SubState -p Result "
        "-p ExecMainStatus -p NRestarts -p ExecMainExitTimestamp "
        "-p Restart --no-pager" % unit).splitlines()
out["journal_tail"] = sh(
    "journalctl -u 'wizard*' -n 40 --no-pager -o short-iso 2>/dev/null"
).splitlines()[-40:]

try:
    out["service_stage"] = (
        R / "curriculum-service-supervisor.stage").read_text().strip()
except OSError as error:
    out["service_stage"] = f"<{error.__class__.__name__}>"

out["procs"] = sh(
    "ps -eo pid,etimes,rss,args --sort=-rss | grep -E "
    "'supervisor|drive_corpora|brain_server|run_programming' "
    "| grep -v grep | head -12").splitlines()

# ---- 2. The named last failure, read rather than inferred -----------------
stderr_logs = sorted(R.glob("deferred-replay-*.stderr.log"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
out["stderr_logs"] = [
    {"name": p.name, "bytes": p.stat().st_size,
     "age_h": round((time.time() - p.stat().st_mtime) / 3600.0, 2)}
    for p in stderr_logs[:6]
]
out["last_stderr_tail"] = {}
for path in stderr_logs[:2]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out["last_stderr_tail"][path.name] = text[-2500:]

# ---- 3. Bucket THIS pass's rejections, per interval ------------------------
# A rejection is only meaningful against the attempt that produced it, so key
# by interval and keep the newest event for each.
newest = {}
kinds = collections.Counter()
health = R / "curriculum-health.jsonl"
if health.is_file():
    for line in health.open(encoding="utf-8", errors="replace"):
        if '"deferred_replay' not in line:
            continue
        try:
            event = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        kind = str(event.get("event") or event.get("kind") or "")
        interval = str(event.get("interval_id") or "")
        when = float(event.get("updated_unix") or event.get("unix") or 0.0)
        kinds[kind] += 1
        if kind == "deferred_replay_failed" and interval:
            if interval not in newest or when > newest[interval]["unix"]:
                newest[interval] = {
                    "unix": when,
                    "age_h": round((time.time() - when) / 3600.0, 2),
                    "error": str(event.get("error") or "")[:900],
                }
out["deferred_event_kinds"] = dict(kinds)


def classify(error: str) -> str:
    low = error.lower()
    if "sigterm" in low or "signal" in low or "killed" in low:
        return "worker_signal"
    if "exited 1" in low or "exit code 1" in low:
        return "worker_exit_1"
    if "memory" in low or "yield" in low or "oom" in low:
        return "resource"
    if "enterprise regression" in low:
        return "enterprise_regression"
    if "semantic" in low:
        return "semantic"
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "checkpoint" in low or "durab" in low:
        return "durability"
    return "other"


by_kind = collections.Counter()
by_corpus = collections.Counter()
per_interval = {}
for interval, record in newest.items():
    kind = classify(record["error"])
    by_kind[kind] += 1
    by_corpus[interval.split(":")[0] + "/" + kind] += 1
    per_interval[interval] = {"kind": kind, "age_h": record["age_h"],
                              "error": record["error"][:300]}
out["newest_failure_by_kind"] = dict(by_kind)
out["newest_failure_by_corpus_kind"] = dict(by_corpus)
out["per_interval"] = dict(sorted(per_interval.items())[:30])

# ---- 4. Memory and brain, because a yield is a memory verdict --------------
out["meminfo"] = {}
for line in open("/proc/meminfo"):
    key, _, value = line.partition(":")
    if key in {"MemTotal", "MemAvailable", "SwapTotal", "Committed_AS"}:
        out["meminfo"][key] = value.strip()

try:
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:18095/stats",
                                timeout=20) as handle:
        stats = json.loads(handle.read().decode("utf-8"))
    out["brain"] = {k: stats.get(k) for k in
                    ("tick", "total_ticks", "resident_terminals",
                     "accepted_episodes", "total_neurons", "total_concepts")}
except Exception as error:  # noqa: BLE001
    out["brain_error"] = str(error)[:200]

# ---- 5. Quarantine bookkeeping still on disk ------------------------------
out["markers"] = sorted(
    p.name for p in R.glob("deferred-replay-*")
    if p.suffix in {".json", ".marker"} )[:20]
out["active_json"] = {}
for name in ("deferred-replay-active.json",
             "curriculum-service-supervisor.status.json"):
    path = R / name
    if path.is_file():
        try:
            out["active_json"][name] = json.loads(path.read_text())
        except Exception as error:  # noqa: BLE001
            out["active_json"][name] = f"<unreadable: {error}>"

print("PROBEJSON " + json.dumps(out, default=str))
PY
