python3 - <<'PY'
"""What the `quarantine_ready` wake-up cannot answer from its own payload.

Three questions, in the order that decides whether there is work to do:

  1. The named `last_failure` arrived as a bare stderr PATH with no tail, even
     though `replay_worker_failure` appends up to 4000 bytes of tail inline.
     Either the worker wrote nothing, or something between the ledger and this
     agent deleted it. Read the file: size, age, and the actual tail.
  2. One named failure is not a population (`named_failure_vs_population`:
     the last one was 6.6% of 288). Bucket every `deferred_replay_failed`
     since the current supervisor started AND since the running binary was
     built, so a repaired cause is not re-debugged
     (`failure_ledger_predates_process`).
  3. `deferred_rows` is 3,044,700 with `forward_remaining_rows` 0, so replay
     is the only remaining producer of admissions. Enumerate the queue: how
     many intervals, which phases, and how many have already been retired
     unrestorable -- 20 admitted against 324 failed is the number that decides
     whether the drought is a queue that cannot drain.

Read-only. Nothing here writes to the runtime or touches the brain's state.
"""
import collections
import json
import os
import re
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}


def sh(command, timeout=60):
    try:
        return subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout
    except Exception as error:
        return f"<{type(error).__name__}: {str(error)[:80]}>"


# --- 1. the stderr log the ledger named --------------------------------------
try:
    binary_built = os.path.getmtime(
        "/srv/wizard/project/target/release/w1z4rd_brain_server")
except OSError:
    binary_built = 0.0
out["binary_built_age_h"] = round((time.time() - binary_built) / 3600.0, 2)

proc_start = 0.0
try:
    boot = 0.0
    for line in open("/proc/stat"):
        if line.startswith("btime"):
            boot = float(line.split()[1])
    hz = os.sysconf("SC_CLK_TCK")
    for pid in sh("pgrep -f programming_curriculum_supervisor.py").split():
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().split()
        proc_start = max(proc_start, boot + int(fields[21]) / hz)
except Exception as error:
    out["proc_start_error"] = str(error)[:120]
out["supervisor_age_h"] = (round((time.time() - proc_start) / 3600.0, 2)
                           if proc_start else -1)

named = f"{R}/deferred-replay-8f4a439a7fc7a772.stderr.log"
if os.path.exists(named):
    out["named_stderr"] = {
        "bytes": os.path.getsize(named),
        "age_h": round((time.time() - os.path.getmtime(named)) / 3600.0, 2),
        "tail": sh(f"tail -c 2500 {named}").splitlines()[-30:],
    }
else:
    out["named_stderr"] = {"missing": True}

# Every replay stderr log, so an empty one can be told from an empty CLASS.
logs = []
for name in sorted(os.listdir(R)):
    if not name.startswith("deferred-replay-") or not name.endswith(
            ".stderr.log"):
        continue
    path = os.path.join(R, name)
    logs.append({
        "name": name,
        "bytes": os.path.getsize(path),
        "age_h": round((time.time() - os.path.getmtime(path)) / 3600.0, 2),
    })
logs.sort(key=lambda entry: entry["age_h"])
out["replay_stderr_logs"] = logs[:12]
out["replay_stderr_log_count"] = len(logs)
out["replay_stderr_logs_empty"] = sum(1 for e in logs if e["bytes"] == 0)

# --- 2. the failure population, not the named one ----------------------------
def classify(error: str) -> str:
    low = error.casefold()
    # Ordered most-specific first: an arm matching a substring as common as
    # "stderr" swallowed every cause once already (`dominant_bucket_is_blind`).
    if "worker exited" in low:
        return "worker_exit"
    if "enterprise regression" in low:
        return "enterprise_regression"
    if "timed out" in low or "timeouterror" in low:
        return "timeout"
    if "recall" in low:
        return "interval_recall"
    if "semantic" in low:
        return "semantic"
    if "checkpoint" in low or "durab" in low:
        return "durability"
    if "memory" in low or "resource" in low:
        return "resource"
    return "other"


buckets_all = collections.Counter()
buckets_deploy = collections.Counter()
buckets_proc = collections.Counter()
phases_proc = collections.Counter()
examples = {}
exit_codes = collections.Counter()
total = 0
first_unix = last_unix = 0.0

for line in open(f"{R}/curriculum-health.jsonl", encoding="utf-8"):
    if '"deferred_replay_failed"' not in line:
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    error = str(event.get("error") or "")
    when = float(event.get("updated_unix") or 0)
    kind = classify(error)
    total += 1
    first_unix = first_unix or when
    last_unix = max(last_unix, when)
    buckets_all[kind] += 1
    if when >= binary_built:
        buckets_deploy[kind] += 1
    if proc_start and when >= proc_start:
        buckets_proc[kind] += 1
        phases_proc[str(event.get("phase"))] += 1
    match = re.search(r"worker exited (-?\d+)", error)
    if match:
        exit_codes[match.group(1)] += 1
    if kind not in examples:
        examples[kind] = error[:700]

out["failures"] = {
    "total": total,
    "buckets_all_time": dict(buckets_all),
    "buckets_since_deploy": dict(buckets_deploy),
    "buckets_since_supervisor_start": dict(buckets_proc),
    "phases_since_supervisor_start": dict(phases_proc),
    "worker_exit_codes": dict(exit_codes),
    "newest_age_h": (round((time.time() - last_unix) / 3600.0, 2)
                     if last_unix else None),
    "examples": examples,
}

# Does a worker-exit failure carry its tail in the LEDGER? The supervisor
# appends up to 4000 bytes; if the ledger has it, the deletion happened in the
# probe that reports to the agent, not in the supervisor.
tail_lengths = []
for line in open(f"{R}/curriculum-health.jsonl", encoding="utf-8"):
    if '"deferred_replay_failed"' not in line:
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    error = str(event.get("error") or "")
    if "worker exited" not in error:
        continue
    tail_lengths.append(len(error))
out["worker_exit_error_lengths"] = {
    "count": len(tail_lengths),
    "max": max(tail_lengths) if tail_lengths else 0,
    "over_180_chars": sum(1 for n in tail_lengths if n > 180),
    "last_5": tail_lengths[-5:],
}

# --- 3. the quarantine queue -------------------------------------------------
queue = collections.Counter()
retired = collections.Counter()
admitted = []
for line in open(f"{R}/curriculum-health.jsonl", encoding="utf-8"):
    for kind, sink in (("deferred_replay_admitted", admitted),
                       ("unrestorable_quarantine_retired", None)):
        if f'"{kind}"' not in line:
            continue
        try:
            event = json.loads(line)
        except Exception:
            continue
        if kind == "deferred_replay_admitted":
            admitted.append({
                "phase": event.get("phase"),
                "interval": event.get("interval_id"),
                "age_h": round(
                    (time.time() - float(event.get("updated_unix") or 0))
                    / 3600.0, 2),
            })
        else:
            retired[str(event.get("phase"))] += 1
out["admitted_recent"] = admitted[-8:]
out["retired_by_phase"] = dict(retired)

for name in sorted(os.listdir(R)):
    if name.startswith("deferred-") and name.endswith(".json"):
        queue[name.split("-")[1].split(".")[0]] += 1
out["deferred_state_files"] = dict(queue)

state = f"{R}/deferred-replay-active.json"
if os.path.exists(state):
    try:
        out["active_state"] = json.load(open(state))
    except Exception as error:
        out["active_state_error"] = str(error)[:120]

# Convergence of the interval running RIGHT NOW: the resume marker, not the
# progress file, is what proves a pass resumed (`replay_pass_resume_discriminator`).
progress = f"{R}/deferred-replay-b01232b593532da2.progress.json"
samples = []
for _ in range(2):
    try:
        samples.append(json.load(open(progress)))
    except Exception as error:
        samples.append({"error": str(error)[:100]})
    time.sleep(6)
out["progress_samples"] = samples
resume = f"{R}/deferred-replay-b01232b593532da2.resume.json"
if os.path.exists(resume):
    try:
        out["resume_marker"] = json.load(open(resume))
    except Exception as error:
        out["resume_marker_error"] = str(error)[:120]
else:
    out["resume_marker_missing"] = True

out["oom_kills"] = sh(
    "dmesg 2>/dev/null | grep -ci 'out of memory\\|oom-kill' || true").strip()

print("PROBEJSON " + json.dumps(out))
PY
