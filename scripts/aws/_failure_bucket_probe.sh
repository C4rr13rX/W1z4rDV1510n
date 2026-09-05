python3 - <<'PY'
import json, os, re, time, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {}

try:
    binary_built = os.path.getmtime(
        "/srv/wizard/project/target/release/w1z4rd_brain_server")
except OSError:
    binary_built = 0
out["binary_built_age_h"] = round((time.time() - binary_built) / 3600.0, 2)

# Newest supervisor process start, so a failure can be dated against the
# generation that produced it -- `failure_ledger_predates_process`.
proc_start = 0
try:
    import subprocess
    pids = subprocess.run(
        ["pgrep", "-f", "programming_curriculum_supervisor.py"],
        capture_output=True, text=True, timeout=30).stdout.split()
    boot = 0.0
    for line in open("/proc/stat"):
        if line.startswith("btime"):
            boot = float(line.split()[1])
    hz = os.sysconf("SC_CLK_TCK")
    for pid in pids:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().split()
        proc_start = max(proc_start, boot + int(fields[21]) / hz)
except Exception as error:
    out["proc_start_error"] = str(error)[:120]
out["supervisor_age_h"] = (round((time.time() - proc_start) / 3600.0, 2)
                           if proc_start else -1)

buckets = collections.Counter()
tick_deltas = collections.Counter()
since_deploy = collections.Counter()
since_proc = collections.Counter()
suite_fails = collections.Counter()
examples = {}
total = 0

def classify(error: str) -> str:
    if "enterprise regression" in error:
        return "enterprise_regression"
    if "semantic" in error:
        return "semantic"
    if "SIGTERM" in error or "signal" in error:
        return "worker_signal"
    if "Timeout" in error or "timed out" in error:
        return "timeout"
    if "checkpoint" in error or "durab" in error:
        return "durability"
    if "structure" in error:
        return "structure"
    return "other"

for line in open(R + "/curriculum-health.jsonl", encoding="utf-8"):
    if '"deferred_replay_failed"' not in line:
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    error = str(event.get("error") or "")
    when = event.get("updated_unix") or 0
    kind = classify(error)
    total += 1
    buckets[kind] += 1
    if when >= binary_built:
        since_deploy[kind] += 1
    if proc_start and when >= proc_start:
        since_proc[kind] += 1
    match = re.search(r"'tick_delta':\s*(-?\d+)", error)
    if match:
        tick_deltas[(kind, "0" if match.group(1) == "0" else "nonzero")] += 1
    for name in re.findall(r"'name':\s*'([^']+)'[^}]*?'passed':\s*False", error):
        suite_fails[name] += 1
    if kind not in examples:
        examples[kind] = error[:600]

out["failures_total"] = total
out["buckets_all_time"] = dict(buckets)
out["buckets_since_deploy"] = dict(since_deploy)
out["buckets_since_supervisor_start"] = dict(since_proc)
out["tick_delta_split"] = {f"{k[0]}:{k[1]}": v for k, v in tick_deltas.items()}
out["failing_suites_top"] = dict(suite_fails.most_common(12))
out["examples"] = examples

# What does the CURRENT brain say -- is the tick actually advancing right now?
try:
    import urllib.request
    def stat():
        with urllib.request.urlopen(
                "http://127.0.0.1:8095/stats", timeout=20) as handle:
            return json.loads(handle.read().decode("utf-8"))
    first = stat()
    time.sleep(8)
    second = stat()
    out["tick_now"] = second.get("tick") or second.get("total_ticks")
    out["tick_delta_8s"] = ((second.get("tick") or second.get("total_ticks") or 0)
                            - (first.get("tick") or first.get("total_ticks") or 0))
    out["resident_terminals"] = second.get("resident_terminals")
    out["accepted_episodes"] = second.get("accepted_episodes")
except Exception as error:
    out["stats_error"] = str(error)[:200]

print("PROBEJSON " + json.dumps(out))
PY
