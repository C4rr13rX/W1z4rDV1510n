set -u
# Why this probe: the watchdog fired `fix_required` at 100.5 h without an
# admitted interval while the supervisor reports `continuous_canary` on
# `go-systems` at row 16,416 of 131,072. Two readings in the same payload
# disagree about whether anything is moving -- the status file is 499 s old
# (fresh) but the newest replay progress file is 100.7 h old (stale). So the
# question is not "is a process up"; it is which of the canary's 110 recorded
# failures is the dominant, current one, and whether the canary row advances
# at all. Bucketing every failure beats re-debugging the single named one
# (the `named_failure_vs_population` lesson: last_failure was 6.6% of 288).
python3 - <<'PY'
import collections, glob, json, os, re, subprocess, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as error:
        return {"_error": str(error)[:120]}


# --- what the supervisor says it is doing ---------------------------------
for name, key in (("curriculum-supervisor.status.json", "supervisor"),
                  ("deferred-replay-active.json", "replay")):
    path = os.path.join(R, name)
    if os.path.exists(path):
        out[key] = read_json(path)
        out[key + "_age_s"] = round(time.time() - os.path.getmtime(path), 1)

# --- does the canary row actually advance? --------------------------------
samples = []
for _ in range(5):
    active = read_json(os.path.join(R, "deferred-replay-active.json"))
    samples.append({
        "t": round(time.time() - out["now"], 1),
        "canary_row": active.get("canary_row"),
        "ram_next_row": active.get("ram_next_row"),
        "durable_next_row": active.get("durable_next_row"),
        "state": active.get("state"),
        "pid": active.get("worker_pid"),
    })
    time.sleep(25)
out["samples"] = samples

# --- bucket EVERY health event, not just the named last failure -----------
rows = []
try:
    with open(os.path.join(R, "curriculum-health.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
except Exception as error:
    out["ledger_error"] = str(error)[:160]
rows.sort(key=lambda r: r.get("updated_unix") or 0)

window = out["now"] - 24 * 3600
day = [r for r in rows if (r.get("updated_unix") or 0) >= window]
out["events_24h"] = collections.Counter(str(r.get("kind")) for r in day).most_common(20)
out["ledger_tail"] = [
    {"kind": r.get("kind"),
     "ago_s": round(out["now"] - (r.get("updated_unix") or 0)),
     "phase": r.get("phase"),
     "interval": str(r.get("interval_id"))[:24],
     "error": str(r.get("error") or "")[:220]}
    for r in rows[-16:]
]


def bucket(error_text):
    """Collapse a failure string to a cause class."""
    text = str(error_text or "")
    if not text:
        return "empty"
    for pattern, label in (
        (r"exited (\d+)", "worker_exit"),
        (r"Killed|SIGKILL|out of memory|Cannot allocate", "oom_kill"),
        (r"SIGTERM|terminated", "sigterm"),
        (r"Connection refused|ConnectionError|timed out|Timeout", "brain_unreachable"),
        (r"Permission denied", "permission"),
        (r"'passed':\s*False", "suite_failed"),
        (r"unknown script|SchemaError", "registry"),
    ):
        if re.search(pattern, text):
            return label
    return text.split(":")[0][:48]


failed_kinds = ("continuous_canary_failed", "deferred_replay_failed",
                "midphase_gate_failed", "completion_gate_failed")
buckets = collections.Counter()
suite_fails = collections.Counter()
for row in rows:
    if str(row.get("kind")) not in failed_kinds:
        continue
    buckets[(str(row.get("kind")), bucket(row.get("error")))] += 1
    for name in re.findall(r"'name':\s*'([^']+)'[^}]*?'passed':\s*False",
                           str(row.get("error") or "")):
        suite_fails[name] += 1
out["failure_buckets"] = [{"kind": k, "cause": c, "n": n}
                          for (k, c), n in buckets.most_common(20)]
out["failing_suites"] = suite_fails.most_common(15)

# The most recent canary failure verbatim -- the population above says which
# cause dominates; this says what the dominant one actually looks like.
for row in reversed(rows):
    if str(row.get("kind")) == "continuous_canary_failed":
        out["last_canary_failure"] = {
            "ago_s": round(out["now"] - (row.get("updated_unix") or 0)),
            "phase": row.get("phase"),
            "error": str(row.get("error") or "")[:1400],
        }
        break

# --- the stderr log the watchdog named ------------------------------------
logs = sorted(glob.glob(os.path.join(R, "deferred-replay-*.stderr.log")),
              key=os.path.getmtime)
out["stderr_logs"] = len(logs)
if logs:
    newest = logs[-1]
    out["stderr_newest"] = os.path.basename(newest)
    out["stderr_age_s"] = round(time.time() - os.path.getmtime(newest), 1)
    try:
        with open(newest, encoding="utf-8", errors="replace") as handle:
            out["stderr_tail"] = handle.read()[-2000:]
    except Exception as error:
        out["stderr_tail"] = str(error)[:160]

# --- processes and memory --------------------------------------------------
out["procs"] = subprocess.run(
    ["bash", "-lc", "ps -eo pid,etimes,rss,stat,comm --sort=-rss | head -8"],
    capture_output=True, text=True).stdout.splitlines()
mem = {}
for line in open("/proc/meminfo"):
    key, _, rest = line.partition(":")
    mem[key] = int(rest.split()[0]) * 1024
out["available_gb"] = round(mem.get("MemAvailable", 0) / 2**30, 2)
out["unit"] = subprocess.run(
    ["systemctl", "is-active", "wizard-curriculum-supervisor.service"],
    capture_output=True, text=True).stdout.strip()
try:
    out["service_log_age_s"] = round(
        time.time() - os.path.getmtime(R + "/curriculum-service.stderr.log"), 1)
    with open(R + "/curriculum-service.stderr.log", encoding="utf-8",
              errors="replace") as handle:
        out["service_log_tail"] = handle.read()[-2500:]
except Exception as error:
    out["service_log_tail"] = str(error)[:160]

print("PROBEJSON " + json.dumps(out))
PY
