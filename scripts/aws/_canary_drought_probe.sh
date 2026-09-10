python3 - <<'PY'
"""Why has `continuous_canary` held the go-systems block for 101.6 hours?

The watchdog reports a drought, not a cause. `state=continuous_canary` freezes
the row by design, so a rate of 0 proves nothing on its own -- but a status
file 500 s stale during a state that is supposed to be writing does. This
probe separates the two: it reads who is alive, what the ledger has recorded
in the recent past, and -- the part a single `last_failure` string cannot give
-- the WHOLE population of canary outcomes bucketed by reason.

`last_failure` names a deferred-replay worker while `service_stage` is
`forward`; that string may predate the running supervisor entirely, so every
process start time is reported next to every failure timestamp.
"""
import collections
import json
import os
import pathlib
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
now = time.time()
out = {"now": now}


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def proc(pid):
    try:
        cmd = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
        cmd = cmd.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        st = os.stat(f"/proc/{pid}")
        rss = 0
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
        return {"pid": int(pid), "age_s": round(now - st.st_mtime, 1),
                "rss_gb": rss, "cmd": cmd[:240]}
    except Exception as exc:  # noqa: BLE001
        return {"pid": pid, "err": str(exc)}


# 1. Who is actually alive. pgrep matches the supervisor for a bare "brain"
#    pattern, so the brain is matched on its binary name only.
procs = {}
for label, pat in (
    ("supervisor", "programming_curriculum_supervisor"),
    ("driver", "drive_corpora_brain"),
    ("worker", "deferred_replay"),
    ("brain", "brain_server"),
):
    procs[label] = [proc(p) for p in sh("pgrep", "-f", pat).split()[:6]]
out["procs"] = procs

# 2. Status + every progress writer, so the freshest one is visible rather
#    than assumed (a forward block writes the status file, not the progress
#    file, and the stale one reads like live throughput).
def stat_file(p):
    try:
        return {"age_s": round(now - os.path.getmtime(p), 1),
                "size": os.path.getsize(p)}
    except OSError:
        return None


status_path = R / "curriculum-supervisor.status.json"
try:
    out["status"] = json.loads(status_path.read_text())
except Exception as exc:  # noqa: BLE001
    out["status"] = {"err": str(exc)}
out["status_stat"] = stat_file(status_path)

writers = []
for p in sorted(R.glob("*.progress.json")) + sorted(R.glob("deferred-replay-active.json")):
    s = stat_file(p)
    if s:
        writers.append({"file": p.name, **s})
writers.sort(key=lambda w: w["age_s"])
out["writers"] = writers[:6]

# 3. The ledger. Bucket EVERY canary outcome, not just the newest, and keep
#    the raw tail so an unbucketed shape is still visible.
ledger = R / "curriculum-health.jsonl"
kinds = collections.Counter()
canary_reasons = collections.Counter()
recent = []
gate_recent = []
last_by_kind = {}
try:
    with ledger.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            kind = str(ev.get("kind") or "")
            kinds[kind] += 1
            ts = ev.get("unix") or ev.get("timestamp") or 0
            try:
                ts = float(ts)
            except Exception:  # noqa: BLE001
                ts = 0.0
            last_by_kind[kind] = ts
            if "canary" in kind:
                canary_reasons[(kind, str(ev.get("reason") or "")[:160])] += 1
            if ts and now - ts < 6 * 3600:
                recent.append({
                    "kind": kind, "age_h": round((now - ts) / 3600.0, 2),
                    "phase": ev.get("phase"), "passed": ev.get("passed"),
                    "reason": str(ev.get("reason") or "")[:220],
                })
            if "gate" in kind or "admit" in kind or "harvest" in kind:
                gate_recent.append({
                    "kind": kind, "age_h": round((now - ts) / 3600.0, 2) if ts else None,
                    "phase": ev.get("phase"), "passed": ev.get("passed"),
                    "first": ev.get("first_passed_suites"),
                    "confirm": ev.get("confirm_passed_suites"),
                    "total": ev.get("total_suites"),
                    "reason": str(ev.get("reason") or "")[:220],
                })
except Exception as exc:  # noqa: BLE001
    out["ledger_err"] = str(exc)

out["ledger_stat"] = stat_file(ledger)
out["recent_6h"] = recent[-40:]
out["gate_events_tail"] = gate_recent[-25:]
out["canary_reason_population"] = [
    {"kind": k, "reason": r, "n": n} for (k, r), n in canary_reasons.most_common(20)
]
out["last_seen_age_h"] = {
    k: round((now - t) / 3600.0, 2)
    for k, t in sorted(last_by_kind.items(), key=lambda kv: -kv[1])[:18] if t
}

# 4. Memory, because a 3 GB floor is what triggers a cooperative yield and a
#    yield mid-canary looks identical to a canary that simply never returns.
mem = {}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
    for key in ("MemTotal:", "MemAvailable:", "SwapTotal:"):
        if line.startswith(key):
            mem[key.strip(":")] = round(int(line.split()[1]) / (1024.0 * 1024.0), 2)
out["mem_gb"] = mem

# 5. The most recent supervisor log tail: the canary runs in-process, so its
#    stall shows up here and nowhere else.
logs = sorted(R.glob("*.log"), key=lambda p: -p.stat().st_mtime)[:4]
out["log_tails"] = {}
for p in logs:
    try:
        with p.open("rb") as fh:
            fh.seek(max(0, p.stat().st_size - 4000))
            tail = fh.read().decode("utf-8", "replace")
        out["log_tails"][p.name] = {
            "age_s": round(now - p.stat().st_mtime, 1),
            "tail": tail[-1800:],
        }
    except Exception as exc:  # noqa: BLE001
        out["log_tails"][p.name] = {"err": str(exc)}

print("PROBE_JSON " + json.dumps(out, default=str))
PY
