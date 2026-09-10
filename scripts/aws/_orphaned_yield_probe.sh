python3 - <<'PY'
"""Why does nothing own terminal state `deferred_replay_resource_yield`?

The wake-up census read wrapper 0 / supervisor 0 / worker 0 with the row frozen
at 241,048 of 262,144 across a 120 s adaptive sample. CLAUDE.md says a worker
census of 0 is the NORMAL reading mid-yield and that liveness is the ROW DELTA
-- but it also says the census is what distinguishes a yield trough from a host
that has lost its supervisor, and a zero WRAPPER count is exactly that second
case. This probe asks what killed the owner and whether a restart resumes at
241,048 or discards the block.

It reads the exit path from systemd rather than inferring it: the wrapper turns
`deferred_replay_failed` into exit 42 (which latches the service stopped) and
every other non-success state into a raw non-zero rc, so the unit's restart
policy and its start-limit counters decide whether this is self-healing or a
permanent stop.
"""
import glob
import json
import os
import subprocess
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


def sh(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {"rc": proc.returncode, "out": proc.stdout[-4000:], "err": proc.stderr[-1500:]}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


# ---- Process census, patterns copied from watch_programming_brain.py.
def census(needle):
    hits = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().decode("utf-8", "replace").replace("\0", " ").strip()
        except Exception:
            continue
        if needle in cmd:
            hits.append({"pid": int(pid), "cmd": cmd[:220]})
    return hits


out["wrapper"] = census("run_programming_curriculum_service.sh")
out["supervisor"] = census("programming_curriculum_supervisor")
out["worker"] = census("tools.training_standard.drive_corpora_brain")
out["brain"] = census("w1z4rd_brain_server")

# ---- Who is supposed to restart it, and did systemd give up?
out["systemd_units"] = sh(
    "systemctl list-units --type=service --all --no-legend --no-pager "
    "| grep -iE 'wizard|curricul|brain' || true"
)
out["service_show"] = sh(
    "systemctl show wizard-programming-curriculum.service "
    "-p ActiveState -p SubState -p Result -p ExecMainStatus -p ExecMainCode "
    "-p NRestarts -p Restart -p RestartUSec -p StartLimitBurst "
    "-p StartLimitIntervalUSec -p ExecMainExitTimestamp --no-pager 2>&1 || true"
)
out["journal_tail"] = sh(
    "journalctl -u wizard-programming-curriculum.service -n 60 --no-pager "
    "-o short-iso 2>&1 || true"
)

# ---- Identity files the wrapper maintains.
for name in (
    "curriculum-service-supervisor.pid",
    "curriculum-service-supervisor.stage",
    "node.pid",
):
    path = os.path.join(R, name)
    try:
        out.setdefault("identity", {})[name] = {
            "age_seconds": round(now - os.path.getmtime(path), 1),
            "text": open(path, encoding="utf-8").read().strip()[:200],
        }
    except Exception as exc:
        out.setdefault("identity", {})[name] = f"{type(exc).__name__}: {exc}"

# ---- Durable interval state: does a restart resume at the frozen row?
active = load(os.path.join(R, "deferred-replay-active.json"))
out["active"] = {
    k: v for k, v in active.items() if k not in {"interval"}
} if isinstance(active, dict) else active
if isinstance(active, dict):
    if active.get("created_unix"):
        out["active"]["created_age_hours"] = round(
            (now - active["created_unix"]) / 3600.0, 2
        )
    interval = active.get("interval")
    if isinstance(interval, dict):
        out["active_interval_id"] = interval.get("interval_id")
        out["active_interval_status"] = interval.get("status")
        if interval.get("updated_unix"):
            out["active_interval_error_age_hours"] = round(
                (now - interval["updated_unix"]) / 3600.0, 2
            )
        err = interval.get("error")
        if err:
            out["active_interval_error"] = str(err)[:400]

# ---- Every progress file that exposes a row, freshest first.
rows = []
for path in glob.glob(os.path.join(R, "*.progress.json")):
    data = load(path)
    if isinstance(data, dict) and data.get("durable_next_row") is not None:
        rows.append(
            {
                "file": os.path.basename(path),
                "age_seconds": round(now - os.path.getmtime(path), 1),
                "durable_next_row": data.get("durable_next_row"),
                "ram_next_row": data.get("ram_next_row"),
                "accepted_episodes": data.get("accepted_episodes"),
                "resume_row": data.get("resume_row"),
                "end_row": data.get("end_row"),
            }
        )
rows.sort(key=lambda r: r["age_seconds"])
out["row_writers"] = rows[:6]

out["status"] = load(os.path.join(R, "curriculum-supervisor.status.json"))
if isinstance(out["status"], dict) and out["status"].get("updated_unix"):
    out["status_age_seconds"] = round(now - out["status"]["updated_unix"], 1)

# ---- The supervisor's own stderr: the reason it exited.
logs = []
for pattern in ("*.stderr.log", "*.log", "*.out"):
    logs.extend(glob.glob(os.path.join(R, pattern)))
logs = sorted(set(logs), key=lambda p: -os.path.getmtime(p))[:6]
out["recent_logs"] = []
for path in logs:
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except Exception as exc:
        out["recent_logs"].append({"file": os.path.basename(path), "_error": str(exc)})
        continue
    out["recent_logs"].append(
        {
            "file": os.path.basename(path),
            "age_seconds": round(now - os.path.getmtime(path), 1),
            "bytes": len(raw),
            "tail": raw[-2500:],
        }
    )

# ---- Did the kernel kill anything? A 3 GB floor beside a 11.77 GB brain on a
# 15.26 GB host is the exact geometry an OOM kill happens in.
out["oom"] = sh("dmesg -T 2>/dev/null | grep -iE 'oom|killed process' | tail -20 || true")
out["memory"] = sh("free -g")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
