python3 - <<'PY'
"""Why does nothing own terminal state `deferred_replay_training`?

The wake-up read wrapper 0 / supervisor 0 / worker 0 against a status file
frozen 2880 s at row 211,688 of 262,144. CLAUDE.md records three distinct
causes for that exact census, and they need different repairs:

  * a cooperative memory yield trough (worker restarting -- NORMAL, and the
    row keeps moving, so the census lies but the block is healthy);
  * an ENOSPC crash loop (`activating (auto-restart)`, never `dead`) -- but
    `df` already reports 502 GB free here, so that one is ruled out;
  * a genuinely absent wrapper, which is the only one that needs a restart.

`last_resource_yield.unix` equals `status.updated_unix` exactly, so the last
thing that happened WAS a yield; the question is whether anything came back.
This measures the unit's real lifecycle state, the wrapper/supervisor/worker
census with the patterns copied from `watch_programming_brain.py` rather than
guessed, the tail of every log that could name an exit, and -- decisively --
whether the row is moving. Liveness is the ROW DELTA; the census only
distinguishes a yield trough from a lost supervisor.
"""
import glob
import json
import os
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-7000:] + (
            ("\n[stderr] " + proc.stderr[-800:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


# 1. Lifecycle state. `auto-restart` is a crash loop, not an absence.
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ActiveState -p SubState -p Result -p NRestarts -p ExecMainStatus "
    "-p ExecMainStartTimestamp -p ExecMainExitTimestamp -p StateChangeTimestamp "
    "-p Restart -p RestartUSec -p StartLimitBurst -p StartLimitIntervalUSec "
    "--no-pager 2>&1"
)
out["unit_is_active"] = sh("systemctl is-active wizard-curriculum-supervisor.service 2>&1")
out["unit_is_enabled"] = sh("systemctl is-enabled wizard-curriculum-supervisor.service 2>&1")

# 2. Census with the REAL patterns (a guessed pgrep reports 0 forever).
census = {}
for name, pat in {
    "wrapper": "run_programming_curriculum_service.sh",
    "supervisor": "run_programming_curriculum",
    "worker": "drive_corpora_brain",
    "brain": "brain_server",
}.items():
    hits = []
    for pid in sorted(os.listdir("/proc")):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if pat in cmd:
            try:
                started = os.stat(f"/proc/{pid}").st_mtime
            except OSError:
                started = None
            hits.append({"pid": int(pid), "cmd": cmd[:300], "started_unix": started})
    census[name] = hits
out["census"] = census

# 3. Does the row move? Two samples far enough apart to clear a commit period.
prog = sorted(
    glob.glob(os.path.join(R, "deferred-replay-*.progress.json")),
    key=lambda p: os.path.getmtime(p),
    reverse=True,
)


def read_rows():
    rows = {}
    for path in prog[:4]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                blob = json.load(handle)
        except Exception as exc:
            rows[os.path.basename(path)] = f"{type(exc).__name__}: {exc}"
            continue
        rows[os.path.basename(path)] = {
            "durable_next_row": blob.get("durable_next_row"),
            "accepted_episodes": blob.get("accepted_episodes"),
            "age_s": round(time.time() - os.path.getmtime(path), 1),
        }
    return rows


out["rows_first"] = read_rows()
time.sleep(90)
out["rows_second"] = read_rows()

# 4. Every log that could name an exit, tail-first (the informative line is LAST).
for label, path in {
    "service_stderr": f"{R}/curriculum-service.stderr.log",
    "service_stdout": f"{R}/curriculum-service.stdout.log",
    "supervisor_log": f"{R}/curriculum-supervisor.log",
}.items():
    out[label] = sh(f"test -f {path} && tail -c 4000 {path} || echo MISSING {path}")

out["journal"] = sh(
    "journalctl -u wizard-curriculum-supervisor.service -n 120 --no-pager 2>&1 | tail -c 6000"
)
out["named_failure_log"] = sh(
    f"tail -c 3000 {R}/deferred-replay-8f4a439a7fc7a772.stderr.log 2>&1"
)
out["recent_replay_stderr"] = sh(
    f"ls -1t {R}/deferred-replay-*.stderr.log 2>/dev/null | head -3 "
    f"| while read f; do echo \"=== $f ($(stat -c %y \"$f\"))\"; tail -c 2500 \"$f\"; done"
)

# 5. Durable interval state -- what is actually owed, and OOM evidence.
out["active_json"] = sh(f"tail -c 2500 {R}/deferred-replay-active.json 2>&1")
out["oom"] = sh("dmesg -T 2>/dev/null | grep -i -E 'oom|killed process' | tail -20 || true")
out["free"] = sh("free -g; df -h /srv 2>&1")
out["uptime"] = sh("uptime; date -u")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
