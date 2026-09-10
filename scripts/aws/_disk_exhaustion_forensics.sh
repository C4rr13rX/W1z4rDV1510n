python3 - <<'PY'
"""What filled the training host's disk, and is the host about to power off?

The wake-up read wrapper 0 / supervisor 0 / worker 0 and called it a terminal
`deferred_replay_resource_yield`. The census was right about the counts and
wrong about the cause: `curriculum-service.stderr.log` is being appended with
`OSError: [Errno 28] No space left on device` on `node.pid.<pid>.tmp` with the
PID incrementing every few hundred milliseconds, and the unit is `activating
(auto-restart)`. So nothing owns the state because the WRAPPER cannot survive
its own identity-publication step -- the supervisor's `--min-free-disk-gb 8`
guard is downstream of a crash that happens before the supervisor is launched.

This measures the filesystem so the reclaim targets the actual consumer instead
of the most conspicuous file, and reports every writer still holding a deleted
inode (space a plain `rm` would not return).
"""
import json
import os
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
now = time.time()
out = {"now": now}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-6000:] + (
            ("\n[stderr] " + proc.stderr[-800:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


out["df"] = sh("df -h -x tmpfs -x devtmpfs")
out["df_inodes"] = sh("df -i -x tmpfs -x devtmpfs")

# Top consumers at each level that matters, largest first.
out["top_srv"] = sh("du -x -m -d 2 /srv 2>/dev/null | sort -rn | head -40")
out["top_runtime_files"] = sh(
    f"find {R} -maxdepth 1 -type f -printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%f\\n' "
    "2>/dev/null | sort -rn | head -40"
)
out["top_root"] = sh("du -x -m -d 1 / 2>/dev/null | sort -rn | head -25")
out["top_var"] = sh("du -x -m -d 2 /var 2>/dev/null | sort -rn | head -20")
out["top_home"] = sh("du -x -m -d 2 /home /tmp 2>/dev/null | sort -rn | head -20")

# Space a plain rm would NOT return: deleted inodes still held open.
out["deleted_open"] = sh(
    "lsof -nP 2>/dev/null | awk '/deleted/ {print $1, $2, $7, $9}' "
    "| sort -k3 -rn | head -25 || true"
)

# Gate runtimes and per-interval artifacts are the historical consumer here.
out["gate_artifacts"] = sh(
    f"find {R} -maxdepth 2 -name '*gate*' -o -maxdepth 2 -name '*polyglot*' "
    "2>/dev/null | head -40"
)
out["runtime_dir_count"] = sh(f"ls -1 {R} | wc -l")
out["runtime_total_mb"] = sh(f"du -x -m -s {R} 2>/dev/null")

# Is the cost guard about to power the host off mid-repair?
out["cost_stop"] = sh(
    "systemctl show wizard-cost-stop.service -p ActiveState -p SubState "
    "-p ExecMainStartTimestamp -p Description --no-pager 2>&1; "
    "systemctl cat wizard-cost-stop.service 2>&1 | head -40"
)
out["curriculum_unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p Result -p NRestarts -p Restart -p RestartUSec "
    "-p StartLimitBurst -p StartLimitIntervalUSec --no-pager 2>&1; "
    "systemctl cat wizard-curriculum-supervisor.service 2>&1 | head -60"
)
out["uptime"] = sh("uptime; date -u")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
