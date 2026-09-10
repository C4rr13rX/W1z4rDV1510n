python3 - <<'PY'
"""Is the stderr-tail fix RUNNING on the host, or only committed here?

`replay_worker_failure` appends up to 4000 bytes of the worker's stderr after
the log path, so a failure's reason travels with its address. Measured
2026-09-10 from `_quarantine_replay_probe.sh`: 65 `worker_exit` failures, max
error length 137 characters, `over_180_chars` 0 -- and the 12 exits recorded
SINCE the current binary was built name stderr logs that are 1116 bytes each.
A 1116-byte reason sitting beside a 135-character record means the append did
not run.

Two candidate causes, and they need different repairs:

  1. The deployed `programming_curriculum_supervisor.py` predates the fix --
     `deploy_is_not_load`, the lesson that a fix copied but never loaded runs
     the OLD code. Repair: redeploy.
  2. The deployed file HAS the fix and the tail still comes back empty --
     then the seek/mark arithmetic is wrong against a log the parent holds
     open in append mode, and the repair is in this repository.

Distinguishing them is one grep on the host, so guessing is not worth it.

Read-only. Nothing here writes to the runtime or touches the brain.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
SUPERVISOR = "/srv/wizard/project/scripts/programming_curriculum_supervisor.py"
out = {}


def sh(command, timeout=60):
    try:
        return subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout).stdout.strip()
    except Exception as error:
        return f"<{type(error).__name__}: {str(error)[:80]}>"


# --- 1. does the deployed supervisor carry the fix at all? ----------------
path = pathlib.Path(SUPERVISOR)
if path.is_file():
    source = path.read_text(encoding="utf-8", errors="replace")
    out["supervisor"] = {
        "exists": True,
        "bytes": len(source),
        "mtime_age_h": round((time.time() - path.stat().st_mtime) / 3600.0, 2),
        "sha256": hashlib.sha256(source.encode("utf-8", "replace")).hexdigest()[:16],
        "inode": path.stat().st_ino,
        # The three names the fix introduced. All three must be present for
        # the append to be reachable; `defines_tail_helper` alone would only
        # mean the helper exists somewhere unused.
        "defines_tail_helper": "def replay_worker_stderr_tail" in source,
        "calls_tail_helper": "replay_worker_stderr_tail(" in source,
        "takes_stderr_mark": "stderr_mark = stderr_path.stat().st_size" in source,
        "appends_tail": 'f"\\n{stderr_tail}" if stderr_tail else ""' in source,
    }
else:
    out["supervisor"] = {"exists": False}

# --- 2. which supervisor source is the RUNNING process actually using? ----
# A redeploy that lands beside a process started before it changes nothing
# until that process restarts; `deploy_is_not_load` cost 96 h once already.
pids = sh("pgrep -f programming_curriculum_supervisor.py || true").split()
running = []
for pid in pids[:6]:
    exe_cwd = sh(f"readlink -f /proc/{pid}/cwd || true")
    cmdline = sh(f"tr '\\0' ' ' < /proc/{pid}/cmdline || true")
    start = sh(f"stat -c %Y /proc/{pid} || true")
    running.append({
        "pid": pid,
        "cwd": exe_cwd,
        "cmdline": cmdline[:200],
        "age_h": round((time.time() - float(start)) / 3600.0, 2) if start.isdigit() else None,
    })
out["running_supervisors"] = running

# --- 3. prove the claim rather than restate it ---------------------------
# For every replay stderr log that has BYTES, find the ledger record that
# names it and report how long that record's error actually is. A log with
# content beside a 135-character record is the append failing; a log with
# content beside a 1200-character record is the append working.
ledger = pathlib.Path(R) / "curriculum-health.jsonl"
by_path = {}
if ledger.is_file():
    with ledger.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or "stderr=" not in line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            error = str(record.get("error") or record.get("detail") or "")
            if "stderr=" not in error:
                continue
            named = error.split("stderr=", 1)[1].split()[0].split("\n")[0]
            by_path.setdefault(named, []).append({
                "error_chars": len(error),
                "carries_tail": "\n" in error.strip(),
                "unix": record.get("unix") or record.get("timestamp"),
            })

paired = []
for log in sorted(pathlib.Path(R).glob("deferred-replay-*.stderr.log")):
    try:
        size = log.stat().st_size
    except OSError:
        continue
    if size <= 0:
        continue
    records = by_path.get(str(log), [])
    paired.append({
        "log": log.name,
        "log_bytes": size,
        "log_age_h": round((time.time() - log.stat().st_mtime) / 3600.0, 2),
        "records": records,
    })
out["logs_with_content_vs_their_records"] = paired[-12:]
out["logs_with_content_total"] = len(paired)
out["records_carrying_a_tail"] = sum(
    1 for entries in by_path.values() for entry in entries if entry["carries_tail"]
)
out["records_naming_a_log_total"] = sum(len(v) for v in by_path.values())

# --- 4. the two loose threads the last probe surfaced --------------------
out["oom"] = {
    "kills_total": sh("dmesg 2>/dev/null | grep -c -i 'killed process' || echo 0"),
    "recent": sh("dmesg 2>/dev/null | grep -i 'killed process' | tail -3 || true")[:600],
}
active = pathlib.Path(R) / "deferred-replay-active.json"
if active.is_file():
    try:
        state = json.loads(active.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        state = {}
    digest = None
    interval = state.get("interval") or {}
    snapshot = str(interval.get("base_snapshot") or "")
    if "/deferred/" in snapshot:
        digest = snapshot.split("/deferred/", 1)[1].split("/", 1)[0]
    out["resume_marker"] = {
        "digest": digest,
        "state": state.get("state"),
        "interval_id": state.get("interval_id"),
        "candidates": sorted(
            p.name for p in pathlib.Path(R).glob(f"*{digest}*") if digest
        )[:12] if digest else [],
    }

print("PROBEJSON " + json.dumps(out, default=str))
PY
