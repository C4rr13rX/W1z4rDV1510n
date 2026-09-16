python3 - <<'PY'
"""Measure why nothing owns terminal state `disk_exhausted_unrecoverable`.

The watchdog reported "162.31 GB free against a 150.0 GB floor; None reclaim
attempts returned None bytes" and blamed the wrapper for being unable to write
its runtime identity files. Both halves need checking before acting:

  * free ABOVE the floor at halt time is not what the supervisor saw. The
    status record carries `disk_free_bytes` from the instant it halted, and
    that number, not the watchdog's later `df`, is the one the guard acted on.
  * "None reclaim attempts" is a schema mismatch, not a measurement: the
    worker-branch status record (supervisor.py:4120) publishes no reclaim
    fields at all, while the ledger carries `disk_floor_reclaim` x3. Read the
    ledger.
  * `wrapper_enospc` is false and the volume has 107M free inodes, so the
    "cannot write its runtime identity files" clause is an inference the
    payload does not support. The unit's own exit status decides this:
    DISK_EXHAUSTED_EXIT is 90 and `RestartPreventExitStatus=42 90`, so a
    deliberate halt and a crashed wrapper look identical from a 0/0/0 census.

Also measures the reclaim the guard never attempts. `reclaim_disk_for_floor`
calls `prune_resolved_deferred_bases` and nothing else, and CLAUDE.md records
that reclaim as exhausted (2 prunable directories, 0.01 GB). The bytes that
actually come back on this volume are the extents unique to `brain.wbrain`
against the last-good guard -- 587.83 GB measured 2026-09-11, and 419G -> 611G
observed live during the queue-ordering deploy. That reclaim is only reachable
through a supervisor RESTART, which is why an operator restarting by hand fixes
a state the automation calls unrecoverable.
"""
import json
import os
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
UNIT = "wizard-curriculum-supervisor.service"
out = {"now": time.time(), "runtime": R}


def sh(cmd, timeout=120, tail=4000):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "rc": proc.returncode,
            "out": (proc.stdout or "").strip()[-tail:],
            "err": (proc.stderr or "").strip()[-600:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except Exception as exc:  # noqa: BLE001
        return {"_error": f"{type(exc).__name__}: {exc}"}


# ---- 1. Does the unit describe a deliberate halt or a crash? --------------
out["unit"] = sh(
    f"systemctl show {UNIT} -p ActiveState -p SubState -p Result "
    "-p ExecMainStatus -p ExecMainCode -p NRestarts "
    "-p RestartPreventExitStatus -p Restart --no-pager"
)["out"]
out["unit_enabled"] = sh(f"systemctl is-enabled {UNIT}")["out"]
out["journal_tail"] = sh(
    f"journalctl -u {UNIT} -n 40 --no-pager -o short-iso"
)["out"]

# ---- 2. Volume, right now -------------------------------------------------
usage = shutil.disk_usage(R)
out["disk"] = {
    "free_gb": round(usage.free / 2**30, 2),
    "total_gb": round(usage.total / 2**30, 2),
    "free_bytes": int(usage.free),
}
out["df"] = sh(f"df -B1 {R} /srv/wizard /")["out"]

# ---- 3. What the supervisor itself published at the halt ------------------
status = read_json(f"{R}/curriculum-supervisor.status.json")
out["status"] = status
out["status_age_seconds"] = round(time.time() - float(status.get("updated_unix") or 0), 1)
out["status_free_gb_at_halt"] = (
    round(int(status["disk_free_bytes"]) / 2**30, 2)
    if isinstance(status.get("disk_free_bytes"), (int, float)) else None
)
out["active_interval"] = read_json(f"{R}/deferred-replay-active.json")

# ---- 4. The reclaim attempts the status record does not carry -------------
ledger = f"{R}/curriculum-health.jsonl"
reclaims, exhausted, yields = [], [], []
try:
    with open(ledger, encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            if kind == "disk_floor_reclaim":
                reclaims.append(event)
            elif kind == "disk_exhausted_unrecoverable":
                exhausted.append(event)
            elif kind == "deferred_replay_resource_yield":
                yields.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)


def trim_reclaim(event):
    return {
        "removed_count": event.get("removed_count"),
        "reclaimed_gb": round((event.get("reclaimed_bytes") or 0) / 2**30, 3),
        "free_gb_before": round((event.get("free_bytes_before") or 0) / 2**30, 2),
        "free_gb_after": round((event.get("free_bytes_after") or 0) / 2**30, 2),
        "cleared_floor": event.get("cleared_floor"),
        "phase": event.get("phase"),
        "error": event.get("error"),
    }


out["disk_floor_reclaims"] = [trim_reclaim(e) for e in reclaims]
out["disk_exhausted_events"] = [
    {
        "phase": e.get("phase"),
        "interval_id": e.get("interval_id"),
        "resume_row": e.get("resume_row"),
        "free_gb": round((e.get("disk_free_bytes") or 0) / 2**30, 2),
        "floor_gb": e.get("minimum_free_disk_gb"),
        "attempts": len(e.get("reclaim_attempts") or []),
        "attempt_gb": [
            round((a.get("reclaimed_bytes") or 0) / 2**30, 3)
            for a in (e.get("reclaim_attempts") or [])
        ],
        "updated_unix": e.get("updated_unix"),
    }
    for e in exhausted[-4:]
]
# Burn rate, measured from the yields' own before/after pairs rather than
# extrapolated from one window.
recent = yields[-8:]
out["recent_yields"] = [
    {
        "interval_id": y.get("interval_id"),
        "disk_before_gb": round((y.get("disk_free_bytes_before") or 0) / 2**30, 2),
        "disk_after_gb": round((y.get("disk_free_bytes_after") or 0) / 2**30, 2),
        "mem_before_gb": round((y.get("available_bytes_before") or 0) / 2**30, 2),
        "mem_after_gb": round((y.get("available_bytes_after") or 0) / 2**30, 2),
        "pressure": y.get("pressure"),
    }
    for y in recent
]

# ---- 5. Process census, with the patterns the watcher actually uses -------
census = {"wrapper": 0, "supervisor": 0, "worker": 0, "brain": 0}
details = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as stream:
            cmd = stream.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if "run_programming_curriculum_service.sh" in cmd:
        census["wrapper"] += 1
    if "programming_curriculum_supervisor.py" in cmd:
        census["supervisor"] += 1
    if "drive_corpora_brain" in cmd:
        census["worker"] += 1
    if "w1z4rdv1510n-node" in cmd or "brain_server" in cmd:
        census["brain"] += 1
        try:
            rss = int(
                [l for l in open(f"/proc/{pid}/status") if l.startswith("VmRSS")][0]
                .split()[1]
            ) * 1024
        except Exception:  # noqa: BLE001
            rss = None
        details.append({"pid": int(pid), "rss_gb": round((rss or 0) / 2**30, 2),
                        "cmd": cmd[:200]})
out["census"] = census
out["brain_processes"] = details

# ---- 6. The reclaim the guard never attempts ------------------------------
for name in ("brain/brain.wbrain", "brain/brain.last-good.wbrain"):
    path = f"{R}/{name}"
    try:
        stat = os.stat(path)
        out.setdefault("brain_files", {})[name] = {
            "size_gb": round(stat.st_size / 2**30, 2),
            "nlink": stat.st_nlink,
            "mtime_age_h": round((time.time() - stat.st_mtime) / 3600, 2),
        }
    except OSError as exc:
        out.setdefault("brain_files", {})[name] = {"error": str(exc)}

# Extents unique to the live brain -- the only number that predicts what a
# rollback returns. Summed file sizes are meaningless on a reflink volume.
out["unique_extents"] = sh(
    "python3 - <<'INNER'\n"
    "import subprocess\n"
    "def extents(p):\n"
    "    try:\n"
    "        o=subprocess.run(['filefrag','-v',p],capture_output=True,text=True,timeout=900).stdout\n"
    "    except Exception as e:\n"
    "        return None,str(e)\n"
    "    s=set()\n"
    "    for line in o.splitlines():\n"
    "        parts=line.replace(':',' ').replace('..',' ').split()\n"
    "        if len(parts)>=6 and parts[0].isdigit():\n"
    "            try:\n"
    "                phys=int(parts[4]); ln=int(parts[6]) if len(parts)>6 and parts[6].isdigit() else 0\n"
    "            except (ValueError,IndexError):\n"
    "                continue\n"
    "            if ln: s.add((phys,ln))\n"
    "    return s,None\n"
    "R='/srv/wizard/runtime/programming-integrated-20260713/brain/'\n"
    "a,ea=extents(R+'brain.wbrain')\n"
    "b,eb=extents(R+'brain.last-good.wbrain')\n"
    "if a is None or b is None:\n"
    "    print('extent_error',ea,eb)\n"
    "else:\n"
    "    bb=set(x[0] for x in b)\n"
    "    uniq=sum(l for p,l in a if p not in bb)\n"
    "    tot=sum(l for p,l in a)\n"
    "    print('wbrain_total_gb',round(tot*4096/2**30,2),'wbrain_unique_gb',round(uniq*4096/2**30,2),'lastgood_extents',len(b))\n"
    "INNER",
    timeout=1800,
)["out"]

# ---- 7. What the queue would select on the next start --------------------
out["queue"] = sh(
    "cd " + P + " && python3 - <<'INNER'\n"
    "import json,sys,pathlib\n"
    "sys.path.insert(0,'.')\n"
    "from scripts.programming_curriculum_supervisor import (\n"
    "    unresolved_deferred_intervals, order_replay_candidates)\n"
    "R=pathlib.Path('" + R + "')\n"
    "pending=unresolved_deferred_intervals(R)\n"
    "try:\n"
    "    ordered=order_replay_candidates(R,pending)\n"
    "except TypeError:\n"
    "    ordered=order_replay_candidates(pending)\n"
    "rows=[{'id':e.get('interval_id'),'span':int(e['end_row'])-int(e['start_row'])} for e in ordered[:12]]\n"
    "print(json.dumps({'pending':len(pending),'head':rows}))\n"
    "INNER",
    timeout=300,
)["out"]

# ---- 8. Which generation of the source is on the host --------------------
out["host_source"] = {
    "head": sh(f"git -C {P} rev-parse --short HEAD 2>/dev/null || echo no-repo")["out"],
    "supervisor_md5": sh(f"md5sum {P}/scripts/programming_curriculum_supervisor.py")["out"],
    "has_queue_span_key": sh(
        f"grep -c 'def order_replay_candidates' {P}/scripts/programming_curriculum_supervisor.py"
    )["out"],
    "has_record_replay_stall": sh(
        f"grep -c 'def record_replay_stall' {P}/scripts/programming_curriculum_supervisor.py"
    )["out"],
}

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
