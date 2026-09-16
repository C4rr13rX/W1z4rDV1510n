python3 - <<'PY'
"""Is the capacity halt stable, and WHICH lever actually binds?

The refusal names three remedies and calls all three user decisions. Two of
them are purchases; the third -- delta-encoded terminal updates -- is code in
this repository, so before reporting "awaiting user" the arithmetic has to say
whether that code could close the gap. Two deficits were published:

  * disk:       2.47 h of window against 46.1 h at the FASTEST measured rate
  * throughput: 9,288 h at each interval's OWN measured rate

A burn fix addresses the first. It only addresses the second if the per-row
cost is dominated by eviction I/O rather than by compute -- so measure the
write volume per row, not just per hour.

Read-only: no unit is started, no file is written.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "").strip()
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)


# ---- 1. Is the halt terminal, or is something restarting underneath it? ----
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ActiveState -p SubState -p Result -p ExecMainStatus -p NRestarts "
    "-p MainPID -p RestartPreventExitStatus -p ExecMainExitTimestamp"
)
out["census"] = {}
for name, pat in (
    ("wrapper", "bash run_programming_curriculum_service.sh"),
    ("supervisor", "curriculum_supervisor"),
    ("worker", "drive_corpora_brain"),
    ("brain", "w1z4rdv1510n-node"),
):
    n = 0
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = pathlib.Path("/proc", pid, "cmdline").read_bytes()
        except Exception:  # noqa: BLE001
            continue
        if pat.encode() in cmd.replace(b"\0", b" "):
            n += 1
    out["census"][name] = n

# ---- 2. Disk truth, and whether the volume is moving with nothing running ----
def df():
    st = os.statvfs("/srv/wizard")
    return {
        "free_gb": round(st.f_bavail * st.f_frsize / 1e9, 2),
        "total_gb": round(st.f_blocks * st.f_frsize / 1e9, 2),
    }


d0 = df()
brain = R / "brain" / "brain.wbrain"
guard = R / "brain" / "brain.last-good.wbrain"
s0 = brain.stat().st_size if brain.exists() else 0
time.sleep(20)
d1 = df()
s1 = brain.stat().st_size if brain.exists() else 0
out["disk"] = {
    "before": d0,
    "after": d1,
    "idle_drain_gb_per_hour": round((d0["free_gb"] - d1["free_gb"]) * 180.0, 3),
    "wbrain_gb": round(s1 / 1e9, 2),
    "wbrain_growth_gb_per_hour": round((s1 - s0) / 1e9 * 180.0, 3),
    "guard_gb": round(guard.stat().st_size / 1e9, 2) if guard.exists() else None,
}

# ---- 3. The census the supervisor published, re-read from its own state ----
status = R / "curriculum-supervisor.status.json"
try:
    out["status"] = json.loads(status.read_text())
except Exception as exc:  # noqa: BLE001
    out["status"] = "%s: %s" % (type(exc).__name__, exc)

# ---- 4. Per-phase cost: is a slow row slow because of BYTES or of COMPUTE? ----
# `replay_stall` records carry rows_trained/hours per generation. Bucket them by
# phase so the 260x spread can be attributed rather than restated.
ledger = R / "curriculum-health.jsonl"
phases = {}
stalls = []
if ledger.exists():
    with ledger.open("rb") as fh:
        for raw in fh:
            try:
                ev = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            kind = ev.get("event") or ev.get("kind") or ""
            if "stall" not in str(kind):
                continue
            iid = str(ev.get("interval_id") or "")
            phase = iid.split(":")[0] if iid else "?"
            rows = ev.get("rows_trained")
            hours = ev.get("hours")
            rec = {
                "phase": phase,
                "interval": iid,
                "rows": rows,
                "hours": hours,
                "rows_per_hour": ev.get("rows_per_hour"),
                "unix": ev.get("unix") or ev.get("timestamp"),
            }
            stalls.append(rec)
            if isinstance(rows, (int, float)) and isinstance(hours, (int, float)) \
                    and rows > 0 and hours > 0:
                b = phases.setdefault(phase, {"rows": 0.0, "hours": 0.0, "n": 0})
                b["rows"] += rows
                b["hours"] += hours
                b["n"] += 1
out["phase_rates"] = {
    k: {
        "rows": v["rows"],
        "hours": round(v["hours"], 3),
        "rows_per_hour": round(v["rows"] / v["hours"], 1),
        "samples": v["n"],
    }
    for k, v in sorted(phases.items())
}
out["stalls_tail"] = stalls[-8:]

# ---- 5. What the queue actually owes, by phase ----
deferred = R / "curriculum-deferred-intervals.jsonl"
pending = {}
if deferred.exists():
    seen = {}
    with deferred.open("rb") as fh:
        for raw in fh:
            try:
                ev = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            iid = ev.get("interval_id")
            if iid:
                seen[iid] = ev
    for iid, ev in seen.items():
        if str(ev.get("state") or "") != "deferred":
            continue
        parts = iid.split(":")
        phase = parts[0]
        try:
            span = int(parts[-1]) - int(parts[-2])
        except Exception:  # noqa: BLE001
            span = 0
        b = pending.setdefault(phase, {"intervals": 0, "rows": 0})
        b["intervals"] += 1
        b["rows"] += span
out["pending_by_phase"] = dict(sorted(pending.items()))
out["pending_rows_total"] = sum(v["rows"] for v in pending.values())

# ---- 6. The store's own counters: how much of the burn was learning? ----
out["brain_metrics"] = sh(
    "curl -s --max-time 8 http://127.0.0.1:18095/stats || "
    "curl -s --max-time 8 http://127.0.0.1:8095/stats || echo no-brain"
)[:1200]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
