python3 - <<'PY'
"""Does ANY unresolved interval fit the disk window at its own phase's rate?

Commit 345b1cb ordered the replay queue by `(stalls, span)` and justified the
span key with "unresolved spans of 14,336-18,432 rows, ~1.0-1.3h at that rate".
That rate was 3.89 rows/s, measured on go-systems. CLAUDE.md's own lesson says
the work unit is NOT sized in rows -- go rows run ~12.6 rows/s end to end and
jupyter-scientific ~0.84-2.4 rows/s -- so a span key is only a proxy for the
thing that matters, and it is a proxy that is wrong by the ratio between two
phases' costs.

The generation that halted at 11:44 UTC is the direct test. It ran
jupyter-scientific-partial:131072:206948 from 07:15 to 11:44 (4.48h) and moved
the durable row 131,072 -> 142,656: 11,584 rows, 0.72 rows/s. Its remaining
64,292 rows therefore need ~24.8h against a window of ~4.5h. Ordering cannot
fix an interval that does not fit; it can only choose which one to fail at.

So this measures, per phase, from authoritative host state:

  * every unresolved interval, its span, and its recorded stall count
  * rows/s per phase, derived from the replay resume records and the health
    ledger rather than from one sample -- an instantaneous rate on this host is
    a duty cycle, so a single window is not a rate
  * the disk window: what a rollback returns, and the burn that spends it
  * ETA = span / rate, against that window, for the head of the queue

The question it answers is binary and decides the fix: if some interval fits,
ordering by ETA is sufficient. If none fits, ordering is inert by construction
and the work unit itself has to be split.
"""
import collections
import json
import os
import pathlib
import shutil
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
P = "/srv/wizard/project"
out = {"now": time.time()}


def sh(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return {"rc": p.returncode,
                "out": (p.stdout or "").strip()[-6000:],
                "err": (p.stderr or "").strip()[-3000:]}
    except Exception as exc:  # noqa: BLE001
        return {"rc": -1, "out": "", "err": f"{type(exc).__name__}: {exc}"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except Exception:  # noqa: BLE001
        return {}


# ---- 1. The queue exactly as the supervisor would order it ---------------
# Import the deployed module in-process. The previous probe shelled out and
# kept only stdout, so the ImportError that actually happened was discarded
# and the field published as "" -- a vacuous empty, not a measurement.
queue = {"error": None}
try:
    import sys
    sys.path.insert(0, P)
    from scripts.programming_curriculum_supervisor import (  # noqa: E402
        unresolved_deferred_intervals, order_replay_candidates,
        replay_stall_counts,
    )
    pending = unresolved_deferred_intervals(R)
    stalls = replay_stall_counts(R)
    ordered = order_replay_candidates(pending, stalls)
    queue["pending"] = len(pending)
    queue["stalls"] = stalls
    queue["ordered"] = [
        {
            "id": e.get("interval_id"),
            "phase": e.get("phase"),
            "span": int(e["end_row"]) - int(e["start_row"]),
            "stalls": stalls.get(str(e.get("interval_id")), 0),
        }
        for e in ordered
    ]
    by_phase = collections.Counter(e.get("phase") for e in pending)
    queue["by_phase"] = dict(by_phase)
    spans = collections.defaultdict(list)
    for e in pending:
        spans[e.get("phase")].append(int(e["end_row"]) - int(e["start_row"]))
    queue["span_min_by_phase"] = {k: min(v) for k, v in spans.items()}
    queue["span_total_by_phase"] = {k: sum(v) for k, v in spans.items()}
except Exception as exc:  # noqa: BLE001
    queue["error"] = f"{type(exc).__name__}: {exc}"
out["queue"] = queue

# ---- 2. Rows/s per phase, from more than one window ----------------------
# Source A: the resume records left by every replay pass. Each carries the
# durable row this generation reached; paired with the interval's own
# progress-file mtimes they bound the rate without re-running anything.
rates = {"resume_records": [], "admissions": []}
for path in sorted(R.glob("deferred-replay-*.resume.json")):
    record = read_json(path)
    if not record:
        continue
    progress = path.with_name(path.name.replace(".resume.json",
                                                ".progress.json"))
    try:
        stat = progress.stat()
        mtime, size = stat.st_mtime, stat.st_size
    except OSError:
        mtime, size = None, None
    rates["resume_records"].append({
        "file": path.name,
        "interval_id": record.get("interval_id"),
        "resume_row": record.get("resume_row") or record.get("row"),
        "start_row": record.get("start_row"),
        "updated_unix": record.get("updated_unix"),
        "age_h": (round((time.time() - record["updated_unix"]) / 3600, 2)
                  if isinstance(record.get("updated_unix"), (int, float))
                  else None),
        "progress_mtime_age_h": (round((time.time() - mtime) / 3600, 2)
                                 if mtime else None),
        "progress_size": size,
    })

# Source B: admissions in the health ledger carry the interval and when it
# resolved. Span / (resolve - first selection) is an END-TO-END rate, which is
# the one that matters against a wall-clock disk window.
ledger_kinds = collections.Counter()
admitted, stall_events, exhausted, yields = [], [], [], []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            ledger_kinds[kind] += 1
            if kind == "deferred_replay_admitted":
                admitted.append(event)
            elif kind == "deferred_replay_interrupted_before_gate":
                stall_events.append(event)
            elif kind == "disk_exhausted_unrecoverable":
                exhausted.append(event)
            elif kind == "deferred_replay_resource_yield":
                yields.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)

out["ledger_kinds"] = dict(ledger_kinds)
out["stall_events"] = [
    {"interval_id": e.get("interval_id"), "phase": e.get("phase"),
     "reason": str(e.get("reason") or "")[:120],
     "age_h": (round((time.time() - e["updated_unix"]) / 3600, 2)
               if isinstance(e.get("updated_unix"), (int, float)) else None)}
    for e in stall_events[-12:]
]
out["admitted_recent"] = [
    {"interval_id": e.get("interval_id"), "phase": e.get("phase"),
     "recovered": e.get("recovered")}
    for e in admitted[-12:]
]

# ---- 3. Burn rate and the window a rollback would return -----------------
# Measured from consecutive yields' own before/after pairs, which straddle the
# whole generation rather than one sample.
window = {}
pairs = [(y.get("disk_free_bytes_before"), y.get("disk_free_bytes_after"))
         for y in yields[-40:]
         if isinstance(y.get("disk_free_bytes_before"), (int, float))]
window["recent_free_gb"] = [round((b or 0) / 2**30, 2) for b, _ in pairs][-20:]
usage = shutil.disk_usage(R)
window["free_gb_now"] = round(usage.free / 2**30, 2)
window["total_gb"] = round(usage.total / 2**30, 2)
for name in ("brain/brain.wbrain", "brain/brain.last-good.wbrain"):
    try:
        stat = os.stat(R / name)
        window.setdefault("files", {})[name] = {
            "size_gb": round(stat.st_size / 2**30, 2),
            "blocks_gb": round(stat.st_blocks * 512 / 2**30, 2),
            "nlink": stat.st_nlink,
        }
    except OSError as exc:
        window.setdefault("files", {})[name] = {"error": str(exc)}

# What the rollback actually returns: blocks allocated to the live brain that
# the guard does not share. `filefrag -v` output is parsed by the documented
# column layout -- the previous probe guessed at the columns and published
# 5,799,882 GB for an 854 GB file, which is a parser bug, not a measurement.
window["unique"] = sh(
    "python3 -c \""
    "import subprocess,re;"
    "def_=None;"
    "p=lambda f: subprocess.run(['filefrag','-v',f],capture_output=True,"
    "text=True,timeout=3600).stdout;"
    "import sys;"
    "rx=re.compile(r'^\\s*\\d+:\\s+(\\d+)\\.\\.\\s*(\\d+):\\s+(\\d+)\\.\\.\\s*(\\d+):\\s+(\\d+)');"
    "ex=lambda t: [(int(m.group(3)),int(m.group(5))) for m in "
    "(rx.match(l) for l in t.splitlines()) if m];"
    "B='/srv/wizard/runtime/programming-integrated-20260713/brain/';"
    "a=ex(p(B+'brain.wbrain'));b=ex(p(B+'brain.last-good.wbrain'));"
    "sb=set();"
    "[sb.update(range(s,s+n)) for s,n in b];"
    "tot=sum(n for _,n in a);"
    "uniq=sum(sum(1 for k in range(s,s+n) if k not in sb) for s,n in a);"
    "print('extents_a',len(a),'extents_b',len(b),"
    "'total_gb',round(tot*4096/2**30,2),'unique_gb',round(uniq*4096/2**30,2))"
    "\"",
    timeout=3600,
)
out["window"] = window

# ---- 4. The halted generation's own rate, end to end --------------------
status = read_json(R / "curriculum-supervisor.status.json")
out["status"] = status
out["unit_since"] = sh(
    "systemctl show wizard-curriculum-supervisor.service "
    "-p ActiveEnterTimestamp -p InactiveEnterTimestamp -p ExecMainStartTimestamp "
    "-p ExecMainExitTimestamp -p ExecMainStatus -p ActiveState --no-pager"
)["out"]
out["argv"] = sh(
    "grep -o -- '--min-free-disk-gb[ =][0-9.]*' "
    "/srv/wizard/runtime/programming-integrated-20260713/*.log 2>/dev/null | tail -3; "
    "systemctl cat wizard-curriculum-supervisor.service --no-pager | "
    "grep -E 'ExecStart|Restart|min-free'"
)["out"]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
