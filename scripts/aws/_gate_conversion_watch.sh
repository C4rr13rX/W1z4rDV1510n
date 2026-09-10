python3 - <<'PY'
"""Watch go-systems:0:131072 through its gate and report the VERDICT.

CLAUDE.md: never report training as working without checking that it CONVERTS.
The measurement is the resolved count rising, not that a process is alive. So
poll until the interval leaves `state:training`, then read what the gate said
at the level that carries a verdict.

This block matters specifically: it is the first to reach a midphase gate
since commit c343443 was deployed (1.4 h ago), and the gate crash that fix
removes is what falsely quarantined the two previous go-systems blocks.

Read-only.
"""
import glob, json, os, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
DEADLINE = time.time() + 660
out = {"start": time.time(), "samples": []}


def jload(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default if default is not None else {}


def health_tail(n=6):
    recs = []
    try:
        with open(f"{R}/curriculum-health.jsonl", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return recs[-n:], len(recs)


base_tail, base_n = health_tail(1)
out["health_lines_at_start"] = base_n

last_state = None
while time.time() < DEADLINE:
    act = jload(f"{R}/deferred-replay-active.json")
    st = jload(f"{R}/curriculum-supervisor.status.json")
    prog = sorted(glob.glob(f"{R}/deferred-replay-*.progress.json"),
                  key=os.path.getmtime)
    row = jload(prog[-1]).get("durable_next_row") if prog else None
    state = act.get("state")
    _, n = health_tail(1)
    snap = {"t": round(time.time() - out["start"]), "active_state": state,
            "row": row, "status_state": st.get("state"),
            "health_lines": n}
    if state != last_state or n != base_n or snap["t"] % 120 < 12:
        out["samples"].append(snap)
        last_state = state
    # the interval has left training: the gate has spoken (or is speaking)
    if state and state != "training" and n > base_n:
        out["transitioned"] = snap
        time.sleep(20)
        break
    time.sleep(12)

tail, n = health_tail(12)
out["health_lines_at_end"] = n
out["new_health_records"] = [
    {"kind": r.get("kind"), "phase": r.get("phase"), "passed": r.get("passed"),
     "interval_id": r.get("interval_id"),
     "age_s": round(time.time() - float(r.get("updated_unix") or time.time())),
     "error_head": str(r.get("error") or "")[:220],
     "error_tail": str(r.get("error") or "")[-220:],
     "note": str(r.get("note") or "")[:120],
     "passed_suites": r.get("passed_suites"), "total_suites": r.get("total_suites")}
    for r in tail]

g = f"{R}/go-systems.enterprise-gate.json"
if os.path.exists(g):
    gd = jload(g)
    out["enterprise_gate"] = {
        "age_h": round((time.time() - os.path.getmtime(g)) / 3600.0, 2),
        "passed": gd.get("passed"), "passed_suites": gd.get("passed_suites"),
        "total_suites": gd.get("total_suites"),
        "failed_suites": [r.get("name") for r in (gd.get("results") or [])
                          if not r.get("passed")]}

out["active_final"] = jload(f"{R}/deferred-replay-active.json")
with open("/proc/meminfo") as fh:
    mi = {l.split(":")[0]: l.split()[1] for l in fh if ":" in l}
out["mem_available_gb"] = round(int(mi["MemAvailable"]) / 1048576.0, 2)

print("PROBE_JSON " + json.dumps(out, default=str)[:12000])
PY
