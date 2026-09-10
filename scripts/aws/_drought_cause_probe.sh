python3 - <<'PY'
"""Why has nothing admitted for 113 h while the gate demonstrably RUNS?

`replay_failures_at_gate` is 259 against 65 before the gate, so this is not
the starved-gate signature -- intervals reach their verdict and are told no.
CLAUDE.md: the ledger record carries only COUNTS; the per-suite verdict lives
in `<phase>.enterprise-gate.json` under `results[].name`, and the per-case
verdict one level further out in each suite's own report. Walk to the level
that carries a verdict, and bucket EVERY failure since the last admission --
`last_failure` was 6.6% of the population once already.

Read-only.
"""
import json, os, time, glob, collections

R = "/srv/wizard/runtime/programming-integrated-20260713"
NOW = time.time()
out = {"now": NOW}


def load(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {str(e)[:140]}"}


def age_h(path):
    try:
        return round((NOW - os.path.getmtime(path)) / 3600.0, 2)
    except Exception:
        return None


# --- the live interval ------------------------------------------------------
act = load(f"{R}/deferred-replay-active.json")
out["active"] = {k: act.get(k) for k in
                 ("interval_id", "state", "start_row", "resume_row", "end_row",
                  "attempts", "phase", "category")} if "_error" not in act else act
out["active_age_h"] = age_h(f"{R}/deferred-replay-active.json")

# --- every health event since the last admission ----------------------------
events, last_admit_idx = [], -1
try:
    with open(f"{R}/curriculum-health.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
except Exception as e:
    out["health_error"] = f"{type(e).__name__}: {str(e)[:120]}"

for i, ev in enumerate(events):
    if "admitted" in str(ev.get("event") or ""):
        last_admit_idx = i
out["health_total"] = len(events)
out["last_admit_index"] = last_admit_idx
if last_admit_idx >= 0:
    la = events[last_admit_idx]
    out["last_admission"] = {k: la.get(k) for k in
                             ("event", "unix", "interval_id", "phase", "reason")}

since = events[last_admit_idx + 1:] if last_admit_idx >= 0 else events
out["events_since_admission"] = len(since)
out["event_mix_since"] = collections.Counter(
    str(e.get("event") or "?") for e in since).most_common(20)

# bucket the FAILURE events by their whole reason text, not a prefix
fails = [e for e in since if "fail" in str(e.get("event") or "")]
out["failures_since"] = len(fails)
buckets = collections.Counter()
for e in fails:
    txt = " ".join(str(e.get(k) or "") for k in
                   ("reason", "detail", "error", "failure", "last_failure"))
    txt = txt.strip()
    key = "empty" if not txt else txt[:200]
    buckets[key] += 1
out["failure_reason_buckets"] = buckets.most_common(12)

# the most recent 8 failures verbatim-ish, with their stage
out["recent_failures"] = [
    {"event": e.get("event"), "unix": e.get("unix"),
     "age_h": round((NOW - float(e.get("unix") or NOW)) / 3600.0, 2),
     "interval_id": e.get("interval_id"), "phase": e.get("phase"),
     "reason": str(e.get("reason") or e.get("detail") or "")[:300],
     "stage": e.get("stage") or e.get("gate")}
    for e in fails[-8:]
]

# --- every gate artifact, per-suite ----------------------------------------
arts = sorted(glob.glob(f"{R}/*enterprise-gate.json"), key=os.path.getmtime)
out["gate_artifact_count"] = len(arts)
out["gate_artifacts"] = []
for p in arts[-4:]:
    g = load(p)
    rec = {"file": os.path.basename(p), "age_h": age_h(p)}
    if "_error" not in g:
        rec.update({"passed": g.get("passed"),
                    "passed_suites": g.get("passed_suites"),
                    "total_suites": g.get("total_suites"),
                    "tick_delta": g.get("tick_delta")})
        rec["suites"] = [{"name": r.get("name"), "passed": r.get("passed"),
                          "detail": str(r.get("detail") or r.get("reason") or "")[:120]}
                         for r in (g.get("results") or [])]
    else:
        rec.update(g)
    out["gate_artifacts"].append(rec)

print("PROBE_JSON " + json.dumps(out, default=str))
PY
