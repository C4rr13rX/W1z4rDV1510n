python3 - <<'PY'
"""Is the 260x per-row spread a property of the PHASE, or of the CLOCK?

The capacity refusal prices the queue at "each interval's own measured rate"
and gets 9,288 h, treating rate as a phase attribute: go-systems 59,778 rows/h
against jupyter-scientific-para4 230.5. If that is really a phase property then
more RAM plus a bigger volume finishes the curriculum. If instead rate has been
DECAYING with wall-clock, the cause is structural -- every tick walks the
terminals of the hub atoms it fires, and those atoms grow with every row
trained -- and then no purchase finishes it, because the cost per row rises as
fast as the rows are consumed.

Those two readings recommend opposite actions to the user, so the distinction
has to be measured rather than argued.

Dumps the real record shapes first: the previous pass at this guessed field
names and reported a vacuous empty dict for every phase.

Read-only.
"""
import collections
import json
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
LEDGER = R / "curriculum-health.jsonl"
out = {"now": time.time(), "ledger": str(LEDGER), "exists": LEDGER.exists()}

rows = []
kinds = collections.Counter()
if LEDGER.exists():
    with LEDGER.open("rb") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            rows.append(ev)
            for key in ("event", "kind", "type", "status", "state"):
                if isinstance(ev.get(key), str):
                    kinds[(key, ev[key])] += 1
                    break

out["records"] = len(rows)
out["kind_counts"] = [[f"{k[0]}={k[1]}", n] for k, n in kinds.most_common(40)]
out["sample_keys"] = sorted({k for ev in rows[-400:] for k in ev.keys()})[:60]

# One verbatim example per kind, so field names stop being guessed.
seen = {}
for ev in rows:
    for key in ("event", "kind", "type"):
        v = ev.get(key)
        if isinstance(v, str) and v not in seen:
            seen[v] = {k: ev[k] for k in list(ev)[:24]}
        if isinstance(v, str):
            break
out["examples"] = {k: v for k, v in list(seen.items())
                   if any(t in k for t in ("stall", "admit", "resolve", "yield",
                                           "advanced", "recycle", "census",
                                           "window", "gate"))}


def ts(ev):
    for key in ("unix", "timestamp", "observed_unix", "created_unix",
                "updated_unix", "time"):
        v = ev.get(key)
        if isinstance(v, (int, float)) and v > 1e9:
            return float(v)
    return None


def iid(ev):
    for key in ("interval_id", "interval", "id"):
        v = ev.get(key)
        if isinstance(v, str) and ":" in v:
            return v
        if isinstance(v, dict):
            inner = v.get("interval_id")
            if isinstance(inner, str) and ":" in inner:
                return inner
    return None


# Every record that carries BOTH a row count and an elapsed span is a rate
# sample, whatever the event is called. Collect them all rather than filtering
# on a name that may not be the one this supervisor writes.
samples = []
for ev in rows:
    t = ts(ev)
    if t is None:
        continue
    r = None
    for key in ("rows_trained", "trained_rows", "rows"):
        if isinstance(ev.get(key), (int, float)):
            r = float(ev[key])
            break
    h = None
    for key in ("hours", "elapsed_hours", "duration_hours"):
        if isinstance(ev.get(key), (int, float)):
            h = float(ev[key])
            break
    rph = ev.get("rows_per_hour")
    if not isinstance(rph, (int, float)):
        rph = (r / h) if (r and h and h > 0) else None
    if rph is None or rph <= 0:
        continue
    name = ev.get("event") or ev.get("kind") or ev.get("type") or "?"
    samples.append({
        "unix": t,
        "age_hours": round((out["now"] - t) / 3600.0, 2),
        "event": name,
        "interval": iid(ev),
        "phase": (iid(ev) or ":").split(":")[0] or None,
        "rows": r,
        "hours": round(h, 4) if h else None,
        "rows_per_hour": round(float(rph), 1),
    })

samples.sort(key=lambda s: s["unix"])
out["rate_samples"] = len(samples)
out["series"] = samples[-60:]

by_phase = collections.defaultdict(list)
for s in samples:
    by_phase[s["phase"] or "?"].append(s)
out["per_phase"] = {
    p: {
        "samples": len(v),
        "first_age_hours": v[0]["age_hours"],
        "last_age_hours": v[-1]["age_hours"],
        "first_rate": v[0]["rows_per_hour"],
        "last_rate": v[-1]["rows_per_hour"],
        "median_rate": sorted(x["rows_per_hour"] for x in v)[len(v) // 2],
    }
    for p, v in sorted(by_phase.items())
}

# The discriminator: within a SINGLE phase, does rate fall with wall-clock?
# A phase attribute predicts a flat line; a structural blowup predicts decay.
trends = {}
for p, v in by_phase.items():
    if len(v) < 4:
        continue
    half = len(v) // 2
    early = sorted(x["rows_per_hour"] for x in v[:half])[half // 2]
    late = sorted(x["rows_per_hour"] for x in v[half:])[(len(v) - half) // 2]
    trends[p] = {
        "early_median_rate": early,
        "late_median_rate": late,
        "ratio_early_over_late": round(early / late, 3) if late else None,
        "early_age_hours": v[0]["age_hours"],
        "late_age_hours": v[-1]["age_hours"],
    }
out["within_phase_trend"] = trends

print("PROBE_JSON " + json.dumps(out, default=str)[:60000])
PY
