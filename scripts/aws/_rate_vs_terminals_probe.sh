python3 - <<'PY'
"""Does per-row cost track TOTAL TERMINALS inside a single phase?

This is the measurement that decides what the user should buy.

  * If rate is flat within a phase while terminals grow, the 260x spread is a
    phase attribute (notebook rows are simply longer than Go rows). Then the
    cost is eviction I/O, more RAM removes most of it, and the curriculum can
    finish on bought hardware.

  * If rate falls as terminals rise WITHIN one phase, the cost is the per-tick
    walk over the terminals of whatever fired -- `apply_heterosynaptic_ltd`
    scans every neuron's every terminal each tick -- and that walk is unaffected
    by RAM. Total work is then quadratic in rows trained and no purchase
    finishes the curriculum.

Controlling for phase is the whole point: comparing go-systems to
jupyter-scientific confounds row size with brain size, and this repository has
already priced a queue on exactly that confound.

`settled_node_memory_recycle` and `deferred_replay_resource_yield` records both
carry a `topology` block with `total_terminals` alongside `trained_rows` and
`updated_unix`, so consecutive records inside one phase give a rate and the
brain size it was measured at.

Read-only. Compact output.
"""
import collections
import json
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
LEDGER = R / "curriculum-health.jsonl"
out = {"now": time.time()}

points = collections.defaultdict(list)
with LEDGER.open("rb") as fh:
    for raw in fh:
        try:
            ev = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        topo = ev.get("topology")
        if not isinstance(topo, dict):
            rec = ev.get("recycled")
            topo = rec.get("topology") if isinstance(rec, dict) else None
        if not isinstance(topo, dict):
            continue
        t = ev.get("updated_unix")
        rows = ev.get("trained_rows")
        term = topo.get("total_terminals")
        tick = topo.get("tick")
        phase = ev.get("phase")
        if not all(isinstance(x, (int, float)) for x in (t, rows, term, tick)):
            continue
        points[str(phase)].append({
            "unix": float(t), "rows": float(rows),
            "terminals": int(term), "tick": int(tick),
            "neurons": topo.get("total_neurons"),
        })

series = {}
for phase, pts in points.items():
    pts.sort(key=lambda p: p["unix"])
    segs = []
    for a, b in zip(pts, pts[1:]):
        dh = (b["unix"] - a["unix"]) / 3600.0
        dr = b["rows"] - a["rows"]
        dt = b["tick"] - a["tick"]
        # Only forward progress inside one generation is a rate. A rollback or
        # a worker restart resets the counter and shows as a decrease -- the
        # counter-reset trap this repository has already published as a
        # negative rate twice.
        if dh <= 0.02 or dr <= 0 or dt <= 0:
            continue
        segs.append({
            "age_h": round((out["now"] - b["unix"]) / 3600.0, 1),
            "rows_per_hour": round(dr / dh, 1),
            "ticks_per_row": round(dt / dr, 2),
            "terminals_millions": round(b["terminals"] / 1e6, 1),
            "ticks_per_hour": round(dt / dh, 1),
        })
    if len(segs) >= 3:
        series[phase] = segs

out["phases"] = {}
for phase, segs in series.items():
    segs_by_term = sorted(segs, key=lambda s: s["terminals_millions"])
    half = len(segs_by_term) // 2
    lo = segs_by_term[:half] or segs_by_term[:1]
    hi = segs_by_term[half:] or segs_by_term[-1:]

    def med(xs, key):
        v = sorted(x[key] for x in xs)
        return v[len(v) // 2]

    out["phases"][phase] = {
        "segments": len(segs),
        "terminals_low_millions": round(med(lo, "terminals_millions"), 1),
        "terminals_high_millions": round(med(hi, "terminals_millions"), 1),
        "rows_per_hour_at_low": med(lo, "rows_per_hour"),
        "rows_per_hour_at_high": med(hi, "rows_per_hour"),
        "ticks_per_hour_at_low": med(lo, "ticks_per_hour"),
        "ticks_per_hour_at_high": med(hi, "ticks_per_hour"),
        "ticks_per_row_at_low": med(lo, "ticks_per_row"),
        "ticks_per_row_at_high": med(hi, "ticks_per_row"),
        "slowdown_factor": round(med(lo, "rows_per_hour") /
                                 max(1e-9, med(hi, "rows_per_hour")), 2),
        "terminal_growth_factor": round(
            med(hi, "terminals_millions") / max(1e-9, med(lo, "terminals_millions")), 2),
        "tick_throughput_slowdown": round(
            med(lo, "ticks_per_hour") / max(1e-9, med(hi, "ticks_per_hour")), 2),
    }
    out["phases"][phase]["head"] = segs[:4]
    out["phases"][phase]["tail"] = segs[-4:]

print("PROBE_JSON " + json.dumps(out, default=str)[:20000])
PY
