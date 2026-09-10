python3 - <<'PY'
"""Ask whether the 138 GB/h of appends is LEARNING or THRASH.

The append-only `.wbrain` grows by one full neuron body per page-out. That is
expected under memory pressure. It is NOT expected three minutes after a
`settled_node_memory_recycle`, when the brain holds ~2 GB against ~12 GB free --
yet 4.6 GB of neuron bodies landed in 120 s at exactly that moment.

Two readings separate the possibilities:

* `store_page_outs` climbing with `store_page_ins` climbing about as fast is a
  THRASH loop -- the same neurons are being slept and immediately woken, so the
  file grows without the brain learning anything. `brain_api` already documents
  the mechanism that would cause it: the `.wbrain` store keeps its own
  full-Neuron cache, separate from the pool's resident map and invisible to the
  tier orchestrator, and `get()` inserts a clone on every read miss.
* `store_page_outs` climbing alone is genuine one-way eviction, and the repair
  is the missing compactor rather than the eviction policy.

The curriculum is stopped, so this measures the brain at rest. A non-zero
page-out rate with no worker POSTing rows would itself be a finding.
"""
import json
import time
import urllib.request

PORTS = (18095, 8095, 8090)
PATHS = ("/memory/residency", "/stats")
out = {}


def fetch(port, path, timeout=20):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


live_port = None
for port in PORTS:
    probe = fetch(port, "/stats", timeout=10)
    if isinstance(probe, dict) and "_error" not in probe:
        live_port = port
        out["stats"] = {
            k: probe.get(k)
            for k in ("total_neurons", "total_concepts", "tick", "current_tick")
            if probe.get(k) is not None
        }
        break
out["brain_port"] = live_port

if live_port is None:
    out["unreachable"] = True
else:
    def totals():
        snap = fetch(live_port, "/memory/residency")
        if not isinstance(snap, dict) or "_error" in snap:
            return None, snap
        pools = snap.get("pools")
        if not isinstance(pools, dict):
            return None, {"_error": "no pools object"}
        agg = {
            "store_page_ins": 0,
            "store_page_outs": 0,
            "store_cached_neurons": 0,
            "resident_neurons": 0,
            "logical_neurons": 0,
        }
        per_pool = {}
        for pid, body in pools.items():
            if not isinstance(body, dict):
                continue
            for key in agg:
                value = body.get(key)
                if isinstance(value, (int, float)):
                    agg[key] += value
            per_pool[pid] = {
                "page_outs": body.get("store_page_outs"),
                "page_ins": body.get("store_page_ins"),
                "cached": body.get("store_cached_neurons"),
                "resident": body.get("resident_neurons"),
                "logical": body.get("logical_neurons"),
            }
        return agg, per_pool

    first, per_pool_first = totals()
    t0 = time.time()
    if first is None:
        out["residency_error"] = per_pool_first
    else:
        out["totals_start"] = first
        time.sleep(90)
        second, per_pool_second = totals()
        elapsed = time.time() - t0
        out["elapsed_seconds"] = round(elapsed, 1)
        out["totals_end"] = second
        if second:
            deltas = {k: second[k] - first[k] for k in first}
            out["deltas"] = deltas
            out["page_outs_per_second"] = round(
                deltas["store_page_outs"] / max(elapsed, 1e-9), 3
            )
            out["page_ins_per_second"] = round(
                deltas["store_page_ins"] / max(elapsed, 1e-9), 3
            )
            # A ratio near 1 means every sleep is undone by a wake: the file
            # grows while the brain learns nothing.
            if deltas["store_page_outs"] > 0:
                out["page_in_to_out_ratio"] = round(
                    deltas["store_page_ins"] / deltas["store_page_outs"], 3
                )
            busiest = sorted(
                (
                    (
                        (per_pool_second.get(pid, {}).get("page_outs") or 0)
                        - (body.get("page_outs") or 0),
                        pid,
                    )
                    for pid, body in per_pool_first.items()
                ),
                reverse=True,
            )[:6]
            out["busiest_pools_by_page_out_delta"] = [
                {
                    "pool": pid,
                    "page_out_delta": delta,
                    "cached": per_pool_second.get(pid, {}).get("cached"),
                    "resident": per_pool_second.get(pid, {}).get("resident"),
                    "logical": per_pool_second.get(pid, {}).get("logical"),
                }
                for delta, pid in busiest
            ]

print("PROBE_JSON " + json.dumps(out, sort_keys=True))
PY
