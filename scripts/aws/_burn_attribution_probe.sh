python3 - <<'PY'
"""Attribute the 206 GB/h disk burn to a FILE before repairing a mechanism.

The standing explanation is eviction churn: the brain runs against its memory
floor, so every tick sleeps neurons and each sleep appends a body the
append-only `.wbrain` store never reclaims. That story predicts the burn should
be near zero right after a `deferred_replay_resource_yield`, when the brain has
just restarted at ~2 GB RSS with ~12 GB free. It was NOT: 8.591 GB in 150 s
measured 3-6 minutes past a recycle.

So the mechanism is not (only) memory pressure, and repairing eviction would be
repairing the wrong thing. This samples every large file under the runtime at
two instants and reports the growth of each, so the burn is charged to a name
rather than to a hypothesis. `st_size` deltas are used deliberately: for an
append the apparent-size delta equals the allocated delta, and unlike a `du`
total it is not inflated by the reflink sharing that made a 560 GB reclaim
prediction return 0.00 GB.

The brain's own counters are read across the same window so growth can be
divided by ticks and by neurons, which distinguishes "storing what it learned"
from "rewriting what it already knew".
"""
import json
import os
import shutil
import stat as statmod
import time
import urllib.request

R = "/srv/wizard/runtime/programming-integrated-20260713"
WINDOW = 120.0
out = {"window_seconds": WINDOW}


def stats(port_list=(18095, 8095)):
    for port in port_list:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/stats", timeout=8) as r:
                return port, json.loads(r.read().decode("utf-8"))
        except Exception:
            continue
    return None, None


def walk_sizes(root):
    """Apparent size of every regular file, plus its link count."""
    sizes = {}
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        # Do not descend into other filesystems or into the corpora inputs,
        # which are static and cannot be the source of a live burn.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                st = os.lstat(path)
            except Exception:
                continue
            # Symlinks carry the size of their TARGET path string, so counting
            # them would charge growth to a name that owns no bytes.
            if not statmod.S_ISREG(st.st_mode):
                continue
            sizes[path] = (st.st_size, st.st_nlink)
    return sizes


free0 = shutil.disk_usage(R).free
port, s0 = stats()
sizes0 = walk_sizes(R)
t0 = time.time()

time.sleep(WINDOW)

sizes1 = walk_sizes(R)
free1 = shutil.disk_usage(R).free
_, s1 = stats()
t1 = time.time()

elapsed = t1 - t0
out["elapsed_seconds"] = round(elapsed, 1)
out["df_consumed_gb"] = round((free0 - free1) / 1e9, 3)
out["df_free_end_gb"] = round(free1 / 1e9, 3)
if free0 > free1:
    burn = (free0 - free1) / elapsed * 3600.0 / 1e9
    out["df_burn_gb_per_hour"] = round(burn, 2)
    out["df_runway_hours"] = round(free1 / 1e9 / burn, 2) if burn > 0 else None

grew = []
for path, (size1, nlink) in sizes1.items():
    size0 = sizes0.get(path, (0, nlink))[0]
    delta = size1 - size0
    if delta > 1_000_000:
        grew.append(
            {
                "path": path.replace(R + "/", ""),
                "grew_gb": round(delta / 1e9, 4),
                "size_gb": round(size1 / 1e9, 3),
                "nlink": nlink,
                "new_file": path not in sizes0,
            }
        )
grew.sort(key=lambda r: -r["grew_gb"])
out["grew"] = grew[:15]
out["grew_total_gb"] = round(sum(r["grew_gb"] for r in grew), 3)

gone = []
for path, (size0, _n) in sizes0.items():
    if path not in sizes1 and size0 > 1_000_000_000:
        gone.append({"path": path.replace(R + "/", ""), "was_gb": round(size0 / 1e9, 3)})
out["deleted_during_window"] = gone[:10]

biggest = sorted(sizes1.items(), key=lambda kv: -kv[1][0])[:12]
out["largest"] = [
    {"path": p.replace(R + "/", ""), "gb": round(sz / 1e9, 3), "nlink": nl}
    for p, (sz, nl) in biggest
]

out["brain_port"] = port
for label, snap in (("stats_start", s0), ("stats_end", s1)):
    if isinstance(snap, dict):
        out[label] = {
            k: snap.get(k)
            for k in (
                "total_neurons",
                "total_concepts",
                "tick",
                "current_tick",
                "resident_terminals",
                "evictions",
                "sleeps",
            )
            if snap.get(k) is not None
        }

if isinstance(s0, dict) and isinstance(s1, dict):
    def delta(key):
        a, b = s0.get(key), s1.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return b - a
        return None

    d_tick = delta("tick") if delta("tick") is not None else delta("current_tick")
    d_neurons = delta("total_neurons")
    out["tick_delta"] = d_tick
    out["neuron_delta"] = d_neurons
    consumed = free0 - free1
    if d_tick:
        out["mb_per_tick"] = round(consumed / d_tick / 1e6, 2)
    if d_neurons:
        out["mb_per_new_neuron"] = round(consumed / d_neurons / 1e6, 2)

print("PROBE_JSON " + json.dumps(out, sort_keys=True))
PY
