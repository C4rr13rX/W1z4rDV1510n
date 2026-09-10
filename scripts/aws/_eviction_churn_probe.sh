python3 - <<'PY'
"""Is the 115 GB/h disk burn eviction churn, or the training itself?

The distinction decides the fix. If the bytes are neuron bodies re-appended by
page-out under memory pressure, the cause is a RAM-starved host converting
thrash into permanent disk growth, and more disk is the wrong purchase. If the
bytes track trained rows independently of paging, it is the training write path.

The store exposes `page_ins`/`page_outs`; correlate their delta against the
checkpoint's growth over the same window and divide to get bytes per page-out.
A figure near a plausible neuron body size confirms eviction churn.
"""
import json
import os
import time
import urllib.request

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = os.path.join(R, "brain", "brain.wbrain")
ENDPOINT = "http://127.0.0.1:18095"
WINDOW = 300.0

out = {}


def stats():
    for path in ("/stats", "/brain/stats", "/health"):
        try:
            with urllib.request.urlopen(ENDPOINT + path, timeout=20) as resp:
                return path, json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            out.setdefault("stats_errors", []).append(f"{path}: {exc}")
    return None, {}


def flatten(obj, prefix=""):
    flat = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}{key}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        flat[prefix.rstrip(".")] = obj
    return flat


path, first_raw = stats()
out["stats_path"] = path
first = flatten(first_raw)
out["stats_keys"] = sorted(k for k in first
                           if any(t in k.lower() for t in
                                  ("page", "evict", "resident", "neuron",
                                   "concept", "tick", "sleep")))[:30]

first_bytes = os.stat(BRAIN).st_blocks * 512
first_free = os.statvfs(R).f_bavail * os.statvfs(R).f_frsize
began = time.time()
time.sleep(WINDOW)
_, last_raw = stats()
last = flatten(last_raw)
last_bytes = os.stat(BRAIN).st_blocks * 512
last_free = os.statvfs(R).f_bavail * os.statvfs(R).f_frsize
elapsed = time.time() - began

out["window_seconds"] = round(elapsed, 1)
grew = last_bytes - first_bytes
out["checkpoint_growth_gb"] = round(grew / 1e9, 3)
out["checkpoint_growth_gb_per_hour"] = round(grew / elapsed * 3600 / 1e9, 2)
out["volume_loss_gb_per_hour"] = round(
    (first_free - last_free) / elapsed * 3600 / 1e9, 2)

deltas = {}
for key, value in last.items():
    if key in first and isinstance(first[key], (int, float)):
        change = value - first[key]
        if change:
            deltas[key] = change
out["counter_deltas"] = dict(sorted(
    deltas.items(), key=lambda kv: -abs(kv[1]))[:25])

# Bytes per page-out, if the counter exists under any name.
for key in list(deltas):
    if "page_out" in key.lower() or "pageout" in key.lower():
        if deltas[key] > 0:
            out["bytes_per_page_out"] = round(grew / deltas[key], 1)
            out["page_out_key"] = key
            out["page_outs_in_window"] = deltas[key]
for key in list(deltas):
    if "page_in" in key.lower() or "pagein" in key.lower():
        out["page_ins_in_window"] = deltas[key]

# Memory pressure over the same window: churn is a consequence of it.
try:
    info = {}
    for line in open("/proc/meminfo", encoding="utf-8"):
        name, _, rest = line.partition(":")
        info[name] = int(rest.split()[0])
    out["mem_available_gb"] = round(info.get("MemAvailable", 0) / 2**20, 2)
    out["mem_total_gb"] = round(info.get("MemTotal", 0) / 2**20, 2)
except Exception:
    pass

rss = 0
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    try:
        if open(f"/proc/{entry}/comm").read().strip() != "w1z4rd_brain_se":
            continue
        for line in open(f"/proc/{entry}/status"):
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1])
    except Exception:
        continue
out["brain_rss_gb"] = round(rss / 2**20, 2)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
