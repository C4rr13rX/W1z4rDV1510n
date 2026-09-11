python3 - <<'PY'
"""What is 40 MB? Walk the store's own records and read the sizes off disk.

The counters said 215 page-outs grew the volume 8 GB in 300 s -- 39,971,683
bytes per body written. That is three orders of magnitude above the 71 KB mean
this repository has been reasoning with, and the mean was not wrong: 363 GB
across 5.09 M neurons really is ~71 KB. Both can hold only if the neurons being
EVICTED are not typical neurons, so the burn is a small number of enormous
bodies rewritten over and over rather than millions of small ones.

`clean_skips` is 0, which does NOT establish that bodies differ between
evictions: at 235 page-outs against 5.08 M neurons almost nothing has had the
opportunity to be evicted twice, and a count of zero across a window where the
path could not run is not a defect -- that lesson is already in CLAUDE.md and
it applies to my own fix here.

So read the tail of the container directly. Each record is
`W1ZNEUR1` + pool:u32 + id:u32 + len:u64 + body, which gives size, owner and
identity per record with no interpretation. Then:

  * the size distribution says whether the burn is a few hubs or broad churn;
  * repeated ids say whether the same neuron is being rewritten -- the
    thrashing case, where suppression or delta encoding would both pay;
  * distinct ids at 40 MB say the fix is fan-out or encoding, not eviction
    policy.
"""
import collections
import json
import os
import struct
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = f"{R}/brain/brain.wbrain"
MARKER = b"W1ZNEUR1"
HEADER = 24  # marker(8) + pool(4) + id(4) + len(8)
out = {"now": time.time()}

size = os.path.getsize(BRAIN)
out["wbrain_gb"] = round(size / 2**30, 2)

# Walk the last few GB, which is what the current generation has written.
WINDOW = 6 * 1024 ** 3
start = max(0, size - WINDOW)
records = []
with open(BRAIN, "rb") as handle:
    handle.seek(start)
    chunk = handle.read(1 << 20)
    first = chunk.find(MARKER)
    if first < 0:
        out["error"] = "no neuron record marker in the scanned window"
        print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
        raise SystemExit(0)
    offset = start + first
    while offset + HEADER < size and len(records) < 20000:
        handle.seek(offset)
        head = handle.read(HEADER)
        if len(head) < HEADER or head[:8] != MARKER:
            break
        pool, neuron_id = struct.unpack("<II", head[8:16])
        (length,) = struct.unpack("<Q", head[16:24])
        if length == 0 or length > size:
            break
        records.append((offset, pool, neuron_id, length))
        offset += HEADER + length

out["records_walked"] = len(records)
if not records:
    print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
    raise SystemExit(0)

lengths = sorted(r[3] for r in records)
total = sum(lengths)
out["scanned_bytes_gb"] = round(total / 2**30, 3)
out["mean_body_bytes"] = round(total / len(lengths))
out["median_body_bytes"] = lengths[len(lengths) // 2]
out["max_body_bytes"] = lengths[-1]
out["min_body_bytes"] = lengths[0]
out["p90_body_bytes"] = lengths[int(len(lengths) * 0.90)]
out["p99_body_bytes"] = lengths[int(len(lengths) * 0.99)]

# How concentrated is the burn? If a handful of records own most of the bytes,
# the fix targets those neurons and not the eviction rate.
top = sorted(records, key=lambda r: r[3], reverse=True)[:15]
out["largest_records"] = [
    {"pool": p, "id": i, "mb": round(n / 2**20, 2)} for _, p, i, n in top
]
out["top15_share_of_bytes"] = round(sum(r[3] for r in top) / total, 4)

by_pool = collections.Counter()
bytes_by_pool = collections.Counter()
for _, pool, _, length in records:
    by_pool[pool] += 1
    bytes_by_pool[pool] += length
out["records_by_pool"] = dict(by_pool.most_common())
out["bytes_gb_by_pool"] = {
    str(pool): round(count / 2**30, 3)
    for pool, count in bytes_by_pool.most_common()
}

# Is the SAME neuron being rewritten? That is the thrashing case.
ids = collections.Counter((p, i) for _, p, i, _ in records)
repeats = [(k, v) for k, v in ids.most_common(15) if v > 1]
out["distinct_neurons"] = len(ids)
out["repeat_rewrites_top"] = [
    {"pool": k[0], "id": k[1], "times": v} for k, v in repeats
]
rewritten = sum(v for v in ids.values() if v > 1)
out["records_that_are_rewrites"] = rewritten - len(
    [1 for v in ids.values() if v > 1]
)
out["rewrite_fraction"] = round(
    out["records_that_are_rewrites"] / len(records), 4
)
# Bytes attributable to rewriting a neuron already present in this window.
wasted = 0
seen = {}
for _, pool, neuron_id, length in records:
    key = (pool, neuron_id)
    if key in seen:
        wasted += seen[key]
    seen[key] = length
out["bytes_gb_superseded_within_window"] = round(wasted / 2**30, 3)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
