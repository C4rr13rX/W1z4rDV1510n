python3 - <<'PY'
"""WHICH neurons consume the burn -- measured on the region appended live.

The existing census walks the container's TAIL and finds that the twelve
largest records are 85.9 % of the bytes. That answers "which records are
biggest", which is NOT the same question as "which records are being written
now", and only the second one tells you whether pinning a handful of atoms
would cut the burn. A fix aimed at the wrong term is inert -- the clean-skip
suppression already proved that here, landing against a workload whose writes
are not redundant.

So: record the size at t0, sleep, then walk ONLY [t0, t1) and tally bytes by
(pool, neuron id). Records are appended, so that window is exactly what the
burn wrote. Only the 24-byte headers are read; the walk seeks over every body.

The `df` delta is measured over the same window, so the neuron-body share of
the burn is reported rather than assumed -- `bytes_per_body_written` once
divided a `df` delta by `page_outs` and silently attributed every other writer
on the volume to eviction.
"""
import collections
import json
import os
import shutil
import struct
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = f"{R}/brain/brain.wbrain"
NEUR, AUX, MANI = b"W1ZNEUR1", b"W1ZAUX01", b"W1ZMANI1"
HEADER = 24
WINDOW = 240.0
out = {"now": time.time()}

start_size = os.path.getsize(BRAIN)
start_ino = os.stat(BRAIN).st_ino
free_a = shutil.disk_usage(R).free
t0 = time.time()
time.sleep(WINDOW)
elapsed = time.time() - t0
end_size = os.path.getsize(BRAIN)
end_ino = os.stat(BRAIN).st_ino
free_b = shutil.disk_usage(R).free

out["elapsed_seconds"] = round(elapsed, 1)
out["inode_stable"] = start_ino == end_ino
out["start_gb"] = round(start_size / 2**30, 3)
out["end_gb"] = round(end_size / 2**30, 3)
out["appended_gb"] = round((end_size - start_size) / 2**30, 3)
out["df_burn_gb_per_hour"] = round(
    (free_a - free_b) / 2**30 / (elapsed / 3600.0), 2)
out["append_gb_per_hour"] = round(
    (end_size - start_size) / 2**30 / (elapsed / 3600.0), 2)

# A rollback replaces the file; the window then describes two different files.
if not out["inode_stable"] or end_size < start_size:
    out["note"] = "container was replaced or truncated during the window"
    print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
    raise SystemExit(0)

# Find the first record boundary at or after t0's size. The offset recorded at
# t0 IS a boundary, because every append lands whole -- but verify the marker
# rather than trusting it, and scan forward if a partial write was in flight.
with open(BRAIN, "rb") as handle:
    offset = start_size
    handle.seek(offset)
    probe_head = handle.read(8)
    if probe_head not in (NEUR, AUX, MANI):
        handle.seek(offset)
        buffer = handle.read(min(256 * 1024 * 1024, end_size - offset))
        found = [buffer.find(m) for m in (NEUR, AUX, MANI)]
        found = [i for i in found if i >= 0]
        if not found:
            out["error"] = "no record boundary found after the t0 offset"
            print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
            raise SystemExit(0)
        offset += min(found)
    out["first_boundary_skew_bytes"] = offset - start_size

    neuron_bytes = collections.Counter()
    neuron_writes = collections.Counter()
    aux_bytes = collections.Counter()
    kinds = collections.Counter()
    walked = 0
    while offset + HEADER <= end_size and walked < 500000:
        handle.seek(offset)
        head = handle.read(HEADER)
        if len(head) < HEADER:
            break
        marker = head[:8]
        if marker not in (NEUR, AUX, MANI):
            break
        pool, second = struct.unpack("<II", head[8:16])
        (length,) = struct.unpack("<Q", head[16:24])
        if length > end_size:
            break
        kinds[marker.decode("ascii")] += 1
        if marker == NEUR:
            neuron_bytes[(pool, second)] += length
            neuron_writes[(pool, second)] += 1
        elif marker == AUX:
            aux_bytes[(pool, f"0x{second:08x}")] += length
        offset += HEADER + length
        walked += 1

out["records_walked"] = walked
out["reached_end"] = offset >= end_size - HEADER
out["records_by_kind"] = dict(kinds)
total_neuron = sum(neuron_bytes.values())
total_aux = sum(aux_bytes.values())
out["neuron_body_gb"] = round(total_neuron / 2**30, 3)
out["auxiliary_gb"] = round(total_aux / 2**30, 3)
appended = max(1, end_size - start_size)
out["neuron_share_of_append"] = round(total_neuron / appended, 4)

# The whole question: is the burn concentrated in a few ids, or spread wide?
top = neuron_bytes.most_common(20)
out["distinct_neurons_written"] = len(neuron_bytes)
out["top_writers"] = [
    {"pool": pool, "id": nid, "mb": round(size / 2**20, 2),
     "writes": neuron_writes[(pool, nid)],
     "share": round(size / max(1, total_neuron), 4)}
    for (pool, nid), size in top
]
out["top12_share_of_neuron_bytes"] = round(
    sum(size for _, size in neuron_bytes.most_common(12)) / max(1, total_neuron), 4)
out["rewritten_neurons"] = sum(1 for c in neuron_writes.values() if c > 1)
out["max_rewrites_of_one_neuron"] = max(neuron_writes.values(), default=0)
out["top_auxiliary"] = [
    {"pool": pool, "tag": tag, "mb": round(size / 2**20, 2)}
    for (pool, tag), size in aux_bytes.most_common(8)
]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
