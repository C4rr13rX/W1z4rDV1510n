python3 - <<'PY'
"""Account the store's tail by RECORD KIND, because the last probe found none.

Scanning the last 6 GB for `W1ZNEUR1` returned no marker at all. That is itself
evidence: whatever is consuming ~96 GB/h at the tail of this container is not
neuron bodies, so the "40 MB mean body" the counters implied was never a body
at all -- `bytes_per_body_written` divides a `df` delta by `page_outs`, and it
silently attributes every OTHER writer on the volume to eviction.

Three record kinds share the file and all three append:
  W1ZNEUR1  marker(8) pool(4) id(4)   len(8)  body   -- a neuron body
  W1ZAUX01  marker(8) pool(4) kind(4) len(8)  body   -- slot tables, label
                                                        indexes, posting-index
                                                        generations
  W1ZMANI1                                            -- the manifest

`binding_posting_generations` was 193 and `fingerprint_state_generations` 193
at the last reading, and both of those are auxiliary records that are rewritten
WHOLE. A brain with 4.43 M bindings whose posting index is republished once per
generation would produce exactly this shape: few writes, enormous each, no
neuron records in sight.

Walks forward from a marker found anywhere in a large search buffer and
classifies every record until EOF, so the answer does not depend on guessing
which kind sits at the tail.
"""
import collections
import json
import os
import struct
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = f"{R}/brain/brain.wbrain"
NEUR = b"W1ZNEUR1"
AUX = b"W1ZAUX01"
MANI = b"W1ZMANI1"
HEADER = 24
out = {"now": time.time()}

size = os.path.getsize(BRAIN)
out["wbrain_gb"] = round(size / 2**30, 2)

# Find a record boundary by searching a large buffer near the tail.
SEARCH = 512 * 1024 * 1024
start = max(0, size - SEARCH)
with open(BRAIN, "rb") as handle:
    handle.seek(start)
    buffer = handle.read(SEARCH)
    positions = []
    for marker in (NEUR, AUX, MANI):
        index = buffer.find(marker)
        if index >= 0:
            positions.append((start + index, marker))
    if not positions:
        out["error"] = "no record marker of any kind in the last 512 MB"
        print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
        raise SystemExit(0)
    offset, found = min(positions)
    out["first_marker"] = found.decode("ascii")
    out["first_marker_offset_gb"] = round(offset / 2**30, 3)

    records = []
    while offset + HEADER <= size and len(records) < 200000:
        handle.seek(offset)
        head = handle.read(HEADER)
        if len(head) < HEADER:
            break
        marker = head[:8]
        if marker not in (NEUR, AUX, MANI):
            break
        pool, second = struct.unpack("<II", head[8:16])
        (length,) = struct.unpack("<Q", head[16:24])
        if length > size:
            break
        records.append((marker.decode("ascii"), pool, second, length))
        offset += HEADER + length

out["records_walked"] = len(records)
out["walked_to_gb"] = round(offset / 2**30, 3)
out["reached_eof"] = offset >= size - HEADER
if not records:
    print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
    raise SystemExit(0)

total = sum(r[3] for r in records)
out["scanned_bytes_gb"] = round(total / 2**30, 3)

by_kind = collections.Counter()
bytes_by_kind = collections.Counter()
for kind, _, _, length in records:
    by_kind[kind] += 1
    bytes_by_kind[kind] += length
out["records_by_kind"] = dict(by_kind)
out["bytes_gb_by_kind"] = {
    kind: round(count / 2**30, 3) for kind, count in bytes_by_kind.items()
}

# For auxiliary records the second u32 is the KIND tag, not a neuron id.
# 0x534C4F54 "SLOT", 0x4C41424C "LABL"; anything else is brain-layer metadata.
TAGS = {0x534C_4F54: "SLOT", 0x4C41_424C: "LABL"}
aux = collections.Counter()
aux_bytes = collections.Counter()
for kind, pool, second, length in records:
    if kind != "W1ZAUX01":
        continue
    tag = TAGS.get(second, f"0x{second:08x}")
    aux[(pool, tag)] += 1
    aux_bytes[(pool, tag)] += length
out["auxiliary_counts"] = {
    f"pool{pool}:{tag}": count for (pool, tag), count in aux.most_common(20)
}
out["auxiliary_bytes_gb"] = {
    f"pool{pool}:{tag}": round(count / 2**30, 3)
    for (pool, tag), count in aux_bytes.most_common(20)
}

neuron_lengths = sorted(r[3] for r in records if r[0] == "W1ZNEUR1")
if neuron_lengths:
    out["neuron_mean_bytes"] = round(sum(neuron_lengths) / len(neuron_lengths))
    out["neuron_median_bytes"] = neuron_lengths[len(neuron_lengths) // 2]
    out["neuron_max_bytes"] = neuron_lengths[-1]

largest = sorted(records, key=lambda r: r[3], reverse=True)[:12]
out["largest_records"] = [
    {
        "kind": kind,
        "pool": pool,
        "tag": TAGS.get(second, f"0x{second:08x}") if kind == "W1ZAUX01"
        else second,
        "mb": round(length / 2**20, 2),
    }
    for kind, pool, second, length in largest
]
out["top12_share_of_bytes"] = round(sum(r[3] for r in largest) / total, 4)

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
