python3 - <<'PY'
"""What SHARE of the burn's bytes is delta-compressible?

The shape probe found two populations and a five-pair sample cannot weight
them: pool 5 atoms grow by ~53 KB onto an 87 MB body and leave every earlier
byte untouched (identical_fraction 0.999394), while pool 1 neurons rewrite a
34 MB body end to end with ZERO growth (identical_fraction 0.000002). A delta
encoding is ~1,650x on the first and inert on the second, so the value of the
whole change is the byte-weighted split -- not the mean of five pairs, which
this repository has already been burned by ("the twelve largest are 85.9 % of
the bytes" was an all-time figure that a live window flattened to 22.6 %).

Classify by SAMPLING rather than by reading bodies: if the two records agree at
several interior probe points, the earlier bytes survived and the record is
append-shaped. Three 4 KB reads per pair instead of 87 MB makes the census wide
enough to be weighted.

Read-only.
"""
import collections
import json
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
BRAIN = R / "brain" / "brain.wbrain"
MARK = b"W1ZNEUR1"
HDR = 24
CHUNK = 8 * 1024 * 1024
WINDOW = 96 * 1024**3
SCAN_SECONDS = 260.0
PROBE = 4096

out = {"now": time.time()}
size = BRAIN.stat().st_size
out["brain_gb"] = round(size / 1e9, 2)
fh = open(BRAIN, "rb")


def marker_at(off):
    if off + 8 > size:
        return False
    fh.seek(off)
    return fh.read(8) == MARK


def resync(from_off, limit=3 * 1024**3):
    pos, end = from_off, min(size, from_off + limit)
    while pos < end:
        fh.seek(pos)
        buf = fh.read(CHUNK)
        if not buf:
            return None
        i = 0
        while True:
            i = buf.find(MARK, i)
            if i < 0:
                break
            cand = pos + i
            fh.seek(cand)
            head = fh.read(HDR)
            if len(head) == HDR:
                blen = int.from_bytes(head[16:24], "little")
                if 0 < blen < 2 * 1024**3 and marker_at(cand + HDR + blen):
                    return cand
            i += 1
        pos += max(1, len(buf) - len(MARK))
    return None


anchor = resync(max(0, size - WINDOW))
if anchor is None:
    print("PROBE_JSON " + json.dumps({**out, "error": "no anchor"}))
    raise SystemExit(0)

records = collections.defaultdict(list)
off, scanned, resyncs = anchor, 0, 0
deadline = time.time() + SCAN_SECONDS
while off < size:
    if time.time() > deadline:
        out["scan_truncated"] = True
        break
    fh.seek(off)
    head = fh.read(HDR)
    blen = int.from_bytes(head[16:24], "little") if len(head) == HDR else 0
    if len(head) < HDR or head[:8] != MARK or blen <= 0 or off + HDR + blen > size:
        nxt = resync(off + 8)
        if nxt is None:
            break
        resyncs += 1
        off = nxt
        continue
    records[(int.from_bytes(head[8:12], "little"),
             int.from_bytes(head[12:16], "little"))].append((off + HDR, blen))
    off += HDR + blen
    scanned += 1

out["records_scanned"] = scanned
out["resyncs"] = resyncs
out["window_gb"] = round((off - anchor) / 1e9, 2)


def read_at(off, n):
    fh.seek(off)
    return fh.read(n)


def classify(off_a, len_a, off_b, len_b):
    """append-shaped, rewrite-shaped, or identical -- from interior samples."""
    n = min(len_a, len_b)
    if n <= PROBE * 4:
        a, b = read_at(off_a, len_a), read_at(off_b, len_b)
        same = sum(1 for x, y in zip(a, b) if x == y)
        return ("append" if same / max(1, n) > 0.9 else "rewrite"), same / max(1, n)
    hits = 0
    for frac in (0.25, 0.5, 0.75):
        p = int(n * frac)
        if read_at(off_a + p, PROBE) == read_at(off_b + p, PROBE):
            hits += 1
    return ("append" if hits >= 2 else "rewrite"), hits / 3.0


buckets = collections.defaultdict(lambda: {"bytes": 0, "pairs": 0, "grew": 0})
per_pool = collections.defaultdict(lambda: collections.defaultdict(int))
examples = collections.defaultdict(list)
pairs_examined = 0
deadline2 = time.time() + 200.0
for (pool, nid), locs in sorted(records.items(), key=lambda kv: -max(l for _, l in kv[1])):
    if len(locs) < 2:
        continue
    if time.time() > deadline2:
        out["classify_truncated"] = True
        break
    (off_a, len_a), (off_b, len_b) = locs[-2], locs[-1]
    kind, score = classify(off_a, len_a, off_b, len_b)
    # Weight by every rewrite this window actually paid for, not just one pair.
    paid = sum(l for _, l in locs[1:])
    buckets[kind]["bytes"] += paid
    buckets[kind]["pairs"] += 1
    buckets[kind]["grew"] += max(0, len_b - len_a)
    per_pool[pool][kind] += paid
    if len(examples[kind]) < 6:
        examples[kind].append({
            "pool": pool, "id": nid, "records": len(locs),
            "len_b": len_b, "grew": len_b - len_a,
            "paid_bytes": paid, "score": round(score, 4),
        })
    pairs_examined += 1

fh.close()
total = sum(v["bytes"] for v in buckets.values())
out["pairs_examined"] = pairs_examined
out["byte_weighted"] = {
    k: {
        "bytes": v["bytes"],
        "gb": round(v["bytes"] / 1e9, 2),
        "share": round(v["bytes"] / max(1, total), 4),
        "pairs": v["pairs"],
        "growth_bytes": v["grew"],
    }
    for k, v in sorted(buckets.items())
}
out["repeat_bytes_total_gb"] = round(total / 1e9, 2)
out["per_pool"] = {
    str(p): {k: round(b / 1e9, 2) for k, b in sorted(d.items())}
    for p, d in sorted(per_pool.items())
}
out["examples"] = {k: v for k, v in examples.items()}

# Projected burn if append-shaped records cost their growth instead of a body.
app = buckets.get("append", {"bytes": 0, "grew": 0})
rew = buckets.get("rewrite", {"bytes": 0})
projected = app["grew"] + rew["bytes"]
out["projection"] = {
    "bytes_today": total,
    "bytes_with_delta_encoding": projected,
    "reduction_factor": round(total / max(1, projected), 2),
    "note": "append-shaped records charged their measured growth; "
            "rewrite-shaped records charged in full (delta cannot help them)",
}
print("PROBE_JSON " + json.dumps(out, default=str))
PY
