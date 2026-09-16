python3 - <<'PY'
"""What CHANGES between two successive durable bodies of the same neuron?

The capacity halt names delta-encoded terminal updates as the one remedy that
is code rather than a purchase. Whether that code can work is an empirical
question this repository has twice answered by assumption and paid for: the
clean-skip digest suppression was built on "most evictions re-append an
identical body" and turned out inert because the hot atoms are exactly the
neurons training mutates every tick.

So measure the mutation, do not characterise it. `Neuron`'s bincode layout is
positional:

    id .. label .. kind .. members .. terminals(len:u64, elements) ..
    born_tick, last_fired_tick, use_count, prediction_error_ema,
    salience, salience_ema

If training APPENDS terminals and leaves existing weights alone, then body B is
body A with (a) a changed 8-byte vector length, (b) the same element bytes, (c)
extra elements, (d) changed trailing scalars -- so a common prefix that stops at
the length field, and a SECOND long common run once that field is stepped over.

If training REWRITES weights throughout -- lazy decay scaling every terminal on
access would do exactly that -- the second common run is short and a delta
encoding saves nothing. That is the outcome that must be ruled out before a
storage format is changed.

Read-only. Walks the tail of the container and compares bytes; deserializes
nothing.
"""
import collections
import json
import os
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
BRAIN = R / "brain" / "brain.wbrain"
MARK = b"W1ZNEUR1"
HDR = 8 + 4 + 4 + 8  # marker, pool:u32, id:u32, len:u64

# A whole-brain sleep writes each resident neuron ONCE -- measured
# `max_rewrites_of_one_neuron` 1 over a 240 s window -- so a window narrower
# than one full sleep cycle (~12.8 GB measured) contains no repeat at all and
# reports a vacuous zero. Index wide enough to span several cycles.
WINDOW = 96 * 1024**3       # bytes of tail to index
MAX_PAIR_BYTES = 400 * 1024**2
SCAN_SECONDS = 240.0
out = {"now": time.time(), "brain_gb": round(BRAIN.stat().st_size / 1e9, 2)}

size = BRAIN.stat().st_size
start = max(0, size - WINDOW)

# ---- index records in the tail, resyncing on the marker ----------------------
# Starting mid-file means the first marker found may be a false positive inside
# a body, so every candidate is validated by requiring that offset+HDR+len lands
# on another marker. A record that does not chain is skipped, not trusted.
records = collections.defaultdict(list)  # (pool,id) -> [(offset,len)]
scanned = 0
fh = open(BRAIN, "rb")


def marker_at(off):
    if off + 8 > size:
        return False
    fh.seek(off)
    return fh.read(8) == MARK


# resync: find a marker whose declared length chains to another marker
pos = start
anchor = None
CHUNK = 8 * 1024 * 1024
while pos < size and anchor is None and pos - start < 512 * 1024**2:
    fh.seek(pos)
    buf = fh.read(CHUNK)
    if not buf:
        break
    i = 0
    while True:
        i = buf.find(MARK, i)
        if i < 0:
            break
        off = pos + i
        fh.seek(off)
        head = fh.read(HDR)
        if len(head) == HDR:
            blen = int.from_bytes(head[16:24], "little")
            if 0 < blen < 2 * 1024**3 and marker_at(off + HDR + blen):
                anchor = off
                break
        i += 1
    pos += max(1, len(buf) - len(MARK))

out["anchor"] = anchor
if anchor is None:
    print("PROBE_JSON " + json.dumps({**out, "error": "no anchor found"}))
    raise SystemExit(0)

def resync(from_off, limit=3 * 1024**3):
    """Next validated neuron-record offset at or after `from_off`.

    The container interleaves auxiliary records -- slot tables, label indexes --
    between neuron records, so a pure chain-walk stops at the first one and
    reports a vacuous zero (measured: 18 records across a 96 GB window). Scan
    forward for a marker whose declared length lands on another marker, which
    rejects the marker bytes that occur INSIDE a body.
    """
    pos = from_off
    end = min(size, from_off + limit)
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


off = anchor
scan_deadline = time.time() + SCAN_SECONDS
resyncs = 0
while off < size:
    if time.time() > scan_deadline:
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
    pool = int.from_bytes(head[8:12], "little")
    nid = int.from_bytes(head[12:16], "little")
    records[(pool, nid)].append((off + HDR, blen))
    off += HDR + blen
    scanned += 1

out["resyncs"] = resyncs

out["records_scanned"] = scanned
out["distinct_neurons"] = len(records)
repeats = {k: v for k, v in records.items() if len(v) >= 2}
out["neurons_with_two_or_more_records"] = len(repeats)


def common_prefix(a, b):
    n = min(len(a), len(b))
    lo, hi = 0, n
    # binary search on equality of prefixes -- memcmp per step, no byte loop
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def common_suffix(a, b):
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[len(a) - mid:] == b[len(b) - mid:]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def read_body(offset, length):
    fh.seek(offset)
    return fh.read(length)


# Compare the LAST two records of the neurons that carry the most bytes: those
# are the records the burn is actually made of.
ranked = sorted(repeats.items(), key=lambda kv: -max(l for _, l in kv[1]))
pairs = []
budget = MAX_PAIR_BYTES
for (pool, nid), locs in ranked:
    (off_a, len_a), (off_b, len_b) = locs[-2], locs[-1]
    if len_a + len_b > budget:
        continue
    budget -= len_a + len_b
    a = read_body(off_a, len_a)
    b = read_body(off_b, len_b)
    lcp1 = common_prefix(a, b)
    # Step over the changed vector-length field and look for a second long run.
    # The exact skip is not assumed: try a small set and keep the best.
    best_skip, lcp2 = 0, 0
    for skip in (8, 4, 1, 2, 16, 0):
        if lcp1 + skip >= min(len_a, len_b):
            continue
        run = common_prefix(a[lcp1 + skip:], b[lcp1 + skip:])
        if run > lcp2:
            best_skip, lcp2 = skip, run
    csuf = common_suffix(a, b)
    identical = lcp1 + lcp2
    pairs.append({
        "pool": pool,
        "id": nid,
        "records_in_window": len(locs),
        "len_a": len_a,
        "len_b": len_b,
        "grew_bytes": len_b - len_a,
        "common_prefix": lcp1,
        "second_run_skip": best_skip,
        "second_run": lcp2,
        "common_suffix": csuf,
        "identical_bytes": identical,
        "identical_fraction": round(identical / max(1, len_b), 6),
        "delta_bytes_if_encoded": max(0, len_b - identical),
    })
    if len(pairs) >= 24:
        break

fh.close()
out["pairs"] = pairs
if pairs:
    tot_b = sum(p["len_b"] for p in pairs)
    tot_d = sum(p["delta_bytes_if_encoded"] for p in pairs)
    out["summary"] = {
        "pairs": len(pairs),
        "bytes_rewritten_today": tot_b,
        "bytes_if_delta_encoded": tot_d,
        "reduction_factor": round(tot_b / max(1, tot_d), 1),
        "mean_identical_fraction": round(
            sum(p["identical_fraction"] for p in pairs) / len(pairs), 6),
        "mean_growth_bytes": int(sum(p["grew_bytes"] for p in pairs) / len(pairs)),
    }
print("PROBE_JSON " + json.dumps(out, default=str))
PY
