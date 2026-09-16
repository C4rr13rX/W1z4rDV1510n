python3 - <<'PY'
"""Is the per-terminal weight change a UNIFORM SCALAR, or independent learning?

The structure probe showed the pool-1 population is perturbed, not rewritten:
~77 % byte agreement, three disagreeing runs per element, and a stride cycle of
{4, 12, 5} summing to 21 -- exactly bincode's fixint width for
Terminal{target: (u32,u32), weight: f32, consolidation: u8, last_fired_tick: u64}.
The 8-byte target never moves, so element i in both bodies is the same edge and
the two can be compared field by field.

Everything turns on the weight ratio:

  * ratio constant across terminals  -> this is lazy decay being MATERIALISED
    into 86 MB of body on every page-out. A scalar kept symbolically would make
    consecutive bodies byte-identical, and the clean-skip suppression already
    in the store would then write nothing at all. That is the 58.5 % of the
    burn the append-delta cannot touch.

  * ratio varies per terminal -> genuine independent plasticity, no encoding
    helps, and the remedy really is a purchase.

`clean_skips: 0` was read as "the hot atoms genuinely differ every eviction".
They do differ; this asks WHY, because a uniform multiply is not information.

Read-only.
"""
import collections
import json
import math
import pathlib
import struct
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
BRAIN = R / "brain" / "brain.wbrain"
MARK = b"W1ZNEUR1"
HDR = 24
CHUNK = 8 * 1024 * 1024
WINDOW = 96 * 1024**3
ELEM = 21
SAMPLE = 21 * 20000          # whole number of elements

out = {"now": time.time(), "elem_width": ELEM}
size = BRAIN.stat().st_size
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
records = collections.defaultdict(list)
off, deadline = anchor, time.time() + 230.0
while off < size and time.time() < deadline:
    fh.seek(off)
    head = fh.read(HDR)
    blen = int.from_bytes(head[16:24], "little") if len(head) == HDR else 0
    if len(head) < HDR or head[:8] != MARK or blen <= 0 or off + HDR + blen > size:
        nxt = resync(off + 8)
        if nxt is None:
            break
        off = nxt
        continue
    records[(int.from_bytes(head[8:12], "little"),
             int.from_bytes(head[12:16], "little"))].append((off + HDR, blen))
    off += HDR + blen


def read_at(o, n):
    fh.seek(o)
    return fh.read(n)


def find_alignment(a, b):
    """Offset k where the 8-byte target field of each element begins.

    Chosen by maximising agreement on the target bytes: the target is the one
    field the structure probe showed never changes, so the correct phase is the
    one under which those bytes agree almost everywhere. Guessing the phase
    would silently misparse every weight.
    """
    best, best_score = 0, -1.0
    n = min(len(a), len(b))
    for k in range(ELEM):
        agree = tot = 0
        for e in range(200):
            p = k + e * ELEM
            if p + 8 > n:
                break
            agree += 1 if a[p:p + 8] == b[p:p + 8] else 0
            tot += 1
        if tot and agree / tot > best_score:
            best, best_score = k, agree / tot
    return best, best_score


results = []
cands = [(k, v) for k, v in records.items() if k[0] == 1 and len(v) >= 2]
cands.sort(key=lambda kv: -max(l for _, l in kv[1]))
for (pool, nid), locs in cands[:4]:
    (off_a, len_a), (off_b, len_b) = locs[-2], locs[-1]
    n = min(len_a, len_b)
    p = (n // 2) & ~1
    a = read_at(off_a + p, SAMPLE)
    b = read_at(off_b + p, SAMPLE)
    k, score = find_alignment(a, b)
    ratios, same_w, zero_w, tick_moves, cons_moves = [], 0, 0, 0, 0
    pairs = 0
    e = 0
    while k + (e + 1) * ELEM <= min(len(a), len(b)):
        base = k + e * ELEM
        e += 1
        ta = a[base:base + 8]
        tb = b[base:base + 8]
        if ta != tb:                       # not the same edge; skip
            continue
        wa = struct.unpack_from("<f", a, base + 8)[0]
        wb = struct.unpack_from("<f", b, base + 8)[0]
        ca = a[base + 12]
        cb = b[base + 12]
        ka = struct.unpack_from("<Q", a, base + 13)[0]
        kb = struct.unpack_from("<Q", b, base + 13)[0]
        pairs += 1
        if ca != cb:
            cons_moves += 1
        if ka != kb:
            tick_moves += 1
        if wa == wb:
            same_w += 1
        elif wa == 0.0 or not math.isfinite(wa) or not math.isfinite(wb):
            zero_w += 1
        else:
            ratios.append(wb / wa)
    ratios.sort()
    summary = {
        "pool": pool, "id": nid, "records": len(locs),
        "len_a": len_a, "len_b": len_b,
        "alignment_offset": k, "alignment_target_agreement": round(score, 4),
        "elements_compared": pairs,
        "weight_unchanged": same_w,
        "weight_changed": len(ratios),
        "weight_degenerate": zero_w,
        "last_fired_tick_changed": tick_moves,
        "consolidation_changed": cons_moves,
    }
    if ratios:
        q = lambda f: ratios[min(len(ratios) - 1, int(len(ratios) * f))]  # noqa: E731
        mean = sum(ratios) / len(ratios)
        var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
        summary["ratio"] = {
            "min": round(ratios[0], 9),
            "p01": round(q(0.01), 9),
            "median": round(q(0.5), 9),
            "p99": round(q(0.99), 9),
            "max": round(ratios[-1], 9),
            "mean": round(mean, 9),
            "stdev": round(var ** 0.5, 12),
            "coefficient_of_variation": round((var ** 0.5) / abs(mean), 9) if mean else None,
            "distinct_rounded_9dp": len({round(r, 9) for r in ratios}),
            "share_within_1e6_of_median": round(
                sum(1 for r in ratios if abs(r - q(0.5)) < 1e-6) / len(ratios), 6),
        }
    results.append(summary)

fh.close()
out["neurons"] = results
print("PROBE_JSON " + json.dumps(out, default=str))
PY
