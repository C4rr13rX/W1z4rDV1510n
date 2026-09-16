python3 - <<'PY'
"""Is the pool-1 "rewrite" population really rewritten, or only PERTURBED?

The share probe classified 58.5 % of repeat bytes as rewrite-shaped using exact
4 KB equality. That predicate cannot tell "every byte changed" from "every 16th
byte changed", and lazy decay -- which scales the weight of every terminal on
access -- produces precisely the second while looking identical to the first
under both exact-equality and common-prefix tests. The earlier
identical_fraction of 0.000002 is a common PREFIX, which also stops at the
first differing byte, so neither existing measurement distinguishes them.

The difference decides the architecture: scattered small perturbations are
captured by a changed-region delta (and so is the append population), whereas a
genuine reorder needs a semantic, terminal-keyed diff or cannot be helped at
all.

So measure positional agreement and the RUN STRUCTURE of the disagreements.
  * high agreement + many short runs  -> scattered field updates (decay)
  * low agreement                     -> wholesale reorder/rebuild
  * high agreement after a shift      -> insertion, not rewrite

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
SAMPLE = 65536

out = {"now": time.time()}
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
off, deadline = anchor, time.time() + 240.0
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


def structure(a, b):
    n = min(len(a), len(b))
    agree = 0
    runs = []              # lengths of maximal DISAGREEING runs
    cur = 0
    for i in range(n):
        if a[i] == b[i]:
            agree += 1
            if cur:
                runs.append(cur)
                cur = 0
        else:
            cur += 1
    if cur:
        runs.append(cur)
    # Period between disagreement-run starts: a fixed stride is the signature of
    # one mutated field inside a fixed-width repeated element.
    starts, idx, run_i = [], 0, 0
    i = 0
    while i < n and len(starts) < 4096:
        if a[i] != b[i]:
            starts.append(i)
            while i < n and a[i] != b[i]:
                i += 1
        else:
            i += 1
    strides = [starts[k + 1] - starts[k] for k in range(len(starts) - 1)]
    stride_hist = collections.Counter(strides).most_common(5)
    return {
        "agreement": round(agree / max(1, n), 6),
        "disagreeing_runs": len(runs),
        "mean_run_len": round(sum(runs) / max(1, len(runs)), 2),
        "max_run_len": max(runs) if runs else 0,
        "stride_mode": stride_hist,
    }


def shifted_agreement(a, b, shift):
    if shift >= 0:
        x, y = a[: len(a) - shift], b[shift:]
    else:
        x, y = a[-shift:], b[: len(b) + shift]
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    return round(sum(1 for i in range(n) if x[i] == y[i]) / n, 6)


results = []
# The pool-1 population the share probe called rewrite-shaped, biggest first.
cands = [(k, v) for k, v in records.items() if k[0] == 1 and len(v) >= 2]
cands.sort(key=lambda kv: -max(l for _, l in kv[1]))
for (pool, nid), locs in cands[:6]:
    (off_a, len_a), (off_b, len_b) = locs[-2], locs[-1]
    n = min(len_a, len_b)
    per_pos = []
    for frac in (0.1, 0.35, 0.6, 0.85):
        p = int(n * frac)
        a = read_at(off_a + p, SAMPLE)
        b = read_at(off_b + p, SAMPLE)
        st = structure(a, b)
        st["at_fraction"] = frac
        per_pos.append(st)
    p = int(n * 0.5)
    a = read_at(off_a + p, SAMPLE)
    b = read_at(off_b + p, SAMPLE)
    shifts = {str(s): shifted_agreement(a, b, s)
              for s in (-32, -16, -8, -4, 0, 4, 8, 16, 32)}
    results.append({
        "pool": pool, "id": nid, "records": len(locs),
        "len_a": len_a, "len_b": len_b, "grew": len_b - len_a,
        "windows": per_pos,
        "shift_agreement": shifts,
    })

# One append-shaped control, to prove the instrument separates the populations.
ctrl = [(k, v) for k, v in records.items() if k[0] == 5 and len(v) >= 2]
ctrl.sort(key=lambda kv: -max(l for _, l in kv[1]))
control = []
for (pool, nid), locs in ctrl[:2]:
    (off_a, len_a), (off_b, len_b) = locs[-2], locs[-1]
    n = min(len_a, len_b)
    p = int(n * 0.5)
    st = structure(read_at(off_a + p, SAMPLE), read_at(off_b + p, SAMPLE))
    control.append({"pool": pool, "id": nid, "len_b": len_b,
                    "grew": len_b - len_a, **st})

fh.close()
out["rewrite_population"] = results
out["append_control"] = control
print("PROBE_JSON " + json.dumps(out, default=str))
PY
