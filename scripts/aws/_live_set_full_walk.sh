python3 - <<'PY'
"""How many bytes in the 472 GB container are LIVE?

Two numbers in the operating record disagree by 38x and the whole capacity
decision hangs on which is right:

  * `.wbrain compaction is net-negative` rests on 363.34 GB of live bodies --
    5,086,800 neurons at a mean body of 71 KB. Compaction would then rewrite
    363 GB of unshareable blocks to reclaim less, and is correctly refused.

  * The brain's own `/stats` reports 454,873,768 total terminals. A terminal is
    a confirmed 21 bytes (bincode fixint over target u32+u32, weight f32,
    consolidation u8, last_fired_tick u64 -- verified by a stride cycle of
    {4,12,5} summing to 21 and a target-field agreement of 1.0), so every
    terminal in the fabric is 9.55 GB. Reaching 363 GB would need ~17.3 billion
    terminals, 38x what the brain reports. The measured RSS of 11.57 GB fits
    the small number, not the large one.

The append-only container makes this decidable without trusting either: the
newest record for each (pool, id) is what the slot table points at, so summing
the LAST length per key is the live set and everything else is garbage a
compaction returns. Walk headers only -- seek over every body -- and report the
coverage actually achieved rather than extrapolating from a sample, because a
sample of an append-only file is biased toward whatever was being rewritten.

Read-only.
"""
import json
import os
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
BRAIN = R / "brain" / "brain.wbrain"
MARK = b"W1ZNEUR1"
HDR = 24
CHUNK = 4 * 1024 * 1024
BUDGET = float(os.environ.get("WALK_BUDGET", "1150"))

size = BRAIN.stat().st_size
out = {"now": time.time(), "container_bytes": size,
       "container_gb": round(size / 1e9, 2)}
fh = open(BRAIN, "rb", buffering=1024 * 1024)


def marker_at(off):
    if off + 8 > size:
        return False
    fh.seek(off)
    return fh.read(8) == MARK


def resync(from_off, limit=8 * 1024**3):
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


start = resync(0)
out["first_record"] = start
if start is None:
    print("PROBE_JSON " + json.dumps({**out, "error": "no neuron record"}))
    raise SystemExit(0)

# key = (pool << 32) | id  -> length of the most recent record seen for it.
last_len = {}
neuron_bytes = 0          # every neuron-record body byte, live or superseded
records = 0
resyncs = 0
aux_bytes = 0
off = start
deadline = time.time() + BUDGET
t0 = time.time()
while off < size:
    if (records & 0x3FFF) == 0 and time.time() > deadline:
        out["truncated"] = True
        break
    fh.seek(off)
    head = fh.read(HDR)
    blen = int.from_bytes(head[16:24], "little") if len(head) == HDR else 0
    if len(head) < HDR or head[:8] != MARK or blen <= 0 or off + HDR + blen > size:
        nxt = resync(off + 8)
        if nxt is None:
            break
        aux_bytes += nxt - off
        resyncs += 1
        off = nxt
        continue
    pool = int.from_bytes(head[8:12], "little")
    nid = int.from_bytes(head[12:16], "little")
    last_len[(pool << 32) | nid] = blen
    neuron_bytes += blen + HDR
    records += 1
    off += HDR + blen

live = 0
for v in last_len.values():
    live += v
covered = off - start
fh.close()

out.update({
    "walk_seconds": round(time.time() - t0, 1),
    "bytes_covered": covered,
    "coverage_fraction": round(covered / max(1, size), 4),
    "records": records,
    "resyncs": resyncs,
    "auxiliary_bytes_skipped_gb": round(aux_bytes / 1e9, 2),
    "distinct_neurons": len(last_len),
    "neuron_record_bytes_gb": round(neuron_bytes / 1e9, 2),
    "live_bytes": live,
    "live_gb": round(live / 1e9, 2),
    "garbage_gb": round((neuron_bytes - live) / 1e9, 2),
    "live_fraction_of_neuron_records": round(live / max(1, neuron_bytes), 6),
    "mean_live_body_bytes": int(live / max(1, len(last_len))),
})
# Cross-check against the fabric's own counters: a live set consistent with
# /stats is evidence the walk found the real thing rather than a parse artifact.
out["expected_terminal_bytes_gb"] = round(454873768 * 21 / 1e9, 2)
print("PROBE_JSON " + json.dumps(out, default=str))
PY
