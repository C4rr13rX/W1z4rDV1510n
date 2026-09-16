set -e
pkill -f /tmp/live_walk.py 2>/dev/null || true
cat > /tmp/live_walk.py <<'PY'
"""Exact live/garbage split of the .wbrain container.

The capacity halt's remedies rest on a live-set figure of 363.34 GB, obtained
by `compaction::estimate` as mean_sampled_body x live_neurons. That estimator
is unbiased only for a light-tailed distribution, and this one has a dozen
86 MB hub atoms against a median near 1.5 KB. The brain's own /stats reports
454,873,768 terminals at a confirmed 21 bytes each -- 9.55 GB of terminal
payload in the entire fabric -- so 363 GB of live bodies would require ~17.3
billion terminals. Those two numbers cannot both be right, and which one is
decides whether compaction reclaims the volume or wastes it.

An earlier pass resynchronised by scanning for the neuron marker and validating
that a record's declared length lands on another marker. That rejects the last
neuron record before every auxiliary record, so it dead-ended at 3.02 %
coverage and its mean body (11.6 KB) is a contiguous-region sample, biased
toward whatever was being rewritten.

The container has exactly three record shapes and all are self-describing:

    W1ZNEUR1 | pool:u32 | id:u32   | len:u64 | body
    W1ZAUX01 | pool:u32 | kind:u32 | len:u64 | body
    W1ZMANI1 | len:u64  | body

so the walk needs no scanning at all -- every record says where the next one
starts. Read SEQUENTIALLY in large blocks rather than seeking per record: the
bodies average ~14 KB, so 32 M seeks would be IOPS-bound for hours while a
straight read is throughput-bound.

The newest record for each (pool, id) is what the slot table addresses, so
summing the LAST length per key is the live set and the remainder is garbage a
compaction returns.
"""
import json
import os
import time

BRAIN = "/srv/wizard/runtime/programming-integrated-20260713/brain/brain.wbrain"
OUT = "/tmp/live_walk.json"
CHUNK = 64 * 1024 * 1024
NEUR = b"W1ZNEUR1"
AUX = b"W1ZAUX01"
MANI = b"W1ZMANI1"

size = os.path.getsize(BRAIN)
last_len = {}
neuron_bytes = 0
aux_bytes = 0
mani_bytes = 0
records = 0
aux_records = 0
t0 = time.time()
pos = 4096                      # HEADER_BYTES
state = {"done": False}


def publish(extra=None):
    live = 0
    for v in last_len.values():
        live += v
    doc = {
        "container_bytes": size,
        "container_gb": round(size / 1e9, 2),
        "bytes_walked": pos,
        "coverage_fraction": round(pos / size, 6),
        "elapsed_seconds": round(time.time() - t0, 1),
        "neuron_records": records,
        "auxiliary_records": aux_records,
        "distinct_neurons": len(last_len),
        "neuron_record_bytes_gb": round(neuron_bytes / 1e9, 2),
        "auxiliary_bytes_gb": round(aux_bytes / 1e9, 2),
        "manifest_bytes_gb": round(mani_bytes / 1e9, 3),
        "live_bytes": live,
        "live_gb": round(live / 1e9, 2),
        "garbage_gb": round((neuron_bytes - live) / 1e9, 2),
        "mean_live_body_bytes": int(live / len(last_len)) if last_len else 0,
        "expected_terminal_bytes_gb": round(454873768 * 21 / 1e9, 2),
    }
    if extra:
        doc.update(extra)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh)
    os.replace(tmp, OUT)


fh = open(BRAIN, "rb", buffering=0)
fh.seek(pos)
buf = b""
cur = 0                         # cursor INTO buf; never re-slice the buffer
last_pub = time.time()
bad = 0
while True:
    # Re-slicing the buffer per record copies the whole remaining chunk each
    # time: at 64 MB chunks and ~14 KB records that is ~288 GB of memcpy per
    # chunk, which measured 0.76 MB/s and would have taken 172 h. Carry a
    # cursor and only compact the buffer when the header runs off the end.
    if len(buf) - cur < 24:
        buf = buf[cur:]
        cur = 0
        more = fh.read(CHUNK)
        if not more:
            break
        buf += more
        if len(buf) < 24:
            break
    marker = buf[cur:cur + 8]
    if marker == NEUR or marker == AUX:
        blen = int.from_bytes(buf[cur + 16:cur + 24], "little")
        head = 24
    elif marker == MANI:
        blen = int.from_bytes(buf[cur + 8:cur + 16], "little")
        head = 16
    else:
        bad += 1
        break
    total = head + blen
    if marker == NEUR:
        key = (int.from_bytes(buf[cur + 8:cur + 12], "little") << 32) | \
              int.from_bytes(buf[cur + 12:cur + 16], "little")
        last_len[key] = blen
        neuron_bytes += total
        records += 1
    elif marker == AUX:
        aux_bytes += total
        aux_records += 1
    else:
        mani_bytes += total
    if cur + total <= len(buf):
        cur += total
    else:
        remaining = total - (len(buf) - cur)
        buf = b""
        cur = 0
        fh.seek(remaining, os.SEEK_CUR)
    pos += total
    if time.time() - last_pub > 20:
        publish()
        last_pub = time.time()

fh.close()
publish({"done": True, "unparsed_marker": bad,
         "trailing_bytes": size - pos})
PY
nohup python3 /tmp/live_walk.py > /tmp/live_walk.log 2>&1 &
echo "launched pid $!"
sleep 25
cat /tmp/live_walk.json 2>/dev/null || echo "no json yet"
echo
tail -3 /tmp/live_walk.log 2>/dev/null || true
