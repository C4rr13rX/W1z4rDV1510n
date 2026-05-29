"""Measure rows/sec on real corpora through the current substrate.
Drives /observe/observe/tick per row directly, no extra book-keeping.
Reports per-corpus speed and projected days for the full 1.14M-row set.
"""

import base64, http.client, json, sys, time
from pathlib import Path

HOST, PORT = "127.0.0.1", 8195
POOL_TEXT, POOL_ACTION = 1, 4

_conn = None
def conn():
    global _conn
    if _conn is None:
        _conn = http.client.HTTPConnection(HOST, PORT, timeout=120)
    return _conn

def post(path, payload):
    try:
        c = conn()
        c.request("POST", path, json.dumps(payload).encode(), {"Content-Type":"application/json"})
        r = c.getresponse()
        out = r.read()
        return r.status
    except Exception:
        global _conn
        _conn = None
        return None

def observe(pool, frame_bytes):
    b64 = base64.urlsafe_b64encode(frame_bytes).decode().rstrip("=")
    return post("/observe", {"pool_id": pool, "frame": b64})

def tick(): return post("/tick", {})

def get(path):
    c = conn()
    c.request("GET", path)
    r = c.getresponse()
    out = r.read()
    try: return json.loads(out)
    except: return out.decode()


CORPORA = [
    # (label, path, expected_rows)
    ("greetings_001",          "data/training/greetings_001.jsonl",          8),
    ("agent_planning_001",     "data/training/agent_planning_001.jsonl",     140),
    ("conversation_basics_001","data/training/conversation_basics_001.jsonl",2120),
]


def drive(path, max_rows):
    p = Path(path)
    if not p.exists():
        return None
    rows_done = 0
    t0 = time.time()
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rows_done >= max_rows: break
            line = line.strip()
            if not line: continue
            try: row = json.loads(line)
            except: continue
            # Schema: { prompt: str, response: str }
            prompt = row.get("prompt") or row.get("input") or row.get("question") or ""
            response = row.get("response") or row.get("output") or row.get("answer") or ""
            if not prompt or not response: continue
            observe(POOL_TEXT, prompt.encode("utf-8")[:512])
            observe(POOL_ACTION, response.encode("utf-8")[:512])
            tick()
            rows_done += 1
    return rows_done, time.time() - t0


def drive_chunked(path, max_rows, chunk):
    """Like drive(), but reports rate per chunk so we can see whether
    speed decays as the brain grows."""
    p = Path(path)
    if not p.exists(): return None
    chunk_times = []
    chunk_start = time.time()
    rows_done = 0
    rows_in_chunk = 0
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if rows_done >= max_rows: break
            line = line.strip()
            if not line: continue
            try: row = json.loads(line)
            except: continue
            prompt = row.get("prompt") or row.get("input") or row.get("question") or ""
            response = row.get("response") or row.get("output") or row.get("answer") or ""
            if not prompt or not response: continue
            observe(POOL_TEXT, prompt.encode("utf-8")[:512])
            observe(POOL_ACTION, response.encode("utf-8")[:512])
            tick()
            rows_done += 1
            rows_in_chunk += 1
            if rows_in_chunk >= chunk:
                dt = time.time() - chunk_start
                chunk_times.append((rows_done, dt, rows_in_chunk / dt))
                rows_in_chunk = 0
                chunk_start = time.time()
    if rows_in_chunk > 0:
        dt = time.time() - chunk_start
        chunk_times.append((rows_done, dt, rows_in_chunk / dt))
    return chunk_times


def main():
    print("Warming up with 50 rows...")
    drive("data/training/conversation_basics_001.jsonl", 50)

    print(f"\n{'corpus':>30} {'rows':>6} {'sec':>8} {'rows/sec':>10} {'sec/row':>10}")
    print("-" * 70)

    rates = []
    for label, path, expected in CORPORA:
        result = drive(path, max_rows=min(expected, 500))
        if result is None:
            print(f"  {label:>28}  (missing)")
            continue
        rows, sec = result
        rate = rows / sec if sec > 0 else 0
        rates.append(rate)
        print(f"{label:>30} {rows:>6} {sec:>8.2f} {rate:>10.2f} {1000*sec/rows:>9.1f}ms")

    if rates:
        avg = sum(rates) / len(rates)
        print(f"\n  mean rate (small-brain phase): {avg:.2f} rows/sec\n")

    # Scale test: drive 5000 rows in chunks of 500 to see where ticks
    # start slowing as the brain grows.
    print("=== Scale test: 5000 rows of k12_subjects (5 chunks of 1000) ===")
    chunks = drive_chunked("data/training/k12_subjects_001.jsonl",
                           max_rows=5000, chunk=1000)
    if chunks:
        print(f"\n{'rows_total':>12} {'chunk_sec':>10} {'rows/sec':>10}")
        print("-" * 40)
        for total, sec, rate in chunks:
            print(f"{total:>12,} {sec:>10.2f} {rate:>10.2f}")

    # Brain-side metrics.
    stats = get("/stats")
    if isinstance(stats, dict):
        print(f"\nBrain after scale test:")
        print(f"  tick={stats.get('tick'):,}  pools={stats.get('pool_count')}  "
              f"neurons={stats.get('total_neurons'):,}  terminals={stats.get('total_terminals'):,}")

    full = 1_144_148
    small_med = 32_496
    if chunks:
        last_rate = chunks[-1][2]
        first_rate = chunks[0][2]
        # Project using LAST chunk (large-brain regime).
        print(f"\nLarge-brain regime rate (last chunk): {last_rate:.2f} rows/sec")
        print(f"Small-brain regime rate (first chunk): {first_rate:.2f} rows/sec\n")
        print(f"Projection using large-brain rate ({last_rate:.2f} rows/sec):")
        print(f"  small/medium 19 corpora  ({small_med:>10,} rows): {small_med/last_rate/3600:>8.2f} hours")
        print(f"  full 1.14M-row set       ({full:>10,} rows): {full/last_rate/86400:>8.2f} days")

    # Brain-side metrics.
    stats = get("/stats")
    if isinstance(stats, dict):
        print(f"\nBrain after run:")
        print(f"  tick={stats.get('tick')}  pools={stats.get('pool_count')}  "
              f"neurons={stats.get('total_neurons')}  terminals={stats.get('total_terminals')}")
    tp = get("/tick_profile")
    if isinstance(tp, dict):
        ticks = tp.get('ticks', 1)
        if ticks:
            print(f"  per-tick avg: atom={tp.get('cross_pool_atom_wiring_ns',0)//max(ticks,1)//1000:>6}us  "
                  f"concept={tp.get('cross_pool_concept_wiring_ns',0)//max(ticks,1)//1000:>6}us  "
                  f"temporal={tp.get('within_pool_temporal_ns',0)//max(ticks,1)//1000:>6}us  "
                  f"hk={tp.get('housekeeping_ns',0)//max(ticks,1)//1000:>6}us  "
                  f"total={tp.get('total_ns',0)//max(ticks,1)//1000:>6}us")

    return 0


if __name__ == "__main__":
    sys.exit(main())
