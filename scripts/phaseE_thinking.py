"""Phase E — autonomous thinking loop validation.

Train the brain, enable /thinking/start, watch the brain chain through
its own knowledge graph in the background, then interrupt with an
inference probe AND a training observation to verify both preempt
cleanly.
"""

import base64, http.client, json, sys, time

HOST, PORT = "127.0.0.1", 8195
POOL_TEXT, POOL_ACTION = 1, 4


def conn(): return http.client.HTTPConnection(HOST, PORT, timeout=30)


def post(p, j):
    c = conn(); c.request("POST", p, json.dumps(j).encode(), {"Content-Type":"application/json"})
    r = c.getresponse(); out = r.read(); c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def get(p):
    c = conn(); c.request("GET", p)
    r = c.getresponse(); out = r.read(); c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def observe(pool, frame):
    b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
    return post("/observe", {"pool_id": pool, "frame": b64})


def tick(): post("/tick", {})


def train_pair(p, r, n=12):
    for _ in range(n):
        observe(POOL_TEXT, p); observe(POOL_ACTION, r); tick()


def decode(b64):
    if not b64: return None
    pad = "=" * (-len(b64) % 4)
    try: return base64.urlsafe_b64decode(b64 + pad).decode("utf-8","replace")
    except: return b64


def probe(prompt):
    """Inference probe that should preempt the thinking loop."""
    observe(POOL_TEXT, prompt)
    s, r = post("/integrate", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    if isinstance(r, dict):
        ans = r.get("answer")
        if isinstance(ans, list): return bytes(ans).decode("utf-8","replace")
        if isinstance(ans, str):  return decode(ans)
        return None
    return None


CORPUS = [
    ("hi", "hello"), ("hey", "hello"), ("yo", "hello"),
    ("cat", "animal"), ("dog", "animal"),
    ("car", "vehicle"),
    ("red", "color"), ("blue", "color"),
]


def main():
    s, _ = get("/health")
    if s != 200: print("brain not up"); return 1

    print("=== Phase E: continuous thinking + interruptibility ===\n")
    print("Training the canonical corpus...")
    for p, r in CORPUS:
        train_pair(p, r, n=12)
    tick()

    _, qa = get("/qa_db_stats")
    print(f"  qa_db={qa['count']}\n")

    # Snapshot before turning on thinking.
    _, st0 = get("/thinking/status")
    print(f"Before /thinking/start: enabled={st0['enabled']} hops={st0['hops_taken']}\n")

    print("Starting thinking loop, idling 2 seconds...")
    post("/thinking/start", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    time.sleep(2.0)

    _, st1 = get("/thinking/status")
    print(f"After 2s: enabled={st1['enabled']} hops={st1['hops_taken']}")
    print(f"  last_seed='{decode(st1['last_seed'])}'  last_answer='{decode(st1['last_answer'])}'")

    print("\nWhile thinking, fire 5 back-to-back inference probes:")
    probe_t0 = time.time()
    for prompt in ["hi", "cat", "red", "blue", "yo"]:
        t0 = time.time()
        ans = probe(prompt)
        dt = (time.time() - t0) * 1000
        print(f"  probe '{prompt}' -> '{ans}' ({dt:.1f} ms)")
    probe_dt = (time.time() - probe_t0) * 1000
    print(f"  5 probes in {probe_dt:.1f} ms total\n")

    _, st2 = get("/thinking/status")
    print(f"After probes: hops={st2['hops_taken']} (grew {st2['hops_taken'] - st1['hops_taken']} during probes)\n")

    print("While thinking, train a new pair (training preemption):")
    t0 = time.time()
    train_pair("ping", "pong", n=4)
    dt = (time.time() - t0) * 1000
    print(f"  4-rep train of ping/pong took {dt:.1f} ms\n")

    time.sleep(1.0)
    _, st3 = get("/thinking/status")
    print(f"After training + 1s idle: hops={st3['hops_taken']}")
    print(f"  last_seed='{decode(st3['last_seed'])}'  last_answer='{decode(st3['last_answer'])}'")

    # Verify new pair was learned even while thinking.
    new_ans = probe("ping")
    print(f"\nPost-train probe 'ping' -> '{new_ans}' (expected 'pong')")

    print("\nStopping thinking loop...")
    post("/thinking/stop", {})
    time.sleep(0.5)
    _, st4 = get("/thinking/status")
    print(f"After stop: enabled={st4['enabled']} hops={st4['hops_taken']}")
    time.sleep(1.0)
    _, st5 = get("/thinking/status")
    print(f"After +1s: enabled={st5['enabled']} hops={st5['hops_taken']} (should be flat)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
