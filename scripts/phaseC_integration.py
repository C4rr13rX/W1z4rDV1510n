"""Phase C — cross-domain integration test.

Three probes:
 1. Exact recall (sanity): trained prompt -> trained response, all islands.
 2. Paraphrase robustness: input with a single trailing byte, e.g. "hi " ->
    expect "hello".  Tests the binding-concept's 0.95 precision×recall
    threshold, since the firing query set now contains an extra atom.
 3. Chain integration: train A->B and B->C in different reps.  Probe A.
    Does the brain reach C via two-hop propagation?  This is the test
    of "answers that exist through the integration of training."
"""

import base64
import http.client
import json
import sys
import time

HOST = "127.0.0.1"
PORT = 8195
POOL_TEXT = 1
POOL_ACTION = 4


def conn():
    return http.client.HTTPConnection(HOST, PORT, timeout=30)


def post(path, payload):
    c = conn()
    c.request("POST", path, json.dumps(payload).encode(),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    out = r.read()
    c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def get(path):
    c = conn()
    c.request("GET", path)
    r = c.getresponse()
    out = r.read()
    c.close()
    try: return r.status, json.loads(out)
    except: return r.status, out.decode(errors="replace")


def observe(pool, frame):
    b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
    return post("/observe", {"pool_id": pool, "frame": b64})


def tick():
    return post("/tick", {})


def train_pair(prompt, response, reps):
    for _ in range(reps):
        observe(POOL_TEXT, prompt)
        observe(POOL_ACTION, response)
        tick()


def decode_answer(ans):
    """The /integrate endpoint serialises Vec<u8> as base64url (no padding).
    Decode it back to a Python string."""
    if ans is None: return None
    if isinstance(ans, list):
        try: return bytes(ans).decode("utf-8", errors="replace")
        except: return repr(ans)
    if isinstance(ans, str):
        pad = "=" * (-len(ans) % 4)
        try: return base64.urlsafe_b64decode(ans + pad).decode("utf-8", errors="replace")
        except: return ans
    return repr(ans)


def probe(prompt):
    """Fire prompt into text pool, integrate to action pool, return decoded bytes."""
    observe(POOL_TEXT, prompt)
    s, r = post("/integrate", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    if isinstance(r, dict):
        return decode_answer(r.get("answer")), r.get("confidence_tier")
    return None, str(r)


CORPUS_DIRECT = [
    ("hi",    "hello"),
    ("hey",   "hello"),
    ("yo",    "hello"),
    ("cat",   "animal"),
    ("dog",   "animal"),
    ("car",   "vehicle"),
    ("red",   "color"),
    ("blue",  "color"),
]

# Chain: train A->B in one set of reps, then B->C in another set.
# Probe A: does it integrate to C?  The brain has never seen A and C in
# the same tick.  C is reachable from A only via the B intermediary.
CHAIN_AB = [
    ("alpha", "bravo"),
    ("delta", "echo"),
]
CHAIN_BC = [
    ("bravo", "charlie"),
    ("echo",  "foxtrot"),
]


def banner(s):
    print(f"\n=== {s} ===")


def flush_tick():
    """Single advance_tick — clears fabric.current.fired so probes
    don't leave stale atoms in the next training fingerprint."""
    tick()


def main():
    s, _ = get("/health")
    if s != 200:
        print("brain not up at :8195"); return 1

    # Direct training.
    banner("Phase 1: direct training")
    for p, r in CORPUS_DIRECT:
        train_pair(p, r, reps=12)
    flush_tick()
    _, qa = get("/qa_db_stats")
    _, lk = get("/consolidation_stats")
    print(f"qa={qa['count']} locked={lk['locked_terminals']}")

    # Exact recall.
    banner("Probe 1: exact recall on trained corpus")
    direct_hits = 0
    for p, expected in CORPUS_DIRECT:
        got, tier = probe(p)
        ok = (got == expected)
        direct_hits += int(ok)
        print(f"  '{p}' -> '{got}' (expected '{expected}', tier={tier}, {'OK' if ok else 'MISS'})")
    print(f"  direct exact: {direct_hits}/{len(CORPUS_DIRECT)}")

    # Paraphrase: trailing space.
    banner("Probe 2: paraphrase robustness (trailing space)")
    para_hits = 0
    for p, expected in CORPUS_DIRECT:
        got, tier = probe(p + " ")
        ok = (got == expected)
        para_hits += int(ok)
        print(f"  '{p} ' -> '{got}' (expected '{expected}', tier={tier}, {'OK' if ok else 'MISS'})")
    print(f"  paraphrase exact: {para_hits}/{len(CORPUS_DIRECT)}")
    flush_tick()  # clear accumulated probe state before chain training

    # Chain integration setup: train A->B, then train B->C.
    banner("Phase 2: chain training (A->B, then B->C)")
    for p, r in CHAIN_AB:
        train_pair(p, r, reps=12)
    flush_tick()
    for p, r in CHAIN_BC:
        train_pair(p, r, reps=12)
    flush_tick()

    # Probe direct A->B (should hit).
    banner("Probe 3a: direct A->B recall (sanity for chain corpus)")
    ab_hits = 0
    for p, expected in CHAIN_AB:
        got, tier = probe(p)
        ok = (got == expected)
        ab_hits += int(ok)
        print(f"  '{p}' -> '{got}' (expected '{expected}', tier={tier}, {'OK' if ok else 'MISS'})")
    print(f"  A->B exact: {ab_hits}/{len(CHAIN_AB)}")

    # Probe direct B->C (sanity).
    banner("Probe 3b: direct B->C recall (sanity)")
    bc_hits = 0
    for p, expected in CHAIN_BC:
        got, tier = probe(p)
        ok = (got == expected)
        bc_hits += int(ok)
        print(f"  '{p}' -> '{got}' (expected '{expected}', tier={tier}, {'OK' if ok else 'MISS'})")
    print(f"  B->C exact: {bc_hits}/{len(CHAIN_BC)}")

    # Chain probe via single-shot integrate (no recursion).
    banner("Probe 3c: single-shot probe — A, expect direct B (never trained A->C)")
    chain_expected_c = {"alpha": "charlie", "delta": "foxtrot"}
    direct_after = 0
    for p in chain_expected_c.keys():
        got, tier = probe(p)
        direct_after += int(got in (chain_expected_c[p], {"alpha":"bravo","delta":"echo"}[p]))
        print(f"  '{p}' -> '{got}' (tier={tier})")

    # Chain probe via /integrate_chain (feeds answer back as next query).
    banner("Probe 3d: CHAIN INTEGRATION via /integrate_chain — A, expect C through B")
    chain_hits = 0
    for seed, want in chain_expected_c.items():
        b64 = base64.urlsafe_b64encode(seed.encode()).decode().rstrip("=")
        s, r = post("/integrate_chain", {
            "query_pool": POOL_TEXT, "target_pool": POOL_ACTION,
            "seed": b64, "max_hops": 4,
        })
        steps = r.get("steps", []) if isinstance(r, dict) else []
        decoded_steps = []
        for st in steps:
            q = decode_answer(st.get("query"))
            a = decode_answer(st.get("answer"))
            decoded_steps.append((q, a))
        # Pass criterion: did C appear ANYWHERE in the chain?  The
        # chain may overshoot past C into junk hops once no further
        # trained pairing exists, but the cross-domain integration
        # succeeded as soon as C was reached.
        reached_c = any(a == want for _, a in decoded_steps)
        chain_hits += int(reached_c)
        chain_str = " -> ".join(f"'{q}'->'{a}'" for q, a in decoded_steps)
        print(f"  chain '{seed}': {chain_str}")
        print(f"    reached '{want}': {'INTEGRATED' if reached_c else 'NOT INTEGRATED'}")
    print(f"  chain integration: {chain_hits}/{len(chain_expected_c)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
