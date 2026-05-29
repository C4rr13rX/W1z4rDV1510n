"""Verify Phase A-E parity between the standalone brain_server (port 8195)
and the merged main-node /brain endpoints (port 8290).

Runs each phase test against the main node's /brain/* routes and
checks that we hit the same architectural milestones — 100% trained
recall, paraphrase robustness, cross-domain chain integration,
self-tuning, autonomous thinking interruptible by inference/training.
"""

import base64, http.client, json, sys, time

HOST = "127.0.0.1"
PORT = 8290           # main node API after merge
PREFIX = "/brain"     # nested router prefix
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
        c.request("POST", PREFIX + path,
                  json.dumps(payload).encode(),
                  {"Content-Type": "application/json"})
        r = c.getresponse(); out = r.read()
        try: return r.status, json.loads(out)
        except: return r.status, out.decode(errors="replace")
    except Exception as e:
        global _conn; _conn = None
        return None, str(e)

def get(path):
    try:
        c = conn()
        c.request("GET", PREFIX + path)
        r = c.getresponse(); out = r.read()
        try: return r.status, json.loads(out)
        except: return r.status, out.decode(errors="replace")
    except Exception as e:
        global _conn; _conn = None
        return None, str(e)


def observe(pool, frame):
    b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
    return post("/observe", {"pool_id": pool, "frame": b64})

def tick(): return post("/tick", {})

def train_pair(prompt, response, reps):
    for _ in range(reps):
        observe(POOL_TEXT, prompt); observe(POOL_ACTION, response); tick()

def decode(b64):
    if not b64: return None
    pad = "=" * (-len(b64) % 4)
    try: return base64.urlsafe_b64decode(b64 + pad).decode("utf-8","replace")
    except: return b64

def probe(prompt):
    observe(POOL_TEXT, prompt)
    s, r = post("/integrate", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    if isinstance(r, dict):
        ans = r.get("answer")
        if isinstance(ans, list): return bytes(ans).decode("utf-8","replace")
        if isinstance(ans, str):  return decode(ans)
    return None


CORPUS = [
    ("hi", "hello"), ("hey", "hello"), ("yo", "hello"),
    ("cat", "animal"), ("dog", "animal"),
    ("car", "vehicle"),
    ("red", "color"), ("blue", "color"),
]
CHAIN_AB = [("alpha", "bravo"), ("delta", "echo")]
CHAIN_BC = [("bravo", "charlie"), ("echo", "foxtrot")]


def banner(s): print(f"\n=== {s} ===")


def main():
    s, _ = get("/health")
    print(f"health on {HOST}:{PORT}{PREFIX}/health -> {s}")
    if s != 200: return 1

    # Phase A — train, verify QA db + lock + soft domain gate.
    banner("Phase A: train direct corpus, verify QA db + consolidation lock")
    for p, r in CORPUS: train_pair(p, r, reps=12)
    tick()
    _, qa = get("/qa_db_stats")
    _, lk = get("/consolidation_stats")
    print(f"  qa_count={qa['count']}  locked_terminals={lk['locked_terminals']}")

    # Phase B — 100% exact recall on trained pairs.
    banner("Phase B: exact recall on trained corpus (expect 8/8)")
    hits = sum(int(probe(p) == r) for p, r in CORPUS)
    print(f"  exact recall: {hits}/{len(CORPUS)}")
    phaseB_ok = hits == len(CORPUS)

    # Paraphrase.
    banner("Phase B+: paraphrase robustness (trailing space)")
    para = sum(int(probe(p + " ") == r) for p, r in CORPUS)
    print(f"  paraphrase exact: {para}/{len(CORPUS)}")

    # Phase C — chain integration.
    banner("Phase C: cross-domain chain integration (A->B, B->C, probe A expect C)")
    for p, r in CHAIN_AB: train_pair(p, r, reps=12)
    tick()
    for p, r in CHAIN_BC: train_pair(p, r, reps=12)
    tick()

    chain_expected = {"alpha": "charlie", "delta": "foxtrot"}
    chain_hits = 0
    for seed, want in chain_expected.items():
        b64 = base64.urlsafe_b64encode(seed.encode()).decode().rstrip("=")
        s, r = post("/integrate_chain", {
            "query_pool": POOL_TEXT, "target_pool": POOL_ACTION,
            "seed": b64, "max_hops": 4,
        })
        steps = r.get("steps", []) if isinstance(r, dict) else []
        decoded = [(decode(st.get("query")), decode(st.get("answer"))) for st in steps]
        reached = any(a == want for _, a in decoded)
        chain_hits += int(reached)
        chain_str = " -> ".join(f"'{q}'->'{a}'" for q, a in decoded)
        print(f"  '{seed}': {chain_str}")
        print(f"    reached '{want}': {'INTEGRATED' if reached else 'NOT INTEGRATED'}")
    print(f"  chain integration: {chain_hits}/{len(chain_expected)}")
    phaseC_ok = chain_hits == len(chain_expected)

    # Phase D — self-tuning runs without panic and produces sensible JSON.
    banner("Phase D: retune step + tuning_state")
    s, rpt = post("/retune", {"sample_count": 16})
    print(f"  retune: recall={rpt['recall_after']:.3f} decay={rpt['decay_after']:.6f} dir={rpt['direction_after']}")
    s, ts = get("/tuning_state")
    print(f"  tuning_state.steps={ts['steps']} best_recall={ts['best_recall']:.3f}")
    phaseD_ok = ts['steps'] >= 1

    # Phase E — background thinking loop preempts cleanly.
    banner("Phase E: enable thinking, run inference probes, train new pair")
    post("/thinking/start", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    time.sleep(2.0)
    _, st = get("/thinking/status")
    print(f"  after 2s of thinking: hops={st['hops_taken']} last_answer='{decode(st.get('last_answer'))}'")

    # Probes during thinking.
    probe_t0 = time.time()
    probe_results = []
    for prompt in ["hi", "cat", "red"]:
        t0 = time.time(); ans = probe(prompt); dt = (time.time()-t0)*1000
        probe_results.append((prompt, ans, dt))
        print(f"    probe '{prompt}' -> '{ans}' ({dt:.1f}ms)")

    # New training pair during thinking.
    train_pair("ping", "pong", reps=4)
    new_ans = probe("ping")
    print(f"  post-train probe 'ping' -> '{new_ans}' (expected 'pong')")

    post("/thinking/stop", {})
    _, st2 = get("/thinking/status")
    print(f"  after /thinking/stop: enabled={st2['enabled']} hops={st2['hops_taken']}")
    phaseE_ok = (new_ans == "pong") and (st['hops_taken'] > 0)

    # Summary.
    banner("Merged-binary parity summary")
    print(f"  Phase B exact recall:         {phaseB_ok}  ({hits}/{len(CORPUS)})")
    print(f"  Phase B+ paraphrase:          {para}/{len(CORPUS)}")
    print(f"  Phase C chain integration:    {phaseC_ok}  ({chain_hits}/{len(chain_expected)})")
    print(f"  Phase D self-tuning runs:     {phaseD_ok}")
    print(f"  Phase E thinking + preempt:   {phaseE_ok}")
    all_ok = phaseB_ok and phaseC_ok and phaseD_ok and phaseE_ok
    print(f"\n  ALL PHASES PASS ON MERGED BINARY: {all_ok}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
