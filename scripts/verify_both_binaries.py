"""Run the full Phase A-E parity check against BOTH binaries — the
main node (port 8290, /brain/* prefix) and the standalone brain_server
(port 8195, no prefix).  After the cleanup refactor, the Phase A-E
handlers are sourced from crates/node/src/brain_api.rs in both cases,
so both runs should produce identical pass results.
"""
import base64, http.client, json, sys, time

POOL_TEXT, POOL_ACTION = 1, 4

CORPUS = [
    ("hi", "hello"), ("hey", "hello"), ("yo", "hello"),
    ("cat", "animal"), ("dog", "animal"),
    ("car", "vehicle"),
    ("red", "color"), ("blue", "color"),
]
CHAIN_AB = [("alpha", "bravo"), ("delta", "echo")]
CHAIN_BC = [("bravo", "charlie"), ("echo", "foxtrot")]


def run_against(label, host, port, prefix):
    # Reconnecting client — the brain holds a single async mutex, so under
    # intensive bursts the persistent HTTP connection occasionally gets
    # reset.  Recreate on any send-side error and retry once.
    state = {"conn": None}
    def conn():
        if state["conn"] is None:
            state["conn"] = http.client.HTTPConnection(host, port, timeout=120)
        return state["conn"]
    def request(method, path, payload=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        headers = {"Content-Type": "application/json"} if payload is not None else {}
        for attempt in range(2):
            try:
                c = conn()
                c.request(method, prefix + path, body, headers)
                r = c.getresponse(); out = r.read()
                try: return r.status, json.loads(out)
                except: return r.status, out.decode(errors="replace")
            except Exception:
                state["conn"] = None
                if attempt == 1: return None, None
    def post(p, j): return request("POST", p, j)
    def get(p):     return request("GET",  p)
    def observe(pool, frame):
        b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
        return post("/observe", {"pool_id": pool, "frame": b64})
    def tick(): return post("/tick", {})
    def train_pair(p, r, n):
        for _ in range(n):
            observe(POOL_TEXT, p); observe(POOL_ACTION, r); tick()
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

    s, _ = get("/health")
    if s != 200:
        print(f"  {label}: health failed ({s})"); conn.close(); return False

    # Phase A.
    for p, r in CORPUS: train_pair(p, r, n=12)
    tick()

    # Phase B exact.
    hits = sum(int(probe(p) == r) for p, r in CORPUS)
    # Phase B+ paraphrase.
    para = sum(int(probe(p + " ") == r) for p, r in CORPUS)

    # Phase C chain.
    for p, r in CHAIN_AB: train_pair(p, r, n=12)
    tick()
    for p, r in CHAIN_BC: train_pair(p, r, n=12)
    tick()
    chain_want = {"alpha": "charlie", "delta": "foxtrot"}
    chain_hits = 0
    for seed, want in chain_want.items():
        b64 = base64.urlsafe_b64encode(seed.encode()).decode().rstrip("=")
        s, r = post("/integrate_chain", {
            "query_pool": POOL_TEXT, "target_pool": POOL_ACTION,
            "seed": b64, "max_hops": 4,
        })
        steps = r.get("steps", []) if isinstance(r, dict) else []
        decoded = [decode(st.get("answer")) for st in steps]
        chain_hits += int(want in decoded)

    # Phase D.
    s, rpt = post("/retune", {"sample_count": 16})
    phaseD = isinstance(rpt, dict) and rpt.get("recall_after") is not None

    # Phase E.
    post("/thinking/start", {"query_pool": POOL_TEXT, "target_pool": POOL_ACTION})
    time.sleep(2.0)
    _, st = get("/thinking/status")
    hops_during = st.get("hops_taken", 0)
    train_pair("ping", "pong", n=4)
    new_ans = probe("ping")
    post("/thinking/stop", {})
    phaseE = (hops_during > 0) and (new_ans == "pong")

    if state["conn"] is not None: state["conn"].close()
    print(f"  {label:>32}: exact={hits}/8  para={para}/8  chain={chain_hits}/2  D={phaseD}  E={phaseE}")
    return hits == 8 and para == 8 and chain_hits == 2 and phaseD and phaseE


def main():
    print("Phase A-E parity check — main node /brain/* AND standalone brain_server\n")

    print("Main node (port 8290, prefix /brain):")
    main_ok = run_against("main node /brain/*", "127.0.0.1", 8290, "/brain")

    print("\nStandalone brain_server (port 8195, no prefix):")
    standalone_ok = run_against("standalone :8195", "127.0.0.1", 8195, "")

    print(f"\nMAIN NODE all phases pass:    {main_ok}")
    print(f"STANDALONE all phases pass:   {standalone_ok}")
    print(f"\nBOTH BINARIES IDENTICAL:      {main_ok and standalone_ok}")
    return 0 if (main_ok and standalone_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
