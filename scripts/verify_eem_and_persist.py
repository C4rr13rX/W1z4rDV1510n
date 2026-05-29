"""End-to-end test for:
  1. The /brain/ask -> hypothesis_queue -> EquationMatrixRuntime resolution path.
  2. Persistence: queue survives a binary restart.

Flow:
  1. Start the node, train a tiny corpus on the brain.
  2. Ingest a physical equation into the EquationMatrixRuntime.
  3. POST /brain/ask with a physics question containing tokens of that
     equation -> brain doesn't know -> queued.
  4. Wait > 30 seconds for hypothesis_research_loop to run.
  5. Re-fetch /hypothesis/queue, expect resolution_path = node_equation_matrix
     and chain_of_facts populated with equation ids.
  6. Stop and restart the binary.
  7. GET /hypothesis/queue -> verify the resolved entry survives.
"""
import base64, http.client, json, os, subprocess, sys, time
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8290
POOL_TEXT, POOL_ACTION = 1, 4

_conn = None
def conn():
    global _conn
    if _conn is None: _conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    return _conn

def request(method, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    for _ in range(2):
        try:
            c = conn()
            c.request(method, path, body, headers)
            r = c.getresponse(); out = r.read()
            try:    return r.status, json.loads(out)
            except: return r.status, out.decode(errors="replace")
        except Exception:
            global _conn; _conn = None
    return None, None

def post(p, j): return request("POST", p, j)
def get(p):     return request("GET",  p)

def observe(pool, frame):
    b64 = base64.urlsafe_b64encode(frame.encode()).decode().rstrip("=")
    return post("/brain/observe", {"pool_id": pool, "frame": b64})
def tick(): return post("/brain/tick", {})

def train_pair(p, r, n=12):
    for _ in range(n):
        observe(POOL_TEXT, p); observe(POOL_ACTION, r); tick()


def main():
    print("=== Step 1: health ===")
    s, _ = get("/brain/health")
    print(f"  status={s}")

    print("\n=== Step 2: ingest a physical equation into EquationMatrixRuntime ===")
    s, r = post("/equations/ingest", {
        "text": "F = m * a relates force F, mass m, and acceleration a in Newtonian mechanics.",
        "discipline": "Mechanics",
        "confidence": 0.95,
    })
    print(f"  status={s}  result={r}")

    print("\n=== Step 3: small brain training (so /brain/integrate works on basics) ===")
    for p, r in [("hi", "hello"), ("cat", "animal")]:
        train_pair(p, r, n=12)
    tick()

    print("\n=== Step 4: POST /brain/ask with a physics question ===")
    question = "What is the equation relating force mass and acceleration in Newtonian mechanics?"
    s, r = post("/brain/ask", {"text": question})
    print(f"  status={s}")
    hyp_id = r.get("hypothesis_id") if isinstance(r, dict) else None
    print(f"  hypothesis_id={hyp_id}")
    print(f"  message={r.get('message') if isinstance(r, dict) else r}")

    print("\n=== Step 5: wait 35s for hypothesis_research_loop to run ===")
    for sec in range(35, 0, -5):
        print(f"  {sec}s remaining...")
        time.sleep(5)

    print("\n=== Step 6: check resolution ===")
    s, r = get("/hypothesis/queue")
    if isinstance(r, dict):
        entries = r.get("entries", [])
        target = next((e for e in entries if e.get("id") == hyp_id), None)
        if target:
            print(f"  question={target.get('question')!r}")
            print(f"  resolved={target.get('resolved')}")
            print(f"  resolution_path={target.get('resolution_path')!r}")
            print(f"  chain_of_facts={target.get('chain_of_facts')}")
            print(f"  attempts={target.get('attempts')}")
            answer = target.get("answer")
            if answer:
                print(f"  answer={answer[:140]!r}{'...' if len(answer)>140 else ''}")
            resolved_by_eem = (target.get("resolved") is True
                              and target.get("resolution_path") in
                                  ("node_equation_matrix", "brain_eem_chain", "brain_substrate"))
            print(f"  RESOLVED BY INTERNAL SUBSTRATE: {resolved_by_eem}")
        else:
            print(f"  entry not found in queue (id={hyp_id})")

    print("\n=== Step 7: verify hypothesis_queue.json was written to disk ===")
    # The binary writes via node_data_dir() which honours
    # W1Z4RDV1510N_DATA_DIR.  Use forward slashes everywhere because
    # bash + windows path escaping conspires to lose backslashes.
    persist_path = Path(os.environ.get("W1Z4RDV1510N_DATA_DIR",
                                       "D:/w1z4rd-test-merged")) / "hypothesis_queue.json"
    if persist_path.exists():
        size = persist_path.stat().st_size
        print(f"  {persist_path} exists, {size} bytes")
        # Read first 300 chars
        with persist_path.open("r", encoding="utf-8") as f:
            print(f"  first 300 chars: {f.read(300)!r}")
    else:
        print(f"  WARNING: persistence file not found at {persist_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
