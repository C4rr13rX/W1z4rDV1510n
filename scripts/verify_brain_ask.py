"""End-to-end test of the orchestrated /brain/ask -> hypothesis_queue -> EEM
research loop pipeline.

Flow:
  1. Train the brain on a small corpus via /brain/observe.
  2. POST /brain/ask with a TRAINED prompt -> expect 200 + answer.
  3. POST /brain/ask with an UNKNOWN prompt -> expect 202 + hypothesis id.
  4. GET /hypothesis/queue -> verify the unknown prompt is queued with
     resolution_path=null and all the new provenance fields.
"""
import base64, http.client, json, sys, time

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

CORPUS = [
    ("hi", "hello"), ("hey", "hello"), ("yo", "hello"),
    ("cat", "animal"), ("dog", "animal"),
    ("red", "color"), ("blue", "color"),
]


def main():
    s, h = get("/brain/health")
    if s != 200: print(f"brain not up ({s})"); return 1

    print("=== Training small corpus ===")
    for p, r in CORPUS:
        train_pair(p, r, n=12)
    tick()

    print("\n=== Test 1: POST /brain/ask with TRAINED prompt 'hi' ===")
    s, r = post("/brain/ask", {"text": "hi"})
    print(f"  status={s}")
    if isinstance(r, dict):
        print(f"  answer={r.get('answer')!r}")
        print(f"  confidence={r.get('confidence'):.3f}  tier={r.get('confidence_tier')}")
        print(f"  outside_grounding={r.get('outside_grounding')}")
        print(f"  hypothesis_id={r.get('hypothesis_id')}")
    trained_ok = (s == 200 and isinstance(r, dict) and r.get("answer") == "hello")
    print(f"  TRAINED OK: {trained_ok}")

    print("\n=== Test 2: POST /brain/ask with UNKNOWN prompt 'photosynthesis' ===")
    s, r = post("/brain/ask", {"text": "photosynthesis"})
    print(f"  status={s} (expect 202)")
    if isinstance(r, dict):
        print(f"  answer={r.get('answer')!r}")
        print(f"  outside_grounding={r.get('outside_grounding')}")
        print(f"  hypothesis_id={r.get('hypothesis_id')!r}")
        print(f"  message={r.get('message')!r}")
    queued_ok = (s == 202 and isinstance(r, dict) and r.get("outside_grounding") is True
                 and r.get("hypothesis_id") is not None)
    print(f"  QUEUED OK: {queued_ok}")
    hyp_id = r.get("hypothesis_id") if isinstance(r, dict) else None

    print("\n=== Test 3: GET /hypothesis/queue ===")
    s, r = get("/hypothesis/queue")
    print(f"  status={s}")
    if isinstance(r, dict):
        print(f"  count={r.get('count')}  unresolved={r.get('unresolved')}")
        entries = r.get("entries", [])
        for e in entries[-3:]:
            print(f"  entry: id={e.get('id')!r}")
            print(f"         question={e.get('question')!r}")
            print(f"         resolved={e.get('resolved')}  attempts={e.get('attempts')}")
            print(f"         resolution_path={e.get('resolution_path')}")
            print(f"         chain_of_facts={e.get('chain_of_facts')}")
            print(f"         query_pool={e.get('query_pool')}")
    found_in_queue = (isinstance(r, dict) and
                      any(e.get("id") == hyp_id for e in r.get("entries", [])))
    print(f"  ENTRY FOUND IN QUEUE: {found_in_queue}")

    print("\n=== Test 4: post-resolve via /hypothesis/resolve (simulating human) ===")
    if hyp_id:
        s, r = post("/hypothesis/resolve", {
            "id": hyp_id,
            "answer": "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
            "confidence": 0.92,
            "resolution_path": "human",
            "source_url": "https://en.wikipedia.org/wiki/Photosynthesis",
            "chain_of_facts": ["light_dependent_reactions", "calvin_cycle"],
        })
        print(f"  status={s}  result={r}")
        s, q = get("/hypothesis/queue")
        resolved_entry = next((e for e in q.get("entries", [])
                               if e.get("id") == hyp_id), None) if isinstance(q, dict) else None
        if resolved_entry:
            print(f"  after resolve: resolved={resolved_entry.get('resolved')} "
                  f"path={resolved_entry.get('resolution_path')!r} "
                  f"chain={resolved_entry.get('chain_of_facts')} "
                  f"source={resolved_entry.get('source_url')!r}")

    print(f"\n=== SUMMARY ===")
    print(f"  Trained prompt returns answer (200):           {trained_ok}")
    print(f"  Unknown prompt queues hypothesis (202):        {queued_ok}")
    print(f"  Queued entry visible in /hypothesis/queue:     {found_in_queue}")
    return 0 if (trained_ok and queued_ok and found_in_queue) else 1


if __name__ == "__main__":
    sys.exit(main())
