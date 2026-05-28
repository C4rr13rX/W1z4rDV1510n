"""Phase A validation: feed a small QA corpus to the brain, then verify
   the QA buffer, consolidation lock, and self-test report behave as
   designed.  No external evaluation set; the brain grades itself.
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
    body = json.dumps(payload).encode()
    c.request("POST", path, body, {"Content-Type": "application/json"})
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, json.loads(out) if out else None


def get(path, parse=True):
    c = conn()
    c.request("GET", path)
    r = c.getresponse()
    out = r.read()
    c.close()
    if not parse:
        return r.status, out.decode(errors="replace")
    try:
        return r.status, json.loads(out)
    except Exception:
        return r.status, out.decode(errors="replace")


def observe(pool_id, frame_bytes):
    b64 = base64.urlsafe_b64encode(frame_bytes).decode().rstrip("=")
    return post("/observe", {"pool_id": pool_id, "frame": b64})


def tick():
    return post("/tick", {})


def train_pair(prompt, response, reps):
    # CRITICAL: observe BOTH pools in the SAME tick so cross-pool
    # firing co-occurs and binding concepts can emerge.  Observing in
    # separate ticks means pool A's firing is wiped before pool B
    # fires → moment fingerprint has only one pool → no binding.
    for _ in range(reps):
        observe(POOL_TEXT, prompt.encode())
        observe(POOL_ACTION, response.encode())
        tick()


CORPUS = [
    ("hi",    "hello"),
    ("hey",   "hi"),
    ("yo",    "hello"),
    ("hello", "hi"),
    ("ping",  "pong"),
    ("cat",   "animal"),
    ("dog",   "animal"),
    ("car",   "vehicle"),
    ("red",   "color"),
    ("blue",  "color"),
]

def main():
    # Baseline.
    s, h = get("/health")
    assert s == 200, f"health failed: {s} {h!r}"
    _, qa0 = get("/qa_db_stats")
    _, lock0 = get("/consolidation_stats")
    print(f"[baseline] qa_count={qa0['count']} locked={lock0['locked_terminals']}")

    # Phase 1: light training.  3 reps each — should NOT lock (lock=3
    # distinct ticks AND co-firing path requires the target terminal to
    # be hit 3 times on different ticks; each pair is hit reps times).
    for prompt, resp in CORPUS:
        train_pair(prompt, resp, reps=3)
    _, qa1 = get("/qa_db_stats")
    _, lock1 = get("/consolidation_stats")
    print(f"[after light train] qa_count={qa1['count']} locked={lock1['locked_terminals']}")

    # Phase 2: heavy training — push past lock threshold.
    for prompt, resp in CORPUS:
        train_pair(prompt, resp, reps=10)
    _, qa2 = get("/qa_db_stats")
    _, lock2 = get("/consolidation_stats")
    print(f"[after heavy train] qa_count={qa2['count']} locked={lock2['locked_terminals']}")

    # Phase 3: self-test.
    s, report = post("/self_test", {"sample_count": 16})
    print(f"[self_test] sampled={report['sampled']} exact={report['exact_recall']}/{report['sampled']} mean_byte_match={report['mean_byte_match']:.3f}")
    for pair in report["per_pair"][:5]:
        print(f"  '{pair['prompt']}' -> '{pair['decoded']}' (expected '{pair['expected']}', bm={pair['byte_match_ratio']:.2f}, exact={pair['exact']})")

    # Phase 4: domain training in island 2 — should NOT erase recall in
    # island 1.  Set domain=2, ingest a different corpus, then re-run self-test.
    post("/set_domain", {"domain_id": 2})
    other_corpus = [("apple", "fruit"), ("steel", "metal"), ("piano", "music")]
    for prompt, resp in other_corpus:
        train_pair(prompt, resp, reps=10)
    post("/set_domain", {"domain_id": 1})  # reset to original island

    _, qa3 = get("/qa_db_stats")
    _, lock3 = get("/consolidation_stats")
    s, report2 = post("/self_test", {"sample_count": 16})
    print(f"[after island-2 train] qa_count={qa3['count']} locked={lock3['locked_terminals']}")
    print(f"[self_test #2]  exact={report2['exact_recall']}/{report2['sampled']} mean_byte_match={report2['mean_byte_match']:.3f}")

    # Print domain histogram.
    s, hist = get("/domain_stats")
    print(f"[domain_stats] {hist}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
