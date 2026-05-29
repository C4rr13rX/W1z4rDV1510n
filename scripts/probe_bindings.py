"""Quick probe — train the chain corpus, then inspect bindings in pool 0."""
import base64, http.client, json, sys

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

# Train
for p, r in [("alpha","bravo"),("delta","echo")]: train_pair(p, r)
for p, r in [("bravo","charlie"),("echo","foxtrot")]: train_pair(p, r)

# Inspect binding pool concepts.
s, r = post("/pool/concepts", {"pool_id": 0})
print(f"binding pool concepts (count={len(r.get('concepts',[]))}):")
for c in r.get("concepts", [])[:30]:
    print(f"  id={c['neuron_id']} mc={c['member_count']} decoded={c['decoded']!r} label[:80]={c['label'][:80]} use={c.get('use_count',0)}")

# Inspect pool 4 concepts ending in 'bravo' or 'charlie' atoms.
s, r4 = post("/pool/concepts", {"pool_id": 4})
print(f"\npool 4 concepts (count={len(r4.get('concepts',[]))}):")
for c in r4.get("concepts", []):
    if any(s in c.get('decoded','') for s in ['bravo','charlie','echo','foxtrot','obravo']):
        print(f"  id={c['neuron_id']} mc={c['member_count']} decoded={c['decoded']!r}")
