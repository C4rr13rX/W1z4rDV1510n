#!/usr/bin/env python3
"""
scripts/train_json_responses.py

Trains the W1z4rD neural fabric to return ONLY valid JSON when asked.
Covers: abstract specs, exact specs, nested objects, arrays, configs,
API responses, schemas, envelopes, scopes, type-tagged values, and more.

Each QA pair: question = plain English description, answer = ONLY the JSON.
No preamble, no explanation, no markdown fences — just raw JSON.

Usage:
  python scripts/train_json_responses.py              # 800 frames
  python scripts/train_json_responses.py --frames 400 # quick run
  python scripts/train_json_responses.py --check      # verify qa_store count
"""
import argparse, json, random, time, urllib.request

NODE = "http://localhost:8090"

def post(path, body, timeout=20):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{NODE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

# ---------------------------------------------------------------------------
# Corpus builder — (question, json_answer) pairs
# ---------------------------------------------------------------------------

def _j(obj):
    return json.dumps(obj, separators=(',', ':'))

def _jp(obj):
    return json.dumps(obj, indent=2)

PAIRS = []

def pair(q, a):
    PAIRS.append((q.strip(), a if isinstance(a, str) else _j(a)))

# ── 1. Primitive scalars ────────────────────────────────────────────────────
pair("Return a JSON null value", "null")
pair("Return a JSON true boolean", "true")
pair("Return a JSON false boolean", "false")
pair("Return the number 42 as JSON", "42")
pair("Return the floating point number 3.14 as JSON", "3.14")
pair("Return the string hello world as JSON", '"hello world"')
pair("Return an empty JSON object", "{}")
pair("Return an empty JSON array", "[]")

# ── 2. Simple flat objects ───────────────────────────────────────────────────
pair("Return a JSON object with a name field set to Alice",
     _j({"name": "Alice"}))
pair("Return a JSON object with fields id equal to 1 and status equal to active",
     _j({"id": 1, "status": "active"}))
pair("Return a JSON object representing a user with username bob and age 30",
     _j({"username": "bob", "age": 30}))
pair("Return a JSON object with a boolean field enabled set to true",
     _j({"enabled": True}))
pair("Return a JSON config object with host localhost and port 8080",
     _j({"host": "localhost", "port": 8080}))
pair("Return a JSON object with a null value for the field error",
     _j({"error": None}))
pair("Return a JSON key-value pair where the key is color and value is blue",
     _j({"color": "blue"}))
pair("Return a JSON object with three fields: x equal to 10, y equal to 20, z equal to 30",
     _j({"x": 10, "y": 20, "z": 30}))

# ── 3. Arrays ───────────────────────────────────────────────────────────────
pair("Return a JSON array containing the numbers 1, 2, and 3",
     _j([1, 2, 3]))
pair("Return a JSON array of strings: apple, banana, cherry",
     _j(["apple", "banana", "cherry"]))
pair("Return a JSON array of three boolean values: true, false, true",
     _j([True, False, True]))
pair("Return a JSON array of objects, each with an id and name. Use id 1 name Alice and id 2 name Bob",
     _j([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]))
pair("Return a JSON array with a mix of string, number, and boolean: hello, 99, true",
     _j(["hello", 99, True]))
pair("Return a JSON array of five even numbers starting from 2",
     _j([2, 4, 6, 8, 10]))

# ── 4. Nested objects ────────────────────────────────────────────────────────
pair("Return a JSON object with a nested address object containing street and city",
     _j({"address": {"street": "123 Main St", "city": "Springfield"}}))
pair("Return a JSON user object with a nested profile object containing bio and avatar fields",
     _j({"user": {"profile": {"bio": "Engineer", "avatar": "https://example.com/a.png"}}}))
pair("Return a deeply nested JSON object three levels deep with keys a, b, c where c has value 42",
     _j({"a": {"b": {"c": 42}}}))
pair("Return a JSON object where the metadata field contains created_at and version",
     _j({"metadata": {"created_at": "2026-01-01T00:00:00Z", "version": "1.0.0"}}))

# ── 5. API response envelopes ────────────────────────────────────────────────
pair("Return a JSON API success response envelope with status ok and a data field containing result hello",
     _j({"status": "ok", "data": {"result": "hello"}}))
pair("Return a JSON API error response with status error, code 404, and message not found",
     _j({"status": "error", "code": 404, "message": "not found"}))
pair("Return a JSON paginated list response with page 1, per_page 20, total 100, and an empty items array",
     _j({"page": 1, "per_page": 20, "total": 100, "items": []}))
pair("Return a JSON response wrapping an array of results under a results key with a count field",
     _j({"count": 3, "results": [{"id": 1}, {"id": 2}, {"id": 3}]}))
pair("Return a JSON webhook payload with event user.created and a data object containing the user id",
     _j({"event": "user.created", "data": {"user_id": "u_abc123"}, "timestamp": "2026-01-01T00:00:00Z"}))

# ── 6. Config objects ────────────────────────────────────────────────────────
pair("Return a JSON database config with host, port, user, password, and database fields",
     _j({"host": "db.example.com", "port": 5432, "user": "admin", "password": "secret", "database": "mydb"}))
pair("Return a JSON server config with debug false, workers 4, and a cors array of allowed origins",
     _j({"debug": False, "workers": 4, "cors": ["https://example.com", "https://app.example.com"]}))
pair("Return a JSON feature flags config with three flags: dark_mode true, beta_ui false, analytics true",
     _j({"dark_mode": True, "beta_ui": False, "analytics": True}))
pair("Return a JSON logging config with level info and format json",
     _j({"logging": {"level": "info", "format": "json"}}))
pair("Return a JSON timeout config with connect timeout 5 and read timeout 30 in seconds",
     _j({"timeouts": {"connect_s": 5, "read_s": 30}}))

# ── 7. Schema-like structures ────────────────────────────────────────────────
pair("Return a JSON schema object describing a string field named email that is required",
     _j({"type": "string", "name": "email", "required": True, "format": "email"}))
pair("Return a JSON schema for a user object with fields id integer, name string, email string",
     _j({"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}, "email": {"type": "string"}}}))
pair("Return a JSON schema for an array of numbers",
     _j({"type": "array", "items": {"type": "number"}}))
pair("Return a JSON field definition with type enum and values pending, active, inactive",
     _j({"type": "enum", "values": ["pending", "active", "inactive"]}))
pair("Return a JSON schema with required fields and their types: id number, name string, active boolean",
     _j({"fields": [{"name": "id", "type": "number", "required": True}, {"name": "name", "type": "string", "required": True}, {"name": "active", "type": "boolean", "required": False}]}))

# ── 8. Structured data records ────────────────────────────────────────────────
pair("Return a JSON record for a product with id, name, price, and in_stock fields",
     _j({"id": "prod_001", "name": "Widget", "price": 9.99, "in_stock": True}))
pair("Return a JSON event record with event type, timestamp, user_id, and properties object",
     _j({"type": "click", "timestamp": 1700000000, "user_id": "u123", "properties": {"element": "button", "page": "/home"}}))
pair("Return a JSON transaction record with from address, to address, amount, and currency",
     _j({"from": "0xabc...", "to": "0xdef...", "amount": "1.5", "currency": "ETH"}))
pair("Return a JSON log entry with level, message, and context",
     _j({"level": "error", "message": "Connection refused", "context": {"host": "db.example.com", "port": 5432}}))
pair("Return a JSON health check response showing service up with version and uptime",
     _j({"status": "up", "version": "2.1.0", "uptime_s": 86400}))

# ── 9. Scope-aware: ask for specific scope/namespace ────────────────────────
pair("Return a JSON object scoped to the auth namespace with token and expires_at fields",
     _j({"auth": {"token": "eyJhbGc...", "expires_at": "2026-12-31T23:59:59Z"}}))
pair("Return a JSON response with a top-level scope key set to global and a config object inside",
     _j({"scope": "global", "config": {"max_connections": 100, "timeout_ms": 5000}}))
pair("Return a JSON object with user scope containing only the fields that belong to the user: name, email, role",
     _j({"user": {"name": "Alice", "email": "alice@example.com", "role": "admin"}}))
pair("Return a JSON envelope with a data scope and a meta scope, each containing relevant fields",
     _j({"data": {"id": 1, "value": "hello"}, "meta": {"request_id": "req_abc", "took_ms": 12}}))
pair("Return a JSON permission object scoped to resource posts with read true and write false",
     _j({"resource": "posts", "permissions": {"read": True, "write": False}}))

# ── 10. Collections and lists ─────────────────────────────────────────────────
pair("Return a JSON list of five cities as an array of strings",
     _j(["Tokyo", "London", "New York", "Paris", "Sydney"]))
pair("Return a JSON array of three user objects each with id and email",
     _j([{"id": 1, "email": "a@x.com"}, {"id": 2, "email": "b@x.com"}, {"id": 3, "email": "c@x.com"}]))
pair("Return a JSON map of country codes to country names for US, GB, and DE",
     _j({"US": "United States", "GB": "United Kingdom", "DE": "Germany"}))
pair("Return a JSON array of key-value pairs as objects with key and value fields",
     _j([{"key": "color", "value": "red"}, {"key": "size", "value": "large"}]))
pair("Return a JSON object mapping weekday names to their index numbers starting from 0",
     _j({"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}))

# ── 11. Computational and scientific data ─────────────────────────────────────
pair("Return a JSON matrix as an array of arrays representing a 2x2 identity matrix",
     _j([[1, 0], [0, 1]]))
pair("Return a JSON bounding box object with x, y, width, height fields",
     _j({"x": 10, "y": 20, "width": 100, "height": 50}))
pair("Return a JSON vector with x, y, and z components",
     _j({"x": 1.0, "y": 0.0, "z": 0.0}))
pair("Return a JSON color object with red, green, blue integer components",
     _j({"red": 255, "green": 128, "blue": 0}))
pair("Return a JSON date range object with start and end ISO date strings",
     _j({"start": "2026-01-01", "end": "2026-12-31"}))
pair("Return a JSON statistics summary with mean, median, min, max, and count",
     _j({"mean": 42.5, "median": 41.0, "min": 10, "max": 99, "count": 100}))

# ── 12. Abstract / instruction-only specs ────────────────────────────────────
pair("I need a JSON object. It should have a string field called message and a boolean field called success.",
     _j({"message": "Operation completed", "success": True}))
pair("Give me JSON. I want an array of objects. Each object should describe a file with a name and size in bytes.",
     _j([{"name": "readme.txt", "size": 1024}, {"name": "main.py", "size": 8192}]))
pair("I want a JSON response. Structure it as a config. It needs a section for database and a section for cache.",
     _j({"database": {"url": "postgres://localhost/mydb"}, "cache": {"backend": "redis", "ttl_s": 300}}))
pair("Return structured JSON. The response should represent a graph edge with source node, target node, and weight.",
     _j({"source": "A", "target": "B", "weight": 0.75}))
pair("Output JSON only. I want an authentication result: whether it succeeded, a token if it did, and an error message if it failed.",
     _j({"success": True, "token": "abc123", "error": None}))
pair("JSON response please. Give me a task queue item with id, priority, payload, and status fields.",
     _j({"id": "task_001", "priority": 5, "payload": {"action": "send_email", "to": "user@example.com"}, "status": "queued"}))
pair("I need the JSON for a rate limit status: how many requests are allowed, how many remain, and when the limit resets.",
     _j({"limit": 100, "remaining": 43, "reset_at": "2026-01-01T00:00:00Z"}))
pair("Return only JSON. Describe a machine learning model: name, version, input shape, output shape, and accuracy.",
     _j({"name": "image_classifier", "version": "3.2", "input_shape": [224, 224, 3], "output_shape": [1000], "accuracy": 0.9245}))

# ── 13. Exact specification prompts ──────────────────────────────────────────
pair('Return a JSON object with exactly these fields: {"status": "ok", "code": 200}',
     _j({"status": "ok", "code": 200}))
pair('Return a JSON array of exactly three null values',
     _j([None, None, None]))
pair('Return {"error": null, "data": [1, 2, 3]}',
     _j({"error": None, "data": [1, 2, 3]}))
pair('Return a JSON object where every field value is an empty array',
     _j({"a": [], "b": [], "c": []}))
pair('Return a JSON array containing exactly one object with key "done" and value true',
     _j([{"done": True}]))
pair('Return exactly: {"ok": true}',
     '{"ok":true}')
pair('Output only this JSON: {"version": "2.0", "features": ["json", "streaming", "tools"]}',
     _j({"version": "2.0", "features": ["json", "streaming", "tools"]}))

# ── 14. Blockchain / crypto specific ─────────────────────────────────────────
pair("Return a JSON wallet summary with address, balance in ETH, and network name",
     _j({"address": "0x1234...abcd", "balance_eth": "2.450000", "network": "mainnet"}))
pair("Return a JSON token info object with symbol, name, decimals, and contract address",
     _j({"symbol": "USDC", "name": "USD Coin", "decimals": 6, "contract": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"}))
pair("Return a JSON swap quote with from token, to token, input amount, output amount, and slippage",
     _j({"from_token": "ETH", "to_token": "USDC", "input_amount": "1.0", "output_amount": "2450.00", "slippage_pct": 0.5}))
pair("Return a JSON transaction status with hash, confirmed boolean, block number, and gas used",
     _j({"hash": "0xabc...", "confirmed": True, "block_number": 19000000, "gas_used": 21000}))

# ── 15. Instruction phrasing variants ────────────────────────────────────────
for q, a in [
    ("When asked to return JSON, output only the JSON with no explanation. Example: return a greeting object.",
     _j({"greeting": "hello", "language": "en"})),
    ("Respond with only a JSON object. No markdown, no code fences, no explanation. The object should have a type field set to response.",
     _j({"type": "response"})),
    ("Your output should be raw JSON. Nothing before it. Nothing after it. Return a single field object with key ping and value pong.",
     _j({"ping": "pong"})),
    ("JSON only — no text: return an object with fields for first name, last name, and full name.",
     _j({"first_name": "John", "last_name": "Doe", "full_name": "John Doe"})),
    ("Do not include any text outside the JSON. Return a config object with timeout of 30 and retries of 3.",
     _j({"timeout": 30, "retries": 3})),
    ("Output: JSON. Format: object. Fields: success (bool), message (string). No extra text.",
     _j({"success": True, "message": "OK"})),
]:
    pair(q, a)

# ── 16. Varied phrasing + more types ─────────────────────────────────────────
def _many_variants():
    types = [
        ("an API key object with key and created_at",    {"key": "sk_live_abc123", "created_at": "2026-01-01"}),
        ("a geolocation point with latitude and longitude", {"lat": 40.7128, "lng": -74.0060}),
        ("a session object with session_id, user_id, and expires_at", {"session_id": "sess_xyz", "user_id": 42, "expires_at": "2026-01-01T12:00:00Z"}),
        ("a search result with query, total_hits, and first hit title", {"query": "python json", "total_hits": 10000, "hits": [{"title": "JSON in Python"}]}),
        ("a device registration with device_id, platform, and push_token", {"device_id": "dev_abc", "platform": "ios", "push_token": "abc...xyz"}),
        ("a shipping address with all required fields", {"name": "Jane Smith", "line1": "456 Oak Ave", "city": "Portland", "state": "OR", "zip": "97201", "country": "US"}),
        ("an invoice with invoice_id, amount, currency, and line items array", {"invoice_id": "inv_001", "amount": 150.00, "currency": "USD", "line_items": [{"description": "Consulting", "quantity": 3, "unit_price": 50.00}]}),
        ("a notification payload for a push notification", {"title": "New message", "body": "You have 3 unread messages", "data": {"screen": "inbox"}}),
        ("a pagination cursor object with has_more and next_cursor", {"has_more": True, "next_cursor": "cursor_abc123"}),
        ("a deployment status with service name, version, environment, and healthy boolean", {"service": "api-gateway", "version": "v2.3.1", "environment": "production", "healthy": True}),
    ]
    prefixes = [
        "Return a JSON object representing", "Give me JSON for", "Output only JSON:",
        "JSON response for", "Return structured JSON describing",
        "Provide a JSON representation of", "Respond with JSON only:",
        "I need a JSON object for", "Return the JSON for",
    ]
    for desc, obj in types:
        for prefix in prefixes[:4]:
            pair(f"{prefix} {desc}", obj)

_many_variants()

# ── 17. Null/missing/error handling patterns ──────────────────────────────────
pair("Return a JSON object representing a failed operation with error details and no data",
     _j({"success": False, "data": None, "error": {"code": "VALIDATION_ERROR", "details": ["name is required"]}}))
pair("Return a JSON response where some optional fields are null",
     _j({"id": 1, "name": "Alice", "phone": None, "website": None}))
pair("Return a JSON partial update payload where only the changed fields are included",
     _j({"name": "New Name", "updated_at": "2026-01-01T00:00:00Z"}))

# ── 18. Instruction compliance reinforcement ──────────────────────────────────
for _ in range(20):
    examples = [
        ("Return JSON. Just JSON. A simple object with one field.",    _j({"value": 1})),
        ("JSON only response. Object with two string fields.",          _j({"a": "hello", "b": "world"})),
        ("Output raw JSON array of integers.",                          _j([1, 2, 3, 4, 5])),
        ("No text before or after. JSON object with status field.",     _j({"status": "ok"})),
        ("Pure JSON output. Nested object two levels.",                 _j({"outer": {"inner": "value"}})),
    ]
    pair(*random.choice(examples))

print(f"Total corpus size: {len(PAIRS)} QA pairs", flush=True)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

BATCH = 10  # candidates per /qa/ingest call

def train(n_frames, delay=0.08):
    shuffled = list(PAIRS)
    random.shuffle(shuffled)
    ok = 0
    print(f"Training {n_frames} batches over {len(shuffled)} pairs...", flush=True)
    for i in range(n_frames):
        batch_pairs = [shuffled[(i * BATCH + j) % len(shuffled)] for j in range(BATCH)]
        candidates = [
            {"qa_id": f"json_{i:05d}_{j:02d}", "question": q, "answer": a,
             "book_id": "json_training_corpus", "confidence": 1.0,
             "evidence": "synthetic", "review_status": "approved"}
            for j, (q, a) in enumerate(batch_pairs)
        ]
        try:
            post("/qa/ingest", {"candidates": candidates})
            ok += 1
        except Exception as e:
            print(f"  batch {i}: {e}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_frames}", flush=True)
        time.sleep(delay)
    print(f"Done: {ok}/{n_frames} batches accepted", flush=True)

def check():
    try:
        r = post("/qa/query", {"question": "Return a JSON object with key status and value ok"})
        print(f"Test query answer: {r.get('report', {}).get('best_answer', {}).get('answer', '(none)')[:120]}", flush=True)
    except Exception as e:
        print(f"Query check failed: {e}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=800)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--node", default="http://localhost:8090")
    args = ap.parse_args()
    NODE = args.node.rstrip("/")

    if args.check:
        check()
    else:
        check()
        train(args.frames)
        check()
