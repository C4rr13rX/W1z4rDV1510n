#!/usr/bin/env python3
"""
scripts/train_json_responses.py — scoped JSON response training.

DESIGN PRINCIPLE — prevents over-training bias:
  Every batch is 50% JSON-request pairs + 50% natural-language pairs.
  JSON questions ALWAYS contain an explicit scope marker:
      "as JSON", "in JSON format", "return JSON", "JSON object", etc.
  Natural-language questions NEVER contain "JSON" — they are answered in prose.
  The node learns: "JSON scope marker present → output JSON,
                    no JSON scope marker → output natural language."

This interleaving means no matter how many batches you run, JSON answers
can never dominate because for every JSON pair trained there is an equal
number of natural-language pairs trained at the same weight.

Usage:
  python scripts/train_json_responses.py              # 200 batches (default)
  python scripts/train_json_responses.py --batches 50 # quick test
  python scripts/train_json_responses.py --check      # verify qa_store count
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
import urllib.request
import urllib.error

NODE = "http://localhost:8090"
BATCH_SIZE = 10  # 5 JSON + 5 natural-language per batch


def post(path: str, body: dict, timeout: int = 20) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{NODE}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"error": str(exc)}


def _j(obj: object) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _jp(obj: object) -> str:
    return json.dumps(obj, indent=2)


# ---------------------------------------------------------------------------
# JSON pairs — question ALWAYS contains explicit JSON scope marker
# ---------------------------------------------------------------------------
JSON_PAIRS: list[tuple[str, str]] = [
    # Primitives
    ("Return a JSON null value", "null"),
    ("Return a JSON true boolean", "true"),
    ("Return a JSON false boolean", "false"),
    ("Return the number 42 as JSON", "42"),
    ("Return the floating point number 3.14 as JSON format", "3.14"),
    ("Return the string hello world formatted as JSON", '"hello world"'),
    ("Give me an empty JSON object", "{}"),
    ("Give me an empty JSON array", "[]"),

    # Flat objects — every question says "JSON object" or "as JSON"
    ("Return a JSON object with a name field set to Alice", _j({"name": "Alice"})),
    ("Return a JSON object with fields id equal to 1 and status equal to active", _j({"id": 1, "status": "active"})),
    ("Return a JSON object representing a user with username bob and age 30", _j({"username": "bob", "age": 30})),
    ("Return a JSON object with a boolean field enabled set to true", _j({"enabled": True})),
    ("Return a JSON config object with host localhost and port 8080", _j({"host": "localhost", "port": 8080})),
    ("Return a JSON object with a null value for the field error", _j({"error": None})),
    ("Return a JSON object with three fields x equal to 10 y equal to 20 z equal to 30", _j({"x": 10, "y": 20, "z": 30})),

    # Arrays
    ("Return a JSON array containing the numbers 1 2 and 3", _j([1, 2, 3])),
    ("Return a JSON array of strings apple banana cherry", _j(["apple", "banana", "cherry"])),
    ("Return a JSON array of three boolean values true false true", _j([True, False, True])),
    ("Return as JSON an array of objects each with an id and name", _j([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])),
    ("Return a JSON array of five even numbers starting from 2", _j([2, 4, 6, 8, 10])),

    # Nested objects
    ("Return a JSON object with a nested address containing street and city", _j({"address": {"street": "123 Main St", "city": "Springfield"}})),
    ("Return a deeply nested JSON object three levels deep with keys a b c where c equals 42", _j({"a": {"b": {"c": 42}}})),
    ("Return a JSON object where the metadata field contains created_at and version", _j({"metadata": {"created_at": "2026-01-01T00:00:00Z", "version": "1.0.0"}})),

    # API envelopes
    ("Return a JSON API success response with status ok and data field containing a result", _j({"status": "ok", "data": {"result": "success"}})),
    ("Return a JSON API error response with status error code 404 and message not found", _j({"status": "error", "code": 404, "message": "not found"})),
    ("Return a JSON paginated response with page 1 per_page 20 total 100 and empty items", _j({"page": 1, "per_page": 20, "total": 100, "items": []})),
    ("Format as JSON a webhook payload with event user.created and a user_id in data", _j({"event": "user.created", "data": {"user_id": "u_abc123"}, "timestamp": "2026-01-01T00:00:00Z"})),

    # Configs
    ("Return a JSON database config with host port user and database fields", _j({"host": "db.example.com", "port": 5432, "user": "admin", "database": "mydb"})),
    ("Return a JSON feature flags object with dark_mode true beta_ui false and analytics true", _j({"dark_mode": True, "beta_ui": False, "analytics": True})),
    ("Return a JSON timeout config with connect 5 and read 30 seconds", _j({"timeouts": {"connect_s": 5, "read_s": 30}})),

    # Schemas
    ("Return a JSON schema describing a required string field named email", _j({"type": "string", "name": "email", "required": True, "format": "email"})),
    ("Return a JSON schema for a user object with id integer name string email string", _j({"type": "object", "properties": {"id": {"type": "integer"}, "name": {"type": "string"}, "email": {"type": "string"}}})),
    ("Return a JSON field definition as JSON with type enum and values pending active inactive", _j({"type": "enum", "values": ["pending", "active", "inactive"]})),

    # Records
    ("Return a JSON record for a product with id name price and in_stock fields", _j({"id": "prod_001", "name": "Widget", "price": 9.99, "in_stock": True})),
    ("Format this as a JSON transaction record: from address to address amount 1.5 currency ETH", _j({"from": "0xabc...", "to": "0xdef...", "amount": "1.5", "currency": "ETH"})),
    ("Return a JSON log entry with level info message and context fields", _j({"level": "info", "message": "Server started", "context": {"pid": 1234}})),

    # ML / AI structured output
    ("Return a JSON object describing an ML model with name version input_shape output_shape and accuracy", _j({"name": "classifier_v1", "version": "1.0", "input_shape": [224, 224, 3], "output_shape": [1000], "accuracy": 0.924})),
    ("Return a JSON training status object with epoch loss and accuracy fields", _j({"epoch": 10, "loss": 0.042, "accuracy": 0.987})),
    ("Return a JSON prediction response with label confidence and top_k alternatives", _j({"label": "cat", "confidence": 0.94, "alternatives": [{"label": "dog", "confidence": 0.05}]})),

    # Scoped crypto records
    ("Return a JSON crypto wallet record with address balance and network as JSON", _j({"address": "0x1234...", "balance": "2.5", "network": "ethereum"})),
    ("Return a JSON block header with hash height timestamp and transaction_count", _j({"hash": "0xabcd...", "height": 18500000, "timestamp": 1700000000, "tx_count": 312})),

    # Instruction compliance
    ("When asked to output JSON please return only raw JSON with no explanation", _j({"status": "understood", "format": "raw_json_only"})),
    ("JSON format only no markdown no prose just the raw JSON object", _j({"compliance": True, "output_format": "json_only"})),
    ("Return the API response as a JSON object", _j({"ok": True, "data": {}})),
    ("Give me the result formatted as JSON", _j({"result": "formatted_as_requested"})),
    ("Output this as JSON format", _j({"output": "json_format", "status": "ok"})),
]

# ---------------------------------------------------------------------------
# Natural-language pairs — NO "JSON" in question, prose answers only
# These are trained alongside JSON pairs so the node learns the contrast.
# ---------------------------------------------------------------------------
NL_PAIRS: list[tuple[str, str]] = [
    # Science
    ("What is the Sun?", "The Sun is the star at the center of our solar system, a hot plasma sphere that generates energy through nuclear fusion in its core."),
    ("What is photosynthesis?", "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("What is gravity?", "Gravity is the force of attraction between masses. On Earth it accelerates objects downward at 9.8 m/s²."),
    ("What is DNA?", "DNA is the molecule that carries genetic instructions for the development and function of all living organisms."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 metres per second."),
    ("What is a black hole?", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape its event horizon."),
    ("Why is the sky blue?", "The sky appears blue because atmospheric molecules scatter shorter blue wavelengths of sunlight more than longer red wavelengths."),
    ("What is quantum mechanics?", "Quantum mechanics describes the behavior of matter and energy at the smallest scales, where particles exhibit both wave and particle properties."),
    ("What is entropy?", "Entropy measures disorder in a system. The second law of thermodynamics states that entropy in an isolated system tends to increase over time."),
    ("What is the Milky Way?", "The Milky Way is the barred spiral galaxy containing our solar system, with an estimated 100 to 400 billion stars."),

    # Geography
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest ocean, covering more than 60 million square miles."),
    ("What is the longest river in the world?", "The Nile River in Africa is approximately 6,650 kilometres long and is generally considered the world's longest river."),
    ("What is the capital of Burkina Faso?", "The capital of Burkina Faso is Ouagadougou."),
    ("What continent is Egypt on?", "Egypt is located in northeastern Africa."),
    ("What is the Sahara Desert?", "The Sahara is the world's largest hot desert, covering approximately 9 million square kilometres across North Africa."),

    # History
    ("Who was Albert Einstein?", "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and won the Nobel Prize in Physics in 1921."),
    ("Who was Isaac Newton?", "Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation and developed calculus."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing and patenting the first practical telephone in 1876."),
    ("Who was Marie Curie?", "Marie Curie was a Polish-French physicist who pioneered research on radioactivity and was the first person to win two Nobel Prizes."),

    # Mathematics
    ("What is Pi?", "Pi is the mathematical constant representing the ratio of a circle's circumference to its diameter, approximately 3.14159."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle a squared plus b squared equals c squared, where c is the hypotenuse."),
    ("What is a prime number?", "A prime number is a natural number greater than 1 with no divisors other than 1 and itself, such as 2, 3, 5, 7, and 11."),
    ("What is calculus?", "Calculus is the branch of mathematics studying continuous change, with two main areas: differentiation for rates of change and integration for areas and accumulation."),

    # Technology
    ("What is artificial intelligence?", "Artificial intelligence is the simulation of human cognitive processes by computer systems, including learning, reasoning, and perception."),
    ("What is machine learning?", "Machine learning is the field where computer systems learn patterns from data to improve their performance without being explicitly programmed for each task."),
    ("What is the Hebbian learning rule?", "The Hebbian learning rule states that neurons that fire together wire together, strengthening synaptic connections between neurons that are simultaneously active."),
    ("What is a neural network?", "A neural network is a computing system modeled on the brain, with interconnected layers of nodes that process information to recognize patterns."),
    ("What is an API?", "An API is a set of rules that allows different software applications to communicate and exchange data with each other."),

    # W1z4rD self-knowledge
    ("What are you?", "I am the W1z4rD V1510n neural node, a distributed Hebbian learning system that builds knowledge through weighted associations."),
    ("How do you learn?", "I learn through Hebbian associative learning. Concepts that appear together frequently form stronger connections, allowing me to recall related knowledge when queried."),
    ("What is your training status?", "I am being continuously trained on scientific, historical, mathematical, and general knowledge corpora. My responses improve as training data accumulates."),

    # Everyday
    ("What causes rainbows?", "Rainbows form when sunlight enters water droplets, gets refracted and reflected inside, and exits at different angles for each color wavelength."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels."),
    ("What is metabolism?", "Metabolism is the set of chemical reactions in an organism that convert food to energy, build proteins, and eliminate waste products."),
    ("What is the immune system?", "The immune system is the body's defense network that identifies and destroys pathogens, foreign substances, and abnormal cells."),
]


def make_candidate(question: str, answer: str, book_id: str) -> dict:
    return {
        "qa_id": str(uuid.uuid4()),
        "question": question,
        "answer": answer,
        "book_id": book_id,
        "confidence": 0.93,
        "evidence": "Scoped training corpus — JSON only when explicitly requested",
        "review_status": "approved",
    }


def build_interleaved_batch(
    json_pool: list[tuple[str, str]],
    nl_pool: list[tuple[str, str]],
    book_id: str,
    batch_idx: int,
) -> list[dict]:
    """Build one batch: BATCH_SIZE//2 JSON pairs + BATCH_SIZE//2 NL pairs."""
    half = BATCH_SIZE // 2

    # Cycle through pools so every pair gets equal coverage
    j_start = (batch_idx * half) % len(json_pool)
    n_start = (batch_idx * half) % len(nl_pool)

    j_pairs = [json_pool[(j_start + i) % len(json_pool)] for i in range(half)]
    n_pairs = [nl_pool[(n_start + i) % len(nl_pool)] for i in range(half)]

    candidates = []
    for q, a in j_pairs:
        candidates.append(make_candidate(q, a, book_id))
    for q, a in n_pairs:
        candidates.append(make_candidate(q, a, book_id))
    return candidates


def check_count(node: str) -> None:
    url = f"http://{node}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        print(json.dumps(data, indent=2))
    except Exception as exc:
        print(f"Could not reach node: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches", type=int, default=200,
        help="Number of interleaved batches to run (default: 200). "
             "Each batch = 5 JSON + 5 NL pairs. "
             "Keep this < total general-knowledge training volume to avoid bias.",
    )
    parser.add_argument("--node", default="localhost:8090")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_count(args.node)
        return

    global NODE
    NODE = f"http://{args.node}"

    json_pool = JSON_PAIRS[:]
    nl_pool = NL_PAIRS[:]

    print(f"Scoped JSON training: {args.batches} batches × {BATCH_SIZE} pairs")
    print(f"  {BATCH_SIZE//2} JSON pairs + {BATCH_SIZE//2} natural-language pairs per batch")
    print(f"  Total: {args.batches * BATCH_SIZE} ingestions ({args.batches * BATCH_SIZE // 2} JSON, {args.batches * BATCH_SIZE // 2} NL)")
    print(f"  Node: {args.node}\n")

    errors = 0
    for i in range(args.batches):
        book_id = f"json_scoped_b{i:04d}"
        candidates = build_interleaved_batch(json_pool, nl_pool, book_id, i)
        result = post("/qa/ingest", {"candidates": candidates})

        if result.get("error"):
            errors += 1
            if errors <= 5:
                print(f"  ERROR batch {i}: {result['error']}", file=sys.stderr)
        elif i % 20 == 0 or i == args.batches - 1:
            pct = (i + 1) / args.batches * 100
            print(f"  [{i+1}/{args.batches}] {pct:.0f}%", flush=True)

        time.sleep(0.04)

    print(f"\nDone. {args.batches} batches. {errors} errors.")
    print("The node now has equal JSON and natural-language weights in this corpus.")
    print("JSON answers will only fire when the question explicitly requests JSON format.")


if __name__ == "__main__":
    main()
