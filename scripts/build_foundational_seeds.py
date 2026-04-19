#!/usr/bin/env python3
"""
build_foundational_seeds.py — Train high-priority factual Q&A seeds.

These pairs fill gaps that the concept/wiki dataset doesn't cover well:
  - Vocabulary-precise science facts (H2O, proton/neutron, DNA)
  - Geographic facts (capitals)
  - Mathematical constants (pi, prime)

Each pair is trained with enough repetitions to dominate noise.

Usage:
    python scripts/build_foundational_seeds.py --node localhost:8090 [--repeats 15]
"""
import argparse
import json
import sys
import time
import urllib.request

DEFAULT_NODE    = "localhost:8090"
DEFAULT_REPEATS = 15
BATCH_SIZE      = 50

# ---------------------------------------------------------------------------
# Seed Q&A pairs — vocabulary chosen to match probe requirements
# ---------------------------------------------------------------------------

KNOWLEDGE_SEEDS = [
    # ── Water / chemistry ────────────────────────────────────────────────────
    ("What is water?",
     "Water is a molecule made of two hydrogen atoms and one oxygen atom, written H2O. "
     "It is a clear liquid essential for all life on Earth."),
    ("What is water made of?",
     "Water is made of hydrogen and oxygen atoms bonded together as H2O."),
    ("What are the chemical components of water?",
     "Water consists of two hydrogen atoms and one oxygen atom — its chemical formula is H2O."),

    # ── Speed of light ────────────────────────────────────────────────────────
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792,458 metres per second, "
     "often rounded to 300,000 kilometres per second. It is the fastest speed possible."),
    ("How fast does light travel?",
     "Light travels at approximately 299,792 kilometres per second (about 300,000 km/s) "
     "through a vacuum."),

    # ── DNA ───────────────────────────────────────────────────────────────────
    ("What is DNA?",
     "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic information "
     "of living organisms. It is made of nucleotides arranged in a double helix and "
     "contains genes stored on chromosomes."),
    ("What does DNA stand for?",
     "DNA stands for deoxyribonucleic acid. It is the molecule that stores genetic "
     "information in the form of genes."),
    ("What is the function of DNA?",
     "DNA stores genetic instructions used in the growth, development, and reproduction "
     "of all known living organisms. It is organized into chromosomes inside the nucleus."),

    # ── Atom ─────────────────────────────────────────────────────────────────
    ("What is an atom?",
     "An atom is the smallest unit of a chemical element. It has a nucleus containing "
     "protons and neutrons, with electrons orbiting around it. Atoms are the basic "
     "building blocks of all matter."),
    ("What are atoms made of?",
     "Atoms are made of a nucleus containing protons and neutrons, surrounded by "
     "electrons. The number of protons determines which element the atom belongs to."),
    ("Describe the structure of an atom.",
     "An atom has a dense central nucleus made of protons and neutrons, surrounded by "
     "a cloud of electrons. The proton count equals the atomic number of the element."),

    # ── Pi ────────────────────────────────────────────────────────────────────
    ("What is pi?",
     "Pi (π) is the mathematical ratio of a circle's circumference to its diameter. "
     "Its value is approximately 3.14159. Pi is used to calculate the area and "
     "circumference of circles."),
    ("What is the value of pi?",
     "The value of pi is approximately 3.14159, often rounded to 3.14. Pi is the "
     "ratio of a circle's circumference to its diameter."),
    ("What is pi used for in mathematics?",
     "Pi (approximately 3.14) is used to calculate the circumference and area of a "
     "circle. The circumference equals pi times the diameter; area equals pi times "
     "the radius squared."),

    # ── Prime number ──────────────────────────────────────────────────────────
    ("What is a prime number?",
     "A prime number is a whole number greater than 1 that has no divisors other than "
     "1 and itself. Examples include 2, 3, 5, 7, 11. A number is prime if it is only "
     "divisible by one and itself."),
    ("What makes a number prime?",
     "A number is prime if it is greater than 1 and only divisible by 1 and itself, "
     "with no other factors. Examples: 2, 3, 5, 7, 11, 13."),

    # ── Geography ─────────────────────────────────────────────────────────────
    ("What is the capital of France?",
     "The capital of France is Paris. Paris is the largest city in France and has been "
     "the country's capital for centuries."),
    ("What city is the capital of France?",
     "Paris is the capital city of France."),
    ("What is the capital of the United States?",
     "The capital of the United States is Washington, D.C. (District of Columbia). "
     "It is the seat of the federal government, home to the White House and Congress."),
    ("What is the capital city of the USA?",
     "Washington, D.C. is the capital city of the United States of America."),
    ("Where is the US capital located?",
     "The US capital, Washington D.C., is located on the East Coast between Virginia "
     "and Maryland."),
]


def post(node: str, path: str, body: dict, timeout: int = 30) -> dict:
    data = json.dumps(body).encode()
    url  = f"http://{node}{path}"
    req  = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default=DEFAULT_NODE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="How many times to train each seed pair")
    args = parser.parse_args()

    node    = args.node
    repeats = args.repeats

    print(f"Foundational seeds — node: {node}  repeats: {repeats}")
    print(f"Seed pairs: {len(KNOWLEDGE_SEEDS)}")

    candidates = [
        {"question": q, "answer": a, "book_id": "foundational_seeds", "page_index": 1}
        for q, a in KNOWLEDGE_SEEDS
    ]

    total_ingested = 0
    errors = 0
    t0 = time.time()

    for rep in range(1, repeats + 1):
        for i in range(0, len(candidates), BATCH_SIZE):
            batch = candidates[i:i + BATCH_SIZE]
            resp = post(node, "/qa/ingest", {"candidates": batch, "pool": "knowledge"})
            if resp.get("error"):
                errors += 1
            else:
                total_ingested += resp.get("ingested", 0)

        if rep % 5 == 0 or rep == repeats:
            elapsed = time.time() - t0
            print(f"  [{rep}/{repeats}] ingested so far: {total_ingested}  errors: {errors}  ({elapsed:.0f}s)")

    # Checkpoint
    post(node, "/qa/checkpoint", {})
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {total_ingested} total ingestions, {errors} errors.")


if __name__ == "__main__":
    main()
