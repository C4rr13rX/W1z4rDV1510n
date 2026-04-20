#!/usr/bin/env python3
"""
build_foundational_seeds.py — Train high-priority factual Q&A seeds.

Full-architecture training: every seed pair goes through ALL endpoints:
  /media/train_sequence  — Q→A temporal STDP sequence
  /qa/ingest             — Q&A store + internal STDP bridge
  /neuro/record_episode  — episodic store (surprise=0, correct answer)
  /equations/ingest      — math/science text into equation matrix
  /knowledge/ingest      — structured knowledge document per fact group

Usage:
    python scripts/build_foundational_seeds.py --node localhost:8090 [--repeats 15]

Dependencies:
  pip install httpx Pillow
"""

import argparse
import asyncio
import sys
import time
import uuid

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

from neuro_client import NeuroClient

DEFAULT_NODE    = "localhost:8090"
DEFAULT_REPEATS = 15
BATCH_SIZE      = 50

# ---------------------------------------------------------------------------
# Seed Q&A pairs — vocabulary chosen to match probe requirements.
# Grouped by topic so knowledge/ingest gets a well-structured document.
# ---------------------------------------------------------------------------

SEED_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    # (group_title, discipline_or_None, [(question, answer), ...])

    ("Water and Chemistry", "classical_mechanics", [
        ("What is water?",
         "Water is a molecule made of two hydrogen atoms and one oxygen atom, written H2O. "
         "It is a clear liquid essential for all life on Earth."),
        ("What is water made of?",
         "Water is made of hydrogen and oxygen atoms bonded together as H2O."),
        ("What are the chemical components of water?",
         "Water consists of two hydrogen atoms and one oxygen atom — its chemical formula is H2O."),
    ]),

    ("Speed of Light", "special_relativity", [
        ("What is the speed of light?",
         "The speed of light in a vacuum is approximately 299,792,458 metres per second, "
         "often rounded to 300,000 kilometres per second. It is the fastest speed possible."),
        ("How fast does light travel?",
         "Light travels at approximately 299,792 kilometres per second (about 300,000 km/s) "
         "through a vacuum."),
    ]),

    ("DNA and Genetics", None, [
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
    ]),

    ("Atoms and Matter", "quantum_mechanics", [
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
    ]),

    ("Pi and Geometry", "information_theory", [
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
    ]),

    ("Prime Numbers", None, [
        ("What is a prime number?",
         "A prime number is a whole number greater than 1 that has no divisors other than "
         "1 and itself. Examples include 2, 3, 5, 7, 11. A number is prime if it is only "
         "divisible by one and itself."),
        ("What makes a number prime?",
         "A number is prime if it is greater than 1 and only divisible by 1 and itself, "
         "with no other factors. Examples: 2, 3, 5, 7, 11, 13."),
    ]),

    ("Geography: France", None, [
        ("What is the capital of France?",
         "The capital of France is Paris. Paris is the largest city in France and has been "
         "the country's capital for centuries."),
        ("What city is the capital of France?",
         "Paris is the capital city of France."),
    ]),

    ("Geography: United States", None, [
        ("What is the capital of the United States?",
         "The capital of the United States is Washington, D.C. (District of Columbia). "
         "It is the seat of the federal government, home to the White House and Congress."),
        ("What is the capital city of the USA?",
         "Washington, D.C. is the capital city of the United States of America."),
        ("Where is the US capital located?",
         "The US capital, Washington D.C., is located on the East Coast between Virginia "
         "and Maryland."),
    ]),

    ("The Sun", "cosmology", [
        ("What is the sun?",
         "The sun is a star at the center of our solar system. It is a massive ball of "
         "plasma held together by gravity, powered by nuclear fusion in its core. "
         "It provides light and energy to Earth."),
        ("What kind of object is the sun?",
         "The sun is a star — a giant sphere of hot plasma powered by nuclear fusion. "
         "It emits light, heat, and solar energy that drives life on Earth."),
    ]),

    ("Gravity", "classical_mechanics", [
        ("What is gravity?",
         "Gravity is a fundamental force that attracts objects with mass toward each other. "
         "On Earth, gravity gives weight to objects and causes them to fall toward the ground. "
         "The gravitational force between two masses is described by Newton's law."),
        ("What causes gravity?",
         "Gravity is caused by mass — every object with mass exerts a gravitational pull "
         "on other objects. The greater the mass and the closer the distance, the stronger "
         "the gravitational attraction."),
    ]),

    ("Photosynthesis", None, [
        ("What is photosynthesis?",
         "Photosynthesis is the process by which plants, algae, and some bacteria use "
         "sunlight, water, and carbon dioxide to produce glucose and oxygen. Chlorophyll "
         "in plant cells absorbs the sunlight that drives this reaction."),
        ("How does photosynthesis work?",
         "During photosynthesis, plants absorb sunlight through chlorophyll, take in "
         "carbon dioxide from the air and water from the soil, and produce glucose "
         "(food) and oxygen as a byproduct."),
    ]),
]


async def train_seeds(node: str, repeats: int) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        nc = NeuroClient(f"http://{node}", client)

        # Verify node
        try:
            r = await client.get(f"http://{node}/health", timeout=5)
            info = r.json()
            print(f"Node: {info.get('node_id','?')}  status={info.get('status')}")
        except Exception as e:
            sys.exit(f"Node not reachable at {node}: {e}")

        total_pairs = sum(len(pairs) for _, _, pairs in SEED_GROUPS)
        print(f"Foundational seeds — {total_pairs} pairs in {len(SEED_GROUPS)} groups")
        print(f"Repeats: {repeats}")
        print(f"Endpoints: train_sequence | qa/ingest | record_episode | equations/ingest | knowledge/ingest\n")

        t0 = time.time()
        stats = {"qa": 0, "ep": 0, "eq": 0, "know": 0, "seq": 0}

        for rep in range(1, repeats + 1):
            for group_title, discipline, pairs in SEED_GROUPS:
                candidates = [
                    {
                        "qa_id":         str(uuid.uuid4()),
                        "question":      q,
                        "answer":        a,
                        "book_id":       "foundational_seeds",
                        "page_index":    1,
                        "confidence":    0.95,
                    }
                    for q, a in pairs
                ]

                # Q&A ingest + Q→A temporal sequence + episodic learning
                n = await nc.ingest_qa_full(
                    candidates, pool="knowledge", record_episodes=True)
                stats["qa"] += n
                stats["seq"] += len(candidates)
                stats["ep"] += len(candidates)

                # Equation matrix — for science-heavy groups
                if discipline:
                    combined_text = " ".join(a for _, a in pairs)
                    n_eq = await nc.ingest_equations(combined_text, discipline=discipline)
                    stats["eq"] += n_eq

                # Knowledge document — structured fact group
                if rep == 1:  # only need to ingest once
                    body = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)
                    ok = await nc.ingest_knowledge(
                        title=group_title,
                        body=body,
                        source="foundational_seeds",
                        tags=["foundation", "seeds"],
                    )
                    if ok:
                        stats["know"] += 1

            if rep % 5 == 0 or rep == repeats:
                elapsed = time.time() - t0
                print(f"  [{rep}/{repeats}]  qa={stats['qa']}  ep={stats['ep']}  "
                      f"eq={stats['eq']}  know={stats['know']}  seq={stats['seq']}  "
                      f"({elapsed:.0f}s)")

        await nc.checkpoint()
        elapsed = time.time() - t0
        print(f"\nDone in {elapsed:.0f}s")
        print(f"  QA ingested  : {stats['qa']}")
        print(f"  Q→A sequences: {stats['seq']}")
        print(f"  Episodes     : {stats['ep']}")
        print(f"  Equations    : {stats['eq']}")
        print(f"  Knowledge doc: {stats['know']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node",    default=DEFAULT_NODE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()
    asyncio.run(train_seeds(args.node, args.repeats))


if __name__ == "__main__":
    main()
