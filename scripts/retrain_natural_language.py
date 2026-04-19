#!/usr/bin/env python3
"""
retrain_natural_language.py — rebalance the node after JSON-heavy training.

The JSON training corpus (800 batches × 10 candidates) caused the node to
return JSON patterns for general questions.  This script injects a large
corpus of plain-English QA pairs across science, history, geography, math,
and everyday knowledge to dilute the JSON-dominant weights.

Run:
    python scripts/retrain_natural_language.py --node localhost:8090
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import urllib.request
import urllib.error

PAIRS: list[tuple[str, str]] = [
    # Science
    ("What is the Sun?", "The Sun is the star at the center of our solar system. It is a nearly perfect sphere of hot plasma, generating energy through nuclear fusion in its core."),
    ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose."),
    ("What is gravity?", "Gravity is a fundamental force of attraction between objects with mass. On Earth, it pulls objects toward the planet's center with an acceleration of 9.8 m/s²."),
    ("What is DNA?", "DNA (deoxyribonucleic acid) is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all living organisms."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second, or about 3×10⁸ m/s."),
    ("What is an atom?", "An atom is the smallest unit of ordinary matter that forms a chemical element. Each atom consists of a nucleus of protons and neutrons surrounded by electrons."),
    ("What is evolution?", "Evolution is the process of change in the heritable characteristics of biological populations over successive generations through natural selection."),
    ("What is the water cycle?", "The water cycle describes the continuous movement of water on, above, and below Earth's surface through evaporation, condensation, precipitation, and runoff."),
    ("What is electricity?", "Electricity is the flow of electric charge, typically electrons, through a conductor. It powers devices and is generated from sources like solar, wind, and fossil fuels."),
    ("What is a black hole?", "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it once past the event horizon."),
    ("What is the Milky Way?", "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy with an estimated 100–400 billion stars."),
    ("What is quantum mechanics?", "Quantum mechanics is the branch of physics that describes the behavior of matter and energy at the smallest scales, where particles exhibit both wave-like and particle-like properties."),
    ("What is entropy?", "Entropy is a measure of disorder or randomness in a system. The second law of thermodynamics states that entropy tends to increase in isolated systems over time."),
    ("What is a neuron?", "A neuron is a nerve cell — the fundamental unit of the brain and nervous system. It transmits information through electrical and chemical signals."),
    ("What is the human genome?", "The human genome is the complete set of DNA in a human cell, consisting of approximately 3 billion base pairs encoding about 20,000–25,000 genes."),

    # Geography
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of Brazil?", "The capital of Brazil is Brasília."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra."),
    ("What is the capital of Germany?", "The capital of Germany is Berlin."),
    ("What is the capital of Burkina Faso?", "The capital of Burkina Faso is Ouagadougou."),
    ("What is the largest ocean?", "The Pacific Ocean is the largest and deepest ocean, covering more than 60 million square miles."),
    ("What is the longest river in the world?", "The Nile River in Africa is generally considered the longest river in the world at approximately 6,650 km."),
    ("What is the highest mountain on Earth?", "Mount Everest in the Himalayas is the highest mountain on Earth, with a peak elevation of 8,849 meters above sea level."),
    ("What is the Amazon rainforest?", "The Amazon rainforest is the world's largest tropical rainforest, covering most of the Amazon basin in South America and hosting extraordinary biodiversity."),
    ("What continent is Egypt in?", "Egypt is located in northeastern Africa."),
    ("What is the Sahara Desert?", "The Sahara is the world's largest hot desert, covering approximately 9 million square kilometers across North Africa."),

    # History
    ("Who was Albert Einstein?", "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity and made major contributions to quantum mechanics. He won the Nobel Prize in Physics in 1921."),
    ("Who was Isaac Newton?", "Isaac Newton was an English mathematician and physicist who formulated the laws of motion and universal gravitation, and developed calculus."),
    ("What was World War II?", "World War II was a global conflict from 1939 to 1945 involving most of the world's nations. It resulted in approximately 70–85 million deaths and ended with the defeat of Nazi Germany and Imperial Japan."),
    ("Who was Abraham Lincoln?", "Abraham Lincoln was the 16th President of the United States. He led the nation through the Civil War and issued the Emancipation Proclamation to abolish slavery."),
    ("What was the Renaissance?", "The Renaissance was a cultural and intellectual movement in Europe during the 14th–17th centuries, characterized by renewed interest in classical art, literature, science, and humanism."),
    ("What was the Industrial Revolution?", "The Industrial Revolution was the transition to new manufacturing processes in Europe and the US from about 1760 to 1840, introducing machinery, factories, and mass production."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing and patenting the first practical telephone in 1876."),
    ("Who was Marie Curie?", "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences."),

    # Mathematics
    ("What is Pi?", "Pi (π) is the mathematical constant representing the ratio of a circle's circumference to its diameter, approximately equal to 3.14159265358979."),
    ("What is the Pythagorean theorem?", "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a² + b² = c²."),
    ("What is a prime number?", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11, 13."),
    ("What is calculus?", "Calculus is a branch of mathematics dealing with continuous change. It has two main branches: differential calculus (rates of change and slopes) and integral calculus (areas and accumulation)."),
    ("What is the Fibonacci sequence?", "The Fibonacci sequence is a series of numbers where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34..."),
    ("What is a derivative?", "A derivative measures how a function changes as its input changes. It is the instantaneous rate of change of a function at a given point."),
    ("What is an integral?", "An integral is a mathematical object that represents the area under a curve or the accumulation of a quantity. It is the inverse operation of differentiation."),
    ("What is linear algebra?", "Linear algebra is the branch of mathematics concerning linear equations, linear functions, and their representations through matrices and vector spaces."),

    # Technology
    ("What is artificial intelligence?", "Artificial intelligence is the simulation of human intelligence processes by computer systems, including learning, reasoning, problem-solving, perception, and language understanding."),
    ("What is machine learning?", "Machine learning is a subset of AI where systems learn from data to improve performance on tasks without being explicitly programmed for each task."),
    ("What is the internet?", "The internet is a global network of interconnected computers that communicate using standardized protocols, enabling the exchange of data, communication, and access to information worldwide."),
    ("What is a neural network?", "A neural network is a computing system inspired by the human brain, consisting of interconnected layers of nodes (neurons) that process information to recognize patterns and make decisions."),
    ("What is the Hebbian learning rule?", "Hebbian learning is a rule in neuroscience and machine learning: neurons that fire together, wire together. It strengthens connections between neurons that are active simultaneously."),
    ("What is blockchain?", "Blockchain is a distributed ledger technology that records transactions across multiple computers in a way that is secure, transparent, and tamper-resistant."),
    ("What is Python?", "Python is a high-level, interpreted programming language known for its readability and simplicity. It is widely used in data science, web development, automation, and AI."),
    ("What is an API?", "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate and share data with each other."),
    ("What is cloud computing?", "Cloud computing is the delivery of computing services — servers, storage, databases, networking, software — over the internet (the cloud) on a pay-as-you-go basis."),
    ("What is a GPU?", "A GPU (Graphics Processing Unit) is a specialized processor originally designed for rendering graphics, now widely used for parallel computation tasks like AI training."),

    # Nature and everyday
    ("Why is the sky blue?", "The sky appears blue because of Rayleigh scattering — molecules in the atmosphere scatter shorter blue wavelengths of sunlight more than longer red wavelengths."),
    ("Why is the ocean salty?", "The ocean is salty because rivers carry dissolved minerals including sodium and chloride from rocks on land into the sea over billions of years."),
    ("How do rainbows form?", "Rainbows form when sunlight enters water droplets, is refracted and reflected inside, then exits at different angles for each wavelength, separating white light into its spectrum of colors."),
    ("What causes earthquakes?", "Earthquakes are caused by the sudden release of energy in Earth's crust due to movement along fault lines, volcanic activity, or tectonic plate interactions."),
    ("What is the food chain?", "A food chain describes the transfer of energy from one organism to another, starting with producers (plants) through consumers (herbivores, carnivores) to decomposers."),
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities like burning fossil fuels and deforestation since the Industrial Revolution."),
    ("What is metabolism?", "Metabolism is the set of chemical reactions in an organism that sustain life, including converting food to energy, building proteins, and eliminating waste."),
    ("What is the immune system?", "The immune system is the body's defense network that identifies and destroys pathogens, foreign substances, and abnormal cells, protecting the organism from disease."),

    # W1z4rD system self-knowledge
    ("What are you?", "I am the W1z4rD V1510n neural node — a distributed Hebbian learning system that builds knowledge through association rather than transformer-based generation."),
    ("How do you learn?", "I learn through Hebbian associative learning: concepts that appear together frequently form stronger connections. Over time, these associations allow me to recall related knowledge when queried."),
    ("What is your training status?", "I am continuously being trained on scientific, historical, mathematical, and general knowledge corpora. My responses improve as my training data grows."),
    ("What is your architecture?", "I use a Hebbian neural fabric — a position accumulator that strengthens associations between co-occurring concepts. I am a distributed node system running on localhost:8090."),
]


def ingest_batch(node: str, pairs: list[tuple[str, str]], book_id: str) -> dict:
    candidates = [
        {
            "qa_id": str(uuid.uuid4()),
            "question": q,
            "answer": a,
            "book_id": book_id,
            "confidence": 0.92,
            "evidence": "Natural language rebalancing corpus",
            "review_status": "approved",
        }
        for q, a in pairs
    ]
    payload = json.dumps({"candidates": candidates}).encode()
    url = f"http://{node}/qa/ingest"
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", default="localhost:8090")
    parser.add_argument("--repeats", type=int, default=200,
                        help="Times to repeat the corpus (more = stronger rebalancing)")
    args = parser.parse_args()

    total = len(PAIRS)
    print(f"Rebalancing with {total} natural-language QA pairs × {args.repeats} repeats = {total * args.repeats} total ingestions")
    print(f"Node: {args.node}")

    batch_size = 10
    batches_done = 0
    total_batches = (total * args.repeats + batch_size - 1) // batch_size

    for repeat in range(args.repeats):
        book_id = f"nl_rebalance_r{repeat:04d}"
        # Shuffle-ish by cycling offset each repeat
        offset = (repeat * 7) % total
        shuffled = PAIRS[offset:] + PAIRS[:offset]

        for i in range(0, len(shuffled), batch_size):
            batch = shuffled[i:i + batch_size]
            result = ingest_batch(args.node, batch, book_id)
            batches_done += 1
            if batches_done % 20 == 0:
                pct = batches_done / total_batches * 100
                err = result.get("error", "")
                status = f"ERROR: {err}" if err else "ok"
                print(f"  [{batches_done}/{total_batches}] {pct:.1f}% — {status}", flush=True)
            time.sleep(0.05)

    print(f"\nDone. {batches_done} batches ingested.")


if __name__ == "__main__":
    main()
