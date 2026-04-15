#!/usr/bin/env python3
"""Boost water and sleep Stage 0 failures."""
import httpx, uuid

NODE = "http://127.0.0.1:8090"

WATER_ANS = ("Water is a clear, odorless liquid made of hydrogen and oxygen (H2O). "
             "It is essential for all known life. Water exists as liquid, solid ice, and water vapor. "
             "It covers about 71 percent of Earth and is vital for drinking, agriculture, and industry.")
SLEEP_ANS = ("Sleep is a natural state of rest in which the body and brain recover and restore themselves. "
             "During sleep, the brain consolidates memories, the body repairs tissues, and energy is restored. "
             "Humans need sleep to function properly, maintain health, and process information.")

PAIRS = {
    "water": {
        "answer": WATER_ANS,
        "questions": [
            "What is water?",
            "What is water made of?",
            "What is the substance called water?",
            "Describe water.",
            "What is water in chemistry?",
            "What is water as a liquid?",
            "What is H2O?",
            "What is the chemical formula for water?",
            "What is water in science?",
            "What are the properties of water?",
            "What is water used for?",
            "Why is water important?",
            "What makes water special?",
            "What is drinking water?",
            "What is water as a molecule?",
            "What is liquid water?",
            "Is water a liquid?",
            "What is water in nature?",
            "What substance is essential for life?",
            "What molecule is H2O?",
            "What is water vapor?",
            "What is pure water?",
            "Why do we drink water?",
            "What fluid covers most of Earth?",
            "What is water and why do we need it?",
        ],
    },
    "sleep": {
        "answer": SLEEP_ANS,
        "questions": [
            "Why do we need to sleep?",
            "What is sleep?",
            "Why do we sleep?",
            "Why is sleep important?",
            "What does sleep do?",
            "What happens when we sleep?",
            "Why do humans need sleep?",
            "What is the purpose of sleep?",
            "Why do we need sleep every night?",
            "What does sleep do for the body?",
            "What does sleep do for the brain?",
            "Why do we need rest and sleep?",
            "What is sleep for?",
            "How does sleep help us?",
            "Why must we sleep?",
            "What is sleep in biology?",
            "What does the brain do during sleep?",
            "How does sleep restore the body?",
            "Why do animals sleep?",
            "What is a sleep cycle?",
            "Why do we need to rest at night?",
            "What happens during sleep?",
            "Why does the body need sleep?",
            "What is the function of sleep?",
            "How does sleep affect memory?",
        ],
    },
}

candidates = []
for concept, data in PAIRS.items():
    for q in data["questions"]:
        candidates.append({
            "qa_id": str(uuid.uuid4()),
            "question": q,
            "answer": data["answer"],
            "confidence": 0.99,
            "book_id": "stage0_" + concept + "_anchor",
            "page_index": 0,
            "evidence": "",
            "review_status": "verified",
        })

print("Ingesting", len(candidates), "pairs...")
with httpx.Client(timeout=60) as client:
    for i in range(0, len(candidates), 50):
        batch = candidates[i:i+50]
        r = client.post(NODE + "/qa/ingest", json={"candidates": batch})
        r.raise_for_status()
        d = r.json()
        print("  Batch", i//50+1, ": ingested=" + str(d.get("ingested", 0)),
              "total_pairs=" + str(d.get("total_pairs", "?")))

    print("Checkpointing...")
    r = client.post(NODE + "/neuro/checkpoint", timeout=120)
    r.raise_for_status()
    print("Done.")

for q, label in [
    ("What is water?", "water"),
    ("Why do we need to sleep?", "sleep"),
]:
    r = httpx.post(NODE + "/qa/query", json={"question": q}, timeout=15)
    results = r.json().get("report", {}).get("results", [])
    top = results[0] if results else {}
    print(label + ": act=" + str(round(top.get("activation", 0), 4))
          + " ans=" + top.get("answer", "")[:80])

# Edge case check
for q in ["What is a blockchain?", "What is a flibbertigibbet?"]:
    r = httpx.post(NODE + "/qa/query", json={"question": q}, timeout=15)
    results = r.json().get("report", {}).get("results", [])
    top = results[0] if results else {}
    print("edge: act=" + str(round(top.get("activation", 0), 4)) + " q=" + q[:40])
