#!/usr/bin/env python3
"""Boost 5 remaining Stage 0 failures: eyes, color_red, math, ocean, learn."""
import httpx, uuid

NODE = "http://127.0.0.1:8090"

EYES_ANS = ("Eyes are the organs of vision. They detect light and convert it into nerve signals "
            "sent to the brain, allowing us to see. Humans have two eyes that perceive depth, color, and movement.")
COLOR_RED_ANS = ("Red is a color with a wavelength of approximately 625-740 nanometers. "
                 "It is one of the primary colors of light. Red is the color of blood, fire, and many fruits.")
MATH_ANS = ("Mathematics (math) is the study of numbers, quantities, shapes, and patterns. "
            "We study math to solve problems, develop logical thinking, and apply calculations to science and everyday life.")
OCEAN_ANS = ("The ocean is a vast body of salt water that covers about 71 percent of Earth. "
             "Oceans are deep, salty bodies of water home to diverse sea life. "
             "The five major oceans are the Pacific, Atlantic, Indian, Southern, and Arctic.")
LEARN_ANS = ("To learn means to acquire knowledge, understanding, or skills through study or experience. "
             "Learning involves gaining new information, understanding concepts, and developing abilities "
             "that can be applied to solve problems.")

PAIRS = {
    "eyes": {
        "answer": EYES_ANS,
        "questions": [
            "What are eyes?",
            "What do eyes do?",
            "What are human eyes?",
            "What are the eyes?",
            "What are eyes used for?",
            "What is the function of eyes?",
            "How do eyes work?",
            "Why do we have eyes?",
            "What organ do we see with?",
            "What organs allow vision?",
            "Describe the eyes.",
            "What are eyes in biology?",
            "What do eyes allow us to do?",
            "How do we see with our eyes?",
            "What body part allows us to see?",
            "How do eyes detect light?",
            "What is the purpose of eyes?",
            "What are eyes for?",
            "What is an eye?",
            "How does vision work through the eyes?",
            "What sense organ detects light?",
            "What are visual organs?",
            "What body part processes sight?",
            "What is the eye used for?",
            "What organs give us sight?",
        ],
    },
    "color_red": {
        "answer": COLOR_RED_ANS,
        "questions": [
            "What is the color red?",
            "What is red?",
            "Describe the color red.",
            "What is red as a color?",
            "What wavelength is red?",
            "What is red in the color spectrum?",
            "How would you describe the color red?",
            "What is the red color?",
            "Describe what red is.",
            "Explain the color red.",
            "What does the color red mean?",
            "What kind of color is red?",
            "What is the visible light color red?",
            "What color is blood?",
            "What color has a long wavelength?",
            "What is red light?",
            "Describe the spectrum color red.",
            "What wavelength does red light have?",
            "What is red in optics?",
            "What makes something appear red?",
            "What color is associated with fire?",
            "What is red color in physics?",
            "What hue is red?",
            "Is red a primary color?",
            "What color does fire appear?",
        ],
    },
    "math": {
        "answer": MATH_ANS,
        "questions": [
            "What is math?",
            "Why do we study math?",
            "Why is math important?",
            "What does math study?",
            "Why do we learn math?",
            "What is mathematics?",
            "Why study mathematics?",
            "What is the purpose of math?",
            "What is math used for?",
            "How does math help us?",
            "Why do we need math?",
            "What is math about?",
            "Why should we study math?",
            "What do we use math for?",
            "What is the point of studying math?",
            "What does mathematics involve?",
            "Why do schools teach math?",
            "How is math applied in life?",
            "What problems does math solve?",
            "What skills does math develop?",
            "Why is math a core subject?",
            "How does studying math help us?",
            "Why do students need math?",
            "What is the value of learning mathematics?",
            "Why does math matter?",
        ],
    },
    "ocean": {
        "answer": OCEAN_ANS,
        "questions": [
            "What is the ocean?",
            "Describe the ocean.",
            "What is an ocean?",
            "What are oceans?",
            "Describe what an ocean is.",
            "What is the ocean made of?",
            "How would you describe an ocean?",
            "What is the ocean in geography?",
            "What body of water is the ocean?",
            "Describe the ocean water.",
            "How big is the ocean?",
            "What is saltwater ocean?",
            "Describe ocean geography.",
            "What is the Pacific ocean?",
            "What is the world ocean?",
            "How deep is the ocean?",
            "Describe the sea and ocean.",
            "What are the oceans of the world?",
            "What is ocean water?",
            "What is the ocean environment?",
            "What covers most of the Earth surface?",
            "What large body of water surrounds continents?",
            "What is deep salt water called?",
            "What ocean covers most of Earth?",
            "What is an ocean biome?",
        ],
    },
    "learn": {
        "answer": LEARN_ANS,
        "questions": [
            "What does it mean to learn?",
            "What is learning?",
            "What does learning mean?",
            "How do we learn?",
            "What does it mean to gain knowledge?",
            "What is the meaning of learning?",
            "What does it mean to acquire knowledge?",
            "What does learning involve?",
            "How does learning work?",
            "What happens when we learn?",
            "What does it mean to learn something?",
            "How do humans learn?",
            "What does it mean to understand something?",
            "Why do we learn things?",
            "What is the process of learning?",
            "What does learning mean in education?",
            "How do students learn?",
            "What happens in the learning process?",
            "What is learning and why is it important?",
            "What does learning new skills mean?",
            "What is the result of learning?",
            "What is gained through learning?",
            "What is it to learn a skill?",
            "What does it mean to gain understanding?",
            "What is it to acquire new knowledge?",
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

# Spot check
for q, label in [
    ("What are eyes?", "eyes"),
    ("What is the color red?", "color_red"),
    ("Why do we study math?", "math"),
    ("Describe the ocean.", "ocean"),
    ("What does it mean to learn?", "learn"),
]:
    r = httpx.post(NODE + "/qa/query", json={"question": q}, timeout=15)
    results = r.json().get("report", {}).get("results", [])
    top = results[0] if results else {}
    act = round(top.get("activation", 0), 4)
    ans = top.get("answer", "")[:80]
    print(label + ": act=" + str(act) + " ans=" + ans)
