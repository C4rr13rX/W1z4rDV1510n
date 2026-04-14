#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W1z4rD V1510n — Full Pipeline Trainer
======================================
Trains through EVERY architectural pathway simultaneously for each fact so
the neural fabric builds maximally robust associations:

  1. MEDIA          /media/train          image card + text spans + keyboard
  2. QA INGEST      /qa/ingest            Q→A pair, confidence 0.97
  3. EPISODE        /neuro/record_episode context labels → answer reinforcement
  4. ENTITY OBSERVE /entity/observe       concept as BehaviorInput entity →
                                          Behavior → Physiology → Survival →
                                          Narrative chain; neuro pool gets
                                          entity/species/latent/position labels
  5. DIRECT NEURO   /neuro/train          raw synonym/relation co-activation
  6. SEQUENCE       /media/train_sequence temporal chain of related concepts

Post-training probe (--bench):
  - Every question runs through /qa/query, /chat, /neuro/generate
  - Reports per-pathway accuracy (exact / partial / miss)

Usage:
  python train_full_pipeline.py                    # full training + bench
  python train_full_pipeline.py --rounds 3         # 3 training passes (default 2)
  python train_full_pipeline.py --bench-only       # skip training, run bench
  python train_full_pipeline.py --chat             # interactive chat after bench
  python train_full_pipeline.py --domain physics   # restrict to one domain

Python: C:/Python313/python.exe  (needs Pillow, httpx)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Missing: pip install Pillow")

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
NODE_URL = "http://127.0.0.1:8090"

DOMAIN_COLORS: dict[str, tuple[int, int, int]] = {
    "physics":    (30,  60, 120),
    "biology":    (20, 100,  40),
    "chemistry":  (100, 30,  80),
    "math":       (120, 80,   0),
    "cs":         (0,   80, 120),
    "earth":      (100, 60,  20),
    "anatomy":    (160,  30,  30),
    "general":    (60,  60,  60),
}

DOMAIN_INDEX: dict[str, int] = {k: i for i, k in enumerate(DOMAIN_COLORS)}

# ─────────────────────────────────────────────────────────────────────────────
# Fact dataset  — 50 curated Q/A pairs with domain and related concept chains
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Fact:
    word:     str          # short concept name (used as entity_id)
    question: str
    answer:   str
    domain:   str
    related:  list[str]    # concept words for sequence / synonym training
    synonyms: list[str]    # raw words for direct /neuro/train co-activation


FACTS: list[Fact] = [
    # ── Physics ─────────────────────────────────────────────────────────────
    Fact("gravity",
         "What is gravity?",
         "Gravity is a fundamental force that attracts objects with mass toward each other. "
         "It keeps planets in orbit and holds objects on Earth's surface.",
         "physics", ["gravity", "force", "mass", "orbit", "acceleration"],
         ["gravity", "force", "attraction", "weight", "acceleration"]),

    Fact("force",
         "What is force in physics?",
         "Force is a push or pull on an object that can change its velocity or shape. "
         "Newton's second law states F = ma, where F is force, m is mass, and a is acceleration.",
         "physics", ["force", "mass", "acceleration", "momentum", "work"],
         ["force", "push", "pull", "newton", "momentum"]),

    Fact("energy",
         "What is energy?",
         "Energy is the capacity to do work or produce heat. It exists as kinetic, potential, "
         "thermal, electrical, and chemical energy and is conserved in closed systems.",
         "physics", ["energy", "work", "heat", "power", "entropy"],
         ["energy", "work", "heat", "kinetic", "potential", "power"]),

    Fact("wave",
         "What is a wave?",
         "A wave is a disturbance that travels through space or a medium, transferring energy "
         "without transferring matter. Waves have properties of frequency, wavelength, and amplitude.",
         "physics", ["wave", "frequency", "sound", "light", "oscillation"],
         ["wave", "frequency", "amplitude", "wavelength", "oscillation"]),

    Fact("light",
         "What is light?",
         "Light is electromagnetic radiation visible to the human eye, traveling as waves and "
         "photons at approximately 3×10^8 meters per second in a vacuum.",
         "physics", ["light", "photon", "speed", "electromagnetic", "wave"],
         ["light", "photon", "electromagnetic", "radiation", "optics"]),

    Fact("sound",
         "What is sound?",
         "Sound is a mechanical wave that travels through a medium as vibrations of particles. "
         "It is perceived by the ear and travels at about 343 m/s in air.",
         "physics", ["sound", "wave", "vibration", "frequency", "ear"],
         ["sound", "vibration", "acoustic", "frequency", "pressure"]),

    Fact("entropy",
         "What is entropy?",
         "Entropy is a measure of disorder or randomness in a system. The second law of "
         "thermodynamics states that entropy in an isolated system always increases over time.",
         "physics", ["entropy", "thermodynamics", "disorder", "energy", "heat"],
         ["entropy", "disorder", "thermodynamics", "randomness", "chaos"]),

    Fact("temperature",
         "What is temperature?",
         "Temperature is a measure of the average kinetic energy of particles in a substance, "
         "determining how hot or cold it is. It is measured in Celsius, Fahrenheit, or Kelvin.",
         "physics", ["temperature", "heat", "kinetic", "entropy", "thermodynamics"],
         ["temperature", "heat", "thermal", "kinetic", "celsius", "kelvin"]),

    Fact("friction",
         "What is friction?",
         "Friction is a force that opposes the relative motion between two surfaces in contact. "
         "It is generated by surface roughness and molecular adhesion.",
         "physics", ["friction", "force", "motion", "resistance", "surface"],
         ["friction", "resistance", "motion", "surface", "drag"]),

    Fact("electricity",
         "What is electricity?",
         "Electricity is the flow of electric charge through a conductor, typically electrons. "
         "It is measured in volts, amperes, and watts, and powers most modern devices.",
         "physics", ["electricity", "electron", "current", "voltage", "circuit"],
         ["electricity", "electron", "current", "voltage", "charge", "circuit"]),

    Fact("magnetism",
         "What is magnetism?",
         "Magnetism is a force arising from the motion of electric charges, creating magnetic "
         "fields. Magnets have north and south poles that attract or repel each other.",
         "physics", ["magnetism", "electricity", "force", "field", "poles"],
         ["magnet", "magnetic", "poles", "field", "attraction", "repulsion"]),

    Fact("pressure",
         "What is pressure?",
         "Pressure is the force applied perpendicular to a surface per unit area. "
         "It is measured in pascals (Pa) and determines how liquids and gases behave.",
         "physics", ["pressure", "force", "area", "fluid", "atmosphere"],
         ["pressure", "force", "area", "pascal", "atmospheric"]),

    # ── Biology ─────────────────────────────────────────────────────────────
    Fact("cell",
         "What is a cell?",
         "A cell is the basic structural and functional unit of all living organisms. "
         "It contains a nucleus, cytoplasm, and cell membrane, and carries out life processes.",
         "biology", ["cell", "nucleus", "DNA", "protein", "organism"],
         ["cell", "nucleus", "membrane", "cytoplasm", "organelle"]),

    Fact("dna",
         "What is DNA?",
         "DNA (deoxyribonucleic acid) is a molecule that carries genetic information in all "
         "living organisms. It is structured as a double helix of nucleotide base pairs.",
         "biology", ["dna", "gene", "chromosome", "protein", "cell"],
         ["dna", "genetic", "nucleotide", "helix", "genome", "base"]),

    Fact("gene",
         "What are genes?",
         "Genes are segments of DNA that encode instructions for making proteins, "
         "determining inherited characteristics passed from parents to offspring.",
         "biology", ["gene", "dna", "chromosome", "protein", "evolution"],
         ["gene", "dna", "heredity", "allele", "trait", "inheritance"]),

    Fact("evolution",
         "What is evolution?",
         "Evolution is the process of genetic change in populations over successive generations, "
         "driven by natural selection, mutation, genetic drift, and gene flow.",
         "biology", ["evolution", "gene", "species", "selection", "mutation"],
         ["evolution", "natural", "selection", "mutation", "species", "adaptation"]),

    Fact("photosynthesis",
         "What is photosynthesis?",
         "Photosynthesis is the process by which plants, algae, and some bacteria convert "
         "sunlight, water, and carbon dioxide into glucose and oxygen.",
         "biology", ["photosynthesis", "chloroplast", "glucose", "sunlight", "oxygen"],
         ["photosynthesis", "chlorophyll", "glucose", "sunlight", "oxygen", "plant"]),

    Fact("protein",
         "What is a protein?",
         "A protein is a large molecule composed of amino acids that performs essential "
         "biological functions including catalysis, structure, signaling, and transport.",
         "biology", ["protein", "amino", "enzyme", "dna", "cell"],
         ["protein", "amino", "enzyme", "folding", "structure", "catalyst"]),

    Fact("neuron",
         "What is a neuron?",
         "A neuron is a nerve cell that transmits electrical and chemical signals throughout "
         "the body. It forms the basic unit of the nervous system.",
         "biology", ["neuron", "synapse", "brain", "signal", "nervous"],
         ["neuron", "nerve", "synapse", "signal", "axon", "dendrite"]),

    Fact("vaccine",
         "What is a vaccine?",
         "A vaccine is a biological preparation that trains the immune system to recognize "
         "and fight a specific pathogen, preventing disease.",
         "biology", ["vaccine", "immune", "antibody", "virus", "disease"],
         ["vaccine", "immune", "antibody", "pathogen", "protection", "immunity"]),

    # ── Chemistry ────────────────────────────────────────────────────────────
    Fact("atom",
         "What is an atom?",
         "An atom is the smallest unit of matter that retains the chemical properties of "
         "an element, consisting of a nucleus of protons and neutrons surrounded by electrons.",
         "chemistry", ["atom", "electron", "nucleus", "element", "molecule"],
         ["atom", "proton", "neutron", "electron", "element", "nucleus"]),

    Fact("molecule",
         "What is a molecule?",
         "A molecule is a group of two or more atoms bonded together, representing the "
         "smallest unit of a chemical compound that retains its properties.",
         "chemistry", ["molecule", "atom", "bond", "compound", "chemistry"],
         ["molecule", "atom", "bond", "compound", "chemical"]),

    Fact("acid",
         "What is an acid?",
         "An acid is a substance that donates hydrogen ions (H+) in solution, has a pH "
         "below 7, and typically reacts with bases to form salts and water.",
         "chemistry", ["acid", "base", "ph", "ion", "reaction"],
         ["acid", "ph", "hydrogen", "proton", "corrosive", "base"]),

    Fact("catalyst",
         "What is a catalyst?",
         "A catalyst is a substance that increases the rate of a chemical reaction without "
         "being consumed, by lowering the activation energy required.",
         "chemistry", ["catalyst", "reaction", "enzyme", "activation", "chemistry"],
         ["catalyst", "reaction", "activation", "energy", "enzyme", "speed"]),

    Fact("osmosis",
         "What is osmosis?",
         "Osmosis is the passive movement of water molecules through a semipermeable "
         "membrane from a region of lower solute concentration to higher concentration.",
         "chemistry", ["osmosis", "membrane", "concentration", "water", "diffusion"],
         ["osmosis", "membrane", "diffusion", "concentration", "gradient"]),

    # ── Earth Science ────────────────────────────────────────────────────────
    Fact("atmosphere",
         "What is the atmosphere?",
         "The atmosphere is the layer of gases surrounding Earth, composed mainly of "
         "nitrogen (78%) and oxygen (21%). It protects life from radiation and regulates temperature.",
         "earth", ["atmosphere", "oxygen", "nitrogen", "climate", "pressure"],
         ["atmosphere", "nitrogen", "oxygen", "climate", "stratosphere"]),

    Fact("volcano",
         "What is a volcano?",
         "A volcano is an opening in Earth's crust through which molten rock (magma), "
         "volcanic ash, and gases escape to the surface during an eruption.",
         "earth", ["volcano", "magma", "crust", "eruption", "tectonic"],
         ["volcano", "magma", "eruption", "lava", "crust", "tectonic"]),

    Fact("erosion",
         "What is erosion?",
         "Erosion is the process by which rock and soil are worn away and transported "
         "by wind, water, ice, or gravity to different locations.",
         "earth", ["erosion", "weathering", "water", "soil", "rock"],
         ["erosion", "weathering", "sediment", "rock", "water", "wind"]),

    Fact("tectonic",
         "What are tectonic plates?",
         "Tectonic plates are large rigid segments of Earth's lithosphere that slowly move "
         "on the underlying mantle, causing earthquakes, volcanoes, and mountain formation.",
         "earth", ["tectonic", "earthquake", "volcano", "mantle", "crust"],
         ["tectonic", "plate", "earthquake", "fault", "subduction", "mantle"]),

    Fact("ocean",
         "What is the ocean?",
         "The ocean is the vast body of saltwater covering about 71% of Earth's surface. "
         "It regulates climate, produces oxygen, and supports the majority of life on Earth.",
         "earth", ["ocean", "water", "climate", "salinity", "marine"],
         ["ocean", "sea", "saltwater", "marine", "depth", "current"]),

    # ── Math ─────────────────────────────────────────────────────────────────
    Fact("prime",
         "What is a prime number?",
         "A prime number is a natural number greater than 1 that has no positive divisors "
         "other than 1 and itself. Examples include 2, 3, 5, 7, and 11.",
         "math", ["prime", "number", "divisor", "factor", "integer"],
         ["prime", "number", "divisor", "factor", "integer", "composite"]),

    Fact("pi",
         "What is pi?",
         "Pi (π) is the ratio of a circle's circumference to its diameter, approximately "
         "equal to 3.14159. It is an irrational and transcendental number.",
         "math", ["pi", "circle", "circumference", "geometry", "ratio"],
         ["pi", "circle", "circumference", "diameter", "ratio", "irrational"]),

    Fact("derivative",
         "What is a derivative in calculus?",
         "A derivative is the instantaneous rate of change of a function with respect to "
         "a variable, measuring the slope of the function at a given point.",
         "math", ["derivative", "calculus", "function", "slope", "integral"],
         ["derivative", "calculus", "slope", "rate", "change", "differential"]),

    Fact("vector",
         "What is a vector?",
         "A vector is a mathematical object with both magnitude and direction, "
         "used to represent quantities like velocity, force, and displacement.",
         "math", ["vector", "magnitude", "direction", "force", "physics"],
         ["vector", "magnitude", "direction", "scalar", "displacement"]),

    Fact("probability",
         "What is probability?",
         "Probability is a measure of the likelihood that an event will occur, expressed "
         "as a number between 0 (impossible) and 1 (certain).",
         "math", ["probability", "statistics", "likelihood", "event", "random"],
         ["probability", "likelihood", "event", "statistics", "chance", "random"]),

    Fact("algorithm",
         "What is an algorithm?",
         "An algorithm is a finite, step-by-step set of instructions for solving a problem "
         "or accomplishing a task, used in mathematics and computer science.",
         "math", ["algorithm", "computation", "logic", "binary", "program"],
         ["algorithm", "procedure", "step", "computation", "logic", "instructions"]),

    # ── Computer Science ──────────────────────────────────────────────────────
    Fact("binary",
         "What is binary code?",
         "Binary code represents data using only two digits: 0 and 1. All digital computers "
         "ultimately process information as sequences of binary digits (bits).",
         "cs", ["binary", "bit", "data", "computer", "digit"],
         ["binary", "bit", "digit", "zero", "one", "digital", "computation"]),

    Fact("encryption",
         "What is encryption?",
         "Encryption is the process of converting readable information into a coded form "
         "that can only be decoded with the correct key, protecting data from unauthorized access.",
         "cs", ["encryption", "security", "key", "data", "privacy"],
         ["encryption", "decryption", "key", "cipher", "security", "privacy"]),

    Fact("machine_learning",
         "What is machine learning?",
         "Machine learning is a branch of artificial intelligence where systems learn from "
         "data to improve performance on tasks without being explicitly programmed.",
         "cs", ["machine_learning", "ai", "data", "training", "neural"],
         ["machine", "learning", "artificial", "intelligence", "training", "model"]),

    Fact("network",
         "What is a computer network?",
         "A computer network is a set of interconnected devices that communicate to share "
         "resources and information, using protocols like TCP/IP.",
         "cs", ["network", "internet", "protocol", "data", "communication"],
         ["network", "internet", "protocol", "connection", "data", "communication"]),

    # ── Anatomy ──────────────────────────────────────────────────────────────
    Fact("brain",
         "What is the brain?",
         "The brain is the central organ of the nervous system, controlling thought, memory, "
         "emotion, motor skills, vision, breathing, and all body processes.",
         "anatomy", ["brain", "neuron", "nervous", "memory", "cognition"],
         ["brain", "neuron", "cortex", "nervous", "cognitive", "memory"]),

    Fact("heart",
         "What is the heart?",
         "The heart is a muscular organ that pumps blood through the circulatory system, "
         "supplying the body with oxygen and nutrients and removing waste.",
         "anatomy", ["heart", "blood", "circulation", "oxygen", "muscle"],
         ["heart", "blood", "circulation", "pump", "artery", "vein"]),

    Fact("digestion",
         "What is digestion?",
         "Digestion is the process of breaking down food into smaller molecules that can "
         "be absorbed into the bloodstream and used for energy and growth.",
         "anatomy", ["digestion", "stomach", "enzyme", "absorption", "nutrition"],
         ["digestion", "stomach", "enzyme", "intestine", "absorption", "nutrition"]),

    Fact("immune",
         "What is the immune system?",
         "The immune system is a complex network of cells, tissues, and organs that defend "
         "the body against pathogens including bacteria, viruses, and parasites.",
         "anatomy", ["immune", "antibody", "vaccine", "cell", "pathogen"],
         ["immune", "antibody", "defense", "pathogen", "lymphocyte", "innate"]),

    # ── General Science ───────────────────────────────────────────────────────
    Fact("black_hole",
         "What is a black hole?",
         "A black hole is a region of spacetime where gravity is so strong that nothing, "
         "not even light, can escape from beyond its event horizon.",
         "general", ["black_hole", "gravity", "spacetime", "light", "singularity"],
         ["black", "hole", "gravity", "singularity", "spacetime", "horizon"]),

    Fact("photon",
         "What is a photon?",
         "A photon is the fundamental particle of light and all electromagnetic radiation. "
         "It carries energy and momentum but has no mass and travels at the speed of light.",
         "general", ["photon", "light", "energy", "wave", "electromagnetic"],
         ["photon", "light", "quantum", "energy", "particle", "radiation"]),

    Fact("chromosome",
         "What is a chromosome?",
         "A chromosome is a thread-like structure of DNA and proteins that carries genetic "
         "information. Humans have 46 chromosomes in 23 pairs.",
         "general", ["chromosome", "dna", "gene", "cell", "nucleus"],
         ["chromosome", "dna", "gene", "karyotype", "pairs", "genetic"]),

    Fact("mitosis",
         "What is mitosis?",
         "Mitosis is the type of cell division that produces two genetically identical "
         "daughter cells, each containing the same number of chromosomes as the parent cell.",
         "general", ["mitosis", "cell", "division", "chromosome", "dna"],
         ["mitosis", "cell", "division", "replication", "chromosome", "growth"]),

    Fact("pythagorean",
         "What is the Pythagorean theorem?",
         "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse "
         "equals the sum of the squares of the other two sides: a² + b² = c².",
         "general", ["pythagorean", "triangle", "geometry", "hypotenuse", "math"],
         ["pythagorean", "triangle", "hypotenuse", "geometry", "right", "square"]),
]

# Temporal relationship chains (for /media/train_sequence training)
CHAINS: list[list[str]] = [
    ["atom", "molecule", "compound", "reaction", "catalyst", "chemistry"],
    ["dna", "gene", "chromosome", "cell", "organism", "evolution"],
    ["photosynthesis", "glucose", "energy", "cell", "organism"],
    ["neuron", "brain", "signal", "nervous", "behavior", "cognition"],
    ["force", "mass", "acceleration", "momentum", "energy", "work"],
    ["electron", "electricity", "current", "voltage", "power", "circuit"],
    ["temperature", "heat", "entropy", "thermodynamics"],
    ["water", "erosion", "ocean", "atmosphere", "climate", "tectonic"],
    ["prime", "number", "probability", "statistics", "algorithm", "binary"],
    ["vaccine", "immune", "antibody", "cell", "protein", "dna"],
    ["gravity", "black_hole", "spacetime", "light", "photon"],
    ["derivative", "calculus", "vector", "pi", "geometry", "pythagorean"],
    ["network", "encryption", "binary", "machine_learning", "algorithm"],
    ["heart", "blood", "brain", "neuron", "digestion", "immune"],
    ["sound", "wave", "frequency", "light", "photon", "energy"],
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def tokenize(text: str) -> list[str]:
    stop = {"a","an","the","is","are","was","were","be","been","do","does","did",
            "will","would","can","could","what","which","who","how","when","where",
            "why","of","to","in","for","on","with","at","by","from","about","into",
            "that","this","there","here","have","has","had","it","its","i","you",
            "he","she","we","they","my","your","or","and","but","as","all","any",
            "each","used","using","called","known","given","through","between"}
    words = re.findall(r"[a-z]+", text.lower())
    return [w for w in words if w not in stop and len(w) > 2]

def keyboard_events(text: str, start_t: float = 0.0, delay: float = 0.09) -> list:
    events = []
    t = start_t
    for ch in text[:120]:
        events.append({"key": ch, "t_secs": round(t, 3), "action": "down"})
        t += delay
    return events

def make_concept_card(fact: Fact, width: int = 480, height: int = 320) -> bytes:
    """Generate a colour-coded concept card image as JPEG bytes."""
    color = DOMAIN_COLORS.get(fact.domain, (60, 60, 60))
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)

    # light panel
    panel_color = tuple(min(255, c + 160) for c in color)
    draw.rectangle([20, 20, width - 20, height - 20], fill=panel_color)

    # text
    try:
        font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 36)
        font_body  = ImageFont.truetype("C:/Windows/Fonts/arial.ttf",   18)
    except Exception:
        font_title = ImageFont.load_default()
        font_body  = font_title

    # title
    title = fact.word.replace("_", " ").upper()
    draw.text((width // 2, 55), title, fill=(20, 20, 20),
              font=font_title, anchor="mm")

    # domain badge
    draw.text((width - 28, 28), fact.domain.upper(), fill=color,
              font=font_body, anchor="ra")

    # answer preview (first 160 chars, wrapped at ~55 chars)
    preview = fact.answer[:160]
    words = preview.split()
    lines, line = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 52:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    y = 100
    for ln in lines[:5]:
        draw.text((width // 2, y), ln, fill=(30, 30, 30),
                  font=font_body, anchor="mm")
        y += 26

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _post(client: httpx.Client, path: str, payload: dict,
          label: str = "", retries: int = 2, timeout: float = 30) -> dict | None:
    url = f"{NODE_URL}{path}"
    for attempt in range(retries + 1):
        try:
            r = client.post(url, json=payload, timeout=timeout)
            if r.status_code == 429:
                time.sleep(3 + attempt * 2)
                continue
            if r.status_code >= 500:
                time.sleep(1)
                continue
            return r.json()
        except Exception as e:
            if attempt == retries:
                print(f"  [ERR] {label}: {e}")
    return None

def _get(client: httpx.Client, path: str, params: dict | None = None,
         timeout: float = 15) -> dict | None:
    try:
        r = client.get(f"{NODE_URL}{path}", params=params, timeout=timeout)
        return r.json()
    except Exception:
        return None


def api_media_train(client: httpx.Client, card: bytes, fact: Fact, lr: float = 1.2) -> bool:
    spans = [
        {"text": fact.word.replace("_", " "), "role": "heading", "bold": True,
         "italic": False, "size_ratio": 1.8, "indent": 0,
         "x_frac": 0.5, "y_frac": 0.06, "seq_index": 0, "seq_total": 1},
    ]
    sentences = [s.strip() for s in fact.answer.replace(".", ".|||").split("|||") if s.strip()]
    for i, sent in enumerate(sentences[:4]):
        spans.append({
            "text": sent, "role": "body", "bold": False, "italic": False,
            "size_ratio": 1.0, "indent": 0,
            "x_frac": 0.5, "y_frac": 0.22 + 0.16 * i,
            "seq_index": i + 1, "seq_total": len(sentences) + 1,
        })
    result = _post(client, "/media/train", {
        "modality": "page",
        "data_b64": b64(card),
        "text": f"{fact.word}. {fact.answer}",
        "spans": spans,
        "keys": keyboard_events(fact.word + " " + fact.answer[:60]),
        "lr_scale": lr,
    }, label=f"{fact.word}:media")
    return result is not None


def api_qa_ingest(client: httpx.Client, fact: Fact) -> bool:
    qa_id = hashlib.sha256(
        f"pipeline|{fact.word}|{fact.answer}".encode()).hexdigest()[:16]
    result = _post(client, "/qa/ingest", {"candidates": [{
        "qa_id": qa_id,
        "question": fact.question,
        "answer": fact.answer,
        "book_id": f"pipeline_{fact.domain}",
        "page_index": 0,
        "confidence": 0.97,
        "evidence": fact.answer,
        "review_status": "VERIFIED",
    }]}, label=f"{fact.word}:qa")
    return result is not None


def api_episode(client: httpx.Client, fact: Fact) -> None:
    ctx = [f"txt:word_{w}" for w in fact.synonyms[:4]]
    ctx.append(f"entity:{fact.word}")
    _post(client, "/neuro/record_episode", {
        "context_labels": ctx,
        "predicted": fact.answer[:120],
        "actual": fact.answer[:120],
        "streams": [fact.domain, "qa_pipeline"],
        "surprise": 0.0,
    }, label=f"{fact.word}:episode")


def api_entity_observe(client: httpx.Client, fact: Fact, ts: int) -> dict | None:
    """
    Encode the concept as a BehaviorInput so it flows through the full
    Behavior → Physiology → Survival → Narrative runtime chain.

    Sensor encoding rationale:
      PHYSIOLOGY  arousal  = conceptual abstraction (0=concrete, 1=abstract)
                  valence  = certainty level (high for formulaic facts)
      ENVIRONMENT word presence bits for top domain keywords
      MOTION      semantic-space x/y derived from domain index and word hash
      VOICE       definition length (normalized) — how "verbose" this concept is

    This gives the neuro pool a second independent pathway:
      entity:{word}, entity:{word}:dim{n}:{bin}, entity:{word}:x*, y*, z*
    """
    dom_idx = DOMAIN_INDEX.get(fact.domain, 0)
    dom_norm = dom_idx / max(1, len(DOMAIN_INDEX) - 1)

    # abstraction: longer answers = more abstract
    abstraction = min(1.0, len(fact.answer) / 500.0)

    # certainty: facts with '=' or formulas are very certain
    certainty = 0.9 if any(c in fact.answer for c in ["=", "²", "³", "%", "km/s"]) else 0.6

    # semantic x/y position from domain + first word hash
    wh = int(hashlib.md5(fact.word.encode()).hexdigest()[:8], 16)
    x_pos = ((dom_idx * 40) + (wh % 20)) - 140.0
    y_pos = ((wh >> 8) % 80) - 40.0
    z_pos = abstraction * 20.0

    # environment: word-presence signals for top content words
    answer_words = tokenize(fact.answer.lower())
    env_vals = {}
    for w in fact.synonyms[:6]:
        env_vals[w] = 1.0 if w in answer_words else 0.2
    # clip any key > 32 chars (node rejects overly long sensor key names)
    env_vals = {k[:32]: v for k, v in env_vals.items()}

    sensors = [
        {
            "kind": "PHYSIOLOGY",
            "values": {"arousal": abstraction, "valence": certainty,
                       "complexity": min(1.0, len(fact.synonyms) / 8.0)},
            "quality": 0.9,
        },
        {
            "kind": "ENVIRONMENT",
            "values": env_vals if env_vals else {"signal": dom_norm},
            "quality": 0.8,
        },
        {
            "kind": "MOTION",
            "values": {"velocity_x": 0.0, "velocity_y": 0.0, "velocity_z": 0.0},
            "quality": 1.0,
        },
        {
            "kind": "VOICE",
            "values": {"intensity": min(1.0, len(fact.answer) / 400.0)},
            "quality": 0.7,
        },
    ]

    payload = {
        "entity_id": fact.word,
        "timestamp": {"unix": ts},
        "species": "CONCEPT",   # → SpeciesKind::Other("CONCEPT")
        "sensors": sensors,
        "actions": [{
            "kind": "INTERACTION",
            "values": {"strength": certainty, "target_count": float(len(fact.related))},
            "quality": 0.85,
        }],
        "metadata": {"domain": fact.domain, "question": fact.question[:80]},
    }
    return _post(client, "/entity/observe", payload,
                 label=f"{fact.word}:entity", timeout=20)


def api_neuro_train(client: httpx.Client, labels: list[str], lr: float = 1.1) -> bool:
    result = _post(client, "/neuro/train",
                   {"labels": labels, "lr_scale": lr},
                   label="neuro_train")
    return result is not None


def api_train_sequence(client: httpx.Client, frames: list[dict]) -> bool:
    result = _post(client, "/media/train_sequence",
                   {"frames": frames, "tau_secs": 2.5},
                   label="sequence", timeout=45)
    return result is not None


def api_checkpoint(client: httpx.Client) -> None:
    _post(client, "/qa/checkpoint", {}, label="qa_checkpoint", timeout=40)


# ─────────────────────────────────────────────────────────────────────────────
# Training phases
# ─────────────────────────────────────────────────────────────────────────────

def train_fact(fact: Fact, card: bytes, client: httpx.Client,
               ts_base: int, idx: int, verbose: bool = False) -> dict:
    ts = ts_base + idx
    ok = {"media": False, "qa": False, "entity": False, "neuro": False, "ep": False}

    # 1. Media (image + text spans + keyboard)
    ok["media"] = api_media_train(client, card, fact)

    # 2. QA ingest
    ok["qa"] = api_qa_ingest(client, fact)

    # 3. Episode
    api_episode(client, fact)
    ok["ep"] = True

    # 4. Entity observe (runs through all 4 runtime layers)
    entity_result = api_entity_observe(client, fact, ts)
    ok["entity"] = entity_result is not None

    # 5. Direct neuro — synonym/relation co-activation
    labels = [f"txt:word_{w}" for w in fact.synonyms]
    labels.append(f"entity:{fact.word}")
    ok["neuro"] = api_neuro_train(client, labels)

    if verbose:
        flags = " ".join(k for k, v in ok.items() if v)
        miss  = " ".join(k for k, v in ok.items() if not v)
        ent_conf = (entity_result or {}).get("confidence", 0)
        print(f"    ok=[{flags}]" +
              (f" miss=[{miss}]" if miss else "") +
              (f" entity_conf={ent_conf:.3f}" if ent_conf else ""))
    return ok


def build_chain_frames(chain: list[str], fact_map: dict[str, Fact],
                       card_map: dict[str, bytes]) -> list[dict]:
    frames = []
    t = 0.0
    for word in chain:
        fact = fact_map.get(word)
        if not fact:
            continue
        frame: dict = {
            "t_secs": round(t, 2),
            "modality": "page" if word in card_map else "text",
            "text": f"{word}: {fact.answer[:80]}",
            "spans": [{
                "text": word.replace("_", " "), "role": "label",
                "bold": True, "italic": False, "size_ratio": 1.5,
                "indent": 0, "x_frac": 0.5, "y_frac": 0.5,
                "seq_index": 0, "seq_total": 1,
            }],
            "lr_scale": 0.9,
        }
        if word in card_map:
            frame["data_b64"] = b64(card_map[word])
        frames.append(frame)
        t += 2.0
    return frames


def run_training(facts: list[Fact], client: httpx.Client,
                 rounds: int, verbose: bool) -> None:
    ts_base = int(time.time())
    fact_map  = {f.word: f for f in facts}
    card_map  = {}

    print(f"\nGenerating {len(facts)} concept cards...")
    for f in facts:
        card_map[f.word] = make_concept_card(f)

    for rnd in range(rounds):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd+1}/{rounds} — {len(facts)} facts, 5 pathways each")
        print(f"{'='*60}")

        # ── Per-fact training ────────────────────────────────────────────────
        total_ok = {"media": 0, "qa": 0, "entity": 0, "neuro": 0, "ep": 0}
        for i, fact in enumerate(facts):
            if (i % 10 == 0) or verbose:
                print(f"  [{i+1:2d}/{len(facts)}] {fact.word:<22} ({fact.domain})")
            ok = train_fact(fact, card_map[fact.word], client,
                            ts_base + rnd * 10000, i, verbose=verbose)
            for k in total_ok:
                if ok.get(k):
                    total_ok[k] += 1
            # tiny sleep so the node isn't overwhelmed
            time.sleep(0.04)

        print(f"\n  Pathway hits: " +
              " | ".join(f"{k}:{v}/{len(facts)}" for k, v in total_ok.items()))

        # ── Synonym relation groups (direct neuro) ───────────────────────────
        print("\n  Direct synonym/relation batches...")
        relation_groups = _build_relation_groups(facts)
        rg_ok = 0
        for grp in relation_groups:
            if api_neuro_train(client, grp, lr=1.05):
                rg_ok += 1
        print(f"    {rg_ok}/{len(relation_groups)} groups trained")

        # ── Temporal chains ──────────────────────────────────────────────────
        print("\n  Temporal concept chains...")
        ch_ok = 0
        for chain in CHAINS:
            frames = build_chain_frames(chain, fact_map, card_map)
            if len(frames) >= 2 and api_train_sequence(client, frames):
                ch_ok += 1
        print(f"    {ch_ok}/{len(CHAINS)} chains trained")

    # ── Checkpoint ──────────────────────────────────────────────────────────
    print("\nSaving QA store...")
    api_checkpoint(client)
    print("  Done.")


def _build_relation_groups(facts: list[Fact]) -> list[list[str]]:
    """
    Build cross-concept label groups for direct neuro co-activation.
    Group facts by domain so domain-concept neurons co-fire.
    Also emit per-fact synonym bundles.
    """
    groups: list[list[str]] = []

    # per-fact: synonym bundle
    for f in facts:
        bundle = [f"txt:word_{w}" for w in f.synonyms[:6]]
        bundle.append(f"entity:{f.word}")
        groups.append(bundle)

    # per-domain: entity-level co-activation
    by_domain: dict[str, list[str]] = {}
    for f in facts:
        by_domain.setdefault(f.domain, []).append(f"entity:{f.word}")
    for dom, entity_labels in by_domain.items():
        # train domain entities together in pairs/triples
        for i in range(0, len(entity_labels), 3):
            grp = entity_labels[i:i+3]
            grp.append(f"txt:word_{dom}")
            if len(grp) >= 2:
                groups.append(grp)

    # related chains: co-activate words within each related list
    for f in facts:
        related_words = [f"txt:word_{w}" for w in f.related[:5]]
        if len(related_words) >= 2:
            groups.append(related_words)

    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark probe
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()

def _score_answer(expected: str, got: str) -> tuple[str, float]:
    """Return ('exact'|'partial'|'miss', score 0-1)."""
    if not got:
        return "miss", 0.0
    exp_words = set(_normalize(expected))
    got_words = set(_normalize(got))
    overlap = exp_words & got_words
    if not exp_words:
        return "miss", 0.0
    ratio = len(overlap) / len(exp_words)
    if ratio >= 0.60:
        return "exact", ratio
    elif ratio >= 0.20:
        return "partial", ratio
    return "miss", ratio


def qa_query(client: httpx.Client, question: str) -> tuple[str, float]:
    data = _post(client, "/qa/query", {"question": question}, timeout=12) or {}
    results = data.get("report", {}).get("results", [])
    if not results:
        return "", 0.0
    best = max(results, key=lambda r: r.get("confidence", 0))
    return best.get("answer", ""), best.get("confidence", 0.0)


def chat_query(client: httpx.Client, question: str) -> str:
    data = _post(client, "/chat", {"question": question}, timeout=15) or {}
    return data.get("answer", "") or data.get("response", "")


def generate_query(client: httpx.Client, question: str) -> str:
    data = _post(client, "/neuro/generate",
                 {"text": question, "max_tokens": 20, "hops": 3, "min_strength": 0.04},
                 timeout=15) or {}
    return data.get("response", "")


def run_bench(facts: list[Fact], client: httpx.Client) -> None:
    print(f"\n{'='*60}")
    print(f"BENCHMARK — {len(facts)} questions × 3 pathways")
    print(f"{'='*60}")
    print(f"{'Concept':<22} {'QA':>6} {'Chat':>6} {'Gen':>6}")
    print("-" * 44)

    totals = {"qa": [], "chat": [], "gen": []}

    by_domain: dict[str, list] = {}
    for fact in facts:
        by_domain.setdefault(fact.domain, []).append(fact)

    for domain, domain_facts in sorted(by_domain.items()):
        print(f"\n  [{domain.upper()}]")
        for fact in domain_facts:
            qa_ans, qa_conf  = qa_query(client, fact.question)
            chat_ans         = chat_query(client, fact.question)
            gen_ans          = generate_query(client, fact.question)

            qa_grade,   qa_score   = _score_answer(fact.answer, qa_ans)
            chat_grade, chat_score = _score_answer(fact.answer, chat_ans)
            gen_grade,  gen_score  = _score_answer(fact.answer, gen_ans)

            totals["qa"].append(qa_score)
            totals["chat"].append(chat_score)
            totals["gen"].append(gen_score)

            def _fmt(grade, score, conf=None):
                sym = {"exact": "✓", "partial": "~", "miss": "✗"}.get(grade, "?")
                c = f"{conf:.2f}" if conf is not None else f"{score:.2f}"
                return f"{sym}{c:>5}"

            print(f"  {fact.word:<22} {_fmt(qa_grade,qa_score,qa_conf):>6} "
                  f"{_fmt(chat_grade,chat_score):>6} {_fmt(gen_grade,gen_score):>6}")
            time.sleep(0.02)

    # Summary
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    print(f"\n{'─'*44}")
    print(f"  {'MEAN SCORE':<22} "
          f"{'QA':>6} {'Chat':>6} {'Gen':>6}")
    print(f"  {'':22} "
          f"{avg(totals['qa']):>6.3f} "
          f"{avg(totals['chat']):>6.3f} "
          f"{avg(totals['gen']):>6.3f}")

    # Exact/partial/miss counts
    def counts(scores):
        e = sum(1 for s in scores if s >= 0.60)
        p = sum(1 for s in scores if 0.20 <= s < 0.60)
        m = sum(1 for s in scores if s < 0.20)
        return e, p, m
    for pathway in ("qa", "chat", "gen"):
        e, p, m = counts(totals[pathway])
        n = len(totals[pathway])
        print(f"  {pathway.upper():<22} exact={e}/{n} partial={p}/{n} miss={m}/{n}")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive chat
# ─────────────────────────────────────────────────────────────────────────────

def run_chat(client: httpx.Client) -> None:
    try:
        import readline  # noqa
    except ImportError:
        pass

    print("\n" + "="*60)
    print("Interactive Chat — type /quit to exit, /help for commands")
    print("="*60 + "\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not raw:
            continue
        if raw in ("/quit", "/q", "/exit"):
            print("Bye!")
            break
        if raw == "/help":
            print("  /qa     <q>   — query QA runtime directly")
            print("  /gen    <q>   — auto-regressive generation")
            print("  /quit         — exit")
            continue
        if raw.startswith("/qa "):
            q = raw[4:]
            ans, conf = qa_query(client, q)
            print(f"  QA [{conf:.2f}]: {ans or '(no result)'}\n")
            continue
        if raw.startswith("/gen "):
            q = raw[5:]
            ans = generate_query(client, q)
            print(f"  GEN: {ans or '(no result)'}\n")
            continue

        # Default: use /chat
        qa_ans, qa_conf = qa_query(client, raw)
        chat_ans = chat_query(client, raw)

        if qa_conf >= 0.4 and qa_ans:
            print(f"\nW1z4rD [{qa_conf:.2f}]: {qa_ans}\n")
        elif chat_ans:
            print(f"\nW1z4rD: {chat_ans}\n")
        else:
            gen = generate_query(client, raw)
            if gen:
                print(f"\nW1z4rD (gen): {gen}\n")
            else:
                print("\nW1z4rD: I don't have a clear answer to that yet.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="W1z4rD Full Pipeline Trainer")
    parser.add_argument("--rounds",      type=int,   default=2,
                        help="Training passes over the full dataset (default 2)")
    parser.add_argument("--bench-only",  action="store_true",
                        help="Skip training, run benchmark only")
    parser.add_argument("--chat",        action="store_true",
                        help="Launch interactive chat after benchmark")
    parser.add_argument("--no-bench",    action="store_true",
                        help="Skip benchmark probe")
    parser.add_argument("--domain",      default=None,
                        help="Restrict to one domain (physics|biology|chemistry|...)")
    parser.add_argument("--verbose",     action="store_true",
                        help="Show per-pathway status for every fact")
    parser.add_argument("--node",        default=NODE_URL,
                        help=f"Node URL (default: {NODE_URL})")
    args = parser.parse_args()

    global NODE_URL
    NODE_URL = args.node

    facts = FACTS
    if args.domain:
        facts = [f for f in facts if f.domain == args.domain]
        if not facts:
            sys.exit(f"No facts found for domain '{args.domain}'. "
                     f"Valid: {sorted(set(f.domain for f in FACTS))}")

    print(f"\nW1z4rD Full Pipeline Trainer")
    print(f"  Node:    {NODE_URL}")
    print(f"  Facts:   {len(facts)}")
    print(f"  Rounds:  {args.rounds}")
    print(f"  Domain:  {args.domain or 'all'}")

    with httpx.Client() as client:
        # Health check
        try:
            health = client.get(f"{NODE_URL}/health", timeout=6).json()
            if health.get("status") != "OK":
                sys.exit("Node not healthy")
            print(f"  Node ID: {health.get('node_id')}  "
                  f"uptime={health.get('uptime_secs')}s\n")
        except Exception as e:
            sys.exit(f"Cannot reach node at {NODE_URL}: {e}")

        if not args.bench_only:
            run_training(facts, client, rounds=args.rounds, verbose=args.verbose)

        if not args.no_bench:
            run_bench(facts, client)

        if args.chat:
            run_chat(client)


if __name__ == "__main__":
    main()
