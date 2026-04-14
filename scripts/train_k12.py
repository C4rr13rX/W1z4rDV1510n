#!/usr/bin/env python3
"""
W1z4rD V1510n — K-12 Staged Training Pipeline
===============================================
Trains the neural fabric through three stages:

  Stage 0  Toddler foundations — basic vocabulary, concepts, associations
  Stage 1  Language / introductory — accessible textbooks from the collection
  Stage 2  Full K-12 curriculum — all remaining LibreTexts PDFs

Each stage uses:
  /media/train         — multimodal page training (image + text spans, or plain text)
  /media/train_sequence — STDP-ordered temporal sequence training
  /qa/ingest           — Q&A pair batch ingestion
  /neuro/checkpoint    — periodic pool persistence

Architecture notes (updated for current build):
  - kWTA (top 2% per hop) enforces cortical sparsity — training populates sparse codes
  - STDP is asymmetric: earlier tokens in a sequence become pre-synaptic (LTP on fwd edge,
    LTD on backward edge). Stage 0 sequences are ordered causal→effect for this reason.
  - Homeostatic scaling runs every 500 steps — repeated passes improve consolidation
    without causing saturation; 3 passes over Stage 0 is intentional.
  - Neuromodulator-gated LR: effective_lr = lr_scale × ACh × (1 + NE × 2.0).
    lr_scale here is the base before neuromodulator amplification.

Usage:
  python train_k12.py [--node http://localhost:8080] [--stages 0,1,2]
  python train_k12.py --stages 0              # toddler only
  python train_k12.py --stages 1,2            # language + full curriculum
  python train_k12.py --max-books 10          # limit books per stage for testing
  python train_k12.py --resume                # skip already-processed books
  python train_k12.py --clear-progress        # wipe progress file, start clean
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing: pip install pymupdf")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get("total", "?")
        for i, x in enumerate(it):
            print(f"  [{i+1}/{total}]", end="\r", flush=True)
            yield x
        print()

ROOT = Path(__file__).resolve().parents[1]
TEXTBOOKS_DIR = ROOT.parent / "StateOfLoci" / "textbooks"
PROGRESS_FILE = ROOT / "data" / "k12_progress.json"

# -- PDF rendering --------------------------------------------------------------

DPI = 96           # low but sufficient for visual zone detection; keeps payloads small
JPEG_QUALITY = 55  # enough for hue/edge/zone features; target <200 KB per page

def render_page_jpeg(page: "fitz.Page") -> bytes:
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)

def extract_spans(page: "fitz.Page") -> list[dict]:
    pw, ph = page.rect.width, page.rect.height
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    raw = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                cx = ((bbox[0] + bbox[2]) / 2) / max(pw, 1)
                cy = ((bbox[1] + bbox[3]) / 2) / max(ph, 1)
                raw.append({"text": text, "size": span.get("size", 12.0),
                             "bold": bool(span.get("flags", 0) & 16),
                             "italic": bool(span.get("flags", 0) & 2),
                             "x_frac": cx, "y_frac": cy})
    if not raw:
        return []
    sizes = sorted(s["size"] for s in raw)
    med = sizes[len(sizes) // 2]
    result = []
    for idx, s in enumerate(raw):
        sz = s["size"]
        if sz >= med * 1.5:       role = "heading"
        elif sz >= med * 1.2:     role = "subheading"
        elif s["bold"]:           role = "label"
        elif sz < med * 0.85:     role = "footnote"
        else:                     role = "body"
        result.append({"text": s["text"], "role": role,
                        "size_ratio": sz / max(med, 1), "bold": s["bold"],
                        "italic": s["italic"], "x_frac": s["x_frac"],
                        "y_frac": s["y_frac"], "seq_index": idx,
                        "seq_total": len(raw)})
    return result

def page_plain_text(page: "fitz.Page") -> str:
    return page.get_text("text").strip()

# -- Q&A extraction from page text ---------------------------------------------

_QA_PATTERNS = [
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+is\s+(?P<d>[^.]{8,240})", re.I),
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+are\s+(?P<d>[^.]{8,240})", re.I),
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+refers to\s+(?P<d>[^.]{8,240})", re.I),
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+is defined as\s+(?P<d>[^.]{8,240})", re.I),
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+consists? of\s+(?P<d>[^.]{8,240})", re.I),
    re.compile(r"^(?P<s>[^.]{3,80}?)\s+can be defined as\s+(?P<d>[^.]{8,240})", re.I),
]
_SKIP_SUBJECTS = {"this", "that", "these", "those", "it", "they", "he", "she",
                  "we", "you", "i", "a", "an", "the"}
# Subjects containing these patterns refer to contextual things, not named concepts
_CONTEXT_SUBJECT_RE = re.compile(r"\b(of the|of a|of an|in the|in a|by the|for the)\b", re.I)
# Answers that are purely mathematical/numeric notation
_MATH_ONLY_RE = re.compile(r'^[\d\s\+\-\*\/\=\.\,\(\)\[\]\\<>%°±×÷√π]+$')

def extract_qa_from_text(text: str, book_id: str, page_idx: int,
                         max_per_page: int = 4) -> list[dict]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text.replace("\r", "\n")))
    pairs = []
    for sent in sentences:
        if len(pairs) >= max_per_page:
            break
        sent = sent.strip()
        if len(sent) < 30:
            continue
        for pat in _QA_PATTERNS:
            m = pat.match(sent)
            if not m:
                continue
            subj = m.group("s").strip(" :;-")
            defn = m.group("d").strip(" .;")
            if not subj or not defn:
                continue
            if subj.lower().split()[0] in _SKIP_SUBJECTS:
                continue
            if len(subj.split()) > 6:
                continue
            # Skip context-dependent subjects ("resistance of the circuit", etc.)
            if _CONTEXT_SUBJECT_RE.search(subj):
                continue
            # Answer must be a real definition: long enough and word-rich enough
            defn_words = defn.split()
            if len(defn) < 30 or len(defn_words) < 4:
                continue
            # Reject math/symbol-only answers
            if _MATH_ONLY_RE.match(defn):
                continue
            # Answer must have at least one substantial content word (5+ chars) in its first 8 words
            if not any(len(w.strip(".,;:()[]\"'")) >= 5 for w in defn_words[:8]):
                continue
            verb = "are" if "are" in pat.pattern else "is"
            q = f"What {verb} {subj}?"
            qa_id = hashlib.sha256(f"{book_id}|{page_idx}|{q}|{defn}".encode()).hexdigest()[:16]
            pairs.append({"qa_id": qa_id, "question": q, "answer": defn,
                          "book_id": book_id, "page_index": page_idx,
                          "confidence": 0.82, "evidence": sent[:240],
                          "review_status": "APPROVED"})
            break
    return pairs

# -- Node API helpers -----------------------------------------------------------

def api_post(client: httpx.Client, url: str, payload: dict,
             retries: int = 4, base_delay: float = 8.0) -> dict | None:
    """POST with retry on 429 rate limit."""
    for attempt in range(retries):
        try:
            resp = client.post(url, json=payload, timeout=45)
            if resp.status_code == 429:
                wait = base_delay * (2 ** attempt)
                tqdm.write(f"  Rate limited — waiting {wait:.0f}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            tqdm.write(f"  HTTP {e.response.status_code}: {e.response.text[:80]}")
            return None
        except Exception as e:
            tqdm.write(f"  Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None

def train_page_media(client: httpx.Client, node_url: str,
                     jpeg_bytes: bytes, spans: list[dict]) -> bool:
    result = api_post(client, f"{node_url}/media/train", {
        "modality": "page",
        "data_b64": base64.b64encode(jpeg_bytes).decode(),
        "spans": spans,
        "lr_scale": 1.0,
    })
    return result is not None

def train_text(client: httpx.Client, node_url: str, text: str,
               lr_scale: float = 1.0) -> bool:
    result = api_post(client, f"{node_url}/media/train", {
        "modality": "text",
        "text": text,
        "lr_scale": lr_scale,
    })
    return result is not None

def ingest_qa_batch(client: httpx.Client, node_url: str,
                    candidates: list[dict]) -> int:
    if not candidates:
        return 0
    result = api_post(client, f"{node_url}/qa/ingest", {"candidates": candidates})
    if result:
        return result.get("ingested", 0)
    return 0

def checkpoint(client: httpx.Client, node_url: str) -> None:
    # Save QA store first (fast, < 1 MB) via dedicated endpoint.
    try:
        r = client.post(f"{node_url}/qa/checkpoint", timeout=30)
        if r.is_success:
            tqdm.write(f"  QA store saved -> {r.json().get('qa_path', '?')}")
    except Exception as e:
        tqdm.write(f"  QA checkpoint failed: {e}")
    # Full neuro pool checkpoint — may take several minutes for large pools.
    # Fire the request but don't wait long; the node will finish in background.
    try:
        resp = client.post(f"{node_url}/neuro/checkpoint", timeout=600)
        if resp.is_success:
            path = resp.json().get("pool_path", "?")
            tqdm.write(f"  Neuro pool saved -> {path}")
    except Exception as e:
        tqdm.write(f"  Neuro checkpoint: {e} (pool save may still be running)")

def train_sequence(client: httpx.Client, node_url: str,
                   texts: list[str], tau: float = 1.5,
                   lr_scale: float = 0.8) -> bool:
    """Train a temporal sequence of text frames via STDP-ordered bridge training.

    STDP note: earlier frames become pre-synaptic to later frames. Sequences
    should be ordered causally (cause → effect, context → concept) to produce
    forward LTP edges and backward LTD edges in the pool.

    Also sends the sequence as a flat joined text so TextBitsEncoder builds
    co-occurrence labels for the full span — complements the frame-level STDP.
    """
    frames = [{"modality": "text", "text": t, "lr_scale": lr_scale} for t in texts]
    ok_seq = api_post(client, f"{node_url}/media/train_sequence",
                      {"frames": frames, "temporal_tau": tau}) is not None
    # Flat text pass: lets the encoder build word-level co-occurrence across the
    # whole sequence in one shot, reinforcing span-level Hebbian connections.
    flat = " ".join(texts)
    ok_flat = api_post(client, f"{node_url}/media/train",
                       {"modality": "text", "text": flat, "lr_scale": lr_scale * 0.5}) is not None
    return ok_seq or ok_flat

# -- Progress tracking ----------------------------------------------------------

def load_progress() -> dict:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {"done_books": [], "stages_complete": []}

def save_progress(prog: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(prog, indent=2))

# -- Stage 0: Toddler foundations -----------------------------------------------

# Each entry: (question, answer, training_sentence)
TODDLER_CONCEPTS = [
    # -- Body ------------------------------------------------------------------
    ("What are eyes?", "Eyes are the organs you use to see. They detect light and send images to your brain.",
     "Eyes are the organs that allow you to see. They detect light and color and send signals to the brain."),
    ("What are ears?", "Ears are the organs you use to hear sounds.",
     "Ears are sense organs that detect sound waves and let you hear voices, music, and other sounds."),
    ("What is a nose?", "A nose is the organ on your face that you use to smell and breathe.",
     "A nose is the part of your face used for smelling and breathing. It filters the air you inhale."),
    ("What is a mouth?", "A mouth is the opening in your face used for eating, drinking, and speaking.",
     "A mouth is the opening in the face for eating, drinking, speaking, and breathing."),
    ("What is a hand?", "A hand is the part of your arm at the end, with fingers for grasping and touching.",
     "A hand is attached to the end of the arm. It has four fingers and a thumb used for holding and touching."),
    ("What is a heart?", "A heart is the organ in your chest that pumps blood through your body.",
     "The heart is a muscle in the chest that pumps blood through the entire body to deliver oxygen and nutrients."),
    ("What is a brain?", "A brain is the organ inside your skull that controls your thoughts, feelings, and actions.",
     "The brain is the command center of the body. It controls thinking, memory, movement, emotion, and all body functions."),
    ("What is skin?", "Skin is the outer covering of your body that protects you and helps you feel touch.",
     "Skin is the body's largest organ. It covers and protects the body, regulates temperature, and detects touch and pain."),

    # -- Animals --------------------------------------------------------------
    ("What is a dog?", "A dog is a domestic animal kept as a pet. Dogs are loyal and friendly and can learn commands.",
     "A dog is a domesticated animal and a common pet. Dogs are known for loyalty, intelligence, and their ability to learn commands."),
    ("What is a cat?", "A cat is a small domestic animal with soft fur, sharp claws, and whiskers.",
     "A cat is a small furry animal kept as a pet. Cats have sharp claws, keen senses, and whiskers they use to navigate."),
    ("What is a bird?", "A bird is an animal with wings and feathers that can usually fly.",
     "Birds are animals with feathers, wings, and beaks. Most birds can fly. They hatch from eggs and are warm-blooded."),
    ("What is a fish?", "A fish is an animal that lives in water and breathes through gills.",
     "Fish are animals that live in water. They breathe through gills, are covered in scales, and swim using fins."),
    ("What is a horse?", "A horse is a large four-legged animal used for riding and pulling loads.",
     "A horse is a large animal with hooves and a mane. Horses are used for riding, work, and sport."),
    ("What is an elephant?", "An elephant is a very large animal with a long trunk, big ears, and tusks.",
     "Elephants are the largest land animals. They have a long flexible trunk for grasping, enormous ears, and ivory tusks."),
    ("What is a butterfly?", "A butterfly is an insect with large colorful wings that flies from flower to flower.",
     "A butterfly is an insect known for its beautiful patterned wings. Butterflies begin life as caterpillars."),

    # -- Food and Drink --------------------------------------------------------
    ("What is an apple?", "An apple is a round fruit that grows on trees. Apples can be red, green, or yellow and taste sweet.",
     "An apple is a round fruit grown on apple trees. It can be red, green, or yellow with sweet or tart flesh."),
    ("What is a banana?", "A banana is a long curved yellow fruit with soft sweet flesh inside.",
     "A banana is a tropical fruit with a yellow peel and soft sweet interior. Bananas are rich in potassium."),
    ("What is bread?", "Bread is a food baked from flour, water, and yeast. It is eaten as a staple food around the world.",
     "Bread is made by baking a dough of flour, water, yeast, and salt. It is one of the oldest and most common foods."),
    ("What is milk?", "Milk is a white liquid produced by mammals to feed their young. Cows, goats, and other animals produce milk people drink.",
     "Milk is a white nutritious liquid produced by mammals. Cow's milk is commonly drunk and used to make cheese, butter, and yogurt."),
    ("What is water?", "Water is a clear liquid that all living things need to survive. It makes up most of the human body.",
     "Water is a clear, odorless liquid essential for all life. It covers most of Earth's surface and makes up about 60 percent of the human body."),
    ("What is an egg?", "An egg is an oval object laid by birds and reptiles. Eggs are also a common food.",
     "An egg is laid by birds, fish, reptiles, and insects. Chicken eggs are a nutritious food containing protein and fats."),
    ("What is rice?", "Rice is a grain grown in flooded fields. It is the staple food for more than half the world.",
     "Rice is a cereal grain grown in paddies flooded with water. It is the primary food source for billions of people worldwide."),
    ("What is sugar?", "Sugar is a sweet substance found naturally in fruits and made from sugar cane or sugar beets.",
     "Sugar is a sweet crystalline substance. It occurs naturally in fruit and is refined from sugarcane or sugar beets. Glucose is the body's main energy sugar."),

    # -- Colors ---------------------------------------------------------------
    ("What is the color red?", "Red is a warm, bright color seen in fire, blood, and ripe strawberries.",
     "Red is a primary color associated with fire, heat, danger, and passion. Strawberries and stop signs are red."),
    ("What is the color blue?", "Blue is a cool color like the sky and the ocean.",
     "Blue is a primary color associated with sky, water, and calm. The ocean and a clear sky are blue."),
    ("What is the color green?", "Green is the color of plants and grass. It is made by mixing blue and yellow.",
     "Green is the color of leaves, grass, and growing plants. It is created by mixing blue and yellow light or pigment."),
    ("What is the color yellow?", "Yellow is a bright warm color like the sun and lemons.",
     "Yellow is a warm, cheerful color associated with sunlight, happiness, and caution. The sun and ripe lemons are yellow."),
    ("What is the color white?", "White is a color that reflects all light. Snow, clouds, and paper are white.",
     "White is the color produced when all visible light wavelengths are reflected. Snow, clouds, and blank paper are white."),
    ("What is the color black?", "Black is the absence of reflected light. Night and shadows are black.",
     "Black is produced when all light is absorbed and none is reflected. Night, coal, and shadows are black."),
    ("What is the color orange?", "Orange is a warm color like oranges and autumn leaves. It is made by mixing red and yellow.",
     "Orange is a warm color between red and yellow. Ripe oranges, pumpkins, and autumn leaves are orange."),
    ("What is the color purple?", "Purple is a color made by mixing red and blue. Grapes and lavender flowers are purple.",
     "Purple is a rich color created by mixing red and blue. It is associated with royalty and creativity."),

    # -- Numbers --------------------------------------------------------------
    ("What is the number zero?", "Zero is the number that represents nothing or no amount.",
     "Zero is the number representing the absence of quantity. It is the additive identity in mathematics."),
    ("What is the number one?", "One is the first counting number. It represents a single item.",
     "One is the first positive integer. It represents a single unit or item."),
    ("What is the number two?", "Two is a number that represents a pair of things.",
     "Two is the second positive integer. A pair of items — like two hands or two eyes — represents the number two."),
    ("What is the number ten?", "Ten is the number after nine. We count in groups of ten because humans have ten fingers.",
     "Ten is a significant number in mathematics. Our decimal number system is based on ten because humans have ten fingers."),
    ("What is a hundred?", "A hundred is the number equal to ten times ten, written as 100.",
     "A hundred equals 100, which is ten groups of ten. It is a common unit for counting and percentages."),

    # -- Shapes ---------------------------------------------------------------
    ("What is a circle?", "A circle is a perfectly round shape where every point on the edge is the same distance from the center.",
     "A circle is a round closed curve where all points are equidistant from the center. Wheels, coins, and the sun appear circular."),
    ("What is a square?", "A square is a shape with four equal sides and four right angles.",
     "A square is a rectangle with all four sides of equal length and four 90-degree corners. Tiles and chessboards are made of squares."),
    ("What is a triangle?", "A triangle is a shape with three sides and three angles.",
     "A triangle has three sides and three angles. The angles of a triangle always add up to 180 degrees."),
    ("What is a rectangle?", "A rectangle is a shape with four sides where opposite sides are equal and all corners are right angles.",
     "A rectangle has four sides with opposite sides equal in length and four right angles. Most doors and books are rectangular."),
    ("What is a sphere?", "A sphere is a perfectly round three-dimensional shape like a ball or a planet.",
     "A sphere is a 3D shape where every point on the surface is equidistant from the center. Balls and planets are roughly spherical."),

    # -- Nature and Earth ------------------------------------------------------
    ("What is a tree?", "A tree is a tall plant with a woody trunk, branches, and leaves.",
     "A tree is a large perennial plant with a single woody trunk, branches, and leaves. Trees provide oxygen, shade, and habitat for animals."),
    ("What is the sun?", "The sun is the star at the center of our solar system. It provides light and heat for all life on Earth.",
     "The sun is a star — a massive ball of burning gas — at the center of our solar system. It provides light, warmth, and energy for all life on Earth."),
    ("What is the moon?", "The moon is the large natural satellite that orbits Earth. It causes tides and lights the night sky.",
     "The moon is Earth's only natural satellite. It orbits Earth roughly every 28 days and reflects sunlight to illuminate the night sky."),
    ("What is rain?", "Rain is water that falls from clouds in the sky to the ground.",
     "Rain occurs when water vapor in clouds condenses into droplets that fall to the ground. Rain is essential for plants, rivers, and drinking water."),
    ("What is a river?", "A river is a large stream of fresh water flowing across the land to the sea or a lake.",
     "A river is a natural flowing watercourse moving toward the sea, a lake, or another river. Rivers carve valleys and support ecosystems."),
    ("What is an ocean?", "An ocean is a vast body of salt water that covers most of Earth's surface.",
     "Oceans are immense bodies of salt water. They cover about 71 percent of Earth's surface and regulate climate, support marine life, and generate oxygen."),
    ("What is a mountain?", "A mountain is a large landform that rises steeply above the surrounding land.",
     "A mountain is an elevated landform rising sharply from the surrounding terrain. Mountains are formed by tectonic forces, erosion, and volcanic activity."),
    ("What is the sky?", "The sky is the region of the atmosphere above the Earth. It looks blue during the day because of how air scatters sunlight.",
     "The sky appears blue because air molecules scatter short blue wavelengths of sunlight more than other colors. At sunrise and sunset it appears red and orange."),

    # -- Family and People -----------------------------------------------------
    ("What is a family?", "A family is a group of people related by blood, marriage, or adoption who care for each other.",
     "A family is a group of people connected by kinship or affection. Families provide love, support, and a sense of belonging."),
    ("What is a mother?", "A mother is a female parent who gives birth to or raises children.",
     "A mother is a female parent. Mothers nurture, protect, and raise their children, providing food, comfort, and guidance."),
    ("What is a father?", "A father is a male parent who helps raise and care for children.",
     "A father is a male parent. Fathers protect, provide for, and guide their children through life."),
    ("What is a friend?", "A friend is someone you know well, enjoy spending time with, and care about.",
     "A friend is a person you know, like, and trust. Friendships bring happiness, support, and companionship."),
    ("What is a teacher?", "A teacher is a person whose job is to help others learn and understand new things.",
     "A teacher educates students by sharing knowledge, explaining concepts, and guiding learning. Teachers are essential for passing knowledge between generations."),

    # -- Actions and Verbs ------------------------------------------------------
    ("What does run mean?", "Run means to move quickly on your feet, faster than walking.",
     "Running is fast movement on foot. When you run, both feet leave the ground briefly with each stride."),
    ("What does eat mean?", "Eat means to put food in your mouth, chew it, and swallow it for nourishment.",
     "Eating involves placing food in the mouth, chewing, and swallowing it. The body then digests food to extract energy and nutrients."),
    ("What does sleep mean?", "Sleep is a natural rest state when your body and mind recover and restore themselves.",
     "Sleep is a state of unconscious rest needed by the brain and body. During sleep, the body heals, memories are consolidated, and energy is restored."),
    ("What does think mean?", "Think means to use your mind to reason, form ideas, and solve problems.",
     "Thinking is the mental process of forming ideas, reasoning, imagining, and problem-solving. The brain generates thoughts continuously."),
    ("What does learn mean?", "Learn means to gain new knowledge or skills through study, experience, or being taught.",
     "Learning is the process of acquiring new knowledge, skills, or understanding. It happens through experience, reading, observation, and instruction."),
    ("What does grow mean?", "Grow means to get bigger or develop over time, as plants and animals do.",
     "Growing is the process of increasing in size, complexity, or capability over time. Plants, animals, and people grow throughout their lives."),
    ("What does love mean?", "Love is a strong feeling of deep affection and care for someone or something.",
     "Love is a profound feeling of affection, attachment, and care. It motivates people to nurture, protect, and support those they love."),

    # -- Descriptors ----------------------------------------------------------
    ("What does big mean?", "Big means large in size, more than usual.",
     "Big describes something that is large in size. An elephant is big. A mountain is big. Big is the opposite of small."),
    ("What does small mean?", "Small means little in size, less than usual.",
     "Small describes something with a little size. An ant is small. A seed is small. Small is the opposite of big or large."),
    ("What does hot mean?", "Hot means having a high temperature, like fire or boiling water.",
     "Hot describes high temperature. Fire, boiling water, and the sun are hot. Hot is the opposite of cold."),
    ("What does cold mean?", "Cold means having a low temperature, like ice or winter air.",
     "Cold describes low temperature. Ice, snow, and winter air are cold. Cold is the opposite of hot."),
    ("What does soft mean?", "Soft means easy to press or bend, not hard. Pillows and cotton are soft.",
     "Soft describes something that is easily deformed, not firm or rigid. Pillows, clouds, and cotton are soft."),
    ("What does hard mean?", "Hard means difficult to scratch or bend, solid and firm. Rocks and metal are hard.",
     "Hard describes something resistant to deformation. Rocks, metal, and wood are hard. Hard is the opposite of soft."),
    ("What does fast mean?", "Fast means moving or happening quickly.",
     "Fast describes high speed — moving or happening in a short time. A cheetah is fast. Light travels extremely fast."),
    ("What does slow mean?", "Slow means moving or happening at a low speed.",
     "Slow describes low speed. A turtle moves slowly. Slow is the opposite of fast."),

    # -- Concepts --------------------------------------------------------------
    ("What is language?", "Language is a system of words and rules humans use to communicate with each other.",
     "Language is a structured system of communication using words, grammar, and meaning. Languages let humans share ideas, feelings, and information."),
    ("What is a question?", "A question is a sentence that asks for information or an answer.",
     "A question is a request for information, often ending with a question mark. Questions drive curiosity and learning."),
    ("What is an answer?", "An answer is a response to a question that gives the information requested.",
     "An answer is the information or response given to a question. Finding answers to questions is the basis of all learning."),
    ("What is a word?", "A word is a unit of language that carries meaning.",
     "A word is a basic unit of language. Words represent ideas, objects, actions, and qualities. Languages have thousands of words."),
    ("What is a sentence?", "A sentence is a group of words that expresses a complete thought.",
     "A sentence is a complete unit of language with a subject and predicate. Sentences express complete thoughts and are the building blocks of communication."),
    ("What is a number?", "A number is a symbol used to count, measure, and label things.",
     "Numbers are abstract symbols representing quantity. Mathematics uses numbers to count, measure, calculate, and describe the world."),
    ("What is time?", "Time is the ongoing sequence of events from the past through the present to the future.",
     "Time is the continuous progression of existence. It is measured in seconds, minutes, hours, days, and years. Time flows from past to present to future."),
    ("What is energy?", "Energy is the ability to do work or cause change. Food gives us energy. Sunlight is energy.",
     "Energy is the capacity to do work or cause change. It exists as heat, light, sound, electricity, and motion. Energy cannot be created or destroyed, only transformed."),
    ("What is gravity?", "Gravity is the force that pulls objects toward each other. Earth's gravity keeps us on the ground.",
     "Gravity is the force of attraction between objects with mass. Earth's gravity pulls everything toward its center, keeping us on the ground and the moon in orbit."),
    ("What is science?", "Science is the study of the natural world through observation, experiments, and careful thinking.",
     "Science is the systematic study of the natural world. Scientists observe, form hypotheses, and test them with experiments to build knowledge."),
    ("What is math?", "Math is the study of numbers, shapes, and patterns. It is used to count, measure, and solve problems.",
     "Mathematics is the study of numbers, quantities, shapes, and their relationships. It underlies science, technology, and everyday reasoning."),
    ("What is music?", "Music is sound organized in time with rhythm, melody, and harmony to express feelings.",
     "Music is an art form combining sound and rhythm. It uses melody, harmony, and tempo to evoke emotions and communicate across cultures."),
    ("What is art?", "Art is the expression of human creativity and imagination through painting, sculpture, music, and other forms.",
     "Art encompasses human creativity expressed in visual, auditory, or performance forms. Art communicates ideas, emotions, and cultural values."),
]

# Temporal sequences for neuro fabric — teaches conceptual associations over time
TODDLER_SEQUENCES = [
    ["apple", "fruit", "sweet", "red", "round", "food", "eat", "tree", "grow"],
    ["dog", "animal", "pet", "loyal", "bark", "run", "play", "fur", "friend"],
    ["cat", "animal", "whiskers", "purr", "soft", "hunt", "curious", "fur"],
    ["water", "drink", "rain", "cloud", "river", "ocean", "wet", "clean", "life"],
    ["sun", "light", "warm", "day", "sky", "yellow", "energy", "star", "orbit"],
    ["tree", "plant", "grow", "leaves", "wood", "forest", "oxygen", "shade"],
    ["red", "blue", "green", "yellow", "color", "light", "see", "rainbow"],
    ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
    ["eye", "see", "ear", "hear", "nose", "smell", "mouth", "taste", "skin", "touch"],
    ["mother", "father", "family", "love", "home", "safe", "together", "care"],
    ["morning", "sunrise", "breakfast", "school", "afternoon", "sunset", "evening", "night", "sleep"],
    ["happy", "smile", "laugh", "fun", "play", "friend", "joy", "good"],
    ["baby", "child", "learn", "grow", "young", "adult", "old", "life"],
    ["question", "ask", "think", "find", "answer", "know", "understand", "learn"],
    ["word", "sentence", "language", "speak", "read", "write", "communicate", "idea"],
    ["hot", "fire", "warm", "cold", "ice", "snow", "temperature", "season"],
    ["big", "large", "huge", "small", "tiny", "size", "compare", "measure"],
    ["fast", "run", "speed", "slow", "walk", "move", "time", "distance"],
    ["heart", "blood", "pump", "beat", "body", "alive", "breathe", "life"],
    ["sun", "planet", "earth", "moon", "orbit", "gravity", "space", "universe"],
]

def stage0_toddler(client: httpx.Client, node_url: str) -> None:
    """
    Stage 0: Toddler Foundations

    Three-pass consolidation strategy:
      Pass 1 (lr=1.5): encode definition sentences — establishes initial sparse codes
      Pass 2 (lr=1.0): STDP-ordered Q→A text pairs — pre-synaptic context fires
                       before post-synaptic answer, building causal forward edges
      Pass 3 (lr=0.7): low-rate consolidation pass — homeostatic scaling has now
                       run ≥1 cycle; this pass refines weights without saturation

    kWTA note: with top-2% active per hop, the pool must see each concept from
    multiple angles (definition, Q+A, sequence) before the sparse code stabilizes.
    That is why we use three structurally different passes, not three identical ones.
    """
    print("\n-- Stage 0: Toddler Foundations --")
    n = len(TODDLER_CONCEPTS)

    # ── Pass 1: definition sentences at elevated LR ──────────────────────────
    print(f"  Pass 1 of 3 — definition sentences ({n} concepts, lr=1.5)...")
    trained = 0
    for _q, _a, definition in tqdm(TODDLER_CONCEPTS, desc="  Defs-P1", unit="def"):
        if train_text(client, node_url, definition, lr_scale=1.5):
            trained += 1
    print(f"  Pass 1 complete: {trained}/{n}")

    # ── Pass 2: STDP-ordered Q+A text pairs ──────────────────────────────────
    # Send each as a single text block: definition first, then "Q: ... A: ..."
    # This makes the definition context pre-synaptic to the Q→A binding —
    # the causal order (definition → question → answer) builds forward LTP edges.
    print(f"  Pass 2 of 3 — STDP Q+A pairs ({n} concepts, lr=1.0)...")
    p2 = 0
    for question, answer, definition in tqdm(TODDLER_CONCEPTS, desc="  QA-P2  ", unit="qa"):
        # Pack: context (definition) fires before question fires before answer
        combined = f"{definition} {question} {answer}"
        if train_text(client, node_url, combined, lr_scale=1.0):
            p2 += 1
    print(f"  Pass 2 complete: {p2}/{n}")

    # ── Ingest Q&A pairs into the QA store ───────────────────────────────────
    print(f"  Ingesting {n} Q&A pairs into QA store...")
    candidates = []
    for question, answer, _ in TODDLER_CONCEPTS:
        qa_id = hashlib.sha256(f"toddler|{question}|{answer}".encode()).hexdigest()[:16]
        candidates.append({"qa_id": qa_id, "question": question, "answer": answer,
                            "book_id": "toddler_foundations", "page_index": 0,
                            "confidence": 0.95, "evidence": answer,
                            "review_status": "VERIFIED"})
    ingested = ingest_qa_batch(client, node_url, candidates)
    print(f"  Ingested {ingested} Q&A pairs")

    # ── Train temporal concept sequences (STDP-ordered chains) ───────────────
    # Sequences are ordered causally: apple→fruit→sweet→... so earlier tokens
    # become pre-synaptic to later ones — forward edges get LTP, backward LTD.
    print(f"  Training {len(TODDLER_SEQUENCES)} STDP-ordered concept chains...")
    for seq in tqdm(TODDLER_SEQUENCES, desc="  Chains ", unit="seq"):
        train_sequence(client, node_url, seq, tau=1.5, lr_scale=0.9)
    print("  Sequences complete")

    checkpoint(client, node_url)

    # ── Pass 3: low-rate consolidation after homeostatic scaling ─────────────
    # By now the pool has done at least a couple of homeostatic cycles (every 500
    # steps). This light pass refines the sparse codes that survived kWTA without
    # re-saturating neurons that homeostasis just brought back to baseline.
    print(f"  Pass 3 of 3 — consolidation sweep ({n} concepts, lr=0.7)...")
    p3 = 0
    for _q, _a, definition in tqdm(TODDLER_CONCEPTS, desc="  Defs-P3", unit="def"):
        if train_text(client, node_url, definition, lr_scale=0.7):
            p3 += 1
    print(f"  Pass 3 complete: {p3}/{n}")

    checkpoint(client, node_url)
    print("  Stage 0 complete.\n")

# -- PDF book processing --------------------------------------------------------

def process_book(client: httpx.Client, node_url: str, pdf_path: Path,
                 ckpt_every: int = 80) -> dict:
    book_id = re.sub(r"[^a-z0-9]+", "_", pdf_path.stem.lower()).strip("_")[:48]
    print(f"  Book: {pdf_path.name}  [{book_id}]")
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    trained_pages = 0
    skipped = 0
    qa_candidates: list[dict] = []

    pages_iter = tqdm(range(total), total=total, desc=book_id[:30], unit="pg")
    for page_idx in pages_iter:
        page = doc[page_idx]
        try:
            jpeg = render_page_jpeg(page)
            spans = extract_spans(page)
            text = page_plain_text(page)

            if not spans and len(jpeg) < 3000:
                skipped += 1
                continue

            if train_page_media(client, node_url, jpeg, spans):
                trained_pages += 1
            else:
                skipped += 1

            # Extract Q&A from this page's text
            qa_candidates.extend(
                extract_qa_from_text(text, book_id, page_idx, max_per_page=3)
            )

            if trained_pages > 0 and trained_pages % ckpt_every == 0:
                tqdm.write(f"  [{trained_pages} pages] checkpointing…")
                checkpoint(client, node_url)

        except Exception as e:
            tqdm.write(f"  Page {page_idx+1} error: {e}")
            skipped += 1

    doc.close()

    # Ingest all Q&A for this book in one call
    if qa_candidates:
        ingested = ingest_qa_batch(client, node_url, qa_candidates)
        tqdm.write(f"  QA ingested: {ingested}/{len(qa_candidates)} pairs")
    else:
        ingested = 0

    checkpoint(client, node_url)
    return {"book_id": book_id, "pages_trained": trained_pages,
            "pages_skipped": skipped, "qa_pairs": ingested}

# -- Stage filtering ------------------------------------------------------------

_LANGUAGE_KEYWORDS = [
    "beginning", "basic", "introductory", "introduction", "primer",
    "elementary", "fundamentals", "principles", "essentials", "overview",
    "general", "survey", "core", "simple",
]

def is_language_book(pdf_path: Path) -> bool:
    name = pdf_path.stem.lower()
    return any(kw in name for kw in _LANGUAGE_KEYWORDS)

def collect_pdfs(textbook_dir: Path) -> list[Path]:
    if not textbook_dir.exists():
        return []
    return sorted(textbook_dir.glob("*.pdf"))

# -- Main pipeline --------------------------------------------------------------

def run_stages(node_url: str, stages: list[int], max_books: int | None,
               resume: bool, textbook_dir: Path, ckpt_every: int) -> None:
    prog = load_progress() if resume else {"done_books": [], "stages_complete": []}

    with httpx.Client() as client:
        # Health check — retry up to 12 times (60s total) to handle node under load
        for attempt in range(12):
            try:
                health = client.get(f"{node_url}/health", timeout=30).json()
                print(f"Node: {health.get('node_id', '?')} — {health.get('status', '?')}")
                break
            except Exception as e:
                if attempt < 11:
                    print(f"  Waiting for node ({attempt+1}/12): {e}", flush=True)
                    time.sleep(5)
                else:
                    sys.exit(f"Node unreachable at {node_url} after 60s: {e}")

        if 0 in stages:
            if "stage0" not in prog["stages_complete"]:
                stage0_toddler(client, node_url)
                prog["stages_complete"].append("stage0")
                save_progress(prog)
            else:
                print("Stage 0 already complete — skipping (use --no-resume to repeat)")

        all_pdfs = collect_pdfs(textbook_dir)
        if not all_pdfs and (1 in stages or 2 in stages):
            print(f"Warning: no PDFs found in {textbook_dir}")

        if 1 in stages:
            print(f"\n-- Stage 1: Language / Introductory --")
            lang_books = [p for p in all_pdfs if is_language_book(p)]
            if max_books:
                lang_books = lang_books[:max_books]
            print(f"  Found {len(lang_books)} introductory books")
            for pdf in lang_books:
                if resume and pdf.name in prog["done_books"]:
                    print(f"  Skipping (done): {pdf.name}")
                    continue
                result = process_book(client, node_url, pdf, ckpt_every)
                prog["done_books"].append(pdf.name)
                save_progress(prog)
                print(f"  Done: {result}")
            prog["stages_complete"].append("stage1")
            save_progress(prog)
            print("Stage 1 complete.")

        if 2 in stages:
            print(f"\n-- Stage 2: Full K-12 Curriculum --")
            # Exclude books already processed in Stage 1
            stage1_names = {p.name for p in all_pdfs if is_language_book(p)}
            remaining = [p for p in all_pdfs if p.name not in stage1_names]
            if max_books:
                remaining = remaining[:max_books]
            print(f"  Found {len(remaining)} curriculum books")
            for pdf in remaining:
                if resume and pdf.name in prog["done_books"]:
                    print(f"  Skipping (done): {pdf.name}")
                    continue
                result = process_book(client, node_url, pdf, ckpt_every)
                prog["done_books"].append(pdf.name)
                save_progress(prog)
                print(f"  Done: {result}")
            prog["stages_complete"].append("stage2")
            save_progress(prog)
            print("Stage 2 complete.")

    print("\nAll requested stages complete.")


def main():
    parser = argparse.ArgumentParser(description="K-12 staged training pipeline")
    parser.add_argument("--node", default="http://127.0.0.1:8090",
                        help="Node API base URL (default: http://localhost:8080)")
    parser.add_argument("--stages", default="0,1,2",
                        help="Comma-separated stages to run: 0,1,2 (default: all)")
    parser.add_argument("--max-books", type=int, default=None,
                        help="Limit books per stage (useful for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed books (reads data/k12_progress.json)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore progress file and reprocess everything")
    parser.add_argument("--clear-progress", action="store_true",
                        help="Delete data/k12_progress.json before starting (fresh run)")
    parser.add_argument("--textbooks", default=str(TEXTBOOKS_DIR),
                        help=f"Textbooks directory (default: {TEXTBOOKS_DIR})")
    parser.add_argument("--checkpoint-every", type=int, default=100, dest="ckpt_every",
                        help="Checkpoint pool every N pages (default 100)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    if args.clear_progress and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print(f"Cleared progress file: {PROGRESS_FILE}")

    stages = [int(s.strip()) for s in args.stages.split(",") if s.strip().isdigit()]
    if not stages:
        sys.exit("No valid stages specified. Use --stages 0,1,2")

    textbook_dir = Path(args.textbooks)
    print(f"Textbook directory: {textbook_dir}")
    print(f"Stages to run: {stages}")
    if args.max_books:
        print(f"Max books per stage: {args.max_books}")

    run_stages(
        node_url=args.node,
        stages=stages,
        max_books=args.max_books,
        resume=args.resume,
        textbook_dir=textbook_dir,
        ckpt_every=args.ckpt_every,
    )


if __name__ == "__main__":
    main()
