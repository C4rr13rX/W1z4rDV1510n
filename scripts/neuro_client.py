#!/usr/bin/env python3
"""
neuro_client.py -- Full-architecture client for the W1z4rD node API.

Every training call goes through here so nothing is skipped.
Covers ALL endpoints that affect learning (pure neural pool -- no Q&A store):
  /media/train             -- single-frame multimodal Hebbian
  /media/train_sequence    -- temporal STDP multi-frame
  /neuro/record_episode    -- episodic learning from confirmed observations
  /equations/ingest        -- Environmental Equation Matrix
  /knowledge/ingest        -- structured knowledge documents
  /neuro/checkpoint        -- pool persistence

Import and use NeuroClient in every training script.
"""

from __future__ import annotations

import base64
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# ---------------------------------------------------------------------------
# Discipline detection -- applied to every text block before equations/ingest
# ---------------------------------------------------------------------------

_DISCIPLINE_RULES: list[tuple[str, list[str]]] = [
    ("classical_mechanics",   ["force", "mass", "acceleration", "velocity", "momentum",
                                "newton", "torque", "friction", "gravity", "kinetic energy",
                                "potential energy", "inertia"]),
    ("thermodynamics",        ["heat", "temperature", "entropy", "thermal", "thermodynamic",
                                "enthalpy", "specific heat", "boiling point", "kelvin"]),
    ("electromagnetism",      ["electric", "magnetic", "current", "voltage", "charge",
                                "electromagnetic", "capacitor", "inductor", "resistor",
                                "maxwell", "coulomb", "ohm"]),
    ("quantum_mechanics",     ["quantum", "wave function", "schrodinger", "photon",
                                "superposition", "eigenvalue", "heisenberg", "planck",
                                "uncertainty principle", "spin", "orbital"]),
    ("statistical_mechanics", ["probability distribution", "boltzmann", "entropy",
                                "statistical", "partition function", "degrees of freedom"]),
    ("fluid_dynamics",        ["fluid", "viscosity", "pressure", "flow", "bernoulli",
                                "turbulence", "laminar", "reynolds", "navier"]),
    ("general_relativity",    ["spacetime", "curvature", "einstein", "general relativity",
                                "black hole", "geodesic", "metric tensor"]),
    ("special_relativity",    ["special relativity", "lorentz", "time dilation",
                                "length contraction", "e=mc", "rest mass"]),
    ("quantum_field_theory",  ["field theory", "lagrangian density", "feynman",
                                "gauge invariance", "symmetry breaking", "standard model"]),
    ("information_theory",    ["entropy", "information", "shannon", "bit", "channel capacity",
                                "compression", "kolmogorov"]),
    ("cosmology",             ["universe", "big bang", "dark matter", "dark energy",
                                "cosmic", "galaxy", "hubble", "redshift"]),
]


def detect_discipline(text: str) -> Optional[str]:
    """Return the best matching discipline string, or None if no match."""
    lower = text.lower()
    best: Optional[tuple[int, str]] = None
    for discipline, keywords in _DISCIPLINE_RULES:
        score = sum(1 for kw in keywords if kw in lower)
        if score > 0 and (best is None or score > best[0]):
            best = (score, discipline)
    return best[1] if best else None


_FORMULA_RE = re.compile(
    r"(?:"
    r"[A-Za-z]\s*[=]\s*[-\d\w\s*/^().+]+"  # X = ...
    r"|∫|∑|∏|√|∂|∇"                          # integral, sum, product, sqrt, partial, nabla
    r"|[A-Za-z]+\d+(?:\s*[+\-*/]\s*[A-Za-z]+\d+)+"  # H2O, CO2
    r")"
)


def _has_formulas(text: str) -> bool:
    return bool(_FORMULA_RE.search(text))


# ---------------------------------------------------------------------------
# Span extraction -- converts plain text into structured TextSpan list
# ---------------------------------------------------------------------------

def make_spans(text: str, source_label: str = "") -> list[dict]:
    """
    Split text into semantic spans with roles, positions, and emphasis.

    Roles (matching TextRole enum in Rust):
      heading    -- first sentence when it introduces a term (title case, short)
      subheading -- section headers within text
      body       -- regular prose
      caption    -- parenthetical or bracketed notes
      code       -- formulas and equations
      list       -- bullet / numbered list items
      footnote   -- trailing notes, references, footnote markers
    """
    if not text:
        return []

    spans = []
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    total = len(sentences)

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue

        y_frac = i / max(total - 1, 1) if total > 1 else 0.0

        # Determine role
        words = sent.split()
        if i == 0 and len(words) <= 15 and sent[0].isupper():
            # Check for "X is ..." definition pattern
            role = "heading"
            bold = True
            size_ratio = 1.2
        elif sent.startswith(("- ", "* ", "* ", "1.", "2.", "3.")):
            role = "list"
            bold = False
            size_ratio = 0.95
        elif sent.startswith("(") and sent.endswith(")"):
            role = "caption"
            bold = False
            size_ratio = 0.85
        elif len(words) <= 8 and sent.endswith(":"):
            role = "subheading"
            bold = True
            size_ratio = 1.1
        elif re.match(r"^\[?\d+\]", sent) or sent.lower().startswith(("ref", "see also", "note:")):
            role = "footnote"
            bold = False
            size_ratio = 0.75
        elif _has_formulas(sent):
            role = "code"
            bold = False
            size_ratio = 0.9
        else:
            role = "body"
            bold = False
            size_ratio = 1.0

        spans.append({
            "text":       sent,
            "role":       role,
            "size_ratio": size_ratio,
            "bold":       bold,
            "italic":     False,
            "indent":     0,
            "x_frac":     0.0,
            "y_frac":     y_frac,
            "seq_index":  i,
            "seq_total":  total,
        })

    return spans


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resize_image_b64(image_path: str, max_px: int = 512, quality: int = 75) -> Optional[str]:
    if not HAS_PILLOW:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def image_bytes_to_b64(img_bytes: bytes, max_px: int = 512, quality: int = 75) -> Optional[str]:
    if not HAS_PILLOW:
        return base64.b64encode(img_bytes).decode()
    try:
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_px:
            scale = max_px / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return base64.b64encode(img_bytes).decode()


# ---------------------------------------------------------------------------
# NeuroClient -- wraps every API endpoint needed for full-architecture training
# ---------------------------------------------------------------------------

class NeuroClient:
    """
    Async HTTP client for the full W1z4rD node training API.

    Instantiate once per script with an httpx.AsyncClient and reuse.
    All methods are fire-and-report: they return True/False and never raise.
    """

    def __init__(self, node_url: str, client: httpx.AsyncClient):
        self.url    = node_url.rstrip("/")
        self.client = client

    # -- Low-level helpers ----------------------------------------------------

    async def _post(self, path: str, body: dict, timeout: float = 30.0) -> Optional[dict]:
        try:
            r = await self.client.post(f"{self.url}{path}", json=body, timeout=timeout)
            if r.status_code in (200, 204):
                return r.json() if r.status_code == 200 else {}
            return None
        except Exception:
            return None

    # -- Single-frame media training ------------------------------------------

    async def train_text(self, text: str, lr: float = 1.0) -> bool:
        """Plain text -> /media/train with structured spans."""
        spans = make_spans(text)
        payload: dict = {"modality": "text", "lr_scale": lr}
        if spans:
            payload["spans"] = spans
        else:
            payload["text"] = text
        return await self._post("/media/train", payload) is not None

    async def train_image(self, img_b64: str, caption: str = "", lr: float = 1.0) -> bool:
        """Image (base64) + caption -> /media/train page modality with caption spans."""
        payload: dict = {"modality": "page", "data_b64": img_b64, "lr_scale": lr}
        if caption:
            spans = make_spans(caption)
            payload["spans"] = spans if spans else None
            payload["text"] = caption
        return await self._post("/media/train", payload) is not None

    # -- Temporal sequence training -- the primary training call ---------------

    async def train_sequence(
        self,
        frames: list[dict],
        tau: float = 2.0,
    ) -> bool:
        """
        POST /media/train_sequence.

        frames: list of {modality, text?, data_b64?, spans?, lr_scale, t_secs}
        tau: temporal decay constant (seconds).
        """
        if not frames:
            return False
        payload = {"frames": frames, "temporal_tau": tau}
        return await self._post("/media/train_sequence", payload) is not None

    async def train_text_temporal(
        self, text: str, context: str = "", lr: float = 1.0, tau: float = 2.0,
    ) -> bool:
        """
        Train text as a 2-frame temporal sequence:
          frame 0 (t=0): full text with structured spans
          frame 1 (t=0.5): context/structural tags (if provided)

        This gives the system temporal structure even for plain text.
        """
        spans = make_spans(text)
        frames: list[dict] = [{
            "modality": "text",
            "t_secs":   0.0,
            "lr_scale": lr,
            "spans":    spans if spans else None,
            "text":     text if not spans else None,
        }]
        if context and context.strip():
            ctx_spans = make_spans(context)
            frames.append({
                "modality": "text",
                "t_secs":   0.5,
                "lr_scale": lr * 0.7,
                "spans":    ctx_spans if ctx_spans else None,
                "text":     context if not ctx_spans else None,
            })
        return await self.train_sequence(frames, tau=tau)

    async def train_image_text_temporal(
        self, img_b64: str, text: str, structural: str = "",
        lr: float = 1.0, tau: float = 2.0,
    ) -> bool:
        """
        Train image + text as a 3-frame temporal sequence:
          frame 0 (t=0.0): image -- visual perception
          frame 1 (t=0.5): text with spans -- semantic association
          frame 2 (t=1.0): structural context -- where/what
        The STDP bridge links image neurons ↔ text neurons ↔ structural neurons.
        """
        spans = make_spans(text)
        frames: list[dict] = [
            {
                "modality": "image",
                "t_secs":   0.0,
                "lr_scale": lr,
                "data_b64": img_b64,
            },
            {
                "modality": "text",
                "t_secs":   0.5,
                "lr_scale": lr * 0.9,
                "spans":    spans if spans else None,
                "text":     text if not spans else None,
            },
        ]
        if structural and structural.strip():
            struct_spans = make_spans(structural)
            frames.append({
                "modality": "text",
                "t_secs":   1.0,
                "lr_scale": lr * 0.7,
                "spans":    struct_spans if struct_spans else None,
                "text":     structural if not struct_spans else None,
            })
        return await self.train_sequence(frames, tau=tau)

    # -- Temporal pair training + episodic learning ---------------------------

    async def record_episode(
        self, question: str, answer: str, surprise: float = 0.0
    ) -> bool:
        """
        POST /neuro/record_episode -- logs a resolved Q->A episode into the episodic store.

        The fabric's conditional sufficiency tracker and inhibitory Hebbian updates
        fire automatically. Every correct Q&A pair should call this with surprise=0.0.
        For surprising/corrective pairs, use surprise=(1 - p_correct)^2.
        """
        # Encode question as context labels -- word-level tokens from the text
        q_words = [
            f"txt:word_{w.lower().strip('.,;:!?()[]\"\'')}"
            for w in question.split()
            if len(w.strip('.,;:!?()[]\"\'')) > 2
        ]
        if not q_words:
            return False
        payload = {
            "context_labels": q_words,
            "predicted":      answer[:500],
            "actual":         answer[:500],
            "streams":        [],
            "surprise":       float(max(0.0, min(1.0, surprise))),
        }
        return await self._post("/neuro/record_episode", payload) is not None

    async def ingest_qa_full(
        self,
        pairs: list[dict],
        pool: str = "knowledge",
        record_episodes: bool = True,
    ) -> int:
        """
        Full pair training through all neural pool pathways (no Q&A store):
          1. /media/train          -- combined Q+A text, full Hebbian activation
          2. /media/train_sequence -- Q frame (t=0) -> A frame (t=1) temporal STDP
          3. /neuro/record_episode -- episodic store for each pair
          4. /equations/ingest    -- equations extracted from answer text

        Returns count of pairs fully trained.
        """
        if not pairs:
            return 0

        trained = 0
        for pair in pairs:
            q = pair.get("question", "")
            a = pair.get("answer", "")
            if not q or not a:
                continue

            # Pass 1: full combined text through Hebbian pool (single frame)
            combined = f"{q} {a}"
            await self.train_text(combined, lr=1.0)

            # Pass 2: Q->A temporal sequence (question fires, then answer fires)
            q_spans = make_spans(q)
            a_spans = make_spans(a)
            frames = [
                {
                    "modality": "text",
                    "t_secs":   0.0,
                    "lr_scale": 1.0,
                    "spans":    q_spans if q_spans else None,
                    "text":     q if not q_spans else None,
                },
                {
                    "modality": "text",
                    "t_secs":   1.0,
                    "lr_scale": 0.9,
                    "spans":    a_spans if a_spans else None,
                    "text":     a if not a_spans else None,
                },
            ]
            await self.train_sequence(frames, tau=2.0)

            # Pass 3: episodic record
            if record_episodes:
                await self.record_episode(q, a, surprise=0.0)

            trained += 1

        return trained

    # -- Equation matrix ------------------------------------------------------

    async def ingest_equations(
        self, text: str, discipline: Optional[str] = None, confidence: float = 0.6,
    ) -> int:
        """
        POST /equations/ingest -- parses equations from text and feeds them
        into the Environmental Equation Matrix.

        Auto-detects discipline from text if not provided.
        Returns number of equations ingested.
        """
        if not text or not text.strip():
            return 0
        disc = discipline or detect_discipline(text)
        if not disc:
            return 0  # skip non-scientific text
        payload: dict = {"text": text, "confidence": confidence}
        if disc:
            payload["discipline"] = disc
        resp = await self._post("/equations/ingest", payload)
        return (resp or {}).get("ingested", 0)

    # -- Knowledge document ingestion -----------------------------------------

    async def ingest_knowledge(
        self, title: str, body: str, source: str = "", tags: list[str] | None = None,
    ) -> bool:
        """
        POST /knowledge/ingest -- structured knowledge document.

        Feeds into KnowledgeRuntime (separate from Hebbian pool and QA store).
        """
        doc: dict = {
            "title": title,
            "body":  body,
            "source": source,
        }
        if tags:
            doc["tags"] = tags
        payload: dict = {"document": doc}
        return await self._post("/knowledge/ingest", payload) is not None

    # -- Full concept training -- uses every endpoint --------------------------

    async def train_concept_full(
        self,
        concept: str,
        definition: str,
        wiki_text: str = "",
        images_b64: list[str] | None = None,
        qa_pairs: list[dict] | None = None,
        pool: str = "knowledge",
        level: int = 0,
    ) -> dict:
        """
        Train a single concept using the FULL architecture:

          1. /media/train_sequence -- text temporal sequence
               frame 0: definition with structured spans (heading + body)
               frame 1: wiki excerpt if available
          2. /media/train_sequence -- for each image:
               frame 0: image (visual perception)
               frame 1: definition with spans (image ↔ concept association)
               frame 2: structural context tag
          3. /equations/ingest -- definition + wiki text (if scientific)
          4. /knowledge/ingest -- structured document with title + body
          5. /qa/ingest + /neuro/record_episode -- all Q&A pairs
        """
        stats = {"seq": 0, "img": 0, "eq": 0, "qa": 0, "ep": 0, "know": 0}

        full_text = definition
        if wiki_text and len(wiki_text) > len(definition):
            full_text = definition + "  " + wiki_text[:500]

        # 1. Text temporal sequence
        if full_text.strip():
            ok = await self.train_text_temporal(
                full_text,
                context=f"[concept:{concept}] [level:{level}]",
                lr=1.0,
            )
            if ok:
                stats["seq"] += 1

        # 2. Image + text temporal sequences
        for img_b64 in (images_b64 or []):
            structural = f"[concept:{concept}] [level:{level}] [visual]"
            ok = await self.train_image_text_temporal(
                img_b64, definition, structural=structural, lr=1.0,
            )
            if ok:
                stats["img"] += 1

        # 3. Equation matrix
        eq_text = full_text
        disc = detect_discipline(eq_text)
        if disc:
            n = await self.ingest_equations(eq_text, discipline=disc)
            stats["eq"] += n

        # 4. Knowledge document
        if definition.strip():
            ok = await self.ingest_knowledge(
                title=concept,
                body=full_text,
                source="concept_dataset",
                tags=[f"level:{level}"],
            )
            if ok:
                stats["know"] += 1

        # 5. Q&A pairs -- full pipeline (ingest + sequence + episode)
        if qa_pairs:
            n = await self.ingest_qa_full(qa_pairs, pool=pool, record_episodes=True)
            stats["qa"] += n

        return stats

    # -- Checkpoint -----------------------------------------------------------

    async def checkpoint(self) -> bool:
        """POST /neuro/checkpoint -- persists the Hebbian pool to disk."""
        resp = await self._post("/neuro/checkpoint", {}, timeout=120.0)
        return resp is not None
