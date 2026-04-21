#!/usr/bin/env python3
"""
PDF -> images + text training pipeline.

For each PDF page:
  1. Render page to high-res image (300 DPI)
  2. Extract text with positional/structural context
  3. POST to /media/train_sequence:
       frame 0 (t=0.0): image modality -- the rendered page
       frame 1 (t=0.5): text modality  -- extracted text with span annotations
       frame 2 (t=1.0): text modality  -- structural tags (section/appendix/title)

Cross-modal Hebbian fires: text neurons ↔ image neurons ↔ structural neurons.

Dependencies:
    pip install pdf2image pypdf requests Pillow
    Poppler (Windows): place bin/ in PATH or set POPPLER_PATH below
"""

import os
import sys
import json
import base64
import hashlib
import argparse
import tempfile
import time
from io import BytesIO
from pathlib import Path

import requests

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False
    print("[warn] pdf2image not installed -- image frames will be skipped")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    print("[warn] pypdf not installed -- text extraction will be skipped")

try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

# --- Config ---
NODE_API = os.environ.get("NODE_API", "http://localhost:8090")
TEXTBOOK_DIRS = [
    Path("D:/w1z4rdv1510n-data/textbooks"),
    Path("D:/Projects/StateOfLoci/textbooks"),
]
PROGRESS_FILE = Path("D:/w1z4rdv1510n-data/textbook_train_progress.json")
DPI = 300
MAX_IMAGE_BYTES = 4 * 1024 * 1024   # 4 MB per image after JPEG compress
JPEG_QUALITY = 85
POPPLER_PATH = None  # e.g. r"C:\poppler\bin" -- set if not in PATH

SESSION = requests.Session()


# --- Progress tracking ---

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    return {}


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def pdf_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# --- Structural detection ---

APPENDIX_PATTERNS = ["appendix", "appendices", "supplement", "reference", "bibliography"]
TITLE_PATTERNS = ["chapter", "section", "unit", "module", "part "]

def detect_structure(text: str, page_num: int) -> dict:
    lower = text.lower()
    is_appendix = any(p in lower for p in APPENDIX_PATTERNS)
    is_title_page = page_num <= 3 and len(text.strip()) < 500
    section_hints = [p for p in TITLE_PATTERNS if p in lower]
    return {
        "is_appendix": is_appendix,
        "is_title_page": is_title_page,
        "section_hints": section_hints,
        "page": page_num,
    }


def structural_text(info: dict, source_pdf: str) -> str:
    parts = [f"[source:{Path(source_pdf).stem}]", f"[page:{info['page']}]"]
    if info["is_appendix"]:
        parts.append("[appendix]")
    if info["is_title_page"]:
        parts.append("[title-page]")
    for h in info["section_hints"]:
        parts.append(f"[{h.strip()}]")
    return " ".join(parts)


# --- Image helpers ---

def image_to_b64(img) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    data = buf.getvalue()
    # If still too large, reduce quality iteratively
    q = JPEG_QUALITY
    while len(data) > MAX_IMAGE_BYTES and q > 40:
        q -= 10
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        data = buf.getvalue()
    return base64.b64encode(data).decode()


# --- Node API ---

def train_sequence(frames: list[dict], lr: float = 0.8) -> bool:
    payload = {"frames": frames, "lr_scale": lr}
    try:
        r = SESSION.post(f"{NODE_API}/media/train_sequence", json=payload, timeout=30)
        if r.status_code == 200:
            return True
        print(f"  [warn] train_sequence returned {r.status_code}: {r.text[:200]}")
        return False
    except Exception as e:
        print(f"  [error] train_sequence: {e}")
        return False


# --- Main pipeline ---

def process_pdf(pdf_path: Path, progress: dict, dry_run: bool) -> int:
    key = str(pdf_path)
    h = pdf_hash(pdf_path)
    pages_done = progress.get(key, {}).get("pages_done", [])
    stored_hash = progress.get(key, {}).get("hash", "")

    if stored_hash and stored_hash != h:
        print(f"  [info] PDF changed, resetting progress for {pdf_path.name}")
        pages_done = []

    print(f"  PDF: {pdf_path.name}  (pages done: {len(pages_done)})")

    images = []
    if HAS_PDF2IMAGE:
        kwargs = {"dpi": DPI}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
        try:
            images = convert_from_path(str(pdf_path), **kwargs)
        except Exception as e:
            print(f"  [error] pdf2image: {e}")

    text_pages = []
    if HAS_PYPDF:
        try:
            reader = PdfReader(str(pdf_path))
            for pg in reader.pages:
                text_pages.append(pg.extract_text() or "")
        except Exception as e:
            print(f"  [error] pypdf: {e}")

    n_pages = max(len(images), len(text_pages))
    if n_pages == 0:
        print("  [skip] no content extracted")
        return 0

    trained = 0
    for i in range(n_pages):
        page_num = i + 1
        if page_num in pages_done:
            continue

        text = text_pages[i] if i < len(text_pages) else ""
        img = images[i] if i < len(images) else None

        struct_info = detect_structure(text, page_num)
        struct_text = structural_text(struct_info, str(pdf_path))

        frames = []

        # Frame 0: image
        if img is not None and HAS_PILLOW:
            img_b64 = image_to_b64(img)
            frames.append({
                "t_secs": 0.0,
                "modality": "image",
                "data": img_b64,
                "lr_scale": 1.0,
                "span_annotations": [
                    {"label": f"page:{page_num}", "start": 0, "end": 0},
                    {"label": f"source:{pdf_path.stem}", "start": 0, "end": 0},
                ] + ([{"label": "appendix", "start": 0, "end": 0}] if struct_info["is_appendix"] else []),
            })

        # Frame 1: page text
        if text.strip():
            frames.append({
                "t_secs": 0.5,
                "modality": "text",
                "data": text[:8000],  # cap per frame
                "lr_scale": 0.9,
                "span_annotations": [
                    {"label": f"page:{page_num}", "start": 0, "end": len(text)},
                ],
            })

        # Frame 2: structural context
        if struct_text.strip():
            frames.append({
                "t_secs": 1.0,
                "modality": "text",
                "data": struct_text,
                "lr_scale": 0.7,
                "span_annotations": [],
            })

        if not frames:
            continue

        if dry_run:
            print(f"    [dry-run] page {page_num}: {len(frames)} frames")
            trained += 1
            continue

        ok = train_sequence(frames, lr=0.8)
        if ok:
            pages_done.append(page_num)
            trained += 1
            if trained % 5 == 0:
                progress[key] = {"hash": h, "pages_done": pages_done}
                save_progress(progress)
            time.sleep(0.05)
        else:
            print(f"    [warn] failed to train page {page_num}, will retry next run")

    progress[key] = {"hash": h, "pages_done": pages_done}
    save_progress(progress)
    print(f"  Trained {trained} new pages from {pdf_path.name}")
    return trained


def run(dirs: list[Path], limit_pdfs: int | None, dry_run: bool):
    progress = load_progress()
    all_pdfs = []
    for d in dirs:
        if d.exists():
            all_pdfs.extend(sorted(d.rglob("*.pdf")))

    print(f"Found {len(all_pdfs)} PDFs across {len(dirs)} directories.")

    processed = 0
    for pdf in all_pdfs:
        if limit_pdfs is not None and processed >= limit_pdfs:
            print(f"Reached PDF limit ({limit_pdfs}).")
            break
        print(f"\n[{processed+1}/{len(all_pdfs)}] {pdf}")
        count = process_pdf(pdf, progress, dry_run)
        if count > 0:
            processed += 1

    print(f"\nAll done. Processed {processed} PDFs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF -> multimodal training pipeline")
    parser.add_argument("--dirs", nargs="+",
                        default=[str(d) for d in TEXTBOOK_DIRS],
                        help="Directories to scan for PDFs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max PDFs to process this run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trained without posting to API")
    parser.add_argument("--api", default=NODE_API,
                        help=f"Node API base URL (default: {NODE_API})")
    args = parser.parse_args()

    NODE_API = args.api
    SESSION.base_url = NODE_API

    run([Path(d) for d in args.dirs], args.limit, args.dry_run)
