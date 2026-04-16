#!/usr/bin/env python3
"""
W1z4rD V1510n — OpenStax K-12 Ingestion Script
===============================================
Downloads OpenStax textbook PDFs and trains the node on each page using
multimodal co-activation: page image + structured text spans together.

Each page is sent as a single /media/train call with:
  - data_b64: the page rendered as a JPEG (image labels: zones, hues, edges)
  - spans:    extracted text blocks with layout metadata (text labels: words,
              phonemes, roles, positions, sequence indices)

Because both modalities fire together, the pool learns that a chapter heading
image-zone co-activates with its word tokens, and that "photosynthesis" in a
large font at the top of a zone links to the image patch that shows the diagram.

After each book, the pool is checkpointed to disk.

Dependencies:
    pip install httpx pymupdf tqdm

Usage:
    python ingest_openstax.py [--node http://localhost:8090] [--subject biology]
    python ingest_openstax.py --subject algebra --pages 1-50
    python ingest_openstax.py --url https://openstax.org/books/...  # direct PDF URL

Supported subject shortcuts:
    biology, chemistry, physics, algebra, statistics, anatomy, psychology,
    sociology, economics, history, precalculus, calculus

PDF image resolution: 144 DPI (2x screen) for good zone coverage.
"""

import argparse
import base64
import io
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

# ── OpenStax CDN PDF map ───────────────────────────────────────────────────────
# Each entry: display name → direct PDF URL from the OpenStax CDN.
# Verified working as of 2025.  Run with --list to print all available subjects.
OPENSTAX_BOOKS = {
    "biology":       "https://openstax.org/apps/archive/latest/books/185cbf87-c72e-48f5-b51e-f14f21b5eaeb/files/",
    "biology2e":     "https://assets.openstax.org/oscms-prodcms/media/documents/Biology2e-OP_OkHt3pQ.pdf",
    "chemistry":     "https://assets.openstax.org/oscms-prodcms/media/documents/Chemistry2e-OP.pdf",
    "physics":       "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume1-OP.pdf",
    "physics2":      "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume2-OP.pdf",
    "physics3":      "https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVolume3-OP.pdf",
    "algebra":       "https://assets.openstax.org/oscms-prodcms/media/documents/IntermediateAlgebra2e-OP.pdf",
    "precalculus":   "https://assets.openstax.org/oscms-prodcms/media/documents/Precalculus2e-OP.pdf",
    "calculus1":     "https://assets.openstax.org/oscms-prodcms/media/documents/CalculusVolume1-OP.pdf",
    "statistics":    "https://assets.openstax.org/oscms-prodcms/media/documents/IntroductoryStatistics-OP.pdf",
    "anatomy":       "https://assets.openstax.org/oscms-prodcms/media/documents/AnatomyandPhysiology-OP.pdf",
    "psychology":    "https://assets.openstax.org/oscms-prodcms/media/documents/Psychology2e-OP.pdf",
    "sociology":     "https://assets.openstax.org/oscms-prodcms/media/documents/IntroductiontoSociology3e-OP.pdf",
    "economics":     "https://assets.openstax.org/oscms-prodcms/media/documents/Principles-of-Economics-3e-OP.pdf",
    "history":       "https://assets.openstax.org/oscms-prodcms/media/documents/USHistory-OP.pdf",
    "government":    "https://assets.openstax.org/oscms-prodcms/media/documents/AmericanGovernment3e-OP.pdf",
    "microbiology":  "https://assets.openstax.org/oscms-prodcms/media/documents/Microbiology-OP.pdf",
}


# ── Page rendering ─────────────────────────────────────────────────────────────

DPI = 144  # 2× screen density — good balance of zone resolution vs payload size
JPEG_QUALITY = 75


def render_page_jpeg(page: "fitz.Page") -> bytes:
    """Render a PDF page to JPEG bytes at DPI resolution."""
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return pix.tobytes("jpeg", jpg_quality=JPEG_QUALITY)


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_spans(page: "fitz.Page", page_w: float, page_h: float) -> list[dict]:
    """
    Extract text spans from a page as structured TextSpanReq dicts.

    PyMuPDF's get_text("dict") gives us blocks → lines → spans with:
      - bbox: (x0, y0, x1, y1) in points
      - size: font size in points
      - flags: bold/italic bits
      - text: the actual string

    We map these to the node's TextSpanReq schema with normalised x_frac/y_frac
    and inferred roles from font size relative to the median body size.
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    all_spans = []
    for block in blocks:
        if block.get("type") != 0:  # skip image blocks
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                x0, y0, x1, y1 = bbox
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                size = span.get("size", 12.0)
                flags = span.get("flags", 0)
                bold   = bool(flags & 2**4)  # bit 4 = bold in PyMuPDF
                italic = bool(flags & 2**1)  # bit 1 = italic
                all_spans.append({
                    "text": text,
                    "size": size,
                    "bold": bold,
                    "italic": italic,
                    "x_frac": cx / page_w if page_w > 0 else 0.5,
                    "y_frac": cy / page_h if page_h > 0 else 0.5,
                })

    if not all_spans:
        return []

    # Infer font size thresholds for role classification.
    sizes = sorted(s["size"] for s in all_spans)
    median_size = sizes[len(sizes) // 2]
    heading_thresh    = median_size * 1.5
    subheading_thresh = median_size * 1.2

    total = len(all_spans)
    result = []
    for idx, s in enumerate(all_spans):
        size = s["size"]
        if size >= heading_thresh:
            role = "heading"
        elif size >= subheading_thresh:
            role = "subheading"
        elif s["bold"] and not s["italic"]:
            role = "label"
        elif size < median_size * 0.85:
            role = "footnote"
        else:
            role = "body"

        result.append({
            "text":       s["text"],
            "role":       role,
            "size_ratio": size / median_size if median_size > 0 else 1.0,
            "bold":       s["bold"],
            "italic":     s["italic"],
            "x_frac":     s["x_frac"],
            "y_frac":     s["y_frac"],
            "seq_index":  idx,
            "seq_total":  total,
        })

    return result


# ── Node API ───────────────────────────────────────────────────────────────────

def train_page(client: httpx.Client, node_url: str,
               jpeg_bytes: bytes, spans: list[dict],
               lr_scale: float = 1.0) -> dict:
    """POST a single page (image + text spans) to /media/train."""
    payload = {
        "modality": "page",
        "data_b64": base64.b64encode(jpeg_bytes).decode(),
        "spans":    spans,
        "lr_scale": lr_scale,
    }
    resp = client.post(f"{node_url}/media/train", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def checkpoint(client: httpx.Client, node_url: str) -> None:
    """Flush the NeuronPool to disk."""
    resp = client.post(f"{node_url}/neuro/checkpoint", timeout=10)
    if resp.is_success:
        data = resp.json()
        print(f"  Checkpointed -> {data.get('path', '?')}")
    else:
        print(f"  Checkpoint failed: {resp.status_code}")


# ── Download ───────────────────────────────────────────────────────────────────

def download_pdf(url: str, cache_dir: Path) -> Path:
    """Download a PDF from url, caching by filename under cache_dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    if not filename.endswith(".pdf"):
        filename = "book.pdf"
    dest = cache_dir / filename
    if dest.exists():
        print(f"Using cached PDF: {dest}")
        return dest
    print(f"Downloading {url} ...")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        bar = tqdm(total=total, unit="B", unit_scale=True, desc=filename)
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))
        bar.close()
    print(f"Saved to {dest}")
    return dest


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_page_range(spec: str | None, total: int) -> range:
    """Parse '1-50', '10', or None (all pages). Pages are 1-indexed."""
    if not spec:
        return range(0, total)
    if "-" in spec:
        a, b = spec.split("-", 1)
        return range(int(a) - 1, min(int(b), total))
    n = int(spec)
    return range(n - 1, min(n, total))


def main():
    parser = argparse.ArgumentParser(description="Ingest OpenStax PDFs into the W1z4rD node")
    parser.add_argument("--node",    default="http://localhost:8090", help="Node API base URL")
    parser.add_argument("--subject", default="biology2e", choices=list(OPENSTAX_BOOKS),
                        help="OpenStax subject shortcut")
    parser.add_argument("--url",     default=None, help="Direct PDF URL (overrides --subject)")
    parser.add_argument("--file",    default=None, help="Local PDF file path (overrides download)")
    parser.add_argument("--pages",   default=None, help="Page range to ingest, e.g. '1-100'")
    parser.add_argument("--lr",      type=float, default=1.0, help="Hebbian learning rate scale")
    parser.add_argument("--cache",   default="data/pdf_cache", help="Local PDF cache directory")
    parser.add_argument("--checkpoint-every", type=int, default=50, dest="ckpt_every",
                        help="Flush pool to disk every N pages (default 50)")
    parser.add_argument("--list",    action="store_true", help="List available subjects and exit")
    args = parser.parse_args()

    if args.list:
        print("Available subjects:")
        for name, url in OPENSTAX_BOOKS.items():
            print(f"  {name:<18} {url}")
        return

    # -- Load PDF
    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            sys.exit(f"File not found: {pdf_path}")
    else:
        url = args.url or OPENSTAX_BOOKS.get(args.subject)
        if not url:
            sys.exit(f"Unknown subject: {args.subject}. Use --list to see options.")
        pdf_path = download_pdf(url, Path(args.cache))

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    page_range = parse_page_range(args.pages, total_pages)
    print(f"Ingesting {pdf_path.name}: pages {page_range.start+1}-{page_range.stop} of {total_pages}")

    # -- Check node health
    with httpx.Client() as hc:
        try:
            health = hc.get(f"{args.node}/health", timeout=5).json()
            print(f"Node: {health.get('node_id')} — {health.get('status')}")
        except Exception as e:
            sys.exit(f"Node not reachable at {args.node}: {e}")

    # -- Ingest pages
    trained = 0
    skipped = 0
    start_time = time.time()

    with httpx.Client() as client:
        pages_iter = tqdm(page_range, total=len(page_range), desc="Pages", unit="pg")
        for page_idx in pages_iter:
            page = doc[page_idx]
            pw, ph = page.rect.width, page.rect.height

            # Render page to JPEG
            jpeg_bytes = render_page_jpeg(page)

            # Extract text spans
            spans = extract_spans(page, pw, ph)

            if not spans and len(jpeg_bytes) < 2000:
                # Blank or near-blank page — skip
                skipped += 1
                continue

            try:
                result = train_page(client, args.node, jpeg_bytes, spans, lr_scale=args.lr)
                trained += 1
                label_count = result.get("label_count", 0)
                pages_iter.set_postfix(labels=label_count, trained=trained)
            except httpx.HTTPStatusError as e:
                tqdm.write(f"  Page {page_idx+1} error: {e.response.status_code} {e.response.text[:80]}")
                skipped += 1
            except Exception as e:
                tqdm.write(f"  Page {page_idx+1} error: {e}")
                skipped += 1

            # Periodic checkpoint
            if trained > 0 and trained % args.ckpt_every == 0:
                tqdm.write(f"  [{trained} pages trained] checkpointing...")
                checkpoint(client, args.node)

        # Final checkpoint
        print(f"\nFinal checkpoint after {trained} pages...")
        checkpoint(client, args.node)

    elapsed = time.time() - start_time
    print(f"\nDone. trained={trained} skipped={skipped} elapsed={elapsed:.1f}s "
          f"({trained/elapsed:.1f} pages/sec)" if elapsed > 0 else "")


if __name__ == "__main__":
    main()
