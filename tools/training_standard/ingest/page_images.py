#!/usr/bin/env python3
"""Turn rendered textbook pages into scale-varied visual reading rows.

A PDF carries both the picture of a page and, for every word on it, that
word's exact bounding box. Rendering the page and cropping to the box
therefore produces a labelled example *for free*: the crop is the input,
the word is the answer, and nothing was annotated by hand or by a model.

Rendering the same crop at several zooms is the point rather than a
convenience. One word at 7 pixels tall and the same word at 38 pixels is a
single identity presented at several apparent distances, so a fabric that
learns to read it has learned scale invariance from data instead of from a
resizing rule someone wrote down. That is the same invariance a camera needs
to recognise one person near and far, arrived at on a corpus where the
ground truth is exact and free.

Every emitted row is a normal training row -- prompt, response, licence,
provenance -- so this path feeds the same curriculum as the text ingesters
and inherits their per-page licence filtering.
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

from ..row import Row, RowWriter, hash_source
from .document_directory import page_is_commercial_safe, pdf_licence

#: Zoom factors applied to every crop. Spans a ~5x range so the same glyph is
#: seen from "near" and "far"; the midpoint matches the 2.0x used to render
#: the existing page PNGs.
DEFAULT_ZOOMS = (0.6, 1.0, 1.6, 2.4, 3.2)

#: A crop shorter than this carries no legible glyph detail at all -- below
#: roughly six pixels of x-height even a human cannot recover the word, so a
#: row built from it would teach noise.
MIN_CROP_PIXELS = 6

#: Words longer than this are usually a run-on extraction artefact rather
#: than a word, and words shorter than two characters carry little signal.
MIN_WORD_CHARS = 2
MAX_WORD_CHARS = 32

#: Cap per page so one dense page cannot dominate a book, and per book so one
#: book cannot dominate the corpus.
WORDS_PER_PAGE = 12
PAGES_PER_BOOK = 40


def crop_to_png(page, box, zoom: float) -> bytes | None:
    """Render just `box` of `page` at `zoom`, as PNG bytes."""
    import fitz

    matrix = fitz.Matrix(zoom, zoom)
    try:
        pixmap = page.get_pixmap(matrix=matrix, clip=fitz.Rect(*box))
    except Exception:
        return None
    if pixmap.height < MIN_CROP_PIXELS or pixmap.width < MIN_CROP_PIXELS:
        return None
    try:
        return pixmap.tobytes("png")
    except Exception:
        return None


def readable_words(page) -> list[tuple[tuple[float, float, float, float], str]]:
    """Word boxes on a page, filtered to ones worth learning from."""
    try:
        words = page.get_text("words")
    except Exception:
        return []
    out = []
    for x0, y0, x1, y1, text, *_ in words:
        text = str(text).strip()
        if not (MIN_WORD_CHARS <= len(text) <= MAX_WORD_CHARS):
            continue
        # A word with no letters is a page number, a rule, or a stray mark.
        if not any(character.isalpha() for character in text):
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        out.append(((x0, y0, x1, y1), text))
    return out


def rows_from_pdf(path: Path, zooms=DEFAULT_ZOOMS):
    """Yield (prompt, response, licence, meta) for one book's page crops."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return
    try:
        document = fitz.open(path)
    except Exception:
        return
    try:
        licence = pdf_licence(document)
        if licence is None:
            return
        # Sample pages across the whole book rather than the first N, so a
        # book contributes its body text and not just its front matter.
        step = max(1, document.page_count // PAGES_PER_BOOK)
        for index in range(0, document.page_count, step):
            try:
                page = document[index]
                if not page_is_commercial_safe(page.get_text()):
                    continue
            except Exception:
                continue
            words = readable_words(page)
            if not words:
                continue
            # Spread the sample across the page instead of taking the first
            # dozen words, which would only ever be the running header.
            stride = max(1, len(words) // WORDS_PER_PAGE)
            for box, text in words[::stride][:WORDS_PER_PAGE]:
                for zoom in zooms:
                    png = crop_to_png(page, box, zoom)
                    if png is None:
                        continue
                    encoded = base64.b64encode(png).decode("ascii")
                    yield (
                        f"[image png b64] {encoded}",
                        text,
                        licence,
                        {"book": path.stem, "page": index, "zoom": zoom},
                    )
    finally:
        document.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="directory of PDFs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="page_images")
    parser.add_argument("--script-id", default="vision_page_images_001")
    parser.add_argument("--limit-books", type=int, default=0,
                        help="0 = every book in --src")
    parser.add_argument("--zooms", default=",".join(str(z) for z in DEFAULT_ZOOMS))
    args = parser.parse_args()

    zooms = tuple(float(z) for z in args.zooms.split(",") if z.strip())
    books = sorted(args.src.glob("*.pdf"))
    if args.limit_books:
        books = books[:args.limit_books]

    written = skipped = 0
    writer = RowWriter(args.out, script_id=args.script_id, source=args.label)
    try:
        for book in books:
            for prompt, response, licence, meta in rows_from_pdf(book, zooms):
                source = (f"{args.label}:{meta['book']}"
                          f"#p{meta['page']}@{meta['zoom']}x")
                row = Row(
                    prompt=prompt,
                    response=response,
                    ctx={"lang": "image", "intent": "read",
                         "source": args.label},
                    license=licence,
                    source=source,
                    # The crop bytes are the observation, so they -- not the
                    # word alone -- identify the row. Hashing the word only
                    # would make all five zooms of one word collide and the
                    # dedup would silently discard the scale variation this
                    # module exists to produce.
                    source_hash=hash_source(f"{source}\n{prompt}"),
                    script_id=args.script_id,
                )
                if writer.write(row):
                    written += 1
                else:
                    skipped += 1
            print(f"  {book.stem[:52]:<54} rows={written:,}", flush=True)
    finally:
        writer.close()
    print(f"wrote {args.out}\n  rows written {written:,}\n  skipped {skipped:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
