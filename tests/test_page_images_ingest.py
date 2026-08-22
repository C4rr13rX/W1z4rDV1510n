"""Tests for the scale-varied page-image ingester.

The module turns a PDF page into labelled visual reading rows by cropping to
each word's own bounding box. The tests below pin the properties that make
those rows worth training on: the label is exact, the same word survives at
several scales, and nothing restricted or illegible gets through.
"""
from __future__ import annotations

import base64
import io

import pytest

from tools.training_standard.ingest import page_images as pi

fitz = pytest.importorskip("fitz")
Image = pytest.importorskip("PIL.Image")


def _one_page_pdf(tmp_path, text="Photosynthesis converts light energy"):
    """A minimal real PDF carrying a permissive licence stamp."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 200), text, fontsize=18)
    page.insert_text((72, 400),
                     "This page is shared under a CC BY 4.0 license.",
                     fontsize=9)
    path = tmp_path / "book.pdf"
    document.save(path)
    document.close()
    return path


def test_a_word_is_emitted_at_every_zoom(tmp_path):
    """The same word at several scales is the whole point of this path.

    One glyph at 7 pixels and the same glyph at 38 is one identity at
    several apparent distances. If dedup or filtering collapsed the zooms,
    the module would emit ordinary OCR rows and teach no invariance at all.
    """
    path = _one_page_pdf(tmp_path)
    rows = list(pi.rows_from_pdf(path, zooms=(0.6, 1.0, 1.6, 2.4, 3.2)))
    assert rows
    by_word: dict[str, set[float]] = {}
    for _prompt, response, _licence, meta in rows:
        by_word.setdefault(response, set()).add(meta["zoom"])
    assert any(len(zooms) == 5 for zooms in by_word.values()), by_word


def test_the_crop_is_a_real_decodable_image(tmp_path):
    """A row's prompt must carry a PNG that actually opens.

    A silently-empty or truncated crop would still look like a valid row in
    the JSONL and would train the fabric on nothing.
    """
    path = _one_page_pdf(tmp_path)
    prompt, response, _licence, _meta = next(iter(pi.rows_from_pdf(path)))
    assert prompt.startswith("[image png b64] ")
    payload = base64.b64decode(prompt.split(" ", 3)[3])
    with Image.open(io.BytesIO(payload)) as image:
        assert image.width >= pi.MIN_CROP_PIXELS
        assert image.height >= pi.MIN_CROP_PIXELS
    assert response.strip() == response and response


def test_a_restricted_book_yields_nothing(tmp_path):
    """Licence filtering applies to the visual path too.

    The images come from the same books as the text rows, so a page the text
    ingester refuses must not re-enter the corpus as a picture of itself.
    """
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 200), "Restricted material here", fontsize=18)
    page.insert_text((72, 400),
                     "This page is shared under a CC BY-NC 4.0 license.",
                     fontsize=9)
    path = tmp_path / "restricted.pdf"
    document.save(path)
    document.close()
    assert list(pi.rows_from_pdf(path)) == []


def test_unreadable_fragments_are_filtered(tmp_path):
    """Page numbers and stray marks are not words.

    `readable_words` must drop anything with no letters, so a row is never
    built to teach that a crop of "19" reads as the word "19".
    """
    path = _one_page_pdf(tmp_path, text="Chapter 19 42 -- Photosynthesis")
    responses = {response for _p, response, _l, _m in pi.rows_from_pdf(path)}
    assert responses
    assert all(any(c.isalpha() for c in word) for word in responses)
    assert all(pi.MIN_WORD_CHARS <= len(word) <= pi.MAX_WORD_CHARS
               for word in responses)
