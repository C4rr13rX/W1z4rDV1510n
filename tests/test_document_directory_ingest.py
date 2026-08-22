"""Tests for compiling a directory of mixed documents into a corpus.

The tool exists so a folder of arbitrary documents can be turned into the
JSONL the curriculum already consumes. The properties that matter are that it
pairs honestly, refuses what it cannot pair rather than inventing data, and
emits rows that satisfy the same contract every other ingest module does.
"""
import json

from tools.training_standard.ingest import document_directory as dd

REQUIRED_FIELDS = {"prompt", "response", "ctx", "license", "source",
                   "source_hash", "script_id"}


def build_source(tmp_path):
    src = tmp_path / "docs"
    (src / "sub").mkdir(parents=True)
    (src / "guide.md").write_text(
        "# Installing the widget\n"
        "Run the installer and accept the defaults. The service starts\n"
        "automatically and listens on port 8080 when installation ends.\n\n"
        "## Configuring retries\n"
        "Set retry_limit in config.toml. Values above ten are clamped, and\n"
        "zero disables retries entirely for that endpoint.\n",
        encoding="utf-8")
    (src / "helpers.py").write_text(
        '\ndef normalise_email(value):\n'
        '    """Lowercase an email address and strip surrounding whitespace."""\n'
        '    return str(value).strip().lower()\n',
        encoding="utf-8")
    (src / "sub" / "pairs.jsonl").write_text(
        json.dumps({"question": "How do I restart the service safely?",
                    "answer": "Stop it, wait for the port to clear, then "
                              "start it again."}) + "\n",
        encoding="utf-8")
    (src / "notes.bin").write_bytes(b"\x00\x01\x02binary")
    return src


def compile_dir(tmp_path, src):
    out = tmp_path / "corpus.jsonl"
    stats = dd.build(src, out, "cc0-1.0", "docs", "domain_docs_001",
                     "implement")
    rows = [json.loads(line) for line in
            out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stats, rows


def test_mixed_directory_compiles_to_the_row_contract(tmp_path):
    stats, rows = compile_dir(tmp_path, build_source(tmp_path))

    assert stats.rows == len(rows) > 0
    for row in rows:
        assert set(row) == REQUIRED_FIELDS
        assert row["prompt"].strip() and row["response"].strip()
        assert row["license"] == "cc0-1.0"
        assert row["script_id"] == "domain_docs_001"
        assert set(row["ctx"]) == {"lang", "intent", "source"}
    # Provenance names the file and how the pair was derived.
    assert any(r["source"].endswith("guide.md#section") for r in rows)
    assert any(r["source"].endswith("pairs.jsonl#record") for r in rows)


def test_every_pairing_strategy_contributes(tmp_path):
    _, rows = compile_dir(tmp_path, build_source(tmp_path))
    kinds = {row["source"].rsplit("#", 1)[1] for row in rows}
    assert {"section", "record", "definition"} <= kinds


def test_binary_and_unpairable_files_are_ignored_not_guessed(tmp_path):
    src = build_source(tmp_path)
    (src / "opaque.md").write_text("no headings, just a loose sentence.\n",
                                   encoding="utf-8")
    _, rows = compile_dir(tmp_path, src)

    assert not any("notes.bin" in row["source"] for row in rows)
    assert not any("opaque.md" in row["source"] for row in rows)


def test_a_syntax_error_does_not_discard_the_whole_file(tmp_path):
    """One bad definition must not lose the good ones beside it."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "mixed.py").write_text(
        '\ndef normalise_email(value):\n'
        '    """Lowercase an email address and strip surrounding whitespace."""\n'
        '    return str(value).strip().lower()\n\n\n'
        'def broken(:\n'
        '    pass\n',
        encoding="utf-8")

    _, rows = compile_dir(tmp_path, src)

    assert len(rows) == 1
    assert "normalise_email" in rows[0]["response"]
    assert "broken" not in rows[0]["response"]


def test_python_responses_that_do_not_parse_are_refused(tmp_path):
    """The salvage must not lower the bar on what is emitted."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "only_broken.py").write_text(
        'def broken(:\n'
        '    """This claims to do something useful."""\n'
        '    pass\n',
        encoding="utf-8")

    stats, rows = compile_dir(tmp_path, src)

    assert rows == []
    assert stats.rows == 0


def test_identical_pairs_are_deduplicated(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    record = json.dumps({"prompt": "How do I restart the service safely?",
                         "response": "Stop it, wait for the port to clear, "
                                     "then start it again."})
    (src / "a.jsonl").write_text(record + "\n" + record + "\n",
                                 encoding="utf-8")

    stats, rows = compile_dir(tmp_path, src)

    assert len(rows) == 1
    assert stats.skipped_duplicate == 1


def test_a_non_permissive_license_is_rejected(tmp_path):
    """Licence validation is the writer's, and must not be bypassed."""
    src = build_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    try:
        dd.build(src, out, "gpl-3.0", "docs", "domain_docs_001", "implement")
    except Exception as exc:  # RowRejected, surfaced by the writer
        assert "gpl-3.0" in str(exc).lower()
    else:
        raise AssertionError("a copyleft licence must not be accepted")


# --- textbook (PDF) ingestion -------------------------------------------------
#
# A folder of textbooks produced ZERO rows before 2026-08-22: layout segments
# are per-fragment records, so re-keying them one at a time found no pair, and
# PDFs were not read at all.

def test_layout_segments_rebuild_sections(tmp_path):
    """A heading segment opens a section; following fragments are its body."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "book.ndjson").write_text("\n".join(json.dumps(s) for s in [
        {"book": "X", "page": 1, "label": "title", "text": "Operating Ratios"},
        {"book": "X", "page": 1, "label": "list",
         "text": "Operating ratios measure how effectively a company is "
                 "utilizing its assets over the accounting period."},
    ]) + "\n", encoding="utf-8")

    _, rows = compile_dir(tmp_path, src)

    assert len(rows) == 1
    assert rows[0]["prompt"] == "Operating Ratios"
    assert "utilizing its assets" in rows[0]["response"]


def test_a_structural_heading_is_qualified_by_its_subject(tmp_path):
    """"Learning Outcomes" alone teaches nothing about any subject."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "book.ndjson").write_text("\n".join(json.dumps(s) for s in [
        {"book": "X", "page": 1, "label": "title", "text": "Operating Ratios"},
        {"book": "X", "page": 1, "label": "list",
         "text": "Operating ratios measure how effectively assets are used "
                 "across a full accounting period by the business."},
        {"book": "X", "page": 2, "label": "title", "text": "Learning Outcomes"},
        {"book": "X", "page": 2, "label": "list",
         "text": "Compute the average inventory turnover ratio and interpret "
                 "the result for a retail business."},
    ]) + "\n", encoding="utf-8")

    _, rows = compile_dir(tmp_path, src)
    prompts = {r["prompt"] for r in rows}

    assert "Operating Ratios: Learning Outcomes" in prompts
    assert "Learning Outcomes" not in prompts


def test_a_contents_listing_is_not_teaching_content():
    """Calibrated on real rows: ToC 3.57 per 100 chars, prose 0.00."""
    assert dd.is_table_of_contents(
        "9.1: Production Budget 9.2: Direct Materials Budget 9.3: Direct "
        "Labor Budget 9.4: Manufacturing Overhead Budget 9.5: Ending Inventory"
    )
    assert not dd.is_table_of_contents(
        "These are the managers involved in the day to day manufacturing "
        "process. They determine the schedule and staffing."
    )
    # Prose that merely cites one subsection must survive.
    assert not dd.is_table_of_contents(
        "As shown in section 5.11: Strategy Development, accounting affects "
        "how a company plans its direction and allocates its resources."
    )


def test_non_commercial_licences_are_never_permitted():
    """110 of 183 LibreTexts books are CC BY-NC and cannot be used."""
    from tools.training_standard.row import (
        NON_COMMERCIAL_LICENSES, PERMISSIVE_LICENSES,
    )
    assert not (NON_COMMERCIAL_LICENSES & PERMISSIVE_LICENSES)
    for licence in ("cc-by-nc", "cc-by-nc-4.0", "cc-by-nc-sa", "cc-by-nd"):
        assert licence not in PERMISSIVE_LICENSES
    # Commercial CC licences ARE permitted; attribution is carried by `source`.
    for licence in ("cc-by-4.0", "cc-by-sa-4.0", "public-domain"):
        assert licence in PERMISSIVE_LICENSES


def test_a_wrapped_licence_stamp_is_still_read():
    """PDF extraction wraps lines mid-licence.

    Measured 2026-08-22 in FinancialAccountingOpenStax, the stamp extracts as
    "shared under a CC\nBY-NC-SA 4.0". A space-only pattern missed it
    entirely, so the book was refused for the WRONG reason -- no licence found
    rather than a restricted one -- and the same wrap in a CC BY book would
    have silently discarded usable material.
    """
    for text in ("shared under a CC\nBY-NC-SA 4.0 license",
                 "shared under a CC BY 4.0 license",
                 "shared under a CC\nBY-SA 4.0 license",
                 "shared under a\nCC BY-NC 4.0 license"):
        assert dd.PDF_LICENCE_PATTERN.search(text), text


def test_openstax_licences_are_read_per_book_not_per_publisher():
    """The document is the evidence, not the publisher's summary page.

    OpenStax's licensing page states all their textbooks are CC BY-NC-SA.
    Their own PDFs disagree: Chemistry 2e stamps CC BY 4.0 while Financial
    Accounting stamps CC BY-NC-SA 4.0. Trusting the publisher would have
    discarded usable books; trusting a folder-wide flag would have imported
    restricted ones.
    """
    def resolve(stamp):
        import re
        match = dd.PDF_LICENCE_PATTERN.search(stamp)
        if not match:
            return None
        key = re.sub(r"[\s_-]+", " ", match.group(1).strip().lower())
        if "nc" in key.split() or "nd" in key.split():
            return None
        return dd.LICENCE_SPDX.get(key)

    assert resolve("shared under a CC BY 4.0 license") == "cc-by-4.0"
    assert resolve("shared under a CC\nBY-NC-SA 4.0 license") is None
    assert resolve("shared under a CC BY-SA 4.0 license") == "cc-by-sa-4.0"
    assert resolve("shared under a CC BY-ND 4.0 license") is None


def test_pdf_typography_is_normalised():
    """Ligatures and hyphenation are typesetting, not content.

    Measured 2026-08-22 over two CS textbooks before the fix: 578 ligature
    glyphs and 483 hyphen-split words. Left in, "first" is stored as "ﬁrst"
    and "randomization" as "ran- domization", so recall fails on exactly the
    words the section is about.
    """
    assert dd.normalise_pdf_text("the ﬁrst diﬃcult ﬂag") == "the first difficult flag"
    assert dd.normalise_pdf_text("uses ran- domization to") == "uses randomization to"
    assert dd.normalise_pdf_text("the tree\u2019s root") == "the tree's root"
    # A genuine hyphenated compound must survive.
    assert dd.normalise_pdf_text("a red-black tree") == "a red-black tree"


def test_a_prose_only_licence_is_honoured():
    """Open Data Structures never writes "CC BY"; it says so in prose.

    A code-only pattern reported NO licence for a book that states
    "Creative Commons Attribution license... including the right to make
    commercial use of the work", and it was silently discarded.
    """
    assert dd.PROSE_LICENCE_HINTS[0][0].search(
        "released under a Creative Commons Attribution license"
    )
    assert dd.COMMERCIAL_GRANT.search(
        "the right to make commercial use of the work"
    )
    # A restriction still wins when no explicit grant is present.
    assert dd.NON_COMMERCIAL_HINT.search("Attribution-NonCommercial 4.0")


def test_discovered_structural_headings_are_qualified():
    """Found empirically: 23 bare "Discussion and Exercises" prompts."""
    assert "discussion and exercises" in dd.STRUCTURAL_HEADINGS
    assert "chapter notes" in dd.STRUCTURAL_HEADINGS


def test_an_oversized_section_is_split_not_discarded():
    """A chapter past the cap used to be thrown away whole.

    Measured 2026-08-22 on Open Data Structures: 19 sections ran past the
    response cap and took 51% of the book's prose with them, including a
    25,028-character B-Trees chapter. Splitting recovered it -- 76 rows and
    ~221K characters became 152 rows and 726K.
    """
    sentence = "This sentence explains one idea about the data structure. "
    long_body = sentence * 400  # ~23K chars, well past the cap
    chunks = dd.split_long_body(long_body)

    assert len(chunks) > 1
    assert all(len(chunk) <= dd.MAX_RESPONSE for chunk in chunks)
    assert all(len(chunk) >= dd.MIN_RESPONSE for chunk in chunks)
    # Splitting must not lose most of the material.
    assert sum(len(c) for c in chunks) > 0.9 * len(long_body.strip())
    # A body already within the cap is returned untouched.
    assert dd.split_long_body("short body text that fits") == [
        "short body text that fits"
    ]


def test_a_ligature_does_not_truncate_a_heading():
    """A ligature gets its own PDF span, splitting the heading line.

    "SEList: A Space-Efficient Linked List" arrives as four spans broken at
    the "ffi". Judged span-by-span, the 2-character ligature span failed the
    heading test and the heading became "cient Linked List".
    """
    joined = dd.normalise_pdf_text("".join(
        ["SEList: A Space-E", "\ufb03", "cient Linked List"]
    ))
    assert joined == "SEList: A Space-Efficient Linked List"


def test_a_page_declaring_restrictive_rights_is_not_commercial_safe():
    """NC/ND pages are dropped however the page happens to word it.

    Measured across the 63 staged LibreTexts books, three wordings each
    escaped a sentence-shaped licence pattern and put non-commercial content
    into a corpus sold as commercial-safe:

      * "is licensed CC BY-NC-SA 4.0"  -- no "under"
      * "(CC BY-NC-SA 3.0; Anonymous)" -- a figure credit, no verb at all
      * "creativecommons.org/licenses/by-nc-sa/3.0/us/" -- a bare URL

    Together these accounted for 30 leaked pages in OrganicChemistryMorschEtAl
    alone.
    """
    for text in (
        "shared under a CC BY-NC 4.0 license and was authored by",
        "This figure is licensed CC BY-NC-SA 4.0. Originally from",
        "The relative potential energy of atomic orbitals. (CC BY-NC; Lower)",
        "revision=1, http://creativecommons.org/licenses/by-nc-sa/3.0/us/)",
        "figure source http://creativecommons.org/licenses/by-nc- sa/3.0/us/",
        "shared under a not declared license and was authored, remixed",
    ):
        assert not dd.page_is_commercial_safe(text), text


def test_permissive_and_ordinary_pages_stay_commercial_safe():
    """The page filter must not eat the book.

    A surname ("Ndiaye"), a subject-matter mention ("noncommercial
    fisheries"), and an ordinary body page carry no licence at all. An
    earlier audit pattern matched the bare letters ND and NC and flagged 27
    such pages, which would have discarded good prose.
    """
    for text in (
        "shared under a CC BY-SA 4.0 license and was authored by",
        "Figure 3 is licensed CC BY 4.0",
        "described in detail by Ndiaye et al. (2019) for these genotypes",
        "Subsistence fisheries are local, noncommercial fisheries",
        "The mole is the SI unit for amount of substance.",
    ):
        assert dd.page_is_commercial_safe(text), text


def test_regex_escapes_survived_the_source_file():
    """A corrupted escape makes a detector match nothing, silently.

    `\n` written as a literal newline (and `\b` as a backspace) has broken a
    pattern in this file before: it still compiles, still runs, and quietly
    stops matching. Assert on the compiled pattern, not on behaviour.
    """
    for pattern in (dd.BARE_RESTRICTIVE_CODE, dd.PDF_LICENCE_PATTERN,
                    dd.NON_COMMERCIAL_HINT, dd.TOC_ENTRY):
        assert "\n" not in pattern.pattern
        assert "\b" not in pattern.pattern


def test_a_body_size_line_can_still_be_a_heading():
    """Half of all real headings share the body's type size.

    Measured against 1,473 labelled textbook segments: 170 of 341 true
    headings sit at exactly the body font size, alongside 213 body
    paragraphs. A size threshold alone therefore found only 34% of headings
    and scored F1 0.41 at EVERY ratio from 1.02 to 1.5 -- the constant was
    never the problem, the signal was.
    """
    for title in ("The Tragedy of the Commons", "Environmental Justice",
                  "1.4: Framework for Project Management", "Appendix",
                  "Summary", "Chapter 4: Ecosystems"):
        assert dd.is_title_shaped(title), title


def test_a_broken_body_line_is_not_a_heading():
    """Short body lines must not become section prompts.

    A false heading is worse than a missed one: it fabricates a training
    prompt for text that does not answer it. These are the shapes that
    actually appeared when body-size headings were first admitted.
    """
    for line in ("shown in Figure",
                 "as shown in Table",
                 "including certain types of bacteria and algae (Figure",
                 "The standards for these units are",
                 "people.\u201d (Gifford Pinchot, 1913)",
                 "https://chem.libretexts.org/@go/page/358665",
                 "19"):
        assert not dd.is_title_shaped(line), line


def test_a_thin_section_merges_instead_of_being_dropped():
    """Undersized sections fold backwards, so no prose is lost.

    Dropping them tamed ChemistryAtomsFirst2eOpenStax's 9,443-section
    shatter but cost 816k characters -- 34% of the book. Merging keeps
    93-97% of every book's text while still roughly doubling the count of
    correctly-titled sections.
    """
    assert dd.MIN_SECTION_BODY > dd.MIN_RESPONSE
