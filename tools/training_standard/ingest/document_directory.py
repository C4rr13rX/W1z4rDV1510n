"""ingest/document_directory.py — turn a directory of mixed documents into a corpus.

Point it at a folder of documents and it compiles what it finds into the
JSONL the curriculum already consumes, through the same `RowWriter` contract
every other ingest module uses: permissive-license validation, `source_hash`
dedup, and a provenance manifest.

Handles heterogeneous directories rather than one repo shape:

  .md .markdown .rst .txt   prose -> the section that follows it
  .py .js .ts .rs .go       a documented definition -> its source
  .java .cs .rb .sh .sql
  .json .jsonl              already-paired records, re-keyed to the contract
  .csv .tsv                 a prompt-ish and response-ish column pair

Pairing strategy, in the order it is attempted per file:

  1. **Existing pairs.** JSON/JSONL/CSV records that already carry something
     prompt-like and response-like are re-keyed, never re-derived. This is
     the highest-fidelity source and is preferred wherever it exists.
  2. **Documented definitions.** For source files, a function or class whose
     docstring/leading comment states what it does becomes
     `prompt = that description`, `response = the definition`.
  3. **Heading sections.** For prose, a heading becomes the prompt and the
     text under it the response.

Anything that matches none of these is skipped and counted, not guessed at.
A document the tool cannot honestly pair is a document it should not invent
training data from.

Quality gates mirror `markdown_book.py`: prompt 16-2000 chars, response
30-8000 chars, and Python responses must parse. Everything is reported, so a
run tells you what it took and what it left behind.

CLI:
    python -m tools.training_standard.ingest.document_directory \\
        --src D:/docs/my_manuals \\
        --out D:/w1z4rdv1510n-data/training/my_manuals.jsonl \\
        --license cc0-1.0 \\
        --label "my_manuals" \\
        --script-id domain_my_manuals_001
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path

from tools.training_standard.row import Row, RowWriter, hash_source

#: Extension -> the `ctx.lang` atom recorded for it.
SOURCE_LANGUAGES = {
    ".py": "python", ".js": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
    ".rs": "rust", ".go": "go", ".java": "java", ".cs": "csharp",
    ".rb": "ruby", ".sh": "bash", ".bash": "bash", ".sql": "sql",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
}
PROSE_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}
RECORD_SUFFIXES = {".json", ".jsonl", ".ndjson"}
TABLE_SUFFIXES = {".csv", ".tsv"}
DOCUMENT_SUFFIXES = {".pdf"}
#: Formats parsed from the path as binary, like PDF.
WORD_SUFFIXES = {".docx"}
WORKBOOK_SUFFIXES = {".xlsx", ".xlsm"}
#: Decoded as text first, then parsed for structure.
HTML_SUFFIXES = {".html", ".htm", ".xhtml"}
#: Every format read as bytes rather than decoded as text.
BINARY_SUFFIXES = DOCUMENT_SUFFIXES | WORD_SUFFIXES | WORKBOOK_SUFFIXES

#: Heading detection for PDFs: a span this much larger than the page's most
#: common (body) size opens a section. Measured over LibreTexts textbooks,
#: body text sits at ~9.6pt with section titles at 13.5 and subheadings at
#: 11.2, so a 15% margin separates them without needing a trained segmenter.
PDF_HEADING_RATIO = 1.15

#: LibreTexts stamps a per-page attribution line. Read the licence from the
#: document itself rather than assuming one for a whole folder: measured over
#: 183 books, 110 are CC BY-NC (no commercial use) and only 63 are permissive,
#: so a folder-wide assumption would quietly poison a commercial corpus.
#: `\s` rather than a literal space throughout: PDF text extraction wraps
#: lines mid-licence. Measured 2026-08-22 in FinancialAccountingOpenStax the
#: stamp extracts as "shared under a CC\nBY-NC-SA 4.0", which a space-only
#: pattern misses entirely. That book was then refused for the WRONG reason --
#: no licence found rather than a restricted one -- and the same wrap in a
#: CC BY book would have silently discarded usable material.
PDF_LICENCE_PATTERN = re.compile(
    r"(?:shared|licen[cs]ed|released|distributed|published|available)\s+under\s+"
    r"(?:the\s+terms\s+of\s+)?(?:an?\s+|the\s+)?"
    r"(CC\s*[-\s]\s*BY(?:\s*[-\s]\s*SA|\s*[-\s]\s*NC(?:\s*[-\s]\s*ND)?"
    r"|\s*[-\s]\s*ND)?(?:\s*\d\.\d)?"
    r"|Creative\s+Commons\s+Attribution(?:[-\s]+(?:Share\s*Alike|"
    r"Non\s*-?\s*Commercial|No\s*Derivatives))*(?:\s*\d\.\d)?"
    r"|public\s+domain"
    # "shared under a not declared license" is LibreTexts' own wording for a
    # page whose rights were never established. It has to be matchable so it
    # can disqualify a book -- while it was unmatched, 69 such pages in
    # IntroductionToOrganicSpectroscopy were invisible and 5 stray CC BY pages
    # decided the whole book.
    r"|not\s+declared)",
    re.IGNORECASE,
)

#: A licence named in prose rather than by code. Open Data Structures says
#: "Creative Commons Attribution license, meaning that anyone is free to
#: share... even commercially" and never writes "CC BY" at all, so a
#: code-only pattern reported NO licence for a plainly usable book. The
#: document is the evidence; the vocabulary it happens to use is not.
PROSE_LICENCE_HINTS = (
    (re.compile(r"creative\s+commons\s+attribution", re.I), "cc-by-4.0"),
    (re.compile(r"public\s+domain", re.I), "public-domain"),
)

#: An explicit grant of commercial use, stated in prose. Open Data Structures
#: says "including the right to make commercial use of the work" -- the
#: document settling the question in its own words.
COMMERCIAL_GRANT = re.compile(
    r"commercial use of the work|even commercially"
    r"|right to make commercial use",
    re.IGNORECASE,
)

#: Terms that make a licence non-commercial no matter how it is phrased.
#: A restrictive Creative Commons code appearing on its own -- in a figure
#: credit or an inline attribution -- without the "shared under" sentence the
#: main pattern looks for. Matched separately so a page can be dropped on it.
BARE_RESTRICTIVE_CODE = re.compile(
    r"CC\s*[-\s]\s*BY(?:\s*[-\s]\s*(?:SA|NC|ND))*\s*[-\s]\s*(?:NC|ND)"
    r"(?:\s*[-\s]\s*(?:SA|NC|ND))*"
    r"|Creative\s+Commons[^.\n]{0,40}?(?:Non\s*-?\s*Commercial|No\s*Derivatives)"
    # A licence URL, which carries no CC-code text at all. Measured across the
    # 63 books, 11 pages cited only
    # "creativecommons.org/licenses/by-nc-sa/3.0/us/" -- a complete and
    # unambiguous restrictive declaration that every text-shaped pattern above
    # misses. The optional space absorbs a PDF line break inside the path
    # ("by-nc- sa"), seen in OrganicChemistry.
    r"|creativecommons\.org/licenses/by(?:-\s?(?:sa|nc|nd))*-\s?(?:nc|nd)"
    r"(?:-\s?(?:sa|nc|nd))*",
    re.IGNORECASE,
)

NON_COMMERCIAL_HINT = re.compile(
    r"non\s*-?\s*commercial|no\s*n?\s*-?\s*deriv|\bNC\b|\bND\b", re.I
)

#: Field names treated as already carrying a prompt or a response. Ordered:
#: the first match wins, so the most explicit naming is preferred.
PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query",
               "title", "request")
RESPONSE_KEYS = ("response", "answer", "output", "completion", "content",
                 "body", "text", "code")

MIN_PROMPT, MAX_PROMPT = 16, 2000
MIN_RESPONSE, MAX_RESPONSE = 30, 8000


class Stats:
    """What a run took and what it left behind."""

    def __init__(self) -> None:
        self.files_seen = 0
        self.files_used = 0
        self.rows = 0
        self.skipped_unpairable = 0
        self.skipped_too_short = 0
        self.skipped_too_long = 0
        self.skipped_bad_syntax = 0
        self.skipped_duplicate = 0
        self.skipped_contents = 0
        self.unreadable: list[str] = []

    def report(self) -> str:
        return (
            f"  files scanned      {self.files_seen}\n"
            f"  files contributing {self.files_used}\n"
            f"  rows written       {self.rows}\n"
            f"  skipped: unpairable {self.skipped_unpairable}, "
            f"too-short {self.skipped_too_short}, "
            f"too-long {self.skipped_too_long}, "
            f"bad-syntax {self.skipped_bad_syntax}, "
            f"duplicate {self.skipped_duplicate}\n"
            f"  unreadable         {len(self.unreadable)}"
        )


def read_text(path: Path) -> str | None:
    """Decode a file, or None when it is not text we can use."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, OSError):
            continue
    return None


def first_present(record: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        for candidate in (key, key.upper(), key.title()):
            value = record.get(candidate)
            if isinstance(value, str) and value.strip():
                return value
    return None


def pairs_from_records(text: str, path: Path) -> list[tuple[str, str, str]]:
    """Re-key records that are already prompt/response shaped."""
    rows: list[dict] = []
    stripped = text.strip()
    if not stripped:
        return []
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    else:
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return []
        if isinstance(value, dict):
            rows = [value]
        elif isinstance(value, list):
            rows = [item for item in value if isinstance(item, dict)]
    out = []
    for record in rows:
        prompt = first_present(record, PROMPT_KEYS)
        response = first_present(record, RESPONSE_KEYS)
        if prompt and response and prompt.strip() != response.strip():
            out.append((prompt, response, "record"))
    return out


#: Segment labels that act as a heading, i.e. can become a prompt.
HEADING_LABELS = {"title", "heading", "header", "h1", "h2", "h3", "section"}

#: A heading that names a document part rather than a subject. These make
#: useless prompts -- "Learning Outcomes" recalled from three different books
#: teaches the brain nothing about any of them -- so the section is attributed
#: to the nearest preceding subject heading instead.
STRUCTURAL_HEADINGS = {
    "learning outcomes", "learning objectives", "objectives", "summary",
    "chapter summary", "key terms", "glossary", "references", "exercises",
    "practice", "practice problems", "problems", "review questions",
    "questions", "introduction", "overview", "contents", "table of contents",
    "index", "preface", "about this book", "acknowledgements",
    "acknowledgments", "further reading", "conclusion", "answers",
    "solutions", "notes", "footnotes", "appendix", "abstract",
    # Found empirically in CS textbooks: "Discussion and Exercises"
    # appeared as a prompt 23 times across two books, teaching nothing
    # about any subject.
    "discussion and exercises", "discussion", "exercises and problems",
    "further study", "chapter notes", "bibliography", "credits",

}


def pairs_from_layout_segments(text: str, path: Path) -> list[tuple[str, str, str]]:
    """Rebuild sections from per-fragment layout segments.

    A PDF segmenter emits one record per visual block -- `{"book", "page",
    "label", "text"}` -- so no single record is a prompt/response pair. A
    heading opens a section and every following non-heading fragment belongs
    to it, which is the same shape `prose_sections` recovers from markdown.

    A structural heading ("Learning Outcomes", "Summary") is a document part,
    not a subject: it is qualified with the nearest preceding subject heading
    so the prompt still says what the section is about.
    """
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "label" in value and "text" in value:
            records.append(value)
    if not records:
        return []

    out: list[tuple[str, str, str]] = []
    heading: str | None = None
    subject: str | None = None
    body: list[str] = []

    def flush() -> None:
        if not heading or not body:
            return
        prompt = heading
        if heading.strip().lower() in STRUCTURAL_HEADINGS and subject:
            prompt = f"{subject}: {heading}"
        out.append((prompt, "\n".join(body).strip(), "section"))

    for record in records:
        content = str(record.get("text") or "").strip()
        if not content:
            continue
        if str(record.get("label") or "").lower() in HEADING_LABELS:
            flush()
            body = []
            heading = content
            if content.strip().lower() not in STRUCTURAL_HEADINGS:
                subject = content
        elif heading is not None:
            body.append(content)
    flush()
    return out


#: SPDX id per declared licence, and whether it may be used commercially.
#: NC and ND forbid it outright, so they map to None and the book is skipped.
LICENCE_SPDX = {
    "public domain": "public-domain",
    "cc by": "cc-by-4.0", "cc by 2.0": "cc-by-4.0",
    "cc by 3.0": "cc-by-4.0", "cc by 4.0": "cc-by-4.0",
    "cc by sa": "cc-by-sa-4.0", "cc by sa 3.0": "cc-by-sa-4.0",
    "cc by sa 4.0": "cc-by-sa-4.0",
}


#: Longest line still eligible to be a body-size heading.
#: Measured on the labelled segments, true headings run to a 28-character
#: median; body paragraphs to 247. 80 sits well clear of both.
MAX_TITLE_LENGTH = 80

#: A section holding less prose than this is folded into the previous one.
MIN_SECTION_BODY = 400

#: Typographic quotation marks. A line carrying one is running text -- a pull
#: quote or a cited sentence -- not a title.
SMART_QUOTE = re.compile(r"[“”]")

#: Unicode private use area. Icon fonts draw callout markers here, so the
#: codepoint means nothing outside the font that rendered it.
PRIVATE_USE_GLYPH = re.compile(r"[-]")

#: Punctuation left stranded at the start of a heading when a two-column
#: layout splits a line, e.g. ": Deriving Moles from Grams".
LEADING_PUNCTUATION = re.compile(r"^[\s,.;:)\]}]+")

#: A closed-class word left dangling at the end of a line, which means the
#: line was broken mid-phrase rather than written as a title. This is a
#: grammatical property of English function words, not subject vocabulary --
#: it says nothing about what any book is about.
DANGLING_FUNCTION_WORD = re.compile(
    r"\b(?:a|an|the|and|or|but|of|in|on|at|to|for|with|from|by|as|into|"
    r"than|that|which|when|while|is|are|was|were|be|been|has|have|had|"
    r"see|shown|including|such)\s*$",
    re.IGNORECASE,
)

#: A cross-reference word left without its number, e.g. a line broken after
#: "shown in Figure". A real caption or title carries the identifier
#: ("Figure 1.3.6"); a dangling one means the line was cut.
#: Requires something before the word: a line that IS just "Appendix" or
#: "Summary" is a legitimate standalone heading, while "shown in Figure" is a
#: sentence cut off before its number.
DANGLING_CROSS_REFERENCE = re.compile(
    r"\S+\s+(?:figure|fig|table|equation|eq|chapter|section|appendix|box|plate)"
    r"\s*$",
    re.IGNORECASE,
)


def is_title_shaped(line: str) -> bool:
    """True when a body-size line reads like a section title.

    Half of all headings in a LibreTexts book are set at the body type size,
    so type size cannot find them and something else has to. A title is
    short, does not run on into a sentence, and is not a fragment of quoted
    speech -- the discriminators below are exactly those three properties.

    Kept deliberately shape-based rather than vocabulary-based: nothing here
    knows what a chapter is called, so it carries no per-publisher or
    per-subject assumption.
    """
    text = line.strip()
    if not text or len(text) > MAX_TITLE_LENGTH:
        return False
    # Ends mid-sentence: a body line that happened to be short.
    if text.endswith((".", ",", ";", ":")):
        return False
    # Opens as a continuation of something quoted or parenthesised.
    if text[:1] in "”’\")":
        return False
    # Carries quoted speech: a pull quote, not a heading.
    if SMART_QUOTE.search(text):
        return False
    # Breaks off mid-phrase. A body line split before a figure reference
    # ("shown in Figure", "including certain types of algae (Figure") is
    # short and has no terminal punctuation, so only the dangling function
    # word at the end distinguishes it from a title. A real heading does not
    # end on a preposition, article, or conjunction.
    if DANGLING_FUNCTION_WORD.search(text):
        return False
    if DANGLING_CROSS_REFERENCE.search(text):
        return False
    # An unclosed bracket means the line was cut, not titled.
    if text.count("(") != text.count(")"):
        return False
    # Page furniture: a bare URL, or a line with no letters at all (a page
    # number, a rule, a stray figure index).
    if text.startswith(("http://", "https://")):
        return False
    return bool(re.search(r"[A-Za-z]", text))


def page_is_commercial_safe(page_text: str) -> bool:
    """False when a single page declares NC/ND rights, or none at all.

    LibreTexts books are remixes: OrganicChemistryMorschEtAl carries 377
    share-alike pages and 6 CC BY-NC ones. Judging the BOOK forced a bad
    trade -- import the 6 restricted pages, or discard 1,722 good ones.
    Judging the PAGE keeps the licence guarantee exact and costs only the
    pages that actually violate it.

    A page with no licence text at all is safe: most pages are body text and
    carry no stamp. Only an explicit restrictive or undeclared statement
    disqualifies one.
    """
    for match in PDF_LICENCE_PATTERN.finditer(page_text):
        key = re.sub(r"[\s-]+", " ", match.group(1).strip().lower())
        if "not declared" in key or NON_COMMERCIAL_HINT.search(key):
            return False
    # A bare restrictive code, with no "shared/licensed under" lead-in, still
    # binds the page. Measured on OrganicChemistryMorschEtAl, 30 pages escaped
    # the sentence-shaped pattern through two wordings it cannot see:
    # "is licensed CC BY-NC-SA 4.0" (no "under") and the parenthetical figure
    # credit "(CC BY-NC-SA 3.0; Anonymous)" (no verb at all).
    #
    # A figure credit strictly binds only that figure, but extracted page text
    # gives no way to separate the figure from the prose around it, so the
    # page is dropped whole. Over-dropping costs a page; under-dropping puts
    # non-commercial content in a corpus sold as commercial-safe.
    return not BARE_RESTRICTIVE_CODE.search(page_text)


def pdf_licence(document) -> str | None:
    """The SPDX id a PDF declares, or None when it is not commercial-safe.

    Read from the document rather than assumed for a folder: measured over
    183 LibreTexts books, 110 are CC BY-NC and cannot be used commercially at
    all, so a folder-wide assumption would quietly poison the corpus. An
    undeclared or NC/ND book returns None and is skipped.
    """
    # Scan the whole book, not a front-matter window. LibreTexts stamps its
    # licence per page, and AerodynamicsAndAircraftPerformance carries its two
    # "CC BY 4.0" stamps past page 40 -- a 40-page window found nothing and the
    # book fell through to the prose path, which matched an incidental
    # "public domain" and mislabelled it.
    # Tally DISTINCT PAGES, not raw hits. A book stamps its own licence across
    # many pages; a figure credit repeats several times on one page. Measured
    # on ProjectManagement2eWatt: the book's real "CC BY-SA 4.0" appears on 19
    # separate pages, while a borrowed figure's "Creative Commons Attribution
    # 3.0" appears 5 times on a SINGLE page. Counting raw hits let that one
    # credit outvote the book and relabel a share-alike work as plain CC BY --
    # under-restrictive, the one direction that actually causes harm.
    pages: dict[str, set[int]] = {}
    for index in range(document.page_count):
        for match in PDF_LICENCE_PATTERN.finditer(document[index].get_text()):
            key = re.sub(r"[\s-]+", " ", match.group(1).strip().lower())
            pages.setdefault(key, set()).add(index)
    if pages:
        # Take the MOST RESTRICTIVE licence present, not the most common one.
        #
        # LibreTexts books are remixes and routinely carry different licences
        # on different pages. Majority vote is unsound for them: CADSkills has
        # 50 CC BY pages but also 29 CC BY-NC ones, and a vote admitted the
        # whole book -- including the NC chapters -- as commercial-safe.
        # Restrictiveness is the only defensible reading, because the corpus
        # ingests every page, not the winning one.
        #
        # A page whose licence was never declared is disqualifying for the same
        # reason: unestablished rights are not permissive ones. Requiring more
        # than one such page keeps a single stray "not declared" boilerplate
        # line from rejecting an otherwise cleanly licensed book.
        # Restricted and undeclared pages are DROPPED by the reader (see
        # `page_is_commercial_safe`), so their presence no longer condemns the
        # book. What must still hold is that something permissive is actually
        # declared -- a book with no usable licence anywhere is refused below.
        permissive = [key for key in pages
                      if "not declared" not in key
                      and not NON_COMMERCIAL_HINT.search(key)]
        if not permissive:
            return None
        pages = {key: hit for key, hit in pages.items() if key in permissive}
        # Share-alike outranks plain attribution: both are commercial-safe, but
        # mislabelling SA as BY understates the obligation the corpus inherits.
        declared = max(
            pages,
            key=lambda key: ("sa" in key.split()
                             or "share" in key.replace("-", " "),
                             len(pages[key])),
        )
        if NON_COMMERCIAL_HINT.search(declared):
            # Refused explicitly rather than by falling off the end of
            # LICENCE_SPDX, so an unrecognised PERMISSIVE variant stays
            # distinguishable from a genuinely restricted one.
            return None
        resolved = LICENCE_SPDX.get(re.sub(r"[\s_-]+", " ", declared.lower()))
        if resolved:
            return resolved

    # No licence code found. Fall back to what the document SAYS: a book may
    # name its licence only in prose. Measured 2026-08-22, Open Data
    # Structures states "Creative Commons Attribution license, meaning that
    # anyone is free to share... even commercially" and never writes "CC BY",
    # so a code-only pattern reported no licence for a plainly usable book.
    # 24 pages, not 8: Open Data Structures states its licence on page 10,
    # inside the preface rather than on a copyright page. A window sized to
    # one publisher's front matter silently discarded a usable book.
    front = "\n".join(
        document[index].get_text() for index in range(min(document.page_count, 24))
    )
    for pattern, spdx in PROSE_LICENCE_HINTS:
        window = pattern.search(front)
        if not window:
            continue
        nearby = front[max(0, window.start() - 200):window.end() + 600]
        # An explicit commercial grant outranks a stray "non-commercial"
        # nearby: a licence that spells out "the right to make commercial use
        # of the work" has settled the question in the document's own words.
        if COMMERCIAL_GRANT.search(nearby):
            return spdx
        if NON_COMMERCIAL_HINT.search(nearby):
            return None
        return spdx
    return None


#: Typographic ligatures a PDF encodes as single glyphs. Left alone, "first"
#: extracts as "ﬁrst" and never matches a user who types it normally.
#: Measured 2026-08-22 over two CS textbooks: 578 occurrences.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

#: A word split across a line break by hyphenation: "ran- domization".
#: Measured 483 times over the same two books.
HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")


def normalise_pdf_text(text: str) -> str:
    """Undo the artefacts PDF extraction introduces.

    Ligatures and hyphenation are typesetting decisions, not content. Carrying
    them into the corpus teaches the brain spellings no reader would ever
    type, so recall fails on the exact words the section is about.
    """
    for glyph, plain in LIGATURES.items():
        text = text.replace(glyph, plain)
    text = HYPHEN_BREAK.sub(r"\1\2", text)
    # Icon-font characters land in the Unicode private use area, where they
    # carry no meaning outside the font that drew them. OpenStax textbooks
    # prefix callouts with them, so 5.3% of prompts began with a glyph like
    # U+F128 that no reader could ever type -- and that the brain would learn
    # as part of the heading.
    text = PRIVATE_USE_GLYPH.sub("", text)
    return text.replace("’", "'").replace("‘", "'")


def split_long_body(text: str) -> list[str]:
    """Break an over-long section at sentence boundaries.

    A chapter that exceeds the response cap used to be discarded whole, which
    cost 51% of Open Data Structures' prose -- including a 25,028-character
    B-Trees chapter. Splitting keeps the material and each part still reads as
    continuous prose, because the break lands between sentences rather than
    mid-word.
    """
    if len(text) <= MAX_RESPONSE:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        # A single sentence longer than the cap cannot be placed; emitting it
        # would fail the gate anyway, so it is dropped with the rest of the
        # oversize run rather than silently truncated mid-thought.
        if len(sentence) > MAX_RESPONSE:
            continue
        if length + len(sentence) + 1 > MAX_RESPONSE and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return [chunk for chunk in chunks if len(chunk) >= MIN_RESPONSE]


def pairs_from_pdf(path: Path) -> list[tuple[str, str, str]]:
    """Rebuild sections from a PDF using type size to find headings.

    A span meaningfully larger than the page's most common size opens a
    section; everything after it is that section's body. This is the same
    shape `prose_sections` recovers from markdown, derived from layout rather
    than markup, so it needs no per-publisher template.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    try:
        document = fitz.open(path)
    except Exception:
        return []
    try:
        licence = pdf_licence(document)
        if licence is None:
            # Not commercial-safe, or undeclared. Refusing here keeps a
            # restricted book out of the corpus even if the caller passes a
            # permissive --license for the folder.
            return []
        spans: list[tuple[float, str]] = []
        sizes: dict[float, int] = {}
        for index in range(document.page_count):
            try:
                page = document[index]
                # Drop a page whose own stamp is NC/ND or undeclared. This is
                # what lets a mixed-licence remix be ingested at all without
                # ever importing a byte the licence does not cover.
                if not page_is_commercial_safe(page.get_text()):
                    continue
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            for block in blocks:
                for line in block.get("lines", []):
                    # Join a line's spans before judging it. A ligature gets
                    # its own span, so "SEList: A Space-Efficient Linked List"
                    # arrives as four spans split at the "ffi" -- judged
                    # separately, the 2-char ligature span fails the heading
                    # test and the heading became "cient Linked List".
                    parts = [str(span.get("text") or "")
                             for span in line.get("spans", [])]
                    content = normalise_pdf_text("".join(parts)).strip()
                    if not content:
                        continue
                    span_sizes = [round(float(span.get("size") or 0.0), 1)
                                  for span in line.get("spans", [])
                                  if str(span.get("text") or "").strip()]
                    if not span_sizes:
                        continue
                    size = max(span_sizes)
                    spans.append((size, content))
                    sizes[size] = sizes.get(size, 0) + len(content)
        if not spans or not sizes:
            return []
        body_size = max(sizes, key=sizes.get)
        threshold = body_size * PDF_HEADING_RATIO

        sections: list[tuple[str, list[str]]] = []
        heading: str | None = None
        subject: str | None = None
        body: list[str] = []

        def close_section() -> None:
            if not heading or not body:
                return
            prompt = heading
            if heading.strip().lower() in STRUCTURAL_HEADINGS and subject:
                prompt = f"{subject}: {heading}"
            # Strip punctuation stranded at the front by a column split, so
            # the prompt reads as the title it was: ": Deriving Moles from
            # Grams" becomes "Deriving Moles from Grams".
            cleaned = LEADING_PUNCTUATION.sub("", normalise_pdf_text(prompt))
            sections.append((cleaned.strip() or prompt.strip(), list(body)))

        for size, content in spans:
            larger = size >= threshold and len(content) <= 200
            # A heading set at BODY size. Measured against 1,473 labelled
            # segments, 170 of 341 true headings share the body's type size
            # exactly, so a size threshold alone found only 34% of them (F1
            # 0.41 at every ratio from 1.02 to 1.5 -- the constant was never
            # the problem). Their real signature is that they are short and
            # shaped like a title.
            same_size_title = (size >= body_size
                               and is_title_shaped(content)
                               and not TOC_ENTRY.search(content))
            if larger or same_size_title:
                close_section()
                body = []
                heading = content
                if content.strip().lower() not in STRUCTURAL_HEADINGS:
                    subject = content
            elif heading is not None:
                if content.startswith("This page titled"):
                    continue
                body.append(content)
        close_section()

        # Fold a section with too little prose back into the one before it.
        #
        # Admitting body-size headings finds real sections ("The Tragedy of
        # the Commons", "Environmental Justice") that the size rule missed
        # entirely, but a two-column book also emits short body-size lines
        # that look like titles. Left alone, ChemistryAtomsFirst2eOpenStax
        # shattered into 9,443 sections with a 22-byte median body.
        #
        # Merging rather than DROPPING is the point: dropping thin sections
        # tamed the count but cost that book 816k characters -- 34% of its
        # prose. Folding them back keeps 93-97% of every book's text while
        # still roughly doubling the number of correctly-titled sections.
        merged: list[tuple[str, str]] = []
        for prompt, parts in sections:
            text = normalise_pdf_text(" ".join(parts)).strip()
            if merged and len(text) < MIN_SECTION_BODY:
                previous_prompt, previous_text = merged[-1]
                merged[-1] = (previous_prompt,
                              f"{previous_text} {prompt} {text}".strip())
            else:
                merged.append((prompt, text))

        out: list[tuple[str, str, str]] = []
        for prompt, text in merged:
            # A chapter longer than the response cap is SPLIT, not discarded.
            # Measured 2026-08-22 on Open Data Structures, 19 sections ran
            # past the cap and took 51% of the book's prose with them -- a
            # 25,028-character "B-Trees" chapter was dropped whole.
            for number, chunk in enumerate(split_long_body(text), start=1):
                part = prompt if number == 1 else f"{prompt} (part {number})"
                out.append((part, chunk, f"section:{licence}"))
        return out
    finally:
        document.close()


def pairs_from_table(text: str, path: Path) -> list[tuple[str, str, str]]:
    """A prompt-ish and a response-ish column, when both are present."""
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        records = list(reader)
    except (csv.Error, UnicodeDecodeError):
        return []
    out = []
    for record in records:
        clean = {k: v for k, v in record.items() if isinstance(k, str) and v}
        prompt = first_present(clean, PROMPT_KEYS)
        response = first_present(clean, RESPONSE_KEYS)
        if prompt and response and prompt.strip() != response.strip():
            out.append((prompt, response, "table"))
    return out


def python_definitions(text: str) -> list[tuple[str, str, str]]:
    """Documented top-level defs and classes, paired with their source.

    A real directory contains files that do not fully parse -- a stray
    editor artefact, a Python 2 module, a template with placeholders. One
    bad definition should not discard every good one beside it, so a file
    that fails to parse whole is retried a definition at a time.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return _python_definitions_piecewise(text)
    lines = text.splitlines()
    out = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        end = getattr(node, "end_lineno", None)
        if end is None:
            continue
        source = "\n".join(lines[node.lineno - 1:end])
        summary = " ".join(doc.strip().split())
        out.append((f"{summary}", source, "definition"))
    return out


#: Start of a top-level Python definition, used only to salvage a file that
#: does not parse as a whole.
PY_DEFINITION_START = re.compile(
    r"^(?:def|class|async\s+def)\s+\w+", re.MULTILINE
)


def _python_definitions_piecewise(text: str) -> list[tuple[str, str, str]]:
    """Recover the definitions that DO parse from a file that does not.

    Each candidate is parsed on its own, so anything returned is still real
    Python -- the salvage never lowers the bar on what is emitted.
    """
    starts = [match.start() for match in PY_DEFINITION_START.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    out = []
    for index in range(len(starts) - 1):
        chunk = text[starts[index]:starts[index + 1]].rstrip()
        if not chunk:
            continue
        try:
            tree = ast.parse(chunk)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc:
                continue
            out.append((" ".join(doc.strip().split()), chunk, "definition"))
    return out


#: A leading block comment or docstring above a definition, for the languages
#: where a full parse is not available here.
COMMENTED_DEFINITION = re.compile(
    r"(?P<doc>(?:^[ \t]*(?://[^\n]*|\*[^\n]*|/\*\*?)[^\n]*\n){1,20})"
    r"(?P<def>^[ \t]*(?:export\s+)?(?:public\s+|private\s+|async\s+|static\s+|"
    r"pub\s+|func\s+|fn\s+|function\s+|class\s+|def\s+)[^\n]*\{?\s*$)",
    re.MULTILINE,
)


def commented_definitions(text: str) -> list[tuple[str, str, str]]:
    """Non-Python source: a comment block immediately above a definition."""
    out = []
    lines = text.splitlines()
    for match in COMMENTED_DEFINITION.finditer(text):
        doc = match.group("doc")
        summary = " ".join(
            re.sub(r"^[ \t]*(?://+|\*+|/\*+|\*/)", "", line).strip()
            for line in doc.splitlines()
        ).strip()
        if not summary:
            continue
        start = text[:match.start("def")].count("\n")
        # Take the definition plus a bounded body; brace matching is not
        # attempted, because a wrong guess would emit truncated source.
        body = "\n".join(lines[start:start + 40])
        out.append((summary, body, "definition"))
    return out


HEADING = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$", re.MULTILINE)


def prose_sections(text: str) -> list[tuple[str, str, str]]:
    """A heading becomes the prompt, the text beneath it the response."""
    matches = list(HEADING.finditer(text))
    if not matches:
        return []
    out = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        title = match.group("title").strip()
        if title and body:
            out.append((title, body, "section"))
    return out


def pairs_from_docx(path: Path) -> list[tuple[str, str, str]]:
    """Sections from a Word document, using its own heading styles.

    A .docx names its headings outright ("Heading 1"), so unlike a PDF there
    is nothing to infer from type size. Rebuilt into markdown headings and
    handed to `prose_sections`, so all downstream handling -- table-of-
    contents rejection, the response cap, section splitting -- is shared with
    every other prose format rather than reimplemented here.
    """
    try:
        import docx  # python-docx
    except ImportError:
        return []
    try:
        document = docx.Document(str(path))
    except Exception:
        return []
    lines: list[str] = []
    for paragraph in document.paragraphs:
        content = (paragraph.text or "").strip()
        if not content:
            continue
        style = str(getattr(paragraph.style, "name", "") or "")
        if style.lower().startswith("heading") or style.lower() == "title":
            lines.append(f"## {content}")
        else:
            lines.append(content)
    return prose_sections("\n\n".join(lines))


def pairs_from_workbook(path: Path) -> list[tuple[str, str, str]]:
    """Rows from a spreadsheet, one sheet at a time.

    A sheet's header row names its columns, so the same prompt/response
    column detection used for CSV applies once the sheet is rendered as
    delimited text. Sheets are handled individually because a workbook
    routinely holds unrelated tables with different headers.
    """
    try:
        import openpyxl
    except ImportError:
        return []
    try:
        workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception:
        return []
    out: list[tuple[str, str, str]] = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for values in sheet.iter_rows(values_only=True):
                cells = ["" if value is None else str(value).replace("\t", " ")
                         for value in values]
                if any(cell.strip() for cell in cells):
                    rows.append("\t".join(cells))
            if len(rows) < 2:
                continue
            out.extend(pairs_from_table("\n".join(rows),
                                        path.with_suffix(".tsv")))
    finally:
        workbook.close()
    return out


def pairs_from_html(text: str, path: Path) -> list[tuple[str, str, str]]:
    """Sections from HTML, keyed on its heading tags."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        soup = BeautifulSoup(text, "html.parser")
    except Exception:
        return []
    # Script and style bodies are not prose and would otherwise be swept into
    # whichever section preceded them.
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    lines: list[str] = []
    for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        content = " ".join((element.get_text() or "").split())
        if not content:
            continue
        if element.name.startswith("h") and element.name[1:].isdigit():
            lines.append(f"## {content}")
        else:
            lines.append(content)
    return prose_sections("\n\n".join(lines))


def extract(path: Path, text: str) -> list[tuple[str, str, str]]:
    """Best available pairing for one document, most faithful first."""
    suffix = path.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        return pairs_from_pdf(path)
    if suffix in WORD_SUFFIXES:
        return pairs_from_docx(path)
    if suffix in WORKBOOK_SUFFIXES:
        return pairs_from_workbook(path)
    if suffix in HTML_SUFFIXES:
        return pairs_from_html(text, path)
    if suffix in RECORD_SUFFIXES:
        # Layout segments are records, but each one is a FRAGMENT -- a heading,
        # a paragraph, a list item -- not a prompt/response pair. Re-keying
        # them one at a time yields nothing, so a folder of segmented
        # textbooks produced an empty corpus. Reassemble them first.
        segments = pairs_from_layout_segments(text, path)
        if segments:
            return segments
        return pairs_from_records(text, path)
    if suffix in TABLE_SUFFIXES:
        return pairs_from_table(text, path)
    if suffix == ".py":
        return python_definitions(text)
    if suffix in SOURCE_LANGUAGES:
        return commented_definitions(text)
    if suffix in PROSE_SUFFIXES:
        return prose_sections(text)
    return []


#: A response that is mostly "N.N: Title" fragments is a table of contents,
#: not teaching content. Measured over a LibreTexts accounting textbook it is
#: ~8% of extracted sections: the prompt names a chapter and the body just
#: lists its subsections, which teaches nothing about the subject.
TOC_ENTRY = re.compile(r"(?<!\d)\d+\.\d+:\s")


#: Numbered-section references per 100 characters above which a response is a
#: contents listing. Calibrated on real extracted rows rather than guessed:
#:   contents listing            3.57 per 100
#:   prose citing one section    0.67 per 100
#:   ordinary prose              0.00 per 100
#: 1.5 sits in that gap, so a section that merely cites a subsection survives.
TOC_DENSITY_PER_100 = 1.5


def is_table_of_contents(response: str) -> bool:
    """True when a response is a contents listing rather than prose."""
    if not response:
        return False
    density = 100.0 * len(TOC_ENTRY.findall(response)) / len(response)
    return density > TOC_DENSITY_PER_100


def acceptable(prompt: str, response: str, lang: str,
               stats: Stats) -> bool:
    """Gate one candidate pair, counting exactly why it was refused."""
    prompt, response = prompt.strip(), response.strip()
    if len(prompt) < MIN_PROMPT or len(response) < MIN_RESPONSE:
        stats.skipped_too_short += 1
        return False
    if len(prompt) > MAX_PROMPT or len(response) > MAX_RESPONSE:
        stats.skipped_too_long += 1
        return False
    if is_table_of_contents(response):
        stats.skipped_contents += 1
        return False
    if lang == "python":
        try:
            ast.parse(response)
        except SyntaxError:
            stats.skipped_bad_syntax += 1
            return False
    return True


def build(src: Path, out: Path, license_id: str, label: str,
          script_id: str, intent: str) -> Stats:
    stats = Stats()
    writer = RowWriter(out, script_id=script_id, source=label)
    try:
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in (RECORD_SUFFIXES | TABLE_SUFFIXES | PROSE_SUFFIXES
                              | BINARY_SUFFIXES | HTML_SUFFIXES
                              | set(SOURCE_LANGUAGES)):
                continue
            stats.files_seen += 1
            if suffix in BINARY_SUFFIXES:
                # Binary: parsed from the path, never decoded as text.
                text = ""
            else:
                decoded = read_text(path)
                if decoded is None:
                    stats.unreadable.append(str(path))
                    continue
                text = decoded
            pairs = extract(path, text)
            if not pairs:
                stats.skipped_unpairable += 1
                continue
            relative = path.relative_to(src).as_posix()
            lang = SOURCE_LANGUAGES.get(suffix, "text")
            wrote_any = False
            for prompt, response, kind in pairs:
                if not acceptable(prompt, response, lang, stats):
                    continue
                # A document that declares its own licence overrides the
                # folder-wide one. Measured over 183 LibreTexts books the
                # licences differ per book, so one flag for a folder is wrong
                # by construction.
                row_licence = license_id
                if kind.startswith("section:"):
                    row_licence = kind.split(":", 1)[1]
                    kind = "section"
                row = Row(
                    prompt=prompt.strip(),
                    response=response.strip(),
                    ctx={"lang": lang, "intent": intent, "source": label},
                    license=row_licence,
                    source=f"{label}:{relative}#{kind}",
                    source_hash=hash_source(f"{prompt}\n{response}"),
                    script_id=script_id,
                )
                if writer.write(row):
                    stats.rows += 1
                    wrote_any = True
                else:
                    stats.skipped_duplicate += 1
            if wrote_any:
                stats.files_used += 1
    finally:
        writer.close()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="directory of documents to compile")
    parser.add_argument("--out", type=Path, required=True,
                        help="destination .jsonl")
    parser.add_argument("--license", required=True,
                        help="SPDX id; must be permissive")
    parser.add_argument("--label", required=True,
                        help="short provenance label, e.g. my_manuals")
    parser.add_argument("--script-id", required=True)
    parser.add_argument("--intent", default="implement")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"not a directory: {args.src}", file=sys.stderr)
        return 2

    stats = build(args.src, args.out, args.license, args.label,
                  args.script_id, args.intent)
    print(f"wrote {args.out}")
    print(stats.report())
    if stats.rows == 0:
        print("\nno rows produced; nothing here was pairable", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
