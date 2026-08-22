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

#: Heading detection for PDFs: a span this much larger than the page's most
#: common (body) size opens a section. Measured over LibreTexts textbooks,
#: body text sits at ~9.6pt with section titles at 13.5 and subheadings at
#: 11.2, so a 15% margin separates them without needing a trained segmenter.
PDF_HEADING_RATIO = 1.15

#: LibreTexts stamps a per-page attribution line. Read the licence from the
#: document itself rather than assuming one for a whole folder: measured over
#: 183 books, 110 are CC BY-NC (no commercial use) and only 63 are permissive,
#: so a folder-wide assumption would quietly poison a commercial corpus.
PDF_LICENCE_PATTERN = re.compile(
    r"shared under (?:an?\s+)?"
    r"(CC[ -]BY(?:[ -]SA|[ -]NC(?:[ -]ND)?|[ -]ND)?(?:\s*\d\.\d)?"
    r"|public domain)",
    re.IGNORECASE,
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


def pdf_licence(document) -> str | None:
    """The SPDX id a PDF declares, or None when it is not commercial-safe.

    Read from the document rather than assumed for a folder: measured over
    183 LibreTexts books, 110 are CC BY-NC and cannot be used commercially at
    all, so a folder-wide assumption would quietly poison the corpus. An
    undeclared or NC/ND book returns None and is skipped.
    """
    counts: dict[str, int] = {}
    for index in range(min(document.page_count, 40)):
        for match in PDF_LICENCE_PATTERN.finditer(document[index].get_text()):
            key = re.sub(r"[\s-]+", " ", match.group(1).strip().lower())
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    declared = max(counts, key=counts.get)
    return LICENCE_SPDX.get(declared)


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
                blocks = document[index].get_text("dict")["blocks"]
            except Exception:
                continue
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        content = str(span.get("text") or "").strip()
                        if not content:
                            continue
                        size = round(float(span.get("size") or 0.0), 1)
                        spans.append((size, content))
                        sizes[size] = sizes.get(size, 0) + len(content)
        if not spans or not sizes:
            return []
        body_size = max(sizes, key=sizes.get)
        threshold = body_size * PDF_HEADING_RATIO

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
            out.append((prompt, " ".join(body).strip(), f"section:{licence}"))

        for size, content in spans:
            if size >= threshold and len(content) <= 200:
                flush()
                body = []
                heading = content
                if content.strip().lower() not in STRUCTURAL_HEADINGS:
                    subject = content
            elif heading is not None:
                if content.startswith("This page titled"):
                    continue
                body.append(content)
        flush()
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


def extract(path: Path, text: str) -> list[tuple[str, str, str]]:
    """Best available pairing for one document, most faithful first."""
    suffix = path.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        return pairs_from_pdf(path)
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
                              | DOCUMENT_SUFFIXES | set(SOURCE_LANGUAGES)):
                continue
            stats.files_seen += 1
            if suffix in DOCUMENT_SUFFIXES:
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
