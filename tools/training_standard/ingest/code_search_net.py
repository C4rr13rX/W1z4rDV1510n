"""ingest/code_search_net.py — CodeSearchNet → training rows.

CodeSearchNet (CSN) is a publicly hosted corpus of 2M+ permissively
licensed function ↔ docstring pairs across six languages.  We use it
as the first proof-of-pipeline source because:

  - License is already enforced upstream (permissive-only).
  - Dedup against GitHub forks is already done upstream.
  - The (docstring, code) shape maps 1:1 to our (prompt, response)
    schema with no synthesis.

Layout of a CSN row (one per line in `.jsonl` after unzipping):
    {
      "code":           "def foo(...): ...",
      "docstring":      "Multi-line doc text.",
      "repo":           "owner/repo",
      "path":           "src/foo.py",
      "url":            "https://github.com/owner/repo/blob/.../foo.py#L10-L42",
      "language":       "python",
      "func_name":      "foo",
      "partition":      "train" | "valid" | "test",
      ...
    }

Conversion:
    prompt   = first paragraph of docstring + "(in {lang})"
    response = the code itself, with the docstring stripped so the
               model has to *recall* the implementation rather than
               echo the documentation back verbatim.
    ctx      = {lang, intent=implement, source=csn}

We reject rows where:
  - docstring is < 10 chars or > 2000 chars (too short → no signal;
    too long → likely a license blob or auto-generated)
  - code is < 30 chars or > 4000 chars
  - docstring contains obvious license/copyright header text
  - code fails sandbox syntactic check (when sandbox is enabled)

CLI:
    python -m tools.training_standard.ingest.code_search_net \
        --lang python \
        --src D:/w1z4rdv1510n-data/sources/codesearchnet/python \
        --out D:/w1z4rdv1510n-data/training/csn_python.jsonl \
        --script-id programming_literacy_python_001 \
        --limit 5000

Test-set safety: CSN's own `partition=test` rows are EXCLUDED so we
can use them later as a held-out integration eval slice.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Iterator

# Make the project root importable when run as a script.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.row import (
    Row, RowRejected, RowWriter, hash_source, render_ctx,
)
from tools.training_standard.sandbox import get_sandbox

# CSN languages we accept.  Map CSN labels → our sandbox lang ids.
LANG_MAP = {
    "python":     "python",
    "javascript": "javascript",
    "go":         "go",
    "java":       "java",
    "php":        "php",
    "ruby":       "ruby",
}

# Sandbox-supported subset; rows in other langs get accepted without
# the syntactic check (still a valid row, just unverified).
_SANDBOX_SUPPORTED = {"python", "javascript"}

# Docstring rejection patterns — license blobs and auto-gen comments.
_BAD_DOC_PATTERNS = [
    re.compile(r"copyright\s+\(c\)", re.IGNORECASE),
    re.compile(r"all rights reserved", re.IGNORECASE),
    re.compile(r"licensed under", re.IGNORECASE),
    re.compile(r"this file (was|is) auto[- ]?generated", re.IGNORECASE),
    re.compile(r"DO NOT EDIT", re.IGNORECASE),
]

# Strip the docstring out of the code so response ≠ prompt regurgitation.
_PY_DOCSTRING = re.compile(
    r'^(\s*)("""(.*?)"""|\'\'\'(.*?)\'\'\')',
    re.DOTALL | re.MULTILINE,
)


def _iter_csn_files(src_dir: Path) -> Iterator[Path]:
    """Yield .jsonl.gz files under src_dir, recursively.  CSN ships
    one .gz per shard."""
    if not src_dir.exists():
        raise FileNotFoundError(f"CSN source dir does not exist: {src_dir}")
    yield from sorted(src_dir.rglob("*.jsonl.gz"))


def _first_paragraph(text: str) -> str:
    """First non-empty paragraph, normalized."""
    paragraphs = re.split(r"\n\s*\n", text.strip(), maxsplit=1)
    para = paragraphs[0] if paragraphs else ""
    para = re.sub(r"\s+", " ", para).strip()
    return para


def _strip_python_docstring(code: str) -> str:
    """Remove the leading docstring from a Python function/method.
    Best-effort — if it doesn't match the standard pattern, leave the
    code alone (still a valid training row)."""
    m = _PY_DOCSTRING.search(code)
    if not m:
        return code
    # Only strip if the docstring sits at the top of the function
    # body (i.e. after the def line).
    pre = code[: m.start()]
    if "def " not in pre and "class " not in pre:
        return code
    return code[: m.start()] + code[m.end():]


def _build_prompt(docstring: str, lang: str, func_name: str) -> str:
    para = _first_paragraph(docstring)
    if not para:
        return ""
    # If the docstring already reads like an instruction, keep it.
    # Otherwise prepend a soft framing so the user-side text reads as a
    # spec rather than a description-of-existing-code.
    if re.match(r"^(write|implement|create|build|return|compute|given)", para, re.I):
        return para
    return f"Write a {lang} function `{func_name}` that: {para}"


def _accept(prompt: str, response: str, docstring: str) -> tuple[bool, str]:
    if len(docstring) < 10:
        return False, "docstring too short"
    if len(docstring) > 2000:
        return False, "docstring too long"
    if len(response) < 30:
        return False, "code too short"
    if len(response) > 4000:
        return False, "code too long"
    if not prompt.strip():
        return False, "empty prompt after build"
    for pat in _BAD_DOC_PATTERNS:
        if pat.search(docstring):
            return False, f"doc matches bad pattern: {pat.pattern!r}"
    return True, ""


def ingest(
    *,
    src_dir: Path,
    out_path: Path,
    script_id: str,
    lang: str,
    limit: int | None,
    skip_sandbox: bool,
) -> dict:
    """Run the CSN → JSONL conversion.  Returns counters."""
    sb = None if skip_sandbox else get_sandbox()
    sandbox_lang = LANG_MAP.get(lang, lang)
    do_sandbox = (sb is not None) and (sandbox_lang in _SANDBOX_SUPPORTED)

    counters = {
        "seen": 0,
        "rejected_test_partition": 0,
        "rejected_filter": 0,
        "rejected_sandbox": 0,
        "rejected_row_writer": 0,
        "dedup_skipped": 0,
        "written": 0,
    }

    with RowWriter(out_path,
                   script_id=script_id,
                   source=f"codesearchnet:{lang}") as writer:
        for shard in _iter_csn_files(src_dir):
            with gzip.open(shard, "rt", encoding="utf-8") as fh:
                for line in fh:
                    counters["seen"] += 1
                    if limit is not None and counters["written"] >= limit:
                        break
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if rec.get("partition") == "test":
                        counters["rejected_test_partition"] += 1
                        continue
                    if rec.get("language", "").lower() != lang:
                        continue

                    docstring = rec.get("docstring") or ""
                    code = rec.get("code") or ""
                    if lang == "python":
                        code_stripped = _strip_python_docstring(code)
                    else:
                        # Best-effort: for non-Python langs CSN's
                        # docstring is already a separate field, so the
                        # code field doesn't repeat it.  Use as-is.
                        code_stripped = code

                    func_name = rec.get("func_name") or "f"
                    # CSN's func_name often includes the class path
                    # like "ClassName.method" — keep only the leaf.
                    func_name = func_name.split(".")[-1]

                    prompt = _build_prompt(docstring, lang, func_name)
                    ok, reason = _accept(prompt, code_stripped, docstring)
                    if not ok:
                        counters["rejected_filter"] += 1
                        continue

                    if do_sandbox:
                        result = sb.check(sandbox_lang, code_stripped, timeout_s=10.0)
                        if not result.ok:
                            counters["rejected_sandbox"] += 1
                            continue

                    row = Row(
                        prompt=prompt,
                        response=code_stripped,
                        ctx=render_ctx(lang=lang, intent="implement", source="csn"),
                        license="MIT",   # CSN guarantees permissive; default
                                          # to MIT label since the dump doesn't
                                          # carry per-row license metadata.
                        source=f"codesearchnet:{lang}:{rec.get('repo','?')}:{rec.get('path','?')}#{func_name}",
                        source_hash=hash_source(code_stripped),
                        script_id=script_id,
                    )
                    try:
                        accepted = writer.write(row)
                    except RowRejected:
                        counters["rejected_row_writer"] += 1
                        continue
                    if accepted:
                        counters["written"] += 1
                    else:
                        counters["dedup_skipped"] += 1

                if limit is not None and counters["written"] >= limit:
                    break

    return counters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True,
                   help="dir containing CSN .jsonl.gz shards (recursive)")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL path")
    p.add_argument("--lang", required=True, choices=sorted(LANG_MAP),
                   help="CSN language")
    p.add_argument("--script-id", required=True,
                   help="registry script id this corpus feeds")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N accepted rows (for smoke tests)")
    p.add_argument("--skip-sandbox", action="store_true",
                   help="don't run syntactic check; fastest, lowest quality")
    args = p.parse_args(argv)

    counters = ingest(
        src_dir=args.src,
        out_path=args.out,
        script_id=args.script_id,
        lang=args.lang,
        limit=args.limit,
        skip_sandbox=args.skip_sandbox,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
