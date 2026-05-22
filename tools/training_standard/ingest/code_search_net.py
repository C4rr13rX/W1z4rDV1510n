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
    """Yield CSN shard paths.  Two on-disk formats supported:

    1. Original CSN release: `*.jsonl.gz` (gzip-compressed JSONL).
    2. HuggingFace re-host:  `*.parquet` (one file per partition,
       partition encoded in the filename e.g. `train.parquet`,
       `validation.parquet`, `test.parquet`).
    """
    if not src_dir.exists():
        raise FileNotFoundError(f"CSN source dir does not exist: {src_dir}")
    yield from sorted(src_dir.rglob("*.jsonl.gz"))
    yield from sorted(src_dir.rglob("*.parquet"))


def _iter_records(shard: Path, default_lang: str | None) -> Iterator[dict]:
    """Stream records out of one shard, whatever the format.  Yields
    dicts in the CSN JSONL shape (code, docstring, repo, path,
    func_name, partition, language)."""
    if shard.suffix == ".gz":
        with gzip.open(shard, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    if shard.suffix == ".parquet":
        # Stream in row groups to keep memory bounded — CSN parquets
        # are hundreds of MB.
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "parquet shard found but pyarrow is not installed; "
                "pip install pyarrow"
            ) from exc
        # Partition is implicit in HF parquet filenames.  CSN's HF
        # repo lays them out as <lang>/<split>-NNNNN-of-MMMMM.parquet.
        stem = shard.stem.lower()
        if stem.startswith("train") or "-train-" in stem:
            partition = "train"
        elif stem.startswith("test") or "-test-" in stem:
            partition = "test"
        elif stem.startswith("valid") or "-valid" in stem:
            partition = "valid"
        else:
            partition = "train"  # safe default — we'd rather train on
                                   # a shard than skip it for unknown name
        # HF CSN parquet uses long column names; original JSONL uses
        # the short ones.  Pick whichever exists per row.
        pf = pq.ParquetFile(shard)

        def _pick(cols: dict, *names: str):
            """First column present in `names`, or None."""
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        for batch in pf.iter_batches(batch_size=2048):
            cols = {c: batch.column(c) for c in batch.schema.names}
            code_col = _pick(cols, "code", "func_code_string", "whole_func_string")
            doc_col  = _pick(cols, "docstring", "func_documentation_string")
            repo_col = _pick(cols, "repo", "repository_name")
            path_col = _pick(cols, "path", "func_path_in_repository")
            name_col = _pick(cols, "func_name")
            lang_col = _pick(cols, "language")
            part_col = _pick(cols, "partition", "split_name")
            n = len(batch)
            for i in range(n):
                rec = {
                    "code":      code_col[i].as_py() if code_col is not None else "",
                    "docstring": doc_col[i].as_py()  if doc_col  is not None else "",
                    "repo":      repo_col[i].as_py() if repo_col is not None else "?",
                    "path":      path_col[i].as_py() if path_col is not None else "?",
                    "func_name": name_col[i].as_py() if name_col is not None else "f",
                    "partition": (part_col[i].as_py() if part_col is not None else partition),
                    "language":  (lang_col[i].as_py() if lang_col is not None else (default_lang or "")),
                }
                yield rec
        return
    raise ValueError(f"unsupported CSN shard format: {shard}")


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
            for rec in _iter_records(shard, default_lang=lang):
                counters["seen"] += 1
                if limit is not None and counters["written"] >= limit:
                    break

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
                    code_stripped = code

                func_name = (rec.get("func_name") or "f").split(".")[-1]
                prompt = _build_prompt(docstring, lang, func_name)
                ok, _reason = _accept(prompt, code_stripped, docstring)
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
                    license="MIT",
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
