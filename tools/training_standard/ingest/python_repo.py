"""ingest/python_repo.py — walk a Python repo, extract function/class docstrings.

For repos like TheAlgorithms/Python where each .py file contains
one or more well-documented functions/classes.  Each public function
or class with a real docstring becomes one training row:

    prompt   = first paragraph of the docstring
    response = the function/class source code (docstring stripped to
               force the brain to *recall* the implementation)
    ctx      = {lang=python, intent=implement, source=<repo_label>,
                file=<relative_path>, name=<func_name>}

The single repo license (from --license arg, validated against
PERMISSIVE_LICENSES) is applied to every row.

We use ast.parse() to find function/class defs, then ast.get_docstring()
to extract the docstring.  Falls back to a clean "skip this row" on
syntax errors — no half-extracted code reaches the writer.

Quality gates:
  - docstring 20–2000 chars
  - function body 30–4000 chars (after docstring strip)
  - body passes sandbox syntactic check (Python)
  - doctest lines (>>> …) are kept; they're a feature, not noise

CLI:
    python -m tools.training_standard.ingest.python_repo \\
        --src D:/.../sources/repos/the_algorithms_python \\
        --out D:/.../training/the_algorithms.jsonl \\
        --license mit \\
        --repo-label "TheAlgorithms/Python" \\
        --script-id dsa_classical_001
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Iterator

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.row import (
    Row, RowRejected, RowWriter, hash_source, render_ctx,
    PERMISSIVE_LICENSES,
)
from tools.training_standard.sandbox import get_sandbox


def _first_paragraph(s: str) -> str:
    import re as _re
    paragraphs = _re.split(r"\n\s*\n", s.strip(), maxsplit=1)
    para = paragraphs[0] if paragraphs else ""
    para = _re.sub(r"\s+", " ", para).strip()
    return para


def _source_segment(file_text: str, node: ast.AST) -> str:
    """ast.get_source_segment(...) but resilient to nested-indent dedent
    quirks — returns the raw lines spanning the node."""
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return ""
    lines = file_text.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1 : node.end_lineno])


def _strip_top_docstring(src: str) -> str:
    """If the very first statement in `src` is an Expr/Str (docstring),
    drop those lines so the response is body-only."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    if not tree.body:
        return src
    first = tree.body[0]
    if isinstance(first, ast.FunctionDef) or isinstance(first, ast.AsyncFunctionDef) or isinstance(first, ast.ClassDef):
        # Defensive: caller should pass a single def, but if they pass a
        # module-level wrapper, dig into the def's body.
        if first.body and isinstance(first.body[0], ast.Expr) \
                and isinstance(getattr(first.body[0], "value", None), ast.Constant) \
                and isinstance(first.body[0].value.value, str):
            doc = first.body[0]
            lines = src.splitlines(keepends=True)
            return "".join(lines[: doc.lineno - 1] + lines[doc.end_lineno:])
    return src


def _walk_defs(file_path: Path, file_text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (name, docstring, source) for every top-level FunctionDef,
    AsyncFunctionDef, or ClassDef with a real docstring.  Private
    (leading-underscore) names are skipped — they're rarely the
    canonical algorithm and rarely have user-facing docs."""
    try:
        tree = ast.parse(file_text)
    except SyntaxError:
        return
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        name = node.name
        if name.startswith("_") and not name.startswith("__"):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        src = _source_segment(file_text, node)
        if not src:
            continue
        yield name, doc, src


def ingest(
    *,
    src_dir: Path,
    out_path: Path,
    script_id: str,
    license: str,
    repo_label: str,
    limit: int | None,
    skip_sandbox: bool,
) -> dict:
    license = license.strip().lower()
    if license not in PERMISSIVE_LICENSES:
        raise RuntimeError(
            f"--license {license!r} is not in PERMISSIVE_LICENSES "
            f"({sorted(PERMISSIVE_LICENSES)})"
        )

    sb = None if skip_sandbox else get_sandbox()

    counters = {
        "files_seen":           0,
        "rejected_quality":     0,
        "rejected_sandbox":     0,
        "rejected_row_writer":  0,
        "dedup_skipped":        0,
        "written":              0,
    }

    with RowWriter(out_path, script_id=script_id,
                   source=f"python-repo:{repo_label}") as writer:
        for py_file in sorted(src_dir.rglob("*.py")):
            counters["files_seen"] += 1
            if limit is not None and counters["written"] >= limit:
                break
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = py_file.relative_to(src_dir).as_posix()
            for name, doc, src in _walk_defs(py_file, text):
                if limit is not None and counters["written"] >= limit:
                    break
                prompt = _first_paragraph(doc)
                if not (20 <= len(prompt) <= 2000):
                    counters["rejected_quality"] += 1
                    continue
                body = _strip_top_docstring(src).strip()
                if not (30 <= len(body) <= 4000):
                    counters["rejected_quality"] += 1
                    continue
                if sb is not None:
                    r = sb.check("python", body, timeout_s=10.0)
                    if not r.ok:
                        counters["rejected_sandbox"] += 1
                        continue
                row = Row(
                    prompt=prompt,
                    response=body,
                    ctx=render_ctx(
                        lang="python", intent="implement",
                        source=repo_label.split("/")[-1].lower(),
                        file=rel, name=name,
                    ),
                    license=license,
                    source=f"python-repo:{repo_label}:{rel}#{name}",
                    source_hash=hash_source(body),
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
    return counters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True,
                   help="repo root (will be walked recursively for .py files)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--script-id", required=True)
    p.add_argument("--license", required=True,
                   help="SPDX-ish; must be in PERMISSIVE_LICENSES")
    p.add_argument("--repo-label", required=True,
                   help="provenance label (e.g. 'TheAlgorithms/Python')")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-sandbox", action="store_true")
    args = p.parse_args(argv)
    counters = ingest(
        src_dir=args.src, out_path=args.out, script_id=args.script_id,
        license=args.license, repo_label=args.repo_label,
        limit=args.limit, skip_sandbox=args.skip_sandbox,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
