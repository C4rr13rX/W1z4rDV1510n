"""ingest/markdown_book.py — walk a Jupyter Book / MyST repo, extract prose↔code pairs.

For repos like d2l-ai/d2l-en, executablebooks/*, scipy-lectures
(when permissive), or any project where chapters are written as .md
files containing fenced code blocks interleaved with prose:

  - Walk all .md / .myst / .qmd files in the repo
  - For each, split into prose chunks and fenced code blocks
  - For every code block, build a Row from
      prompt   = the prose immediately preceding it (last 1-3 paragraphs)
      response = the code block (language-tagged)
      ctx      = {lang=<extracted from fence>, intent=implement,
                  source=<repo_label>, file=<relative_path>}

Fence patterns we recognise:
    ```python
    ```py
    ```{.python .input}             ← MyST/d2l style
    ```{code-cell} python           ← Jupyter Book MyST
    ```{code-block} python
    ```{ipython3}
    ```{python}

The single repo license (from --license, validated against
PERMISSIVE_LICENSES) is applied to every row.

Quality gates:
    prose 20–2000 chars after stripping markdown noise (img, latex,
        link decoration kept-as-text)
    code  30–4000 chars
    code passes sandbox syntactic check (when lang is python/bash/
        javascript/powershell; other langs accepted unverified)

CLI:
    python -m tools.training_standard.ingest.markdown_book \\
        --src D:/.../sources/repos/d2l_en \\
        --out D:/.../training/d2l_en.jsonl \\
        --license apache-2.0 \\
        --repo-label "d2l-ai/d2l-en" \\
        --script-id domain_ai_ml_002
"""
from __future__ import annotations

import argparse
import json
import re
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

_SANDBOX_SUPPORTED = {"python", "javascript", "bash", "powershell"}

# Fence start: ```<info>   where <info> is parsed to find a language.
_FENCE_START = re.compile(r"^```(.*)$")
_FENCE_END   = re.compile(r"^```\s*$")

# Map fence info → canonical lang id used in ctx + sandbox key.
def _detect_lang(info: str) -> str:
    s = info.strip().lower()
    if not s:
        return ""
    # MyST style: {.python .input} or {code-cell} python
    if s.startswith("{"):
        m = re.search(r"\.python\b|\bpython\b|\bipython\d*\b", s)
        if m:
            return "python"
        m = re.search(r"\bbash\b|\bsh\b|\bshell\b", s)
        if m:
            return "bash"
        m = re.search(r"\bjavascript\b|\bjs\b|\btypescript\b|\bts\b", s)
        if m:
            return "javascript"
        return ""
    # Plain ```python / ```py / ```rust
    aliases = {
        "py": "python", "python": "python", "python3": "python",
        "ipython": "python", "ipython3": "python",
        "js": "javascript", "javascript": "javascript",
        "ts": "javascript", "typescript": "javascript",
        "sh": "bash", "shell": "bash", "bash": "bash",
        "ps1": "powershell", "powershell": "powershell", "pwsh": "powershell",
        "rs": "rust", "rust": "rust",
        "go": "go", "java": "java", "c": "cpp", "cpp": "cpp",
        "c++": "cpp", "cs": "csharp", "csharp": "csharp",
    }
    head = s.split()[0].split(",")[0]
    return aliases.get(head, head if head.isalpha() else "")


# Markdown noise we strip from prose.
_IMG_RE      = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE     = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # keep link text
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_LATEX_RE    = re.compile(r"\$[^$\n]+\$")             # keep as inline math
_REF_RE      = re.compile(r":\w+:`[^`]+`")            # MyST role refs

def _clean_prose(s: str) -> str:
    s = _IMG_RE.sub("", s)
    s = _LINK_RE.sub(r"\1", s)
    s = _HTML_TAG_RE.sub("", s)
    s = _REF_RE.sub("", s)
    # Collapse runs of whitespace inside lines, keep newlines.
    out = []
    for ln in s.splitlines():
        ln = re.sub(r"[ \t]+", " ", ln).strip()
        if ln:
            out.append(ln)
    return "\n".join(out).strip()


def _split_md(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (prose_before, lang, code) for each fenced code block in
    order.  prose_before is the markdown content since the previous
    code block (or the document start), trimmed to the last N paragraphs."""
    lines = text.splitlines()
    i = 0
    last_prose: list[str] = []
    while i < len(lines):
        m = _FENCE_START.match(lines[i])
        if not m:
            last_prose.append(lines[i])
            i += 1
            continue
        # We hit a fence start.  Skip any prior content before yielding.
        lang = _detect_lang(m.group(1))
        # Scan to the matching fence end.
        code_lines: list[str] = []
        j = i + 1
        while j < len(lines) and not _FENCE_END.match(lines[j]):
            code_lines.append(lines[j])
            j += 1
        # Build prose_before from accumulated lines.
        prose_raw = "\n".join(last_prose).strip()
        # Trim to last 3 paragraphs.
        paragraphs = re.split(r"\n\s*\n", prose_raw)
        prose = _clean_prose("\n\n".join(paragraphs[-3:]))
        yield prose, lang, "\n".join(code_lines)
        # Reset prose accumulator; start AFTER the closing fence.
        last_prose = []
        i = j + 1


def ingest(
    *,
    src_dir: Path,
    out_path: Path,
    script_id: str,
    license: str,
    repo_label: str,
    limit: int | None,
    skip_sandbox: bool,
    intent: str,
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
        "blocks_seen":          0,
        "rejected_no_lang":     0,
        "rejected_prose_size":  0,
        "rejected_code_size":   0,
        "rejected_sandbox":     0,
        "rejected_row_writer":  0,
        "dedup_skipped":        0,
        "written":              0,
    }

    md_paths: list[Path] = []
    for ext in ("*.md", "*.myst", "*.qmd"):
        md_paths.extend(src_dir.rglob(ext))
    md_paths.sort()

    with RowWriter(out_path, script_id=script_id,
                   source=f"markdown-book:{repo_label}") as writer:
        for path in md_paths:
            counters["files_seen"] += 1
            if limit is not None and counters["written"] >= limit:
                break
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            rel = path.relative_to(src_dir).as_posix()
            for prose, lang, code in _split_md(text):
                counters["blocks_seen"] += 1
                if limit is not None and counters["written"] >= limit:
                    break
                if not lang:
                    counters["rejected_no_lang"] += 1
                    continue
                if not (20 <= len(prose) <= 2000):
                    counters["rejected_prose_size"] += 1
                    continue
                code = code.strip()
                if not (30 <= len(code) <= 4000):
                    counters["rejected_code_size"] += 1
                    continue
                if sb is not None and lang in _SANDBOX_SUPPORTED:
                    r = sb.check(lang, code, timeout_s=10.0)
                    if not r.ok:
                        counters["rejected_sandbox"] += 1
                        continue
                row = Row(
                    prompt=prose,
                    response=code,
                    ctx=render_ctx(
                        lang=lang, intent=intent,
                        source=repo_label.split("/")[-1].lower(),
                        file=rel,
                    ),
                    license=license,
                    source=f"markdown-book:{repo_label}:{rel}",
                    source_hash=hash_source(f"{prose}|||{code}"),
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
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--script-id", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--repo-label", required=True)
    p.add_argument("--intent", default="implement")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-sandbox", action="store_true")
    args = p.parse_args(argv)
    counters = ingest(
        src_dir=args.src, out_path=args.out, script_id=args.script_id,
        license=args.license, repo_label=args.repo_label,
        limit=args.limit, skip_sandbox=args.skip_sandbox,
        intent=args.intent,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
