"""ingest/commit_pack_ft.py — CommitPackFT → SWE-practice training rows.

CommitPackFT (bigcode/commitpackft, MIT) is a permissive-only,
instruction-tuned subset of CommitPack: real commits where the message
reads like an instruction and the diff is small enough to learn from.
Shape (one JSONL row per commit):

    commit, old_file, new_file, old_contents, new_contents,
    subject, message, lang, license, repos

This is the cleanest replacement for SE's "how do I X" / "how do I
debug Y" / "how do I refactor Z" patterns, but permissive-licensed
end-to-end (each row carries its repo license, e.g. bsd-3-clause,
mit, apache-2.0).

Conversion:
    prompt   = `[ctx lang=... intent=<inferred>] {subject}` with an
               optional `\n\n{message_body}` when present and non-trivial.
    response = `new_contents` — the brain learns what the file looks
               like AFTER the change.  For small files (< MAX_FILE_CHARS)
               we include the full new file so the response is complete.

Quality gates:
  - new_contents length 50–4000 chars (small focused commits only)
  - subject length 8–200 chars
  - message minus subject is either empty or 10–1500 chars
  - response code passes sandbox syntactic check (when supported)
  - row.license must be permissive — enforced by Row writer

Intent inference (cheap regex on the subject):
    "fix" / "bug" / "error"        → intent=debug
    "refactor" / "cleanup" / "rewrite"→ intent=refactor
    "test" / "spec" / "coverage"   → intent=test
    "add" / "implement" / "support" → intent=implement
    "rename" / "move"               → intent=refactor
    "doc" / "comment" / "readme"   → intent=document
    else                            → intent=implement (catch-all)

Lang mapping: CommitPackFT's `lang` is title-cased ("Python", "Rust",
"JavaScript").  We lowercase for ctx, and map to the sandbox-supported
subset for syntactic checks.

CLI:
    python -m tools.training_standard.ingest.commit_pack_ft \\
        --src D:/.../sources/commitpackft/python.jsonl \\
        --out D:/.../training/cpft_python.jsonl \\
        --lang python \\
        --script-id swe_practice_refactor_001 \\
        --limit 50000
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

MAX_FILE_CHARS    = 4000
MIN_FILE_CHARS    = 50
MIN_SUBJECT_CHARS = 8
MAX_SUBJECT_CHARS = 200
MAX_MESSAGE_CHARS = 1500

# CommitPackFT lang label (title case) → our lowercase ctx + sandbox key.
LANG_MAP: dict[str, str] = {
    "python":     "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust":       "rust",
    "bash":       "bash",
    "shell":      "bash",
    "powershell": "powershell",
    "go":         "go",
    "java":       "java",
    "c++":        "cpp",
    "c":          "cpp",
    "c#":         "csharp",
    "ruby":       "ruby",
    "php":        "php",
}

_SANDBOX_SUPPORTED = {"python", "javascript", "bash", "powershell"}

# Intent inference patterns — first match wins.
_INTENT_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^(fix|bug|error|crash|broken|regression|issue|hotfix)\b", re.I), "debug"),
    (re.compile(r"^(refactor|cleanup|rewrite|rename|move|reorganize|simplify)\b", re.I), "refactor"),
    (re.compile(r"^(test|spec|coverage|tests?|tdd)\b", re.I), "test"),
    (re.compile(r"^(doc|comment|readme|docstring|changelog|copyright)", re.I), "document"),
    (re.compile(r"^(add|implement|support|introduce|enable|allow|create)\b", re.I), "implement"),
    (re.compile(r"^(remove|delete|drop|deprecate)\b", re.I), "refactor"),
    (re.compile(r"^(update|upgrade|bump|change|switch)\b", re.I), "implement"),
]


def _infer_intent(subject: str) -> str:
    s = (subject or "").strip()
    for pat, intent in _INTENT_RULES:
        if pat.search(s):
            return intent
    return "implement"


def _normalize_license(lic: str) -> str:
    """CommitPackFT spellings → our permissive set spellings."""
    if not lic:
        return ""
    s = lic.strip().lower()
    # Common spellings: bsd-3-clause, bsd-2-clause, mit, apache-2.0, isc
    if s in PERMISSIVE_LICENSES:
        return s
    # A few aliases CommitPackFT uses.
    aliases = {
        "apache 2.0":           "apache-2.0",
        "apache-licence-2.0":   "apache-2.0",
        "bsd":                  "bsd-3-clause",
        "0bsd":                 "0bsd",
        "public domain":        "cc0-1.0",
        "unlicense":            "unlicense",
        "mit-0":                "mit",
        "wtfpl":                "wtfpl",
        "zlib":                 "zlib",
    }
    return aliases.get(s, s)  # let row writer reject if still bad


def _split_message(subject: str, message: str) -> tuple[str, str]:
    """Return (clean_subject, body_after_subject_or_empty)."""
    subject = (subject or "").strip()
    message = (message or "").strip()
    if message.startswith(subject):
        body = message[len(subject):].strip()
    else:
        body = message
    return subject, body


def ingest(
    *,
    src_path: Path,
    out_path: Path,
    script_id: str,
    lang: str,
    limit: int | None,
    skip_sandbox: bool,
) -> dict:
    sb = None if skip_sandbox else get_sandbox()
    sandbox_lang = LANG_MAP.get(lang, lang)
    do_sandbox = (sb is not None) and (sandbox_lang in _SANDBOX_SUPPORTED)

    counters = {
        "seen":               0,
        "rejected_quality":   0,
        "rejected_lang":      0,
        "rejected_license":   0,
        "rejected_sandbox":   0,
        "rejected_row_writer":0,
        "dedup_skipped":      0,
        "written":            0,
    }

    with RowWriter(out_path, script_id=script_id,
                   source=f"commitpackft:{lang}") as writer:
        with src_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                counters["seen"] += 1
                if limit is not None and counters["written"] >= limit:
                    break

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                row_lang = (rec.get("lang") or "").strip().lower()
                if row_lang != lang:
                    counters["rejected_lang"] += 1
                    continue

                lic_norm = _normalize_license(rec.get("license") or "")
                if lic_norm not in PERMISSIVE_LICENSES:
                    counters["rejected_license"] += 1
                    continue

                subject, body = _split_message(rec.get("subject"), rec.get("message"))
                if not (MIN_SUBJECT_CHARS <= len(subject) <= MAX_SUBJECT_CHARS):
                    counters["rejected_quality"] += 1
                    continue
                if body and len(body) > MAX_MESSAGE_CHARS:
                    body = body[:MAX_MESSAGE_CHARS]

                new_contents = rec.get("new_contents") or ""
                if not (MIN_FILE_CHARS <= len(new_contents) <= MAX_FILE_CHARS):
                    counters["rejected_quality"] += 1
                    continue

                if do_sandbox:
                    r = sb.check(sandbox_lang, new_contents, timeout_s=10.0)
                    if not r.ok:
                        counters["rejected_sandbox"] += 1
                        continue

                intent = _infer_intent(subject)
                prompt = subject if not body else f"{subject}\n\n{body}"

                row = Row(
                    prompt=prompt,
                    response=new_contents,
                    ctx=render_ctx(
                        lang=lang,
                        intent=intent,
                        source="cpft",
                        file=Path(rec.get("new_file") or "").name or "?",
                    ),
                    license=lic_norm,
                    source=f"commitpackft:{lang}:{rec.get('commit','?')}:{rec.get('new_file','?')}",
                    source_hash=hash_source(new_contents),
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
                   help="path to commitpackft data.jsonl for one language")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--script-id", required=True)
    p.add_argument("--lang", required=True, choices=sorted(LANG_MAP),
                   help="language label (matches the per-language data.jsonl)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-sandbox", action="store_true")
    args = p.parse_args(argv)

    counters = ingest(
        src_path=args.src, out_path=args.out, script_id=args.script_id,
        lang=args.lang, limit=args.limit, skip_sandbox=args.skip_sandbox,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
