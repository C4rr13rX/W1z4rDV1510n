"""generate/paraphrase.py — N prompt variants per source row.

Reads a JSONL produced by an ingest script and writes a new JSONL
containing the original prompts PLUS up to `n_variants` paraphrased
versions, each mapping to the *same* response.  The Hebbian brain
learns "different phrasings of the same intent → same code", which
is the foundation for handling user under-specification.

Why deterministic / template-based paraphrase rather than calling an
LLM:
  - We're permissive-only.  Synthesizing rows via a proprietary API
    would taint the corpus license.
  - Deterministic paraphrase is reproducible — the manifest captures
    exactly which variants were generated.
  - The brain is Hebbian, not transformer.  It learns associations;
    it doesn't need linguistic depth in the prompt side.  A handful
    of consistent templates work.

Variant families (each row picks up to n_variants of them):
  1. instruction style    "Write a function that …"
  2. need style           "I need a function that …"
  3. how-do-I style       "How do I write a function that …"
  4. show-me style        "Show me a function that …"
  5. terse style          (verb-leading, no preamble)
  6. context-tag dropped  prompt + response unchanged but ctx loses
                          one atom (handled by partial_context.py,
                          not here)

The original row is always passed through, then up to (n_variants - 1)
new variants are emitted with the same response.  Variants are deduped
within a row (so e.g. if templates collapse to the same string we
only emit it once).

CLI:
    python -m tools.training_standard.generate.paraphrase \\
        --src D:/.../training/csn_python.jsonl \\
        --out D:/.../training/csn_python_para3.jsonl \\
        --n-variants 3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.row import (
    Row, RowRejected, RowWriter, hash_source, iter_jsonl,
)


# Detect what kind of prompt we're paraphrasing.  Most CSN/CPFT-derived
# rows start with "Write a <lang> function `name` that: …" — we
# canonicalise by extracting the descriptive clause after the colon
# and rewrap it in each template style.
_FUNC_INTRO = re.compile(
    r"^\s*write\s+a\s+(?P<lang>\w+)\s+function\s+`?(?P<name>[\w_.]+)`?\s+that:?\s*",
    re.IGNORECASE,
)
# CommitPackFT-style commits look like "Add support for X" — already
# imperative.  We can paraphrase these in a similar way.
_IMPERATIVE = re.compile(
    r"^\s*(?P<verb>add|fix|update|refactor|rename|remove|implement|create|"
    r"improve|test|document|introduce|support|change)\b",
    re.IGNORECASE,
)


def _strip_imperative_subject(subject: str) -> str:
    """For 'Add support for X' returns 'support for X' (lowercased
    object), so we can plug it into question forms etc."""
    m = _IMPERATIVE.match(subject)
    if not m:
        return subject
    rest = subject[m.end():].strip()
    return rest[:1].lower() + rest[1:] if rest else subject


def variants_for(prompt: str, lang_atom: str | None) -> list[str]:
    """Return distinct paraphrases of `prompt`.  Always includes the
    original as the first element."""
    out: list[str] = [prompt.strip()]
    seen: set[str] = {out[0]}

    def _push(text: str) -> None:
        t = text.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)

    m = _FUNC_INTRO.match(prompt)
    if m:
        lang = m.group("lang") or (lang_atom or "function")
        name = m.group("name") or "this"
        clause = prompt[m.end():].strip().rstrip(".")
        # When inserted after "that ", lead with lowercase verb so the
        # sentence reads naturally even though docstrings start
        # capitalised.
        clause_lc = clause[:1].lower() + clause[1:] if clause else ""
        clause_cap = clause[:1].upper() + clause[1:] if clause else ""
        if clause:
            _push(f"Implement `{name}` in {lang} that {clause_lc}.")
            _push(f"I need a {lang} function called `{name}` that {clause_lc}.")
            _push(f"How do I write a {lang} function `{name}` that {clause_lc}?")
            _push(f"Show me a {lang} function `{name}` — it should {clause_lc}.")
            _push(f"In {lang}, give me `{name}` that {clause_lc}.")
            _push(f"{clause_cap}. Implement it as `{name}` in {lang}.")
        return out

    m = _IMPERATIVE.match(prompt)
    if m:
        rest = _strip_imperative_subject(prompt)
        if rest:
            _push(f"Please {m.group('verb').lower()} {rest}.")
            _push(f"Can you {m.group('verb').lower()} {rest}?")
            _push(f"I'd like to {m.group('verb').lower()} {rest}.")
        return out

    # Generic fallback: question / show-me / I-need wrappers.
    body = prompt.strip().rstrip(".?!")
    if body:
        _push(f"Can you do this: {body}.")
        _push(f"I'd like {body[:1].lower()}{body[1:]}.")
        _push(f"How do I: {body}?")
    return out


def generate(
    *,
    src_path: Path,
    out_path: Path,
    n_variants: int,
    limit: int | None,
) -> dict:
    counters = {
        "src_rows":       0,
        "variants_total": 0,
        "rejected":       0,
        "dedup_skipped":  0,
        "written":        0,
    }

    # Streaming RowWriter, but we need to override dedup behaviour: we
    # want (prompt + response) uniqueness, not response-only.  The
    # writer already uses Row.source_hash — we provide that hash
    # explicitly per variant.
    with RowWriter(out_path,
                   script_id=f"{src_path.stem}_paraphrased",
                   source=f"paraphrase:{src_path.stem}") as writer:
        for rec in iter_jsonl(src_path):
            counters["src_rows"] += 1
            if limit is not None and counters["written"] >= limit:
                break
            prompt   = rec.get("prompt") or ""
            response = rec.get("response") or ""
            ctx      = rec.get("ctx") or {}
            lang     = ctx.get("lang") if isinstance(ctx, dict) else None
            licence  = rec.get("license") or ""
            base_src = rec.get("source") or "unknown"
            script_id = rec.get("script_id") or "paraphrase_generic"
            if not prompt.strip() or not response.strip():
                counters["rejected"] += 1
                continue
            variants = variants_for(prompt, lang)[:n_variants]
            counters["variants_total"] += len(variants)
            for i, v in enumerate(variants):
                row = Row(
                    prompt=v,
                    response=response,
                    ctx=ctx if isinstance(ctx, dict) else {},
                    license=licence,
                    source=f"{base_src}|gen=paraphrase:{i}",
                    source_hash=hash_source(f"{v}|||{response}"),
                    script_id=script_id,
                )
                try:
                    accepted = writer.write(row)
                except RowRejected:
                    counters["rejected"] += 1
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
                   help="input JSONL produced by an ingest script")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-variants", type=int, default=4,
                   help="max variants per source row (includes original; "
                        "default 4)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    counters = generate(src_path=args.src, out_path=args.out,
                        n_variants=args.n_variants, limit=args.limit)
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
