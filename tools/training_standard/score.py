"""training_standard/score.py — benchmark response evaluator.

Given a Benchmark and a response string, returns a BenchmarkResult.
Three checks combine into the final score:

  1. Keyword recall  — fraction of `must_include` substrings present
                       (case-insensitive substring match, not word).
  2. Forbidden hit   — any `forbidden` substring kills the pass.
  3. Structural      — if `must_be_valid` is set, AST/syntax-check the
                       extracted code block.  Pass adds full weight;
                       fail caps the score at 0.5.

Final score is a weighted mean; weights chosen so a response can pass
on keyword recall alone for prose benchmarks (must_be_valid=None) and
must clear the structural bar for code benchmarks.
"""
from __future__ import annotations

import ast
import dataclasses
import re
from typing import Iterable

from .schema import Benchmark


_FENCED_RE = re.compile(r"```(?:[a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)

# What makes a string read as shell rather than as a sentence about shell.
# `for`/`if`/`while` are absent on purpose: they are ordinary English words
# ("...one for the word ERROR"), so only their closing halves count.
_BASH_SIGNAL_RE = re.compile(
    r"[|$`]"                        # pipe, expansion, command substitution
    r"|[<>]"                        # redirection
    r"|(?:^|\s)-{1,2}[A-Za-z]"      # a flag argument
    r"|(?:^|\s)\w+="                # assignment
    r"|^#!"                         # shebang
    r"|\b(?:do|then|done|fi|esac|elif)\b",
    re.MULTILINE,
)


@dataclasses.dataclass
class BenchmarkResult:
    benchmark_label: str
    prompt:           str
    response:         str
    score:            float          # 0..1
    passed:           bool
    breakdown:        dict           # keyword_score, forbidden_hits, struct_ok
    error:            str = ""       # populated if probe itself failed


def _extract_code(response: str) -> str:
    """Pull the first fenced code block, or fall back to the full
    response if there's no fence.  Some models emit bare code without
    fences when the prompt was code-only."""
    m = _FENCED_RE.search(response)
    if m:
        return m.group(1)
    return response


def _validate_lang(code: str, lang: str) -> tuple[bool, str]:
    """Return (ok, error_msg).  Best-effort syntax check per language."""
    if lang == "python":
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as exc:
            return False, f"python SyntaxError: {exc}"
    if lang == "json":
        import json
        try:
            json.loads(code)
            return True, ""
        except Exception as exc:
            return False, f"json parse error: {exc}"
    # Rust/JS/Bash/PowerShell — no in-process parser bundled, so we
    # do cheap heuristic checks rather than nothing.
    if lang == "rust":
        if "fn " not in code and "let " not in code and "struct " not in code:
            return False, "rust check: no fn/let/struct keyword found"
        # balanced braces, paren-counted
        if code.count("{") != code.count("}"):
            return False, "rust check: unbalanced braces"
        return True, ""
    if lang == "javascript":
        if not any(k in code for k in ("function", "const ", "let ", "var ", "=>")):
            return False, "js check: no function/declaration keyword"
        if code.count("{") != code.count("}"):
            return False, "js check: unbalanced braces"
        return True, ""
    if lang == "go":
        # `go` has to have a real arm, not just a seat in SUPPORTED_LANGS:
        # the fallthrough at the bottom of this function returns ok for any
        # language it doesn't know, so a bare listing would have handed every
        # Go benchmark the full 0.5 structural weight for any text at all.
        if not any(k in code for k in ("func ", "func(", "package ", "type ", "var ")):
            return False, "go check: no func/package/type/var keyword"
        if code.count("{") != code.count("}"):
            return False, "go check: unbalanced braces"
        return True, ""
    if lang == "bash":
        # This arm used to be unable to fail: a `pass`-bodied nesting check
        # followed by an unconditional `return True, ""`, so every bash
        # benchmark collected the full 0.5 structural weight for any text at
        # all, prose included.  Kept heuristic like rust/js above -- no shell
        # parser is bundled and `evaluate` is documented pure -- but it now
        # discriminates.  See test_every_declared_language_has_a_validator_arm.
        if not _BASH_SIGNAL_RE.search(code):
            # Deliberately keyed on shell *signals* -- operators, flags,
            # variables, block keywords -- and not on "a line that starts with
            # a word", which every English sentence also satisfies.  The cost
            # is that a bare `echo hello` reads as prose; no registry benchmark
            # asks for one, and a false accept here is the more expensive error.
            return False, "bash check: no shell operator, flag or keyword found"
        # Truncation is the failure this exists to catch, and an unterminated
        # block is what truncation looks like in shell.
        for opener, closer in (("do", "done"), ("then", "fi"), ("case", "esac")):
            opens = len(re.findall(rf"\b{opener}\b", code))
            closes = len(re.findall(rf"\b{closer}\b", code))
            if opens != closes:
                return False, f"bash check: {opens} {opener} vs {closes} {closer}"
        if code.count('"') % 2:
            return False, "bash check: unbalanced double quotes"
        return True, ""
    if lang == "powershell":
        if not any(k in code.lower() for k in ("get-", "set-", "$", "function", "param")):
            return False, "powershell check: no cmdlet/var/keyword found"
        return True, ""
    return True, ""  # unsupported lang = skip


def evaluate(response: str, benchmark: Benchmark) -> BenchmarkResult:
    """Score `response` against `benchmark`.  Pure function: no I/O,
    so the runner can call this in tight loops over historical
    responses without side effects."""
    text = response or ""
    lower = text.lower()

    # 1) Keyword recall
    must = benchmark.must_include
    if must:
        hits = sum(1 for kw in must if kw.lower() in lower)
        keyword_score = hits / len(must)
    else:
        # No keywords declared → keyword axis doesn't constrain.  We
        # don't want a benchmark with only must_be_valid to score 0 on
        # the keyword axis, so treat empty must as full credit.
        keyword_score = 1.0

    # 2) Forbidden
    forbidden_hits = [f for f in benchmark.forbidden if f.lower() in lower]

    # 3) Structural
    struct_ok = True
    struct_err = ""
    if benchmark.must_be_valid:
        code = _extract_code(text)
        struct_ok, struct_err = _validate_lang(code, benchmark.must_be_valid)

    # Combined score.  For code benchmarks we weight structure heavily:
    # a syntactically-broken response can't pass even with perfect
    # keyword recall.  For prose benchmarks structure is N/A so it
    # doesn't enter the mean.
    if benchmark.must_be_valid:
        score = 0.5 * keyword_score + (0.5 if struct_ok else 0.0)
    else:
        score = keyword_score

    if forbidden_hits:
        score = min(score, 0.0)

    passed = score >= benchmark.min_score and not forbidden_hits and struct_ok

    return BenchmarkResult(
        benchmark_label=benchmark.label,
        prompt=benchmark.prompt,
        response=text,
        score=round(score, 3),
        passed=passed,
        breakdown={
            "keyword_score":  round(keyword_score, 3),
            "forbidden_hits": forbidden_hits,
            "structural_ok":  struct_ok,
            "structural_err": struct_err,
        },
    )
