"""training_standard/schema.py — dataclasses + TOML loader.

A training script lives in registry/<id>.toml.  The shape is fixed by
this module; the runner refuses to execute anything that doesn't load
into a valid TrainingScript.

The schema is intentionally small.  Adding fields is cheap, but each
field has to mean something the runner can verify — otherwise the
"standard" devolves into commentary.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib  # py311+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# Categories every script must declare.  The runner uses these to
# group benchmark sparklines in the UI and to enforce that the
# overall curriculum covers every target.
CATEGORIES = {
    "conversation",
    "code_generation",
    "agent_planning",
    "automation",
    "security",
    "terminal_proficiency_win",
    "terminal_proficiency_mac",
    "terminal_proficiency_linux",
    # Cross-modal: text ↔ image/audio/pdf paired training so the multi-pool
    # fabric forms cross-pool synapses between modality pools.
    "multimodal_pairs",
}

# Code languages the AST validator understands.  Anything else means
# "skip structural check, fall back to keyword score only".
SUPPORTED_LANGS = {"python", "rust", "javascript", "bash", "powershell"}


@dataclasses.dataclass(frozen=True)
class Input:
    kind: str                       # "corpus" | "generator" | "template"
    path: str                       # repo-relative path or generator-name


@dataclasses.dataclass(frozen=True)
class Benchmark:
    """One verifiable contract this script promises after training.

    The evaluator scores a response on three axes:
      - keyword recall:  fraction of `must_include` substrings present
      - forbidden hit:   penalty for any `forbidden` substring found
      - structural:      if `must_be_valid` is set, AST/syntax check

    Final score is the weighted combo (see score.py).  Pass iff
    score >= min_score AND no forbidden hits.
    """
    prompt:        str
    must_include:  tuple[str, ...] = ()
    forbidden:     tuple[str, ...] = ()
    must_be_valid: str | None = None         # one of SUPPORTED_LANGS or None
    min_score:     float = 0.7
    # Optional: where the answer should land on the structure axis.
    # "code" expects a fenced or bare code block; "prose" expects no
    # code fence; "json" expects parseable JSON.  None = don't check.
    expected_structure: str | None = None
    # Human-readable label so UI sparklines have something to show.
    label: str = ""


@dataclasses.dataclass(frozen=True)
class RegressionProtect:
    """A reference to another script whose benchmarks must continue to
    pass after this script trains.  The runner reruns those benchmarks
    and treats any drop below their min_score as a regression alert."""
    script_id: str


@dataclasses.dataclass(frozen=True)
class TrainingScript:
    id:                 str
    category:           str
    phase:              int
    description:        str
    depends_on:         tuple[str, ...]
    inputs:             tuple[Input, ...]
    benchmarks:         tuple[Benchmark, ...]
    regression_protects: tuple[RegressionProtect, ...]
    source_path:        Path                 # where it was loaded from


# ── Loader ──────────────────────────────────────────────────────────────────


class SchemaError(ValueError):
    """Raised when a TOML file fails to load into a TrainingScript."""


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise SchemaError(f"{ctx}: missing required key '{key}'")
    return d[key]


def _parse_input(raw: dict, ctx: str) -> Input:
    kind = _require(raw, "kind", ctx)
    if kind not in ("corpus", "generator", "template"):
        raise SchemaError(f"{ctx}: input.kind must be corpus|generator|template, got {kind!r}")
    return Input(kind=kind, path=_require(raw, "path", ctx))


def _parse_benchmark(raw: dict, ctx: str) -> Benchmark:
    prompt = _require(raw, "prompt", ctx)
    must_be_valid = raw.get("must_be_valid")
    if must_be_valid is not None and must_be_valid not in SUPPORTED_LANGS:
        raise SchemaError(
            f"{ctx}: must_be_valid={must_be_valid!r} not in {SUPPORTED_LANGS}"
        )
    expected_structure = raw.get("expected_structure")
    if expected_structure is not None and expected_structure not in ("code", "prose", "json"):
        raise SchemaError(
            f"{ctx}: expected_structure={expected_structure!r} must be code|prose|json"
        )
    min_score = float(raw.get("min_score", 0.7))
    if not (0.0 <= min_score <= 1.0):
        raise SchemaError(f"{ctx}: min_score must be in [0,1], got {min_score}")
    return Benchmark(
        prompt=prompt,
        must_include=tuple(raw.get("must_include", ())),
        forbidden=tuple(raw.get("forbidden", ())),
        must_be_valid=must_be_valid,
        min_score=min_score,
        expected_structure=expected_structure,
        label=raw.get("label", prompt[:60]),
    )


def load_script(path: Path) -> TrainingScript:
    """Parse a single TOML file into a TrainingScript.  Raises
    SchemaError with a precise pointer to the offending field if
    anything is malformed."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SchemaError(f"{path}: TOML parse failed — {exc}") from exc

    script = _require(raw, "script", f"{path}")
    sid = _require(script, "id", f"{path}[script]")
    category = _require(script, "category", f"{path}[script]")
    if category not in CATEGORIES:
        raise SchemaError(
            f"{path}[script]: category={category!r} not in {sorted(CATEGORIES)}"
        )
    phase = int(_require(script, "phase", f"{path}[script]"))

    inputs_raw = raw.get("inputs") or []
    if not isinstance(inputs_raw, list):
        raise SchemaError(f"{path}: [[inputs]] must be a TOML array of tables")
    inputs = tuple(
        _parse_input(item, f"{path}[[inputs]][{i}]")
        for i, item in enumerate(inputs_raw)
    )

    benchmarks_raw = raw.get("benchmarks") or []
    if not isinstance(benchmarks_raw, list):
        raise SchemaError(f"{path}: [[benchmarks]] must be a TOML array of tables")
    if not benchmarks_raw:
        raise SchemaError(
            f"{path}: a script with no benchmarks isn't part of the standard — "
            "every script must promise at least one verifiable outcome."
        )
    benchmarks = tuple(
        _parse_benchmark(item, f"{path}[[benchmarks]][{i}]")
        for i, item in enumerate(benchmarks_raw)
    )

    rp_raw = raw.get("regression_protects") or []
    regression_protects = tuple(
        RegressionProtect(script_id=_require(item, "script_id", f"{path}[[regression_protects]]"))
        for item in rp_raw
    )

    return TrainingScript(
        id=sid,
        category=category,
        phase=phase,
        description=script.get("description", ""),
        depends_on=tuple(script.get("depends_on", ())),
        inputs=inputs,
        benchmarks=benchmarks,
        regression_protects=regression_protects,
        source_path=path,
    )


def load_registry(registry_dir: Path) -> dict[str, TrainingScript]:
    """Load every *.toml under registry_dir into a {id: script} map.
    Validates uniqueness of ids and that every depends_on / regression
    target actually exists in the registry."""
    scripts: dict[str, TrainingScript] = {}
    for path in sorted(registry_dir.glob("*.toml")):
        script = load_script(path)
        if script.id in scripts:
            raise SchemaError(
                f"duplicate script id {script.id!r} (also in {scripts[script.id].source_path})"
            )
        scripts[script.id] = script

    # Cross-reference validation.
    for s in scripts.values():
        for dep in s.depends_on:
            if dep not in scripts:
                raise SchemaError(
                    f"{s.source_path}: depends_on references unknown script {dep!r}"
                )
        for rp in s.regression_protects:
            if rp.script_id not in scripts:
                raise SchemaError(
                    f"{s.source_path}: regression_protects references unknown "
                    f"script {rp.script_id!r}"
                )
    return scripts
