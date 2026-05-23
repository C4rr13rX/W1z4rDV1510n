"""ingest/math_reasoning.py — math / scientific reasoning Q&A → training rows.

Several permissive math datasets share a near-identical shape: a
problem statement (English prose, often with LaTeX) and a worked
solution (step-by-step natural language with the final answer).
These teach the brain how to reason about scientific/quantitative
problems, complementing the code-shape corpora (CSN, jupyter, CPFT).

Supported sources (auto-detected by file format and field names):

  - openai/gsm8k (MIT)               .parquet  question / answer
  - meta-math/MetaMathQA (MIT)       .json     query / response
  - TIGER-Lab/MathInstruct (MIT)     .json     instruction / output
  - hendrycks/competition_math (MIT) .parquet  problem / solution / level / type
  - generic JSON arrays              detected via field aliases

Common output shape:

    prompt   = the problem statement (trimmed, LaTeX preserved)
    response = the worked solution (trimmed)
    ctx      = {intent=reason, source=<dataset>, kind=math,
                level=<difficulty if present>, topic=<topic if present>}
    license  = mit (datasets verified MIT-permissive upstream)

Quality gates:
    - problem 20-2000 chars
    - solution 20-4000 chars
    - non-empty after strip

No sandbox check — these rows are prose/math, not executable code.

CLI:
    python -m tools.training_standard.ingest.math_reasoning \\
        --src D:/.../sources/math_reasoning/metamathqa.json \\
        --out D:/.../training/metamath.jsonl \\
        --script-id reasoning_math_001 \\
        --license mit
"""
from __future__ import annotations

import argparse
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

# Field-name aliases — first match wins per row.
_PROMPT_KEYS   = ("query", "question", "instruction", "problem", "prompt")
_RESPONSE_KEYS = ("response", "answer", "output", "solution", "completion")
_TOPIC_KEYS    = ("type", "topic", "category", "subject")
_LEVEL_KEYS    = ("level", "difficulty")
_SOURCE_KEYS   = ("source", "dataset", "origin")

MIN_PROMPT = 20
MAX_PROMPT = 2000
MIN_RESP   = 20
MAX_RESP   = 4000


def _first_match(d: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _iter_records(src_path: Path) -> Iterator[dict]:
    """Stream records from a parquet, JSON array, or JSONL file."""
    if src_path.suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(src_path)
        for batch in pf.iter_batches(batch_size=2048):
            cols = {c: batch.column(c) for c in batch.schema.names}
            n = len(batch)
            for i in range(n):
                yield {k: (v[i].as_py() if v[i].is_valid else "") for k, v in cols.items()}
        return
    if src_path.suffix in (".json", ".jsonl"):
        # JSON array (typical for HF JSON dumps) vs JSONL — detect cheaply.
        with src_path.open("r", encoding="utf-8") as fh:
            first = fh.read(64).lstrip()
        if first.startswith("["):
            # Whole-file JSON array — load it.  Up to ~500MB; fits in
            # RAM on this host.  If we need lazy mode later we'll add
            # ijson; for now keep it simple.
            with src_path.open("r", encoding="utf-8") as fh:
                arr = json.load(fh)
            for item in arr:
                if isinstance(item, dict):
                    yield item
            return
        # JSONL.
        with src_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
        return
    raise ValueError(f"unsupported math-source format: {src_path}")


def ingest(
    *,
    src_path: Path,
    out_path: Path,
    script_id: str,
    license: str,
    dataset_label: str,
    intent: str,
    limit: int | None,
) -> dict:
    lic = license.strip().lower()
    if lic not in PERMISSIVE_LICENSES:
        raise RuntimeError(
            f"--license {license!r} not in PERMISSIVE_LICENSES "
            f"({sorted(PERMISSIVE_LICENSES)})"
        )

    counters = {
        "seen":               0,
        "rejected_quality":   0,
        "rejected_row_writer":0,
        "dedup_skipped":      0,
        "written":            0,
    }

    with RowWriter(out_path, script_id=script_id,
                   source=f"math:{dataset_label}") as writer:
        for rec in _iter_records(src_path):
            counters["seen"] += 1
            if limit is not None and counters["written"] >= limit:
                break

            prompt   = _first_match(rec, _PROMPT_KEYS).strip()
            response = _first_match(rec, _RESPONSE_KEYS).strip()
            if not (MIN_PROMPT <= len(prompt) <= MAX_PROMPT):
                counters["rejected_quality"] += 1
                continue
            if not (MIN_RESP <= len(response) <= MAX_RESP):
                counters["rejected_quality"] += 1
                continue

            ctx_atoms = {
                "intent": intent,
                "source": dataset_label,
                "kind":   "math",
            }
            level = _first_match(rec, _LEVEL_KEYS).strip()
            topic = _first_match(rec, _TOPIC_KEYS).strip()
            if level:
                ctx_atoms["level"] = level
            if topic:
                # Normalize topic — many datasets use long phrases here.
                ctx_atoms["topic"] = topic[:32].lower().replace(" ", "_")

            row = Row(
                prompt=prompt,
                response=response,
                ctx=render_ctx(**ctx_atoms),
                license=lic,
                source=f"math:{dataset_label}:{counters['seen']}",
                source_hash=hash_source(f"{prompt}|||{response}"),
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
                   help="path to .parquet | .json (array) | .jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--script-id", required=True)
    p.add_argument("--license", required=True)
    p.add_argument("--dataset-label", required=True,
                   help="short label for provenance, e.g. 'gsm8k', 'metamathqa'")
    p.add_argument("--intent", default="reason")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    counters = ingest(
        src_path=args.src, out_path=args.out, script_id=args.script_id,
        license=args.license, dataset_label=args.dataset_label,
        intent=args.intent, limit=args.limit,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
