"""generate/partial_context.py — strip [ctx] atoms for graceful-degradation training.

Reads a JSONL produced by an ingest or generator script and emits new
rows where a fraction of the [ctx] metadata atoms have been removed.
The Hebbian brain learns "[ctx ... lang=python ...]" co-occurs with
Python-specific code; if at inference time the agent can only supply
{intent=implement} (no lang), the brain should still recall *broadly*
relevant patterns rather than going OOG.

Plan §4 calls for ~10% of all training rows to carry deliberately
partial context, so this generator's default `--partial-rate 0.10`
gives one stripped variant for every 10 source rows on average.  Pass
--partial-rate 1.0 to emit one partial variant per source row.

Stripping strategies (all preserve the original row unchanged; partial
variants are NEW rows alongside):
  1. drop-one        randomly remove one ctx atom
  2. drop-all-but-intent  keep only intent; the most aggressive degrade
  3. drop-lang       remove the lang atom specifically (user didn't
                     say which language — common in real Q&A)
  4. drop-source     remove source/site provenance atoms (irrelevant
                     to recall but tests robustness)

Each source row emits up to `--variants-per-row` partial variants (one
per strategy that applies — e.g. a row with no lang atom skips
drop-lang).  Variants are deduped within a row by (ctx, prompt,
response) so we don't repeat work.

CLI:
    python -m tools.training_standard.generate.partial_context \\
        --src D:/.../training/csn_python_para4.jsonl \\
        --out D:/.../training/csn_python_para4_partial.jsonl \\
        --partial-rate 0.1 \\
        --variants-per-row 2

Provenance: variant `source` becomes
    <original_source>|gen=partial:<strategy>
so downstream eval can identify partial-context rows in the corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.row import (
    Row, RowRejected, RowWriter, hash_source, iter_jsonl,
)


_STRATEGIES = ("drop_one", "drop_all_but_intent", "drop_lang", "drop_source")


def _apply_strategy(ctx: dict, strategy: str, rng: random.Random) -> dict | None:
    """Return the modified ctx, or None if the strategy doesn't apply
    to this row (so we skip rather than emit an unchanged variant)."""
    if not isinstance(ctx, dict) or not ctx:
        return None
    new_ctx = dict(ctx)
    if strategy == "drop_one":
        if len(new_ctx) <= 1:
            return None
        key = rng.choice(sorted(new_ctx.keys()))
        new_ctx.pop(key, None)
    elif strategy == "drop_all_but_intent":
        if "intent" not in new_ctx or len(new_ctx) <= 1:
            return None
        new_ctx = {"intent": new_ctx["intent"]}
    elif strategy == "drop_lang":
        if "lang" not in new_ctx:
            return None
        new_ctx.pop("lang", None)
    elif strategy == "drop_source":
        if "source" not in new_ctx and "site" not in new_ctx:
            return None
        new_ctx.pop("source", None)
        new_ctx.pop("site", None)
    else:
        return None
    # Empty ctx is allowed — it represents "user gave no metadata"
    return new_ctx


def generate(
    *,
    src_path: Path,
    out_path: Path,
    partial_rate: float,
    variants_per_row: int,
    seed: int,
    limit: int | None,
) -> dict:
    rng = random.Random(seed)
    counters = {
        "src_rows":              0,
        "rolled_into_partial":   0,
        "variants_total":        0,
        "rejected":              0,
        "dedup_skipped":         0,
        "written":               0,
    }
    with RowWriter(out_path,
                   script_id=f"{src_path.stem}_partial",
                   source=f"partial:{src_path.stem}") as writer:
        for rec in iter_jsonl(src_path):
            counters["src_rows"] += 1
            if limit is not None and counters["written"] >= limit:
                break
            if rng.random() >= partial_rate:
                continue
            counters["rolled_into_partial"] += 1

            prompt   = rec.get("prompt") or ""
            response = rec.get("response") or ""
            ctx      = rec.get("ctx") or {}
            licence  = rec.get("license") or ""
            base_src = rec.get("source") or "unknown"
            script_id = rec.get("script_id") or "partial_context_generic"

            if not prompt.strip() or not response.strip():
                counters["rejected"] += 1
                continue

            tried_ctx_hashes: set[str] = set()
            emitted = 0
            for strategy in rng.sample(list(_STRATEGIES), k=len(_STRATEGIES)):
                if emitted >= variants_per_row:
                    break
                new_ctx = _apply_strategy(ctx, strategy, rng)
                if new_ctx is None:
                    continue
                ctx_key = hashlib.sha1(
                    json.dumps(new_ctx, sort_keys=True).encode()
                ).hexdigest()
                if ctx_key in tried_ctx_hashes:
                    continue
                tried_ctx_hashes.add(ctx_key)

                counters["variants_total"] += 1
                row = Row(
                    prompt=prompt,
                    response=response,
                    ctx=new_ctx,
                    license=licence,
                    source=f"{base_src}|gen=partial:{strategy}",
                    source_hash=hash_source(
                        f"{prompt}|||{response}|||{json.dumps(new_ctx, sort_keys=True)}"
                    ),
                    script_id=script_id,
                )
                try:
                    accepted = writer.write(row)
                except RowRejected:
                    counters["rejected"] += 1
                    continue
                if accepted:
                    counters["written"] += 1
                    emitted += 1
                else:
                    counters["dedup_skipped"] += 1
                if limit is not None and counters["written"] >= limit:
                    break

    return counters


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partial-rate", type=float, default=0.10,
                   help="fraction of source rows that yield partial variants "
                        "(default 0.10, matches plan §4)")
    p.add_argument("--variants-per-row", type=int, default=2,
                   help="how many partial-strategy variants to emit per "
                        "rolled-in source row (default 2)")
    p.add_argument("--seed", type=int, default=1510)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    counters = generate(src_path=args.src, out_path=args.out,
                        partial_rate=args.partial_rate,
                        variants_per_row=args.variants_per_row,
                        seed=args.seed, limit=args.limit)
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
