"""ingest/jupyter_code_text_pairs.py — Jupyter notebook md↔code pairs.

bigcode/jupyter-code-text-pairs is 9.3M (markdown, code, output)
triples extracted from real Jupyter notebooks on permissively-licensed
GitHub repos.  Each row carries its source repo's `license` so we can
filter per-row against PERMISSIVE_LICENSES.

This is the primary corpus for "extremely scientific work" — Jupyter
notebooks are overwhelmingly NumPy / SciPy / pandas / sklearn /
PyTorch / TensorFlow / astropy / biopython / sympy code, paired with
researchers' own English explanations of what they're computing and
why.

Schema (one row per markdown↔code pair):

    {
      "markdown":   <str>   — the prose cell that precedes a code cell
      "code":       <str>   — the code cell that followed it
      "output":     <str>   — cell output (we don't train on this; it
                              often contains user data, randomness,
                              huge numpy reprs)
      "license":    <str>   — SPDX-ish license of the source repo
      "path":       <str>   — file path within the repo
      "repo_name":  <str>   — github "owner/repo"
    }

Quality gates:
  - row license in PERMISSIVE_LICENSES (enforced by RowWriter)
  - markdown 20–2000 chars after HTML/img stripping
  - code 30–4000 chars
  - code passes sandbox syntactic check (Python) — most cells are
    Python; non-Python cells fall through unchecked
  - --scientific flag (default ON): code must contain at least one
    scientific-library import (numpy, scipy, sklearn, pandas, torch,
    tensorflow, jax, astropy, biopython, sympy, statsmodels, networkx,
    matplotlib, seaborn, plotly, pymc, stan, gym, transformers, …)

The scientific filter is what makes this corpus "extremely scientific"
rather than "any Python in a notebook".  Pass --no-scientific to ingest
all permissive rows regardless (still high quality, just broader).

CLI:
    python -m tools.training_standard.ingest.jupyter_code_text_pairs \\
        --src D:/.../sources/jupyter-code-text-pairs \\
        --out D:/.../training/jcp_scientific.jsonl \\
        --script-id domain_ai_ml_001 \\
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

# Scientific library imports — code that mentions any of these in an
# import statement is biased toward "extremely scientific work".
_SCIENTIFIC_IMPORTS = (
    # Numerics
    "numpy", "scipy", "sympy", "mpmath", "uncertainties", "numba",
    "cython", "pythran",
    # Data
    "pandas", "polars", "pyarrow", "dask", "xarray", "vaex",
    # ML / DL
    "sklearn", "scikit", "torch", "tensorflow", "keras", "jax",
    "flax", "optax", "transformers", "diffusers", "datasets", "evaluate",
    "huggingface", "tokenizers", "accelerate", "deepspeed",
    "lightning", "pytorch_lightning",
    "xgboost", "lightgbm", "catboost",
    # Stats / probabilistic
    "statsmodels", "pymc", "pyro", "numpyro", "stan", "bambi", "arviz",
    # Science domains
    "astropy", "biopython", "Bio.",  "skimage", "openslide",
    "cv2", "opencv", "pillow", "PIL.",
    "networkx", "graph_tool",
    "nltk", "spacy", "gensim",
    "rdkit", "ase", "pymatgen", "openmm",
    # RL
    "gym", "gymnasium", "stable_baselines",
    # Plotting / viz (scientific bias)
    "matplotlib", "seaborn", "plotly", "bokeh", "altair",
    # Numerical methods
    "scipy.integrate", "scipy.optimize", "scipy.fft", "scipy.signal",
    "scipy.linalg", "scipy.stats", "scipy.sparse",
)
_SCIENTIFIC_IMPORT_RE = re.compile(
    r"(?:^|\n)\s*(?:from\s+(" + "|".join(re.escape(m) for m in _SCIENTIFIC_IMPORTS)
    + r")|import\s+(" + "|".join(re.escape(m) for m in _SCIENTIFIC_IMPORTS) + r"))",
    re.MULTILINE,
)

# Markdown noise we strip out.
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_IMG_RE      = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE     = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # keep link text
_MULTI_WS    = re.compile(r"[ \t]+")


def _normalize_markdown(md: str) -> str:
    """Strip HTML tags, image refs, replace links with their text,
    collapse runs of whitespace.  Empty result → row is rejected."""
    if not md:
        return ""
    s = _IMG_RE.sub("", md)
    s = _LINK_RE.sub(r"\1", s)
    s = _HTML_TAG_RE.sub("", s)
    # Normalise but keep paragraph breaks.
    lines = [_MULTI_WS.sub(" ", ln).strip() for ln in s.splitlines()]
    s = "\n".join(ln for ln in lines if ln)
    return s.strip()


def _normalize_license(lic: str) -> str:
    """Map common SPDX-ish spellings to our permissive set."""
    if not lic:
        return ""
    s = lic.strip().lower().replace("_", "-")
    if s in PERMISSIVE_LICENSES:
        return s
    aliases = {
        "apache-2":           "apache-2.0",
        "apache 2.0":         "apache-2.0",
        "bsd-2":              "bsd-2-clause",
        "bsd-3":              "bsd-3-clause",
        "bsd":                "bsd-3-clause",
        "mit-0":              "mit",
        "mit license":        "mit",
        "public-domain":      "cc0-1.0",
        "public domain":      "cc0-1.0",
    }
    return aliases.get(s, s)


def _looks_scientific(code: str) -> bool:
    return bool(_SCIENTIFIC_IMPORT_RE.search(code or ""))


def _iter_parquet_shards(src_dir: Path) -> Iterator[Path]:
    yield from sorted(src_dir.rglob("*.parquet"))


def _iter_rows(shard: Path) -> Iterator[dict]:
    """Stream rows from one parquet shard."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(shard)
    for batch in pf.iter_batches(batch_size=2048):
        cols = {c: batch.column(c) for c in batch.schema.names}
        n = len(batch)
        for i in range(n):
            yield {
                "markdown":   cols["markdown"][i].as_py()  if "markdown"  in cols else "",
                "code":       cols["code"][i].as_py()      if "code"      in cols else "",
                "license":    cols["license"][i].as_py()   if "license"   in cols else "",
                "path":       cols["path"][i].as_py()      if "path"      in cols else "",
                "repo_name":  cols["repo_name"][i].as_py() if "repo_name" in cols else "",
            }


def ingest(
    *,
    src_dir: Path,
    out_path: Path,
    script_id: str,
    limit: int | None,
    skip_sandbox: bool,
    scientific_only: bool,
    intent: str,
) -> dict:
    sb = None if skip_sandbox else get_sandbox()

    counters = {
        "seen":                 0,
        "rejected_license":     0,
        "rejected_md_size":     0,
        "rejected_code_size":   0,
        "rejected_scientific":  0,
        "rejected_sandbox":     0,
        "rejected_row_writer":  0,
        "dedup_skipped":        0,
        "written":              0,
    }

    with RowWriter(out_path, script_id=script_id,
                   source="jupyter-code-text-pairs") as writer:
        for shard in _iter_parquet_shards(src_dir):
            for rec in _iter_rows(shard):
                counters["seen"] += 1
                if limit is not None and counters["written"] >= limit:
                    break

                lic = _normalize_license(rec.get("license") or "")
                if lic not in PERMISSIVE_LICENSES:
                    counters["rejected_license"] += 1
                    continue

                md = _normalize_markdown(rec.get("markdown") or "")
                if not (20 <= len(md) <= 2000):
                    counters["rejected_md_size"] += 1
                    continue

                code = (rec.get("code") or "").strip()
                if not (30 <= len(code) <= 4000):
                    counters["rejected_code_size"] += 1
                    continue

                if scientific_only and not _looks_scientific(code):
                    counters["rejected_scientific"] += 1
                    continue

                if sb is not None:
                    r = sb.check("python", code, timeout_s=10.0)
                    if not r.ok:
                        counters["rejected_sandbox"] += 1
                        continue

                row = Row(
                    prompt=md,
                    response=code,
                    ctx=render_ctx(
                        lang="python",
                        intent=intent,
                        source="jcp",
                        repo=rec.get("repo_name") or "?",
                    ),
                    license=lic,
                    source=f"jupyter-code-text-pairs:{rec.get('repo_name','?')}:{rec.get('path','?')}",
                    source_hash=hash_source(f"{md}|||{code}"),
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
                   help="dir containing the jupyter-code-text-pairs parquets")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--script-id", required=True)
    p.add_argument("--intent", default="implement")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-sandbox", action="store_true")
    p.add_argument("--no-scientific", dest="scientific_only", action="store_false",
                   help="ingest every permissive row (not just scientific imports)")
    p.set_defaults(scientific_only=True)
    args = p.parse_args(argv)

    counters = ingest(
        src_dir=args.src, out_path=args.out, script_id=args.script_id,
        limit=args.limit, skip_sandbox=args.skip_sandbox,
        scientific_only=args.scientific_only, intent=args.intent,
    )
    print(json.dumps(counters, indent=2))
    return 0 if counters["written"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
