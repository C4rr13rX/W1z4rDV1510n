#!/usr/bin/env python3
"""Compile a directory of documents into a corpus and register it for training.

Onboarding a corpus took seven manual steps, and four of them silently broke
during 2026-08-20/21:

  1. run an ingester to produce .jsonl
  2. copy it to the corpus root
  3. register it in manifest.json with exact bytes/rows/sha256
  4. edit curriculum_phases() with an EXACT hand-counted row total
  5. write a registry .toml
  6. update the authoritative row-total invariant test
  7. redeploy the supervisor and restart

Step 4 is the dangerous one: the supervisor tracks durable progress against
that number, so a stale count mistrains without an error. Step 3 looks like it
does the work but trains nothing on its own -- `curriculum_phases()` never
reads the manifest, which cost a full cycle to discover.

This does the whole thing in one command, and derives every number rather than
asking for it. The phase is declared in the `curriculum_extensions.json`
sidecar, whose row counts the supervisor COUNTS from the file, so no code
change or redeploy is needed to add a corpus.

    python scripts/onboard_corpus.py \\
        --src D:/docs/my_manuals \\
        --corpus-root /srv/wizard/corpora \\
        --name my-manuals \\
        --license cc0-1.0

Use --dry-run to see exactly what it would write first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.training_standard.ingest import document_directory as ingest

EXTENSIONS_FILENAME = "curriculum_extensions.json"
MANIFEST_FILENAME = "manifest.json"


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "corpus"


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def register_manifest(corpus_root: Path, corpus: Path) -> dict:
    """Record provenance. Informational: the trainer reads the sidecar."""
    path = corpus_root / MANIFEST_FILENAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        manifest = {"schema": "wizard-vision-corpora/v1", "files": []}
    raw = corpus.read_bytes()
    entry = {
        "bytes": len(raw),
        "logical_rows": count_rows(corpus),
        "name": corpus.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    files = [f for f in manifest.get("files", []) if f.get("name") != corpus.name]
    files.append(entry)
    manifest["files"] = files
    manifest["logical_rows"] = sum(f.get("logical_rows", 0) for f in files)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return entry


def register_phase(corpus_root: Path, name: str, script_id: str,
                   corpus_name: str, repeats: int) -> dict:
    """Declare the phase as data, so no code edit is needed."""
    path = corpus_root / EXTENSIONS_FILENAME
    try:
        declared = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        declared = {"phases": []}
    if not isinstance(declared.get("phases"), list):
        declared["phases"] = []
    entry = {"name": name, "script_id": script_id, "corpus": corpus_name}
    if repeats > 1:
        entry["repeats"] = repeats
    declared["phases"] = [p for p in declared["phases"]
                          if not (isinstance(p, dict) and p.get("name") == name)]
    declared["phases"].append(entry)
    path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return entry


def suggest_repeats(rows: int) -> int:
    """Small corpora are reinforced rather than duplicated on disk.

    Mirrors the existing plan, where a 1,953-row corpus carries repeats=4 and
    the multi-million-row ones carry 1.
    """
    if rows < 1_000:
        return 4
    if rows < 20_000:
        return 2
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True,
                        help="directory of documents to compile")
    parser.add_argument("--corpus-root", type=Path, required=True,
                        help="where corpora live, e.g. /srv/wizard/corpora")
    parser.add_argument("--name", required=True,
                        help="phase name, e.g. my-manuals")
    parser.add_argument("--license", required=True)
    parser.add_argument("--script-id", default=None,
                        help="defaults to domain_<name>_001")
    parser.add_argument("--intent", default="implement")
    parser.add_argument("--repeats", type=int, default=0,
                        help="0 chooses from the row count")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.src.is_dir():
        print(f"not a directory: {args.src}", file=sys.stderr)
        return 2

    name = slug(args.name)
    script_id = args.script_id or f"domain_{name.replace('-', '_')}_001"
    corpus_name = f"{name.replace('-', '_')}.jsonl"
    corpus = args.corpus_root / corpus_name

    if args.dry_run:
        print("would write:")
        print(f"  corpus    {corpus}")
        print(f"  phase     {name} (script_id={script_id})")
        print(f"  declared  {args.corpus_root / EXTENSIONS_FILENAME}")
        print(f"  manifest  {args.corpus_root / MANIFEST_FILENAME}")
        return 0

    args.corpus_root.mkdir(parents=True, exist_ok=True)
    stats = ingest.build(args.src, corpus, args.license, name, script_id,
                         args.intent)
    print(f"compiled {args.src} -> {corpus}")
    print(stats.report())
    if stats.rows == 0:
        print("\nnothing pairable here; not registering an empty corpus",
              file=sys.stderr)
        corpus.unlink(missing_ok=True)
        return 1

    repeats = args.repeats or suggest_repeats(stats.rows)
    entry = register_manifest(args.corpus_root, corpus)
    phase = register_phase(args.corpus_root, name, script_id, corpus_name,
                           repeats)
    print(f"\nregistered phase {phase['name']} "
          f"({entry['logical_rows']} rows, repeats={repeats})")
    print("the supervisor picks this up on its next pass; no code change or "
          "redeploy is needed.")
    print("\nremaining manual step: a registry .toml under "
          "tools/training_standard/registry/ if you want admission "
          "benchmarks for it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
