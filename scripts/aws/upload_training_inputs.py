#!/usr/bin/env python3
"""Bundle, checksum, and optionally upload the seven production corpora."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUCKET = "wizard-vision-private-321572159829-us-east-1"
KEY_ROOT = "wizard-vision/corpora"
CORPORA = {
    "mathinstruct.jsonl": 245_323,
    "metamathqa.jsonl": 385_524,
    "csn_python_full.jsonl": 421_477,
    "csn_python_full_para5.jsonl": 2_028_816,
    "jupyter_scientific_full.jsonl": 690_175,
    "jupyter_scientific_para4.jsonl": 2_760_496,
    "jupyter_scientific_partial.jsonl": 206_948,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path(r"D:\w1z4rdv1510n-data\training"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "aws" / "wizard-vision-corpora.tar.gz",
    )
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    missing = [
        str(args.corpus_root / name)
        for name in CORPORA
        if not (args.corpus_root / name).is_file()
    ]
    if missing:
        raise SystemExit("missing production corpora: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = [args.corpus_root / name for name in CORPORA]
    manifest = {
        "schema": "wizard-vision-corpora/v1",
        "logical_rows": sum(CORPORA.values()),
        "files": [
            {
                "name": path.name,
                "logical_rows": CORPORA[path.name],
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    with tarfile.open(args.output, "w:gz", compresslevel=3) as archive:
        for path in files:
            archive.add(path, arcname=path.name)
        encoded = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(encoded)
            manifest_path = Path(handle.name)
        try:
            archive.add(manifest_path, arcname="manifest.json")
        finally:
            manifest_path.unlink(missing_ok=True)
    manifest["archive"] = {
        "name": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": digest(args.output),
    }
    print(json.dumps(manifest, indent=2))
    if args.upload:
        subprocess.run(
            [
                "aws", "s3", "cp", str(args.output),
                f"s3://{BUCKET}/{KEY_ROOT}/{args.output.name}",
                "--profile", args.profile,
                "--region", "us-east-1",
                "--only-show-errors",
                "--metadata", f"sha256={manifest['archive']['sha256']}",
            ],
            cwd=ROOT,
            check=True,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump(manifest, handle, indent=2)
            remote_manifest = Path(handle.name)
        try:
            subprocess.run(
                [
                    "aws", "s3", "cp", str(remote_manifest),
                    f"s3://{BUCKET}/{KEY_ROOT}/manifest.json",
                    "--profile", args.profile,
                    "--region", "us-east-1",
                    "--only-show-errors",
                ],
                cwd=ROOT,
                check=True,
            )
        finally:
            remote_manifest.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
