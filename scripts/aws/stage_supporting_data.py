#!/usr/bin/env python3
"""Checksum and stage the small generated benchmark/training support corpora."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUCKET = "wizard-vision-private-321572159829-us-east-1"
KEY_ROOT = "wizard-vision/supporting-data"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    paths = sorted((ROOT / "data" / "training").glob("*.jsonl"))
    if not paths:
        raise SystemExit("no supporting data/training JSONL files exist")
    manifest = {
        "schema": "wizard-vision-supporting-data/v1",
        "files": [
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in paths
        ],
    }
    print(json.dumps(manifest, indent=2))
    if not args.upload:
        return 0
    for item, path in zip(manifest["files"], paths):
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(path),
                f"s3://{BUCKET}/{KEY_ROOT}/{item['path']}",
                "--profile",
                args.profile,
                "--region",
                "us-east-1",
                "--only-show-errors",
                "--metadata",
                f"sha256={item['sha256']}",
            ],
            cwd=ROOT,
            check=True,
        )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2)
        manifest_path = Path(handle.name)
    try:
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(manifest_path),
                f"s3://{BUCKET}/{KEY_ROOT}/manifest.json",
                "--profile",
                args.profile,
                "--region",
                "us-east-1",
                "--only-show-errors",
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
