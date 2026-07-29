#!/usr/bin/env python3
"""Create and optionally upload a reproducible tracked-source bundle."""
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
PREFIX = "wizard-vision/source"


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    )
    return [
        ROOT / name.decode("utf-8")
        for name in output.split(b"\0")
        if name
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "aws" / "wizard-vision-source.tar.gz",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    files = tracked_files()
    with tarfile.open(args.output, "w:gz", compresslevel=6) as archive:
        for path in files:
            if path.is_file():
                archive.add(path, arcname=path.relative_to(ROOT))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest = {
        "commit": commit,
        "file_count": len(files),
        "archive": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": sha256(args.output),
    }
    print(json.dumps(manifest, indent=2))
    if args.upload:
        key = f"{PREFIX}/{commit}/{args.output.name}"
        subprocess.run(
            [
                "aws", "s3", "cp", str(args.output), f"s3://{BUCKET}/{key}",
                "--profile", args.profile,
                "--region", "us-east-1",
                "--only-show-errors",
            ],
            cwd=ROOT,
            check=True,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", encoding="utf-8", delete=False
        ) as handle:
            json.dump({**manifest, "s3_key": key}, handle, indent=2)
            temporary = Path(handle.name)
        try:
            subprocess.run(
                [
                    "aws", "s3", "cp", str(temporary),
                    f"s3://{BUCKET}/{PREFIX}/{commit}/manifest.json",
                    "--profile", args.profile,
                    "--region", "us-east-1",
                    "--only-show-errors",
                ],
                cwd=ROOT,
                check=True,
            )
        finally:
            temporary.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
