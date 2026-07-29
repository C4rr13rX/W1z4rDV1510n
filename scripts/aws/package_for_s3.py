#!/usr/bin/env python3
"""Create and optionally upload a reproducible tracked-source bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUCKET = "wizard-vision-private-321572159829-us-east-1"
PREFIX = "wizard-vision/source"


def tracked_files(revision: str = "HEAD", root: Path = ROOT) -> list[str]:
    output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "-z", revision], cwd=root
    )
    return [
        name.decode("utf-8")
        for name in output.split(b"\0")
        if name
    ]


def write_source_archive(output: Path, revision: str, root: Path = ROOT) -> None:
    """Gzip the immutable Git tree with a reproducible wrapper header."""
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["git", "archive", "--format=tar", revision],
        cwd=root,
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with output.open("wb") as raw, gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=6,
            fileobj=raw,
            mtime=0,
        ) as compressed:
            shutil.copyfileobj(process.stdout, compressed, 8 * 1024 * 1024)
    finally:
        process.stdout.close()
    if process.wait() != 0:
        output.unlink(missing_ok=True)
        raise subprocess.CalledProcessError(process.returncode, process.args)


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

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    files = tracked_files(commit)
    write_source_archive(args.output, commit)
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
