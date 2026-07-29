#!/usr/bin/env python3
"""Create and optionally upload a locked offline Rust dependency bundle."""
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
REGION = "us-east-1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def upload(profile: str, source: Path, key: str, checksum: str) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(source),
            f"s3://{BUCKET}/{key}",
            "--profile",
            profile,
            "--region",
            REGION,
            "--only-show-errors",
            "--metadata",
            f"sha256={checksum}",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runtime" / "aws" / "rust-offline.tar.gz",
    )
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="wizard-rust-vendor-", dir=args.output.parent
    ) as directory:
        staging = Path(directory)
        vendor = staging / "vendor"
        configuration = subprocess.check_output(
            [
                "cargo",
                "vendor",
                "--manifest-path",
                str(ROOT / "Cargo.toml"),
                "--locked",
                "--versioned-dirs",
                "vendor",
            ],
            cwd=staging,
            text=True,
        )
        cargo_directory = staging / ".cargo"
        cargo_directory.mkdir()
        (cargo_directory / "config.toml").write_text(
            configuration, encoding="utf-8"
        )
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        with tarfile.open(temporary, "w:gz", compresslevel=3) as archive:
            archive.add(vendor, arcname="vendor")
            archive.add(cargo_directory, arcname=".cargo")
        temporary.replace(args.output)

    checksum = digest(args.output)
    manifest = {
        "schema": "wizard-vision-rust-offline/v1",
        "source_commit": commit,
        "cargo_lock_sha256": digest(ROOT / "Cargo.lock"),
        "archive": args.output.name,
        "bytes": args.output.stat().st_size,
        "sha256": checksum,
    }
    print(json.dumps(manifest, indent=2))
    if not args.upload:
        return 0

    key_root = f"wizard-vision/build/rust-offline/{commit}"
    upload(args.profile, args.output, f"{key_root}/{args.output.name}", checksum)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2)
        manifest_path = Path(handle.name)
    try:
        upload(
            args.profile,
            manifest_path,
            f"{key_root}/manifest.json",
            digest(manifest_path),
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
