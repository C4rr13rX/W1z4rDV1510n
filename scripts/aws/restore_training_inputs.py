#!/usr/bin/env python3
"""Restore and verify source, corpora, and brain state on the private host."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path


BUCKET = "wizard-vision-private-321572159829-us-east-1"
PREFIX = "wizard-vision"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(32 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def s3_copy(source: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "aws", "s3", "cp", source, str(destination),
            "--region", "us-east-1", "--only-show-errors",
        ],
        check=True,
    )


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract only members that remain beneath *destination*."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError(f"archive path escapes destination: {member.name}")
    archive.extractall(destination)


def rebase_progress_corpora(runtime: Path, corpus_root: Path) -> list[str]:
    """Rewrite verified host-specific corpus paths after a cross-OS restore."""
    changed: list[str] = []
    for path in sorted(runtime.glob("*.progress.json")):
        progress = json.loads(path.read_text(encoding="utf-8"))
        recorded = str(progress.get("corpus") or "")
        if not recorded:
            continue
        basename = Path(recorded.replace("\\", "/")).name
        target = (corpus_root / basename).resolve()
        if not target.is_file() or recorded == str(target):
            continue
        progress["corpus"] = str(target)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        changed.append(path.name)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/srv/wizard"))
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()

    staging = args.root / "staging"
    source_manifest_path = staging / "source-manifest.json"
    source_root = f"s3://{BUCKET}/{PREFIX}/source/{args.source_commit}"
    s3_copy(f"{source_root}/manifest.json", source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_archive = staging / source_manifest["archive"]
    s3_copy(f"{source_root}/{source_archive.name}", source_archive)
    if digest(source_archive) != source_manifest["sha256"]:
        raise RuntimeError("source archive checksum mismatch")
    with tarfile.open(source_archive, "r:gz") as archive:
        safe_extract(archive, args.root / "project")

    corpus_manifest_path = staging / "corpora-manifest.json"
    corpus_root = f"s3://{BUCKET}/{PREFIX}/corpora"
    s3_copy(f"{corpus_root}/manifest.json", corpus_manifest_path)
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    corpus_archive = staging / corpus_manifest["archive"]["name"]
    s3_copy(f"{corpus_root}/{corpus_archive.name}", corpus_archive)
    if digest(corpus_archive) != corpus_manifest["archive"]["sha256"]:
        raise RuntimeError("corpus archive checksum mismatch")
    with tarfile.open(corpus_archive, "r:gz") as archive:
        safe_extract(archive, args.root / "corpora")

    brain_manifest_path = staging / "brain-manifest.json"
    brain_root = (
        f"s3://{BUCKET}/{PREFIX}/brain/programming-integrated-20260713"
    )
    s3_copy(f"{brain_root}/manifest.json", brain_manifest_path)
    brain_manifest = json.loads(brain_manifest_path.read_text(encoding="utf-8"))
    runtime = args.root / "runtime" / brain_manifest["runtime"]
    for item in brain_manifest["files"]:
        destination = runtime / item["path"]
        s3_copy(f"{brain_root}/{item['path']}", destination)
        if destination.stat().st_size != item["bytes"]:
            raise RuntimeError(f"brain size mismatch: {item['path']}")
        if digest(destination) != item["sha256"]:
            raise RuntimeError(f"brain checksum mismatch: {item['path']}")
    rebased = rebase_progress_corpora(runtime, args.root / "corpora")
    print(json.dumps({
        "restored": True,
        "source_commit": args.source_commit,
        "runtime": str(runtime),
        "corpus_files": len(corpus_manifest["files"]),
        "brain_files": len(brain_manifest["files"]),
        "rebased_progress_files": rebased,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
