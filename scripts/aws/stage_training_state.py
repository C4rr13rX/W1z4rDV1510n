#!/usr/bin/env python3
"""Settle or stage a stopped Wizard Vision runtime with verified checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "FountainServer"
BUCKET = "wizard-vision-private-321572159829-us-east-1"
KEY_ROOT = "wizard-vision/brain/programming-integrated-20260713"
DEFAULT_RUNTIME = ROOT / "runtime" / "brains" / "programming-integrated-20260713"


def post(endpoint: str, route: str, timeout: int) -> dict:
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def endpoint_is_live(endpoint: str) -> bool:
    try:
        with urllib.request.urlopen(
            endpoint.rstrip("/") + "/health", timeout=2
        ) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(32 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state_files(runtime: Path) -> list[Path]:
    required = [
        runtime / "brain" / "brain.wbrain",
        runtime / "brain" / "brain.wal",
        runtime / "brain" / "brain.identity.toml",
        runtime / "brain" / "brain.last-good.wbrain",
        runtime / "brain" / "brain.last-good.json",
        runtime / "csn-python-para5.progress.json",
        runtime / "curriculum-supervisor.status.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("brain migration is missing: " + ", ".join(missing))
    controls = [
        path
        for path in runtime.iterdir()
        if path.is_file()
        and (
            path.name.endswith(".progress.json")
            or path.name
            in {
                "brain.deployment.toml",
                "curriculum-deferred-intervals.jsonl",
                "curriculum-health.jsonl",
            }
        )
    ]
    return sorted(set(required + controls))


def upload_file(profile: str, source: Path, key: str, checksum: str) -> None:
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
            "us-east-1",
            "--only-show-errors",
            "--metadata",
            f"sha256={checksum}",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18095")
    parser.add_argument("--settle-live-node", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()

    live = endpoint_is_live(args.endpoint)
    if args.settle_live_node:
        if not live:
            raise SystemExit("brain server is not live; no settlement was performed")
        settlement = {
            "sleep": post(args.endpoint, "/sleep", 1800),
            "checkpoint": post(args.endpoint, "/checkpoint", 3600),
            "flush": post(args.endpoint, "/flush", 300),
        }
        print(json.dumps(settlement, indent=2))
        if args.upload:
            raise SystemExit(
                "settlement completed; stop the brain server, then rerun with --upload"
            )
        return 0
    if live:
        raise SystemExit(
            "brain server is live; use --settle-live-node, stop it, then stage state"
        )

    files = state_files(runtime)
    progress = json.loads(
        (runtime / "csn-python-para5.progress.json").read_text(encoding="utf-8")
    )
    manifest = {
        "schema": "wizard-vision-brain-transfer/v1",
        "created_unix": time.time(),
        "runtime": runtime.name,
        "durable_next_row": progress.get("durable_next_row"),
        "ram_next_row": progress.get("ram_next_row"),
        "files": [
            {
                "path": str(path.relative_to(runtime)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
            for path in files
        ],
    }
    if manifest["durable_next_row"] != manifest["ram_next_row"]:
        raise RuntimeError("RAM progress is ahead of durable progress")
    print(json.dumps(manifest, indent=2))
    if not args.upload:
        return 0

    for item, path in zip(manifest["files"], files):
        upload_file(
            args.profile,
            path,
            f"{KEY_ROOT}/{item['path']}",
            item["sha256"],
        )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2)
        manifest_path = Path(handle.name)
    try:
        upload_file(
            args.profile,
            manifest_path,
            f"{KEY_ROOT}/manifest.json",
            digest(manifest_path),
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
