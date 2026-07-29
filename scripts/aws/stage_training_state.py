#!/usr/bin/env python3
"""Stage a settled Wizard Vision runtime in the private migration bucket."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE = "FountainServer"
BUCKET = "wizard-vision-private-321572159829-us-east-1"


def post(endpoint: str, route: str, timeout: int) -> dict:
    request = urllib.request.Request(
        endpoint.rstrip("/") + route, data=b"{}", method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18095")
    parser.add_argument("--settle-live-node", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    brain = runtime / "brain"
    authoritative = brain / "brain.wbrain"
    progress = runtime / "csn-python-para5.progress.json"
    if not authoritative.is_file() or not progress.is_file():
        parser.error("runtime lacks brain/brain.wbrain or current progress ledger")
    settlement = {}
    if args.settle_live_node:
        settlement["sleep"] = post(args.endpoint, "/sleep", 1800)
        settlement["checkpoint"] = post(args.endpoint, "/checkpoint", 3600)
        settlement["flush"] = post(args.endpoint, "/flush", 300)
    ledger = json.loads(progress.read_text(encoding="utf-8"))
    manifest = {
        "schema": "wizard-vision-aws-migration/v1",
        "created_unix": time.time(),
        "runtime": str(runtime),
        "durable_next_row": ledger.get("durable_next_row"),
        "ram_next_row": ledger.get("ram_next_row"),
        "authoritative": {
            "relative_path": "brain/brain.wbrain",
            "bytes": authoritative.stat().st_size,
            "sha256": digest(authoritative),
        },
        "settlement": settlement,
    }
    manifest_path = runtime / "aws-migration-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    if args.upload:
        destination = f"s3://{BUCKET}/wizard-vision/migrations/programming-integrated-20260713/"
        subprocess.run(
            [
                "aws", "s3", "sync", str(runtime), destination,
                "--profile", PROFILE,
                "--sse", "AES256",
                "--exclude", "*.log",
                "--exclude", "*.pid",
                "--exclude", "*.tmp*",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "aws", "s3api", "head-object", "--bucket", BUCKET,
                "--key",
                "wizard-vision/migrations/programming-integrated-20260713/"
                "brain/brain.wbrain",
                "--profile", PROFILE,
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

