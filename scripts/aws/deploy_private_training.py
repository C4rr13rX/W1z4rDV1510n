#!/usr/bin/env python3
"""Deploy the private Wizard Vision trainer only after a dollar cost guard."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "infra" / "aws" / "wizard-vision-private-training.yaml"
POLICY = ROOT / "infra" / "aws" / "fountain-server-deploy-policy.json"
STACK = "wizard-vision-private-training"
PROFILE = "FountainServer"
REGION = "us-east-1"


def estimate(volume_gib: int, hours: int, spot_cap: float) -> dict[str, float]:
    # us-east-1 public rates as of 2026-07-29. The deployer deliberately
    # assumes every compute hour reaches the Spot max-price ceiling.
    ebs = volume_gib * 0.08 * hours / 720.0
    compute = spot_cap * hours
    endpoints = 3 * 0.01 * hours
    staging = 4.0
    return {
        "compute_max": round(compute, 2),
        "gp3_pro_rated": round(ebs, 2),
        "private_endpoints": round(endpoints, 2),
        "staging_and_requests": staging,
        "guarded_total": round(compute + ebs + endpoints + staging, 2),
    }


def run(*parts: str) -> None:
    subprocess.run(parts, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-gib", type=int, default=1024)
    parser.add_argument("--hours", type=int, default=240)
    parser.add_argument("--spot-max", type=float, default=0.20)
    parser.add_argument("--instance-type", default="m6a.2xlarge")
    parser.add_argument("--acknowledge-max-usd", type=float, required=True)
    parser.add_argument("--bootstrap-permissions", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args()
    if not 512 <= args.volume_gib <= 2048:
        parser.error("--volume-gib must be between 512 and 2048")
    if not 24 <= args.hours <= 240:
        parser.error("--hours must be between 24 and 240")
    costs = estimate(args.volume_gib, args.hours, args.spot_max)
    print(json.dumps(costs, indent=2))
    if args.acknowledge_max_usd < costs["guarded_total"]:
        parser.error(
            f"acknowledged ceiling ${args.acknowledge_max_usd:.2f} is below "
            f"the conservative ${costs['guarded_total']:.2f} guard"
        )
    if args.estimate_only:
        return 0
    if args.bootstrap_permissions:
        run(
            "aws", "iam", "put-user-policy",
            "--user-name", "FountainServer",
            "--policy-name", "WizardVisionPrivateTrainingDeployment",
            "--policy-document", f"file://{POLICY}",
            "--profile", PROFILE,
        )
    run(
        "aws", "cloudformation", "deploy",
        "--stack-name", STACK,
        "--template-file", str(TEMPLATE),
        "--capabilities", "CAPABILITY_NAMED_IAM",
        "--profile", PROFILE,
        "--region", REGION,
        "--parameter-overrides",
        "ArtifactBucket=wizard-vision-private-321572159829-us-east-1",
        f"InstanceType={args.instance_type}",
        f"DataVolumeGiB={args.volume_gib}",
        f"MaximumSpotPrice={args.spot_max:.3f}",
        f"MaximumRuntimeHours={args.hours}",
        "--tags", "Project=WizardVision", "CostGuard=Temporary",
        "--no-fail-on-empty-changeset",
    )
    run(
        "aws", "cloudformation", "describe-stacks",
        "--stack-name", STACK, "--profile", PROFILE, "--region", REGION,
        "--query", "Stacks[0].Outputs", "--output", "table",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

