#!/usr/bin/env python3
"""Plan or deploy the private, cost-capped Wizard Vision training stack."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "infra" / "aws" / "wizard-vision-private-training.yaml"
POLICY = ROOT / "infra" / "aws" / "fountain-server-deploy-policy.json"
STACK = "wizard-vision-private-training"
PROFILE = "FountainServer"
REGION = "us-east-1"
ACCOUNT = "321572159829"
BUCKET = f"wizard-vision-private-{ACCOUNT}-{REGION}"
POLICY_NAME = "WizardVisionPrivateTrainingDeployment"
POLICY_ARN = f"arn:aws:iam::{ACCOUNT}:policy/{POLICY_NAME}"
DEPLOYMENT_GROUP = "WizardVisionDeployers"
AGREED_MAX_USD = 100.0


@dataclass(frozen=True)
class CostEnvelope:
    compute: float
    data_ebs: float
    root_ebs: float
    endpoints: float
    s3_month: float
    contingency: float

    @property
    def total(self) -> float:
        return sum(asdict(self).values())


def estimate_cost(
    *,
    hours: int,
    hourly_compute: float,
    volume_gib: int,
    final_s3_gib: int,
) -> CostEnvelope:
    """Conservative us-east-1 envelope; storage is prorated except retained S3."""
    month_hours = 24 * 30
    return CostEnvelope(
        compute=hours * hourly_compute,
        data_ebs=volume_gib * 0.08 * hours / month_hours,
        root_ebs=30 * 0.08 * hours / month_hours,
        endpoints=3 * 0.01 * hours,
        s3_month=final_s3_gib * 0.023,
        contingency=5.0,
    )


def run(
    *parts: str, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        parts,
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def aws(
    profile: str, *arguments: str, capture: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return run(
        "aws",
        *arguments,
        "--profile",
        profile,
        "--region",
        REGION,
        capture=capture,
        check=check,
    )


def bootstrap_permissions(profile: str) -> None:
    """Publish the repository policy and attach it without consuming a user slot."""
    present = aws(
        profile, "iam", "get-policy", "--policy-arn", POLICY_ARN,
        capture=True, check=False,
    )
    if present.returncode:
        aws(
            profile,
            "iam",
            "create-policy",
            "--policy-name",
            POLICY_NAME,
            "--description",
            "Scoped deployment actions for private Wizard Vision training",
            "--policy-document",
            f"file://{POLICY}",
        )
    else:
        versions = json.loads(
            aws(
                profile,
                "iam",
                "list-policy-versions",
                "--policy-arn",
                POLICY_ARN,
                "--output",
                "json",
                capture=True,
            ).stdout
        )["Versions"]
        removable = sorted(
            (item for item in versions if not item["IsDefaultVersion"]),
            key=lambda item: item["CreateDate"],
        )
        if len(versions) >= 5:
            aws(
                profile,
                "iam",
                "delete-policy-version",
                "--policy-arn",
                POLICY_ARN,
                "--version-id",
                removable[0]["VersionId"],
            )
        aws(
            profile,
            "iam",
            "create-policy-version",
            "--policy-arn",
            POLICY_ARN,
            "--policy-document",
            f"file://{POLICY}",
            "--set-as-default",
        )

    group = aws(
        profile, "iam", "get-group", "--group-name", DEPLOYMENT_GROUP,
        capture=True, check=False,
    )
    if group.returncode:
        aws(profile, "iam", "create-group", "--group-name", DEPLOYMENT_GROUP)
    aws(
        profile,
        "iam",
        "attach-group-policy",
        "--group-name",
        DEPLOYMENT_GROUP,
        "--policy-arn",
        POLICY_ARN,
    )
    membership = aws(
        profile,
        "iam",
        "list-groups-for-user",
        "--user-name",
        PROFILE,
        "--query",
        f"Groups[?GroupName=='{DEPLOYMENT_GROUP}'].GroupName",
        "--output",
        "text",
        capture=True,
    ).stdout.strip()
    if not membership:
        aws(
            profile,
            "iam",
            "add-user-to-group",
            "--group-name",
            DEPLOYMENT_GROUP,
            "--user-name",
            PROFILE,
        )


def assert_private_bucket(profile: str) -> None:
    public = json.loads(
        aws(
            profile,
            "s3api",
            "get-public-access-block",
            "--bucket",
            BUCKET,
            "--output",
            "json",
            capture=True,
        ).stdout
    )["PublicAccessBlockConfiguration"]
    if not all(public.values()):
        raise RuntimeError(f"{BUCKET} does not block every form of public access")
    encryption = json.loads(
        aws(
            profile,
            "s3api",
            "get-bucket-encryption",
            "--bucket",
            BUCKET,
            "--output",
            "json",
            capture=True,
        ).stdout
    )
    rules = encryption.get("ServerSideEncryptionConfiguration", {}).get("Rules", [])
    if not rules:
        raise RuntimeError(f"{BUCKET} has no default server-side encryption")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--hours", type=int, default=192)
    parser.add_argument("--volume-gib", type=int, default=1024)
    parser.add_argument("--final-s3-gib", type=int, default=400)
    parser.add_argument("--hourly-compute-cap", type=float, default=0.1728)
    parser.add_argument("--instance-type", default="m6a.xlarge")
    parser.add_argument("--cost-cap-usd", type=float)
    parser.add_argument("--bootstrap-permissions", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 24 <= args.hours <= 192:
        raise SystemExit("--hours must be between 24 and 192")
    if not 512 <= args.volume_gib <= 2048:
        raise SystemExit("--volume-gib must be between 512 and 2048")
    estimate = estimate_cost(
        hours=args.hours,
        hourly_compute=args.hourly_compute_cap,
        volume_gib=args.volume_gib,
        final_s3_gib=args.final_s3_gib,
    )
    report = {
        "stack": STACK,
        "region": REGION,
        "private_bucket": BUCKET,
        "instance_type": args.instance_type,
        "maximum_running_hours": args.hours,
        "estimate_usd": {
            **{key: round(value, 2) for key, value in asdict(estimate).items()},
            "guarded_total": round(estimate.total, 2),
        },
        "deploy_requested": args.deploy,
    }
    print(json.dumps(report, indent=2))

    if args.bootstrap_permissions:
        bootstrap_permissions(args.profile)
    if not args.deploy:
        return 0
    if args.cost_cap_usd is None:
        raise SystemExit("--deploy requires --cost-cap-usd")
    if args.cost_cap_usd > AGREED_MAX_USD:
        raise SystemExit(
            f"refusing a cost cap above the agreed ${AGREED_MAX_USD:.0f} ceiling"
        )
    if estimate.total > args.cost_cap_usd:
        raise SystemExit(
            f"estimated ${estimate.total:.2f} exceeds "
            f"${args.cost_cap_usd:.2f} cost cap"
        )
    assert_private_bucket(args.profile)
    aws(
        args.profile,
        "cloudformation",
        "deploy",
        "--stack-name",
        STACK,
        "--template-file",
        str(TEMPLATE),
        "--capabilities",
        "CAPABILITY_NAMED_IAM",
        "--parameter-overrides",
        f"ArtifactBucket={BUCKET}",
        f"InstanceType={args.instance_type}",
        f"DataVolumeGiB={args.volume_gib}",
        f"MaximumRuntimeHours={args.hours}",
        "--tags",
        "Project=WizardVision",
        "CostGuard=Temporary",
        "--no-fail-on-empty-changeset",
    )
    aws(
        args.profile,
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        STACK,
        "--query",
        "Stacks[0].Outputs",
        "--output",
        "table",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
