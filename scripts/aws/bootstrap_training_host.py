#!/usr/bin/env python3
"""Bootstrap and restore the private training host through AWS Systems Manager."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "FountainServer"
REGION = "us-east-1"
STACK = "wizard-vision-private-training"
BUCKET = "wizard-vision-private-321572159829-us-east-1"
AWSCLI_KEY = "wizard-vision/bootstrap/awscli-exe-linux-x86_64.zip"


def aws(
    profile: str,
    *arguments: str,
    capture: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["aws", *arguments, "--profile", profile, "--region", REGION],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=capture,
    )


def stack_instance_id(profile: str) -> str:
    result = aws(
        profile,
        "cloudformation",
        "describe-stacks",
        "--stack-name",
        STACK,
        "--query",
        "Stacks[0].Outputs[?OutputKey=='InstanceId'].OutputValue | [0]",
        "--output",
        "text",
    ).stdout.strip()
    if not result or result == "None":
        raise RuntimeError(f"{STACK} has no InstanceId output")
    return result


def wait_for_ssm(profile: str, instance_id: str, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        query = aws(
            profile,
            "ssm",
            "describe-instance-information",
            "--filters",
            f"Key=InstanceIds,Values={instance_id}",
            "--query",
            "InstanceInformationList[0].PingStatus",
            "--output",
            "text",
        ).stdout.strip()
        if query == "Online":
            return
        time.sleep(10)
    raise TimeoutError(f"{instance_id} did not become SSM Online")


def instance_deadline_epoch(
    profile: str, instance_id: str, maximum_hours: int
) -> int:
    launched = aws(
        profile,
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
        "--query",
        "Reservations[0].Instances[0].LaunchTime",
        "--output",
        "text",
    ).stdout.strip()
    launch_time = datetime.fromisoformat(launched.replace("Z", "+00:00"))
    return int((launch_time + timedelta(hours=maximum_hours)).timestamp())


def cost_guard_commands(deadline_epoch: int) -> list[str]:
    stop_script = """cat >/usr/local/sbin/wizard-cost-stop.sh <<'EOF'
#!/bin/bash
set -euo pipefail
DEADLINE_FILE=/srv/wizard/control/cost-deadline.epoch
test -r "$DEADLINE_FILE" || exit 0
if test "$(date +%s)" -ge "$(cat "$DEADLINE_FILE")"; then
  /sbin/shutdown -h now
fi
EOF"""
    service = """cat >/etc/systemd/system/wizard-cost-stop.service <<'EOF'
[Unit]
Description=Stop Wizard Vision training host after its persisted cost deadline
RequiresMountsFor=/srv/wizard

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/wizard-cost-stop.sh
EOF"""
    timer = """cat >/etc/systemd/system/wizard-cost-stop.timer <<'EOF'
[Unit]
Description=Check the Wizard Vision persisted cost deadline

[Timer]
OnBootSec=1min
OnUnitActiveSec=1min
Persistent=true
Unit=wizard-cost-stop.service

[Install]
WantedBy=timers.target
EOF"""
    return [
        "set -euxo pipefail",
        "mkdir -p /srv/wizard/control",
        f"echo {deadline_epoch} >/srv/wizard/control/cost-deadline.epoch",
        stop_script,
        "chmod 750 /usr/local/sbin/wizard-cost-stop.sh",
        service,
        timer,
        "systemctl daemon-reload",
        "systemctl enable --now wizard-cost-stop.timer",
        "systemctl start wizard-cost-stop.service",
    ]


def source_manifest(profile: str, commit: str) -> dict:
    return json.loads(
        aws(
            profile,
            "s3",
            "cp",
            f"s3://{BUCKET}/wizard-vision/source/{commit}/manifest.json",
            "-",
            "--only-show-errors",
        ).stdout
    )


def rust_manifest(profile: str, commit: str) -> dict:
    return json.loads(
        aws(
            profile,
            "s3",
            "cp",
            (
                f"s3://{BUCKET}/wizard-vision/build/rust-offline/"
                f"{commit}/manifest.json"
            ),
            "-",
            "--only-show-errors",
        ).stdout
    )


def bootstrap_commands(
    presigned_url: str,
    commit: str,
    manifest: dict,
    rust: dict,
) -> list[str]:
    archive = manifest["archive"]
    checksum = manifest["sha256"]
    source_root = f"s3://{BUCKET}/wizard-vision/source/{commit}"
    rust_archive = rust["archive"]
    rust_checksum = rust["sha256"]
    rust_root = f"s3://{BUCKET}/wizard-vision/build/rust-offline/{commit}"
    return [
        "set -euxo pipefail",
        "test -d /srv/wizard/control",
        "rm -rf /tmp/aws /tmp/awscliv2.zip",
        (
            "python3 -c \"import urllib.request; "
            f"urllib.request.urlretrieve('{presigned_url}', '/tmp/awscliv2.zip')\""
        ),
        "python3 -m zipfile -e /tmp/awscliv2.zip /tmp/aws",
        "chmod -R a+rX /tmp/aws/aws",
        "chmod a+x /tmp/aws/aws/install /tmp/aws/aws/dist/aws",
        "/tmp/aws/aws/install --update",
        (
            "dnf install -y rust cargo gcc gcc-c++ openssl-devel "
            "pkgconf-pkg-config cmake python3-psutil"
        ),
        "mkdir -p /srv/wizard/project /srv/wizard/staging",
        (
            f"/usr/local/bin/aws s3 cp {source_root}/{archive} "
            f"/srv/wizard/staging/{archive} --region {REGION} --only-show-errors"
        ),
        (
            f"test \"$(sha256sum /srv/wizard/staging/{archive} | "
            f"cut -d' ' -f1)\" = \"{checksum}\""
        ),
        (
            f"tar -xzf /srv/wizard/staging/{archive} "
            "-C /srv/wizard/project"
        ),
        (
            f"/usr/local/bin/aws s3 cp {rust_root}/{rust_archive} "
            f"/srv/wizard/staging/{rust_archive} "
            f"--region {REGION} --only-show-errors"
        ),
        (
            f"test \"$(sha256sum /srv/wizard/staging/{rust_archive} | "
            f"cut -d' ' -f1)\" = \"{rust_checksum}\""
        ),
        (
            f"tar -xzf /srv/wizard/staging/{rust_archive} "
            "-C /srv/wizard/project"
        ),
        (
            "python3 /srv/wizard/project/scripts/aws/restore_training_inputs.py "
            f"--root /srv/wizard --source-commit {commit}"
        ),
        (
            "printf '%s\\n' "
            f"'{{\"state\":\"inputs_restored\",\"source_commit\":\"{commit}\"}}' "
            ">/srv/wizard/control/state.json"
        ),
        "chown -R ec2-user:ec2-user /srv/wizard",
    ]


def send_and_wait(
    profile: str,
    instance_id: str,
    commands: list[str],
    timeout: int,
    *,
    comment: str,
) -> dict:
    parameters = json.dumps({"commands": commands, "executionTimeout": [str(timeout)]})
    command_id = aws(
        profile,
        "ssm",
        "send-command",
        "--instance-ids",
        instance_id,
        "--document-name",
        "AWS-RunShellScript",
        "--comment",
        comment,
        "--parameters",
        parameters,
        "--timeout-seconds",
        str(min(timeout, 172800)),
        "--query",
        "Command.CommandId",
        "--output",
        "text",
    ).stdout.strip()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = aws(
            profile,
            "ssm",
            "get-command-invocation",
            "--command-id",
            command_id,
            "--instance-id",
            instance_id,
            "--output",
            "json",
            check=False,
        )
        if result.returncode:
            if "InvocationDoesNotExist" in result.stderr:
                time.sleep(2)
                continue
            raise RuntimeError(
                f"cannot inspect SSM command {command_id}: {result.stderr.strip()}"
            )
        invocation = json.loads(
            result.stdout
        )
        status = invocation["Status"]
        if status == "Success":
            return invocation
        if status in {"Cancelled", "Failed", "TimedOut", "Undeliverable", "Terminated"}:
            raise RuntimeError(
                f"bootstrap {command_id} ended {status}: "
                f"{invocation.get('StandardErrorContent', '')}"
            )
        time.sleep(10)
    raise TimeoutError(f"bootstrap {command_id} exceeded {timeout} seconds")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument(
        "--source-commit",
        default=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    parser.add_argument("--timeout", type=int, default=14_400)
    parser.add_argument("--maximum-hours", type=int, default=192)
    parser.add_argument("--cost-guard-only", action="store_true")
    args = parser.parse_args()

    instance_id = stack_instance_id(args.profile)
    wait_for_ssm(args.profile, instance_id, min(args.timeout, 1800))
    deadline = instance_deadline_epoch(
        args.profile, instance_id, args.maximum_hours
    )
    guard = send_and_wait(
        args.profile,
        instance_id,
        cost_guard_commands(deadline),
        min(args.timeout, 600),
        comment="Install persisted Wizard Vision cost deadline",
    )
    if args.cost_guard_only:
        print(
            json.dumps(
                {
                    "instance_id": instance_id,
                    "status": guard["Status"],
                    "deadline_epoch": deadline,
                },
                indent=2,
            )
        )
        return 0

    manifest = source_manifest(args.profile, args.source_commit)
    rust = rust_manifest(args.profile, args.source_commit)
    presigned_url = aws(
        args.profile,
        "s3",
        "presign",
        f"s3://{BUCKET}/{AWSCLI_KEY}",
        "--expires-in",
        "3600",
    ).stdout.strip()
    invocation = send_and_wait(
        args.profile,
        instance_id,
        bootstrap_commands(
            presigned_url,
            args.source_commit,
            manifest,
            rust,
        ),
        args.timeout,
        comment="Restore checksummed Wizard Vision training state",
    )
    print(
        json.dumps(
            {
                "instance_id": instance_id,
                "status": invocation["Status"],
                "source_commit": args.source_commit,
                "output": invocation.get("StandardOutputContent", "")[-4000:],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
