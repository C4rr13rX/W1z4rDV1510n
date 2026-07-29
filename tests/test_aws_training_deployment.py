import json
from pathlib import Path

from scripts.aws.bootstrap_training_host import (
    bootstrap_commands,
    cost_guard_commands,
)
from scripts.aws.deploy_private_training import estimate_cost
from scripts.aws.restore_training_inputs import rebase_progress_corpora


ROOT = Path(__file__).resolve().parents[1]


def test_default_cost_envelope_stays_below_agreed_ceiling():
    estimate = estimate_cost(
        hours=192,
        hourly_compute=0.1728,
        volume_gib=1024,
        final_s3_gib=400,
    )
    assert estimate.total < 100


def test_private_template_has_no_public_network_or_elasticache():
    template = (
        ROOT / "infra" / "aws" / "wizard-vision-private-training.yaml"
    ).read_text(encoding="utf-8")
    assert "MapPublicIpOnLaunch: false" in template
    assert "AWS::EC2::InternetGateway" not in template
    assert "AWS::EC2::NatGateway" not in template
    assert "AWS::ElastiCache::" not in template
    assert "AWS::Route53::" not in template
    assert "AWS::EC2::VPCEndpoint" in template


def test_data_volume_is_encrypted_retained_and_reflink_ready():
    template = (
        ROOT / "infra" / "aws" / "wizard-vision-private-training.yaml"
    ).read_text(encoding="utf-8")
    assert "DeletionPolicy: Retain" in template
    assert "Encrypted: true" in template
    assert "mkfs.xfs -f -m reflink=1" in template
    assert "wizard-cost-stop.timer" in template


def test_private_host_bootstrap_verifies_source_then_restores_manifests():
    commands = "\n".join(
        bootstrap_commands(
            "https://example.invalid/presigned",
            "a" * 40,
            {"archive": "source.tar.gz", "sha256": "b" * 64},
            {"archive": "rust.tar.gz", "sha256": "c" * 64},
        )
    )
    assert "sha256sum" in commands
    assert '"b' in commands
    assert '"c' in commands
    assert "chmod a+x /tmp/aws/aws/install /tmp/aws/aws/dist/aws" in commands
    assert "dnf install -y rust cargo" in commands
    assert "rust.tar.gz" in commands
    assert "restore_training_inputs.py" in commands
    assert "--source-commit " + "a" * 40 in commands


def test_deployer_can_apply_and_recover_in_place_instance_updates():
    policy = (
        ROOT / "infra" / "aws" / "fountain-server-deploy-policy.json"
    ).read_text(encoding="utf-8")
    assert '"ec2:ModifyInstanceAttribute"' in policy
    assert '"cloudformation:ContinueUpdateRollback"' in policy


def test_cost_guard_uses_a_persisted_absolute_deadline():
    commands = "\n".join(cost_guard_commands(1_800_000_000))
    assert "echo 1800000000 >/srv/wizard/control/cost-deadline.epoch" in commands
    assert "wizard-cost-stop.timer" in commands
    assert "OnUnitActiveSec=1min" in commands
    assert "/sbin/shutdown -h now" in commands


def test_cross_os_restore_rebases_verified_progress_corpus(tmp_path):
    runtime = tmp_path / "runtime"
    corpora = tmp_path / "corpora"
    runtime.mkdir()
    corpora.mkdir()
    (corpora / "phase.jsonl").write_text("{}\n", encoding="utf-8")
    progress = runtime / "phase.progress.json"
    progress.write_text(
        '{"corpus":"D:\\\\training\\\\phase.jsonl","durable_next_row":7}',
        encoding="utf-8",
    )

    assert rebase_progress_corpora(runtime, corpora) == [progress.name]
    restored = json.loads(progress.read_text(encoding="utf-8"))
    assert restored["corpus"] == str((corpora / "phase.jsonl").resolve())
    assert restored["durable_next_row"] == 7
