from pathlib import Path

from scripts.aws.bootstrap_training_host import bootstrap_commands
from scripts.aws.deploy_private_training import estimate_cost


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
    assert "AssociatePublicIpAddress: false" in template
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
        )
    )
    assert "sha256sum" in commands
    assert '"b' in commands
    assert "restore_training_inputs.py" in commands
    assert "--source-commit " + "a" * 40 in commands
