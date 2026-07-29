from pathlib import Path

from scripts.aws.deploy_private_training import estimate


ROOT = Path(__file__).resolve().parents[1]


def test_cost_guard_stays_below_acknowledged_ceiling():
    costs = estimate(volume_gib=1024, hours=240, spot_cap=0.20)
    assert costs["guarded_total"] < 100
    assert costs["compute_max"] == 48.0


def test_private_template_has_no_public_ingress_or_elasticache():
    template = (
        ROOT / "infra" / "aws" / "wizard-vision-private-training.yaml"
    ).read_text(encoding="utf-8")
    assert "MapPublicIpOnLaunch: false" in template
    assert "SecurityGroupIngress: []" in template
    assert "AWS::ElastiCache" not in template
    assert "AWS::EC2::Volume" in template
    assert "DeletionPolicy: Retain" in template


def test_runbook_preserves_settle_then_upload_invariant():
    runbook = (ROOT / "docs" / "AWS_PRIVATE_TRAINING.md").read_text(
        encoding="utf-8"
    )
    assert "Never copy the mutable container while training is running" in runbook
    assert "--settle-live-node" in runbook
