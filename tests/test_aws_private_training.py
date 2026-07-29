from pathlib import Path

from scripts.aws.deploy_private_training import (
    DEPLOYMENT_GROUP,
    POLICY_ARN,
    estimate_cost,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cost_guard_stays_below_acknowledged_ceiling():
    costs = estimate_cost(
        volume_gib=1024,
        hours=192,
        hourly_compute=0.1728,
        final_s3_gib=400,
    )
    assert costs.total < 100
    assert round(costs.compute, 2) == 33.18
    assert POLICY_ARN.endswith(":policy/WizardVisionPrivateTrainingDeployment")
    assert DEPLOYMENT_GROUP == "WizardVisionDeployers"


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
