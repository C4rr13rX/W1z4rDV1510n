# AWS private training runbook

Wizard Vision's authoritative state is a mutable, randomly addressed
`brain.wbrain` container plus a WAL. S3 is the migration and recovery tier, not
the live filesystem. ElastiCache is intentionally absent: the brain already
owns a neuron-scoped RAM tier, and a remote key/value cache neither supplies
durable random-access storage nor improves local propagation.

The stack in `infra/aws/wizard-vision-private-training.yaml` creates:

- one VPC and subnet with no internet gateway and no public address;
- no inbound security-group rules and no domain;
- private S3, SSM, and SSM Messages endpoints;
- one On-Demand `m6a.xlarge` with 4 vCPU and 16 GiB dedicated RAM;
- one encrypted 1 TiB gp3 data volume retained independently of the instance;
- an instance timer that shuts down and stops compute after at most 192 hours;
  and
- a least-privilege instance role restricted to the Wizard Vision bucket
  prefix and its own cost-stop operations.

The private bucket is
`wizard-vision-private-321572159829-us-east-1`. Public ACLs and policies are
blocked, TLS is required, default SSE-S3 encryption is enabled, and versioning
is enabled.

## Cost boundary

`scripts/aws/deploy_private_training.py` refuses deployment unless the operator
acknowledges a ceiling at or above its conservative estimate. The initial
deployment uses the full On-Demand hourly price rather than assuming Spot
availability. The initial deployment command is:

```text
python scripts/aws/deploy_private_training.py \
  --bootstrap-permissions \
  --deploy \
  --cost-cap-usd 100
```

The EBS volume is retained if the stack is removed accidentally. After final
validation, upload the settled final runtime, verify the object checksum,
explicitly delete the retained volume, and remove the inline deployment policy.
The deployment permission is a customer-managed policy attached through the
dedicated `WizardVisionDeployers` group; it is not an inline user policy.

## Migration invariant

Never copy the mutable container while training is running. First stop the
trainer at its WAL-durable boundary, then run:

```text
python scripts/aws/stage_training_state.py \
  --runtime runtime/brains/programming-integrated-20260713 \
  --settle-live-node
```

After the manifest reports equal RAM and durable offsets and a successful
checkpoint, stop the local brain server and rerun with `--upload`. The upload
includes only the authoritative container, WAL, rollback guard, identity,
corpus ledgers, deferred intervals, and admission metadata enumerated in its
SHA-256 manifest. Do not delete local state until the AWS instance has restored
the exact hashes and passed the retention and enterprise gates.

Before provisioning, run `scripts/aws/stage_rust_dependencies.py --upload` to
vendor the lockfile-pinned Rust dependencies under the same source commit.
Once the source, corpus, Rust, and stopped brain manifests all exist, restore
through Systems Manager without opening ingress or adding a NAT gateway:

```text
python scripts/aws/bootstrap_training_host.py \
  --source-commit <the-pushed-git-commit>
```
