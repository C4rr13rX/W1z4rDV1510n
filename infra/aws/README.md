# Private AWS training environment

This deployment exists only to finish the Wizard Vision programming
curriculum. It deliberately does not create ElastiCache: the brain already
owns a neuron-scoped RAM tier, and a remote key-value cache neither stores the
authoritative `.wbrain` safely nor reduces its random-access latency.

The stack creates one private subnet with no public IP, public ingress,
internet gateway, NAT gateway, load balancer, DNS record, or domain. S3 uses a
free gateway endpoint. Three interface endpoints provide private Systems
Manager control. The instance has a persistent encrypted gp3 data volume and
a seven-day default runtime stop timer. The data volume is retained if the
stack is deleted so cleanup must be explicit after the final verified brain is
copied to S3.

The default cost envelope is:

- `m6a.xlarge` Linux On-Demand, budgeted at `$0.1728/hour`;
- `1024 GiB` gp3 at the included `3000 IOPS / 125 MB/s` baseline;
- at most `192` running hours;
- three single-AZ interface endpoints;
- one month of 400 GiB in a private S3 staging/finalization prefix;
- a `$5` allowance for requests and minor unmodeled overhead.

Run `python scripts/aws/deploy_private_training.py` first. Planning is the
default. Deployment requires both `--deploy` and a cost cap no lower than the
calculated worst-case estimate. It will refuse a public or unencrypted bucket.

`--bootstrap-permissions` publishes
`infra/aws/fountain-server-deploy-policy.json` as a customer-managed policy.
When the established `FountainServer` user's ten direct managed-policy slots
are occupied, the deployer attaches it through the dedicated
`WizardVisionDeployers` IAM group instead. The project policy adds only the
EC2 network/storage actions needed by this stack, CloudFormation stack
control, private host control through Systems Manager, and read-only access to
the public Amazon Linux AMI parameter. Existing S3 permissions supply bucket
access.

The production data migration is a separate transaction:

1. Package and upload tracked source plus the seven corpus files.
2. Stop the local curriculum supervisor.
3. Settle/checkpoint the brain and verify zero resident terminals.
4. Stop the local brain server.
5. Upload the authoritative `.wbrain`, WAL, identity, indexes, progress
   ledgers, quarantine/deferred ledgers, and the current last-good guard.
6. Verify remote sizes and SHA-256 checksums.
7. Restore on the EBS volume, validate topology and retention, then resume from
   the exact durable row.

Never delete the local copy until the restored AWS brain passes those gates.
`scripts/aws/stage_training_state.py` deliberately separates live settlement
from stopped-state hashing/upload so a mutable `.wbrain` can never be uploaded
under a checksum calculated from a different point in time.
