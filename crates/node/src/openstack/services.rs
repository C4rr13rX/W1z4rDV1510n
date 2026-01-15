use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePlan {
    pub vcpus: usize,
    pub memory_mb: usize,
    pub disk_gb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPlan {
    pub bandwidth_mbps: usize,
    pub public_ip: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoragePlan {
    pub volume_gb: usize,
    pub iops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPlan {
    pub enabled: bool,
    pub sample_interval_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenStackNodePlan {
    pub compute: ComputePlan,
    pub network: NetworkPlan,
    pub storage: StoragePlan,
    pub telemetry: TelemetryPlan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionedNode {
    pub instance_id: String,
    pub public_ip: Option<String>,
}

pub trait OpenStackProvider: Send + Sync {
    fn provision_node(&self, plan: OpenStackNodePlan) -> Result<ProvisionedNode>;
}
