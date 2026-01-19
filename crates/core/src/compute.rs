use crate::config::{ClusterConfig, ComputeRoutingConfig, QuantumEndpointConfig};
use crate::hardware::HardwareProfile;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComputeTarget {
    Cpu,
    Gpu,
    Quantum,
    Cluster,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ComputeJobKind {
    TensorHeavy,
    Bitwise,
    Graph,
    Sampling,
    EventIntensity,
    CausalDiscovery,
    QuantumAnneal,
    QuantumCalibration,
}

#[derive(Debug, Clone)]
pub struct ComputeJob {
    pub kind: ComputeJobKind,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ComputeResult {
    pub target: ComputeTarget,
    pub payload: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct QuantumJob {
    pub kind: ComputeJobKind,
    pub payload: Vec<u8>,
    pub timeout_secs: u64,
}

#[derive(Debug, Clone)]
pub struct QuantumResult {
    pub payload: Vec<u8>,
    pub metadata: HashMap<String, String>,
}

pub trait QuantumExecutor: Send + Sync {
    fn submit(&self, job: QuantumJob) -> Result<QuantumResult>;
}

#[derive(Debug, Default)]
pub struct NoopQuantumExecutor;

impl QuantumExecutor for NoopQuantumExecutor {
    fn submit(&self, _job: QuantumJob) -> Result<QuantumResult> {
        anyhow::bail!("quantum executor not configured")
    }
}

pub struct ComputeRouter {
    profile: HardwareProfile,
    compute: ComputeRoutingConfig,
    cluster: ClusterConfig,
}

impl ComputeRouter {
    pub fn new(compute: ComputeRoutingConfig, cluster: ClusterConfig) -> Self {
        let profile = HardwareProfile::detect();
        Self {
            profile,
            compute,
            cluster,
        }
    }

    pub fn route(&self, job: ComputeJobKind) -> ComputeTarget {
        if self.should_use_quantum(job) {
            return ComputeTarget::Quantum;
        }
        if self.should_use_gpu(job) {
            return ComputeTarget::Gpu;
        }
        if self.should_use_cluster(job) {
            return ComputeTarget::Cluster;
        }
        ComputeTarget::Cpu
    }

    pub fn quantum_endpoints(&self) -> &[QuantumEndpointConfig] {
        &self.compute.quantum_endpoints
    }

    pub fn hardware_profile(&self) -> &HardwareProfile {
        &self.profile
    }

    fn should_use_quantum(&self, job: ComputeJobKind) -> bool {
        if !self.compute.allow_quantum {
            return false;
        }
        if self.compute.quantum_endpoints.is_empty() {
            return false;
        }
        matches!(job, ComputeJobKind::QuantumAnneal | ComputeJobKind::QuantumCalibration)
    }

    fn should_use_gpu(&self, job: ComputeJobKind) -> bool {
        if !self.compute.allow_gpu || !self.profile.has_gpu {
            return false;
        }
        matches!(job, ComputeJobKind::TensorHeavy | ComputeJobKind::Bitwise)
    }

    fn should_use_cluster(&self, job: ComputeJobKind) -> bool {
        if !self.cluster.enabled {
            return false;
        }
        matches!(job, ComputeJobKind::Graph | ComputeJobKind::CausalDiscovery)
    }
}
