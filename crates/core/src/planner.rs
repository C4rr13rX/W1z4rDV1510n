use crate::hardware::HardwareProfile;
use crate::tensor::TensorPrecision;
use std::sync::Arc;
use tracing::debug;

#[derive(Debug, Clone, Copy)]
pub enum KernelKind {
    TensorHeavy,
    Bitwise,
    PrePost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
    Hybrid,
}

pub struct BackendPlanner {
    profile: HardwareProfile,
    has_gpu_backend: bool,
    default_precision: TensorPrecision,
}

impl BackendPlanner {
    pub fn new(profile: HardwareProfile, has_gpu_backend: bool) -> Self {
        let default_precision = if profile.cpu_cores >= 8 && profile.total_memory_gb >= 16.0 {
            TensorPrecision::Fp16
        } else {
            TensorPrecision::Fp32
        };
        debug!(
            target: "w1z4rdv1510n::planner",
            cpu_cores = profile.cpu_cores,
            memory_gb = profile.total_memory_gb,
            has_gpu_backend,
            ?default_precision,
            "backend planner initialized"
        );
        Self {
            profile,
            has_gpu_backend,
            default_precision,
        }
    }

    pub fn target_for(&self, kernel: KernelKind) -> ExecutionTarget {
        match kernel {
            KernelKind::TensorHeavy => {
                if self.has_gpu_backend && self.profile.cpu_cores >= 8 {
                    ExecutionTarget::Hybrid
                } else {
                    ExecutionTarget::Cpu
                }
            }
            KernelKind::Bitwise => {
                if self.has_gpu_backend {
                    ExecutionTarget::Hybrid
                } else {
                    ExecutionTarget::Cpu
                }
            }
            KernelKind::PrePost => ExecutionTarget::Cpu,
        }
    }

    pub fn precision_for(&self, kernel: KernelKind) -> TensorPrecision {
        match kernel {
            KernelKind::TensorHeavy => self.default_precision,
            KernelKind::Bitwise => TensorPrecision::Int8,
            KernelKind::PrePost => TensorPrecision::Fp32,
        }
    }

    pub fn threads_for(&self, workload: usize) -> usize {
        let cores = self.profile.cpu_cores.max(1);
        let desired = if workload < cores {
            workload.max(1)
        } else {
            cores
        };
        desired
    }
}

pub type BackendPlannerHandle = Arc<BackendPlanner>;
