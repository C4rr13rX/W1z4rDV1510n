use crate::schema::{ParticleState, Population};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareBackendType {
    Cpu,
}

impl Default for HardwareBackendType {
    fn default() -> Self {
        HardwareBackendType::Cpu
    }
}

pub trait HardwareBackend: Send + Sync {
    fn map_particles(&self, population: &mut Population, func: &mut dyn FnMut(&mut ParticleState));
}

#[derive(Default)]
pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn map_particles(&self, population: &mut Population, func: &mut dyn FnMut(&mut ParticleState)) {
        for particle in &mut population.particles {
            func(particle);
        }
    }
}

pub type HardwareBackendHandle = Arc<dyn HardwareBackend>;

pub fn create_hardware_backend(kind: HardwareBackendType) -> HardwareBackendHandle {
    match kind {
        HardwareBackendType::Cpu => Arc::new(CpuBackend),
    }
}
