use crate::schema::{ParticleState, Population};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareBackendType {
    Cpu,
    MultiThreadedCpu,
    External,
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

pub struct MultiThreadedCpuBackend;

impl HardwareBackend for MultiThreadedCpuBackend {
    fn map_particles(&self, population: &mut Population, func: &mut dyn FnMut(&mut ParticleState)) {
        for particle in &mut population.particles {
            func(particle);
        }
    }
}

pub struct ExternalBackend;

impl HardwareBackend for ExternalBackend {
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
        HardwareBackendType::MultiThreadedCpu => Arc::new(MultiThreadedCpuBackend),
        HardwareBackendType::External => Arc::new(ExternalBackend),
    }
}
