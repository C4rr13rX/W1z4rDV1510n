use crate::schema::{ParticleState, Population};
use rayon::prelude::*;
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
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    );
}

#[derive(Default)]
pub struct CpuBackend;

impl HardwareBackend for CpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        for particle in &mut population.particles {
            func(particle);
        }
    }
}

pub struct MultiThreadedCpuBackend;

impl HardwareBackend for MultiThreadedCpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        population
            .particles
            .par_iter_mut()
            .for_each(|particle| func(particle));
    }
}

pub struct ExternalBackend;

impl HardwareBackend for ExternalBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
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
