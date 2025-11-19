use crate::schema::{ParticleState, Population};
use parking_lot::Mutex;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
    fn noise_source(&self) -> NoiseSourceHandle;
}

pub struct CpuBackend {
    noise: NoiseSourceHandle,
}

impl CpuBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

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

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub struct MultiThreadedCpuBackend {
    noise: NoiseSourceHandle,
}

impl MultiThreadedCpuBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

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

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub struct ExternalBackend {
    noise: NoiseSourceHandle,
}

impl ExternalBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

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

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub type HardwareBackendHandle = Arc<dyn HardwareBackend>;

pub fn create_hardware_backend(kind: HardwareBackendType, seed: u64) -> HardwareBackendHandle {
    match kind {
        HardwareBackendType::Cpu => Arc::new(CpuBackend::new(seed)),
        HardwareBackendType::MultiThreadedCpu => Arc::new(MultiThreadedCpuBackend::new(seed)),
        HardwareBackendType::External => Arc::new(ExternalBackend::new(seed)),
    }
}

pub trait NoiseSource: Send + Sync {
    fn random_float(&self) -> f64;
    fn random_int(&self, upper: u64) -> u64;
    fn entropy_status(&self) -> HashMap<String, String>;
}

pub type NoiseSourceHandle = Arc<dyn NoiseSource>;

struct SoftwareNoiseSource {
    rng: Mutex<StdRng>,
}

impl SoftwareNoiseSource {
    fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

impl NoiseSource for SoftwareNoiseSource {
    fn random_float(&self) -> f64 {
        self.rng.lock().r#gen()
    }

    fn random_int(&self, upper: u64) -> u64 {
        if upper == 0 {
            0
        } else {
            self.rng.lock().gen_range(0..upper)
        }
    }

    fn entropy_status(&self) -> HashMap<String, String> {
        let mut status = HashMap::new();
        status.insert("type".into(), "software_rng".into());
        status.insert("backing".into(), "StdRng".into());
        status
    }
}
