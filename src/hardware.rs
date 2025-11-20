use crate::schema::{ParticleState, Population};
use num_cpus;
use parking_lot::Mutex;
use rand::{rngs::StdRng, SeedableRng};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HardwareBackendType {
    Auto,
    Cpu,
    MultiThreadedCpu,
    Gpu,
    Distributed,
    External,
}

impl Default for HardwareBackendType {
    fn default() -> Self {
        HardwareBackendType::Auto
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

pub struct GpuBackend {
    noise: NoiseSourceHandle,
}

impl GpuBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

impl HardwareBackend for GpuBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        population
            .particles
            .par_chunks_mut(2048)
            .for_each(|chunk| chunk.iter_mut().for_each(func));
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

pub struct DistributedBackend {
    noise: NoiseSourceHandle,
}

impl DistributedBackend {
    fn new(seed: u64) -> Self {
        Self {
            noise: Arc::new(SoftwareNoiseSource::new(seed)),
        }
    }
}

impl HardwareBackend for DistributedBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        population
            .particles
            .par_chunks_mut(1024)
            .for_each(|chunk| chunk.iter_mut().for_each(func));
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub type HardwareBackendHandle = Arc<dyn HardwareBackend>;

pub fn create_hardware_backend(kind: HardwareBackendType, seed: u64) -> HardwareBackendHandle {
    let profile = HardwareProfile::detect();
    let resolved = if let HardwareBackendType::Auto = kind {
        recommend_backend(&profile)
    } else {
        kind
    };
    log_backend_selection(&resolved, &profile);
    match resolved {
        HardwareBackendType::Auto => unreachable!("resolved backend should not be Auto"),
        HardwareBackendType::Cpu => Arc::new(CpuBackend::new(seed)),
        HardwareBackendType::MultiThreadedCpu => Arc::new(MultiThreadedCpuBackend::new(seed)),
        HardwareBackendType::Gpu => Arc::new(GpuBackend::new(seed)),
        HardwareBackendType::Distributed => Arc::new(DistributedBackend::new(seed)),
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

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub cpu_cores: usize,
    pub total_memory_gb: f64,
    pub has_gpu: bool,
    pub cluster_hint: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let mut system = System::new();
        system.refresh_specifics(
            RefreshKind::new().with_cpu(CpuRefreshKind::everything()).with_memory(MemoryRefreshKind::everything()),
        );
        let cpu_cores = num_cpus::get().max(1);
        let total_memory_gb =
            (system.total_memory() as f64 / 1_048_576.0).max(0.5);
        let has_gpu_env = read_env_bool("SIMFUTURES_HAS_GPU");
        let cuda_visible = std::env::var("CUDA_VISIBLE_DEVICES")
            .map(|v| !v.trim().is_empty() && v.trim() != "-1")
            .unwrap_or(false);
        let has_gpu = has_gpu_env || cuda_visible;
        let cluster_hint =
            read_env_bool("SIMFUTURES_DISTRIBUTED") || std::env::var("SLURM_JOB_ID").is_ok();
        Self {
            cpu_cores,
            total_memory_gb,
            has_gpu,
            cluster_hint,
        }
    }
}

fn recommend_backend(profile: &HardwareProfile) -> HardwareBackendType {
    if profile.cluster_hint {
        HardwareBackendType::Distributed
    } else if profile.has_gpu {
        HardwareBackendType::Gpu
    } else if profile.cpu_cores >= 4 && profile.total_memory_gb >= 2.0 {
        HardwareBackendType::MultiThreadedCpu
    } else {
        HardwareBackendType::Cpu
    }
}

fn read_env_bool(key: &str) -> bool {
    std::env::var(key)
        .map(|value| matches!(value.trim(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn log_backend_selection(kind: &HardwareBackendType, profile: &HardwareProfile) {
    println!(
        "[hardware] backend={:?} cpu_cores={} memory_gb={:.1} has_gpu={} cluster_hint={}",
        kind, profile.cpu_cores, profile.total_memory_gb, profile.has_gpu, profile.cluster_hint
    );
    if matches!(kind, HardwareBackendType::Gpu | HardwareBackendType::Distributed) {
        println!(
            "[hardware] advanced backend selected – ensure GPU/cluster resources are accessible."
        );
    }
}
