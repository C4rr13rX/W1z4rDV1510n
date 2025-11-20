//! Experimental hardware backend implementations.

use parking_lot::Mutex;
use std::sync::Arc;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};
use w1z4rdv1510n::config::ExperimentalHardwareConfig;
use w1z4rdv1510n::hardware::{HardwareBackend, HardwareBackendHandle, HardwareBackendType, NoiseSourceHandle};
use w1z4rdv1510n::schema::{ParticleState, Population};

pub struct ExperimentalHardwareBackend {
    noise: NoiseSourceHandle,
    system: Mutex<System>,
    options: ExperimentalHardwareConfig,
    last_sample: Mutex<std::time::Instant>,
}

impl ExperimentalHardwareBackend {
    pub fn new(noise: NoiseSourceHandle, options: ExperimentalHardwareConfig) -> Self {
        Self {
            noise,
            system: Mutex::new(System::new()),
            options,
            last_sample: Mutex::new(std::time::Instant::now()),
        }
    }

    fn maybe_sample(&self) {
        let interval = std::time::Duration::from_millis(750);
        let mut last = self.last_sample.lock();
        if last.elapsed() < interval {
            return;
        }
        *last = std::time::Instant::now();
        let mut system = self.system.lock();
        system.refresh_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::new())
                .with_memory(MemoryRefreshKind::everything()),
        );
        let loads = system.load_average();
        let used_memory_gb = (system.used_memory() as f64 / 1_048_576.0).max(0.0);
        tracing::info!(
            target: "w1z4rdv1510n::hardware::experimental",
            load_avg_one = loads.one,
            used_memory_gb,
            "sampled experimental hardware metrics"
        );
    }
}

impl HardwareBackend for ExperimentalHardwareBackend {
    fn map_particles(
        &self,
        population: &mut Population,
        func: &(dyn Fn(&mut ParticleState) + Send + Sync),
    ) {
        self.maybe_sample();
        for particle in &mut population.particles {
            func(particle);
        }
    }

    fn noise_source(&self) -> NoiseSourceHandle {
        Arc::clone(&self.noise)
    }
}

pub fn create_backend(
    noise: NoiseSourceHandle,
    options: ExperimentalHardwareConfig,
) -> (HardwareBackendHandle, HardwareBackendType) {
    (
        Arc::new(ExperimentalHardwareBackend::new(noise, options)) as HardwareBackendHandle,
        HardwareBackendType::Experimental,
    )
}
