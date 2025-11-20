use crate::hardware::HardwareBackendType;
use crate::ml::MLBackendType;
use crate::proposal::ProposalConfig;
use crate::random::RandomConfig;
use crate::schema::Timestamp;
use crate::search::SearchConfig;
use crate::state_population::InitStrategyConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

fn default_snapshot_path() -> PathBuf {
    PathBuf::from("snapshot.json")
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    #[serde(default = "default_snapshot_path")]
    pub snapshot_file: PathBuf,
    #[serde(default)]
    pub mode: RunMode,
    pub t_end: Timestamp,
    #[serde(default = "default_particles")]
    pub n_particles: usize,
    #[serde(default)]
    pub schedule: AnnealingScheduleConfig,
    #[serde(default)]
    pub energy: EnergyConfig,
    #[serde(default)]
    pub proposal: ProposalConfig,
    #[serde(default)]
    pub init_strategy: InitStrategyConfig,
    #[serde(default)]
    pub resample: ResampleConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub ml_backend: MLBackendType,
    #[serde(default)]
    pub hardware_backend: HardwareBackendType,
    #[serde(default = "default_seed")]
    pub random_seed: u64,
    #[serde(default)]
    pub random: RandomConfig,
    #[serde(default)]
    pub experimental_hardware: ExperimentalHardwareConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub output: OutputConfig,
}

impl RunConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.n_particles > 0, "n_particles must be > 0");
        anyhow::ensure!(
            self.schedule.n_iterations > 0,
            "schedule.n_iterations must be > 0"
        );
        anyhow::ensure!(
            self.energy.w_motion >= 0.0
                && self.energy.w_collision >= 0.0
                && self.energy.w_goal >= 0.0
                && self.energy.w_group_cohesion >= 0.0
                && self.energy.w_ml_prior >= 0.0
                && self.energy.w_path_feasibility >= 0.0
                && self.energy.w_env_constraints >= 0.0,
            "energy weights must be non-negative"
        );
        anyhow::ensure!(
            self.proposal.local_move_prob
                + self.proposal.group_move_prob
                + self.proposal.swap_move_prob
                + self.proposal.path_based_move_prob
                + self.proposal.global_move_prob
                > 0.0,
            "proposal move probabilities must sum to > 0"
        );
        anyhow::ensure!(
            self.search.cell_size > 0.01,
            "search.cell_size must be > 0.01"
        );
        anyhow::ensure!(
            self.resample.mutation_rate >= 0.0 && self.resample.mutation_rate <= 1.0,
            "mutation_rate must be between 0 and 1"
        );
        if matches!(self.mode, RunMode::Production) {
            anyhow::ensure!(
                !self.experimental_hardware.enabled,
                "experimental hardware options must be disabled in production mode"
            );
            anyhow::ensure!(
                self.hardware_backend != HardwareBackendType::Experimental,
                "experimental hardware backend is not allowed in production mode"
            );
            anyhow::ensure!(
                !matches!(
                    self.random.provider,
                    crate::random::RandomProviderType::JitterExperimental
                ),
                "experimental jitter RNG is not allowed in production mode"
            );
        }
        if self.experimental_hardware.enabled {
            anyhow::ensure!(
                self.hardware_backend == HardwareBackendType::Experimental,
                "set hardware_backend to EXPERIMENTAL when experimental hardware options are enabled"
            );
        }
        Ok(())
    }
}

fn default_particles() -> usize {
    512
}

fn default_seed() -> u64 {
    7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnealingScheduleConfig {
    pub t_start: f64,
    pub t_end: f64,
    pub n_iterations: usize,
    #[serde(default)]
    pub schedule_type: ScheduleType,
}

impl Default for AnnealingScheduleConfig {
    fn default() -> Self {
        Self {
            t_start: 5.0,
            t_end: 0.2,
            n_iterations: 200,
            schedule_type: ScheduleType::Linear,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScheduleType {
    Linear,
    Exponential,
    Custom,
}

impl Default for ScheduleType {
    fn default() -> Self {
        ScheduleType::Linear
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConfig {
    pub w_motion: f64,
    pub w_collision: f64,
    pub w_goal: f64,
    pub w_group_cohesion: f64,
    pub w_ml_prior: f64,
    pub w_path_feasibility: f64,
    pub w_env_constraints: f64,
    #[serde(default)]
    pub other_terms: HashMap<String, f64>,
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            w_motion: 1.0,
            w_collision: 5.0,
            w_goal: 1.0,
            w_group_cohesion: 0.5,
            w_ml_prior: 0.0,
            w_path_feasibility: 1.0,
            w_env_constraints: 3.0,
            other_terms: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResampleConfig {
    #[serde(default = "ResampleConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "ResampleConfig::default_threshold")]
    pub effective_sample_size_threshold: f64,
    #[serde(default = "ResampleConfig::default_mutation_rate")]
    pub mutation_rate: f64,
}

impl ResampleConfig {
    fn default_enabled() -> bool {
        true
    }

    fn default_threshold() -> f64 {
        0.3
    }

    fn default_mutation_rate() -> f64 {
        0.05
    }
}

impl Default for ResampleConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            effective_sample_size_threshold: Self::default_threshold(),
            mutation_rate: Self::default_mutation_rate(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "LoggingConfig::default_level")]
    pub log_level: String,
    #[serde(default)]
    pub log_path: Option<PathBuf>,
    #[serde(default)]
    pub json: bool,
}

impl LoggingConfig {
    fn default_level() -> String {
        "INFO".to_string()
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: Self::default_level(),
            log_path: None,
            json: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    #[serde(default)]
    pub save_best_state: bool,
    #[serde(default)]
    pub save_population_summary: bool,
    #[serde(default)]
    pub save_trajectories: bool,
    #[serde(default = "OutputConfig::default_format")]
    pub format: OutputFormat,
    #[serde(default)]
    pub output_path: Option<PathBuf>,
}

impl OutputConfig {
    fn default_format() -> OutputFormat {
        OutputFormat::Json
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            save_best_state: true,
            save_population_summary: true,
            save_trajectories: false,
            format: Self::default_format(),
            output_path: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Msgpack,
    Custom,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Json
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RunMode {
    Production,
    LabExperimental,
}

impl Default for RunMode {
    fn default() -> Self {
        RunMode::Production
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalHardwareConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub use_thermal: bool,
    #[serde(default)]
    pub use_performance_counters: bool,
    #[serde(default = "ExperimentalHardwareConfig::default_interval")]
    pub max_sample_interval_secs: f64,
}

impl ExperimentalHardwareConfig {
    fn default_interval() -> f64 {
        1.0
    }
}

impl Default for ExperimentalHardwareConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            use_thermal: true,
            use_performance_counters: false,
            max_sample_interval_secs: Self::default_interval(),
        }
    }
}
