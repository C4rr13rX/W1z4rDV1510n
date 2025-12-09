use crate::hardware::HardwareBackendType;
use crate::ml::MLBackendType;
use crate::neuro::NeuroRuntimeConfig;
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
    pub quantum: QuantumConfig,
    #[serde(default)]
    pub search: SearchConfig,
    #[serde(default)]
    pub ml_backend: MLBackendType,
    #[serde(default)]
    pub hardware_backend: HardwareBackendType,
    #[serde(default)]
    pub hardware_overrides: HardwareOverrides,
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
    #[serde(default)]
    pub neuro: NeuroRuntimeConfig,
    #[serde(default)]
    pub homeostasis: HomeostasisConfig,
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
                && self.energy.w_relational_prior >= 0.0
                && self.energy.w_neuro_alignment >= 0.0
                && self.energy.w_motif_transition >= 0.0
                && self.energy.w_super_network_prior >= 0.0
                && self.energy.w_path_feasibility >= 0.0
                && self.energy.w_env_constraints >= 0.0
                && self.energy.w_stack_hash >= 0.0
                && self.energy.stack_alignment_weight >= 0.0
                && self.energy.stack_future_weight >= 0.0,
            "energy weights must be non-negative"
        );
        anyhow::ensure!(
            self.energy.stack_alignment_topk > 0,
            "energy.stack_alignment_topk must be > 0"
        );
        if self.quantum.enabled {
            anyhow::ensure!(
                self.quantum.trotter_slices > 0,
                "quantum.trotter_slices must be > 0 when quantum mode is enabled"
            );
            anyhow::ensure!(
                self.quantum.driver_strength >= 0.0,
                "quantum.driver_strength must be >= 0"
            );
            anyhow::ensure!(
                self.quantum.driver_final_strength >= 0.0,
                "quantum.driver_final_strength must be >= 0"
            );
            anyhow::ensure!(
                self.quantum.slice_temperature_scale > 0.0,
                "quantum.slice_temperature_scale must be > 0"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.quantum.worldline_mix_prob),
                "quantum.worldline_mix_prob must be in [0,1]"
            );
        }
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
            (0.0..=1.0).contains(&self.neuro.min_activation),
            "neuro.min_activation must be in [0,1]"
        );
        anyhow::ensure!(
            self.neuro.module_threshold > 0,
            "neuro.module_threshold must be > 0"
        );
        if self.homeostasis.enabled {
            anyhow::ensure!(
                self.homeostasis.patience > 0,
                "homeostasis.patience must be > 0 when enabled"
            );
            anyhow::ensure!(
                self.homeostasis.energy_plateau_tolerance >= 0.0,
                "homeostasis.energy_plateau_tolerance must be >= 0"
            );
            anyhow::ensure!(
                self.homeostasis.mutation_boost >= 0.0,
                "homeostasis.mutation_boost must be >= 0"
            );
            anyhow::ensure!(
                self.homeostasis.reheat_scale >= 0.0,
                "homeostasis.reheat_scale must be >= 0"
            );
        }
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
        if !self.hardware_overrides.allow_gpu {
            anyhow::ensure!(
                self.hardware_backend != HardwareBackendType::Gpu,
                "hardware_backend=GPU is not allowed when hardware_overrides.allow_gpu is false"
            );
        }
        if !self.hardware_overrides.allow_distributed {
            anyhow::ensure!(
                self.hardware_backend != HardwareBackendType::Distributed,
                "hardware_backend=DISTRIBUTED is not allowed when hardware_overrides.allow_distributed is false"
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
#[serde(default)]
pub struct HomeostasisConfig {
    /// Enable the self-adjustment loop that reheats/boosts mutation on plateaus.
    pub enabled: bool,
    /// Minimum relative improvement in best energy to be considered progress.
    pub energy_plateau_tolerance: f64,
    /// Number of consecutive plateau iterations before triggering adjustments.
    pub patience: usize,
    /// Multiplier applied to mutation rate when plateauing (e.g., 0.5 -> +50%).
    pub mutation_boost: f64,
    /// Scale factor applied to temperature when plateauing (e.g., 0.2 -> +20% temp).
    pub reheat_scale: f64,
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            energy_plateau_tolerance: 1e-4,
            patience: 8,
            mutation_boost: 0.5,
            reheat_scale: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergyConfig {
    pub w_motion: f64,
    pub w_collision: f64,
    pub w_goal: f64,
    pub w_group_cohesion: f64,
    pub w_ml_prior: f64,
    /// Weight for domain-aware relational/factor priors (motifs, roles, transitions).
    pub w_relational_prior: f64,
    /// Weight for alignment with emergent neurogenesis motifs/centroids.
    pub w_neuro_alignment: f64,
    /// Weight for motif transition priors (previous frame motif -> current motif).
    pub w_motif_transition: f64,
    /// Weight for boosting relational transitions using active super-networks.
    pub w_super_network_prior: f64,
    /// Additional multiplier for factor priors (role/zone/group) when available.
    #[serde(default = "EnergyConfig::default_factor_weight")]
    pub factor_prior_weight: f64,
    pub w_path_feasibility: f64,
    pub w_env_constraints: f64,
    /// Weight for the sequence stack/hash consistency term (set >0 to fuse prior trajectories).
    pub w_stack_hash: f64,
    /// Optional path to a relational priors JSON (from build_relational_priors.py). If absent, the
    /// relational prior term is disabled even if w_relational_prior > 0.
    pub relational_priors_path: Option<String>,
    /// How many history frames to align against when scoring stack similarity (top-k by overlap distance).
    #[serde(default = "EnergyConfig::default_stack_alignment_topk")]
    pub stack_alignment_topk: usize,
    /// How far ahead (in frames) to look when encouraging future consistency from the matched history frame.
    #[serde(default = "EnergyConfig::default_stack_future_horizon")]
    pub stack_future_horizon: usize,
    /// Internal scale for matching the current state to the closest history frame (before the external w_stack_hash weight).
    #[serde(default = "EnergyConfig::default_stack_alignment_weight")]
    pub stack_alignment_weight: f64,
    /// Internal scale for pulling the state toward the future frame of the matched history (gap-filling / forecasting).
    #[serde(default = "EnergyConfig::default_stack_future_weight")]
    pub stack_future_weight: f64,
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
            w_relational_prior: 0.0,
            w_neuro_alignment: 0.0,
            w_motif_transition: 0.0,
            w_super_network_prior: 0.0,
            factor_prior_weight: EnergyConfig::default_factor_weight(),
            w_path_feasibility: 1.0,
            w_env_constraints: 3.0,
            w_stack_hash: 0.0,
            relational_priors_path: None,
            stack_alignment_topk: EnergyConfig::default_stack_alignment_topk(),
            stack_future_horizon: EnergyConfig::default_stack_future_horizon(),
            stack_alignment_weight: EnergyConfig::default_stack_alignment_weight(),
            stack_future_weight: EnergyConfig::default_stack_future_weight(),
            other_terms: HashMap::new(),
        }
    }
}

impl EnergyConfig {
    fn default_stack_alignment_topk() -> usize {
        3
    }

    fn default_stack_future_horizon() -> usize {
        1
    }

    fn default_stack_alignment_weight() -> f64 {
        1.0
    }

    fn default_stack_future_weight() -> f64 {
        0.75
    }

    fn default_factor_weight() -> f64 {
        1.0
    }

    pub fn get_other_term(&self, key: &str) -> Option<f64> {
        self.other_terms.get(key).copied()
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
pub struct QuantumConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "QuantumConfig::default_trotter")]
    pub trotter_slices: usize,
    #[serde(default = "QuantumConfig::default_driver")]
    pub driver_strength: f64,
    #[serde(default = "QuantumConfig::default_driver_final")]
    pub driver_final_strength: f64,
    #[serde(default = "QuantumConfig::default_worldline")]
    pub worldline_mix_prob: f64,
    #[serde(default = "QuantumConfig::default_temp_scale")]
    pub slice_temperature_scale: f64,
}

impl QuantumConfig {
    fn default_trotter() -> usize {
        4
    }

    fn default_driver() -> f64 {
        0.35
    }

    fn default_driver_final() -> f64 {
        0.35
    }

    fn default_worldline() -> f64 {
        0.05
    }

    fn default_temp_scale() -> f64 {
        1.0
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trotter_slices: Self::default_trotter(),
            driver_strength: Self::default_driver(),
            driver_final_strength: Self::default_driver_final(),
            worldline_mix_prob: Self::default_worldline(),
            slice_temperature_scale: Self::default_temp_scale(),
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
    #[serde(default)]
    pub capture_metrics: bool,
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
            capture_metrics: false,
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
    #[serde(default)]
    pub summary_path: Option<PathBuf>,
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
            summary_path: None,
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
pub struct HardwareOverrides {
    #[serde(default = "HardwareOverrides::default_allow_gpu")]
    pub allow_gpu: bool,
    #[serde(default = "HardwareOverrides::default_allow_distributed")]
    pub allow_distributed: bool,
    #[serde(default = "HardwareOverrides::default_max_threads")]
    pub max_threads: Option<usize>,
}

impl HardwareOverrides {
    fn default_allow_gpu() -> bool {
        env_bool("W1Z4RDV1510N_ALLOW_GPU")
            .or_else(|| env_bool("SIMFUTURES_ALLOW_GPU"))
            .unwrap_or(true)
    }

    fn default_allow_distributed() -> bool {
        env_bool("W1Z4RDV1510N_ALLOW_DISTRIBUTED")
            .or_else(|| env_bool("SIMFUTURES_ALLOW_DISTRIBUTED"))
            .unwrap_or(true)
    }

    fn default_max_threads() -> Option<usize> {
        env_usize("W1Z4RDV1510N_MAX_THREADS")
            .or_else(|| env_usize("SIMFUTURES_MAX_THREADS"))
            .and_then(|value| if value == 0 { None } else { Some(value) })
    }
}

impl Default for HardwareOverrides {
    fn default() -> Self {
        Self {
            allow_gpu: Self::default_allow_gpu(),
            allow_distributed: Self::default_allow_distributed(),
            max_threads: Self::default_max_threads(),
        }
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

fn env_bool(key: &str) -> Option<bool> {
    std::env::var(key)
        .ok()
        .and_then(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Some(true),
            "0" | "false" | "no" | "off" => Some(false),
            _ => None,
        })
}

fn env_usize(key: &str) -> Option<usize> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}
