use crate::hardware::HardwareBackendType;
use crate::bridge::{BridgeVerificationMode, ChainKind};
use crate::ml::MLBackendType;
use crate::neuro::NeuroRuntimeConfig;
use crate::objective::GameObjectiveConfig;
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
    #[serde(default)]
    pub objective: GameObjectiveConfig,
    #[serde(default)]
    pub streaming: StreamingConfig,
    #[serde(default)]
    pub compute: ComputeRoutingConfig,
    #[serde(default)]
    pub governance: GovernanceConfig,
    #[serde(default)]
    pub cluster: ClusterConfig,
    #[serde(default)]
    pub ledger: LedgerConfig,
    #[serde(default)]
    pub blockchain: BlockchainConfig,
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
                && self.energy.stack_future_weight >= 0.0
                && self.energy.w_objective >= 0.0,
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
        if self.quantum.remote_enabled {
            anyhow::ensure!(
                self.compute.allow_quantum,
                "quantum.remote_enabled requires compute.allow_quantum"
            );
            anyhow::ensure!(
                !self.compute.quantum_endpoints.is_empty(),
                "quantum.remote_enabled requires compute.quantum_endpoints"
            );
            anyhow::ensure!(
                self.quantum.remote_timeout_secs > 0,
                "quantum.remote_timeout_secs must be > 0"
            );
            anyhow::ensure!(
                self.quantum.remote_trace_samples > 0,
                "quantum.remote_trace_samples must be > 0"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.quantum.calibration_alpha),
                "quantum.calibration_alpha must be in [0,1]"
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
            self.search.constraint_check_every > 0,
            "search.constraint_check_every must be > 0"
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
        if matches!(self.mode, RunMode::Streaming) {
            anyhow::ensure!(
                self.streaming.enabled,
                "streaming.enabled must be true when mode is STREAMING"
            );
        }
        if self.streaming.enabled {
            anyhow::ensure!(
                self.streaming.temporal_tolerance_secs >= 0.0,
                "streaming.temporal_tolerance_secs must be >= 0"
            );
            anyhow::ensure!(
                (0.0..=1.0).contains(&self.streaming.confidence_gate),
                "streaming.confidence_gate must be in [0,1]"
            );
            anyhow::ensure!(
                self.streaming.ingest.enabled_source_count() > 0,
                "streaming ingest must enable at least one source"
            );
            if self.streaming.hypergraph.enabled {
                anyhow::ensure!(
                    self.streaming.hypergraph.max_edges > 0,
                    "streaming.hypergraph.max_edges must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.hypergraph.edge_decay),
                    "streaming.hypergraph.edge_decay must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.hypergraph.edge_ttl_secs >= 0.0,
                    "streaming.hypergraph.edge_ttl_secs must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.hypergraph.min_weight >= 0.0,
                    "streaming.hypergraph.min_weight must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.hypergraph.max_nodes_per_batch > 0,
                    "streaming.hypergraph.max_nodes_per_batch must be > 0"
                );
            }
            if self.streaming.temporal.enabled {
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.temporal.layer_ema_alpha),
                    "streaming.temporal.layer_ema_alpha must be in [0,1]"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.temporal.coherence_alpha),
                    "streaming.temporal.coherence_alpha must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.temporal.event_decay_tau_secs > 0.0,
                    "streaming.temporal.event_decay_tau_secs must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.temporal.base_rate_alpha),
                    "streaming.temporal.base_rate_alpha must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.temporal.dirichlet_prior > 0.0,
                    "streaming.temporal.dirichlet_prior must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.temporal.intensity_floor >= 0.0,
                    "streaming.temporal.intensity_floor must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.temporal.calm_threshold >= 0.0,
                    "streaming.temporal.calm_threshold must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.temporal.surge_threshold
                        >= self.streaming.temporal.calm_threshold,
                    "streaming.temporal.surge_threshold must be >= calm_threshold"
                );
                anyhow::ensure!(
                    self.streaming.temporal.excitation_boost >= 0.0,
                    "streaming.temporal.excitation_boost must be >= 0"
                );
            }
            if self.streaming.spike.enabled {
                anyhow::ensure!(
                    self.streaming.spike.max_neurons_per_pool > 0,
                    "streaming.spike.max_neurons_per_pool must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.max_inputs_per_pool > 0,
                    "streaming.spike.max_inputs_per_pool must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.max_frames_per_snapshot > 0,
                    "streaming.spike.max_frames_per_snapshot must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.spike.membrane_decay),
                    "streaming.spike.membrane_decay must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.spike.threshold > 0.0,
                    "streaming.spike.threshold must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.intensity_norm > 0.0,
                    "streaming.spike.intensity_norm must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.evidence_norm > 0.0,
                    "streaming.spike.evidence_norm must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.hypergraph_node_norm > 0.0,
                    "streaming.spike.hypergraph_node_norm must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.spike.hypergraph_edge_norm > 0.0,
                    "streaming.spike.hypergraph_edge_norm must be > 0"
                );
            }
            if self.streaming.branching.enabled {
                anyhow::ensure!(
                    self.streaming.branching.max_branches > 0,
                    "streaming.branching.max_branches must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.branching.max_depth > 0,
                    "streaming.branching.max_depth must be > 0"
                );
                if self.streaming.branching.retrodiction_enabled {
                    anyhow::ensure!(
                        self.streaming.branching.retrodiction_min_intensity >= 0.0,
                        "streaming.branching.retrodiction_min_intensity must be >= 0"
                    );
                    anyhow::ensure!(
                        self.streaming.branching.retrodiction_max > 0,
                        "streaming.branching.retrodiction_max must be > 0"
                    );
                }
            }
            if self.streaming.causal.enabled {
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.causal.edge_decay),
                    "streaming.causal.edge_decay must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.causal.min_weight >= 0.0,
                    "streaming.causal.min_weight must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.causal.max_edges > 0,
                    "streaming.causal.max_edges must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.causal.max_nodes > 0,
                    "streaming.causal.max_nodes must be > 0"
                );
            }
            if self.streaming.plasticity.enabled {
                anyhow::ensure!(
                    self.streaming.plasticity.surprise_threshold >= 0.0,
                    "streaming.plasticity.surprise_threshold must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.trust_region >= 0.0,
                    "streaming.plasticity.trust_region must be >= 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.plasticity.ema_teacher_alpha),
                    "streaming.plasticity.ema_teacher_alpha must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.drift_threshold >= 0.0,
                    "streaming.plasticity.drift_threshold must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.rollback_threshold >= 0.0,
                    "streaming.plasticity.rollback_threshold must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.reservoir_capacity > 0,
                    "streaming.plasticity.reservoir_capacity must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.horizon_initial_secs > 0.0,
                    "streaming.plasticity.horizon_initial_secs must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.horizon_max_secs
                        >= self.streaming.plasticity.horizon_initial_secs,
                    "streaming.plasticity.horizon_max_secs must be >= horizon_initial_secs"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.horizon_growth_factor >= 1.0,
                    "streaming.plasticity.horizon_growth_factor must be >= 1"
                );
                anyhow::ensure!(
                    self.streaming.plasticity.horizon_improvement_steps > 0,
                    "streaming.plasticity.horizon_improvement_steps must be > 0"
                );
            }
            if self.streaming.ontology.enabled {
                anyhow::ensure!(
                    self.streaming.ontology.window_minutes > 0,
                    "streaming.ontology.window_minutes must be > 0"
                );
                anyhow::ensure!(
                    !self.streaming.ontology.version_prefix.trim().is_empty(),
                    "streaming.ontology.version_prefix must be non-empty"
                );
            }
            if self.streaming.physiology.enabled {
                anyhow::ensure!(
                    self.streaming.physiology.min_samples > 0,
                    "streaming.physiology.min_samples must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.physiology.update_alpha),
                    "streaming.physiology.update_alpha must be in [0,1]"
                );
                anyhow::ensure!(
                    self.streaming.physiology.covariance_floor >= 0.0,
                    "streaming.physiology.covariance_floor must be >= 0"
                );
                anyhow::ensure!(
                    self.streaming.physiology.max_deviation_update > 0.0,
                    "streaming.physiology.max_deviation_update must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.physiology.max_templates > 0,
                    "streaming.physiology.max_templates must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&self.streaming.physiology.prior_strength),
                    "streaming.physiology.prior_strength must be in [0,1]"
                );
            }
            if self.streaming.analysis.enabled {
                anyhow::ensure!(
                    self.streaming.analysis.interval_batches > 0,
                    "streaming.analysis.interval_batches must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.analysis.max_tokens > 0,
                    "streaming.analysis.max_tokens must be > 0"
                );
                anyhow::ensure!(
                    self.streaming.analysis.max_layers > 0,
                    "streaming.analysis.max_layers must be > 0"
                );
            }
        }
        if self.compute.allow_quantum {
            for endpoint in &self.compute.quantum_endpoints {
                anyhow::ensure!(
                    !endpoint.url.trim().is_empty(),
                    "compute.quantum_endpoints url must be non-empty"
                );
                anyhow::ensure!(
                    endpoint.timeout_secs > 0,
                    "compute.quantum_endpoints timeout_secs must be > 0"
                );
            }
        }
        if self.blockchain.enabled {
            anyhow::ensure!(
                !self.blockchain.chain_id.trim().is_empty(),
                "blockchain.chain_id must be non-empty when enabled"
            );
            anyhow::ensure!(
                !self.blockchain.consensus.trim().is_empty(),
                "blockchain.consensus must be non-empty when enabled"
            );
            if self.blockchain.require_sensor_attestation {
                anyhow::ensure!(
                    !self.blockchain.attestation.endpoint.trim().is_empty(),
                    "blockchain.attestation.endpoint must be set when sensor attestation is required"
                );
            }
            anyhow::ensure!(
                self.blockchain.validator_policy.heartbeat_interval_secs > 0,
                "validator_policy.heartbeat_interval_secs must be > 0"
            );
            anyhow::ensure!(
                self.blockchain.validator_policy.max_missed_heartbeats > 0,
                "validator_policy.max_missed_heartbeats must be > 0"
            );
            anyhow::ensure!(
                self.blockchain.validator_policy.downtime_penalty_score >= 0.0,
                "validator_policy.downtime_penalty_score must be >= 0"
            );
            if self.blockchain.fee_market.enabled {
                let fee = &self.blockchain.fee_market;
                anyhow::ensure!(
                    fee.base_fee.is_finite()
                        && fee.min_base_fee.is_finite()
                        && fee.max_base_fee.is_finite(),
                    "fee_market base_fee/min_base_fee/max_base_fee must be finite"
                );
                anyhow::ensure!(
                    fee.min_base_fee >= 0.0,
                    "fee_market.min_base_fee must be >= 0"
                );
                anyhow::ensure!(
                    fee.min_base_fee <= fee.max_base_fee,
                    "fee_market.min_base_fee must be <= fee_market.max_base_fee"
                );
                anyhow::ensure!(
                    fee.base_fee >= fee.min_base_fee && fee.base_fee <= fee.max_base_fee,
                    "fee_market.base_fee must be within [min_base_fee, max_base_fee]"
                );
                anyhow::ensure!(
                    fee.target_txs_per_window > 0,
                    "fee_market.target_txs_per_window must be > 0"
                );
                anyhow::ensure!(
                    fee.window_secs > 0,
                    "fee_market.window_secs must be > 0"
                );
                anyhow::ensure!(
                    (0.0..=1.0).contains(&fee.adjustment_rate),
                    "fee_market.adjustment_rate must be in [0,1]"
                );
            }
            if self.blockchain.bridge.enabled {
                let bridge = &self.blockchain.bridge;
                anyhow::ensure!(
                    bridge.max_proof_bytes > 0,
                    "bridge.max_proof_bytes must be > 0"
                );
                anyhow::ensure!(
                    !bridge.chains.is_empty(),
                    "bridge.chains must be non-empty when bridge is enabled"
                );
                for chain in &bridge.chains {
                    anyhow::ensure!(
                        !chain.chain_id.trim().is_empty(),
                        "bridge chain_id must be non-empty"
                    );
                    anyhow::ensure!(
                        chain.min_confirmations > 0,
                        "bridge.min_confirmations must be > 0"
                    );
                    anyhow::ensure!(
                        chain.relayer_quorum > 0,
                        "bridge.relayer_quorum must be > 0"
                    );
                    anyhow::ensure!(
                        chain.max_deposit_amount.is_finite() && chain.max_deposit_amount > 0.0,
                        "bridge.max_deposit_amount must be > 0 and finite"
                    );
                    anyhow::ensure!(
                        !chain.allowed_assets.is_empty(),
                        "bridge.allowed_assets must be non-empty"
                    );
                    if let Some(address) = &chain.deposit_address {
                        anyhow::ensure!(
                            !address.trim().is_empty(),
                            "bridge.deposit_address must be non-empty when set"
                        );
                    }
                    if let Some(template) = &chain.recipient_tag_template {
                        anyhow::ensure!(
                            !template.trim().is_empty(),
                            "bridge.recipient_tag_template must be non-empty when set"
                        );
                    }
                    if matches!(chain.verification, BridgeVerificationMode::RelayerQuorum) {
                        anyhow::ensure!(
                            chain.relayer_public_keys.len() as u32 >= chain.relayer_quorum,
                            "bridge.relayer_public_keys must satisfy relayer_quorum"
                        );
                    }
                }
            }
        }
        if self.governance.enforce_public_only {
            anyhow::ensure!(
                self.governance.disable_face_id,
                "governance.disable_face_id must be true when enforce_public_only is set"
            );
            anyhow::ensure!(
                self.governance.disable_pii,
                "governance.disable_pii must be true when enforce_public_only is set"
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
    /// Weight for game-objective scoring (win/step penalties).
    #[serde(default)]
    pub w_objective: f64,
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
            w_objective: 0.0,
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
    #[serde(default)]
    pub remote_enabled: bool,
    #[serde(default = "QuantumConfig::default_remote_timeout")]
    pub remote_timeout_secs: u64,
    #[serde(default = "QuantumConfig::default_remote_trace_samples")]
    pub remote_trace_samples: usize,
    #[serde(default = "QuantumConfig::default_calibration_alpha")]
    pub calibration_alpha: f64,
    #[serde(default)]
    pub calibration_path: Option<PathBuf>,
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

    fn default_remote_timeout() -> u64 {
        45
    }

    fn default_remote_trace_samples() -> usize {
        128
    }

    fn default_calibration_alpha() -> f64 {
        0.2
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
            remote_enabled: false,
            remote_timeout_secs: Self::default_remote_timeout(),
            remote_trace_samples: Self::default_remote_trace_samples(),
            calibration_alpha: Self::default_calibration_alpha(),
            calibration_path: None,
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
    #[serde(default)]
    pub live_frame_path: Option<PathBuf>,
    #[serde(default = "LoggingConfig::default_live_frame_every")]
    pub live_frame_every: usize,
    #[serde(default)]
    pub live_neuro_path: Option<PathBuf>,
    #[serde(default = "LoggingConfig::default_live_neuro_every")]
    pub live_neuro_every: usize,
}

impl LoggingConfig {
    fn default_level() -> String {
        "INFO".to_string()
    }

    fn default_live_frame_every() -> usize {
        1
    }

    fn default_live_neuro_every() -> usize {
        1
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            log_level: Self::default_level(),
            log_path: None,
            json: false,
            capture_metrics: false,
            live_frame_path: None,
            live_frame_every: Self::default_live_frame_every(),
            live_neuro_path: None,
            live_neuro_every: Self::default_live_neuro_every(),
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
    Streaming,
}

impl Default for RunMode {
    fn default() -> Self {
        RunMode::Production
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingConfig {
    pub enabled: bool,
    pub ingest: StreamingIngestConfig,
    /// Temporal alignment tolerance for token/layer fusion in seconds.
    pub temporal_tolerance_secs: f64,
    /// Confidence gate used for per-modality fusion in [0,1].
    pub confidence_gate: f64,
    #[serde(default)]
    pub layer_flags: StreamingLayerFlags,
    #[serde(default)]
    pub hypergraph: StreamingHypergraphConfig,
    #[serde(default)]
    pub temporal: TemporalInferenceConfig,
    #[serde(default)]
    pub spike: StreamingSpikeConfig,
    #[serde(default)]
    pub branching: BranchingFuturesConfig,
    #[serde(default)]
    pub causal: CausalDiscoveryConfig,
    #[serde(default)]
    pub consistency: ConsistencyChunkingConfig,
    #[serde(default)]
    pub plasticity: OnlinePlasticityConfig,
    #[serde(default)]
    pub ontology: OntologyConfig,
    #[serde(default)]
    pub physiology: PhysiologyConfig,
    #[serde(default)]
    pub analysis: StreamingAnalysisConfig,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            ingest: StreamingIngestConfig::default(),
            temporal_tolerance_secs: 2.0,
            confidence_gate: 0.5,
            layer_flags: StreamingLayerFlags::default(),
            hypergraph: StreamingHypergraphConfig::default(),
            temporal: TemporalInferenceConfig::default(),
            spike: StreamingSpikeConfig::default(),
            branching: BranchingFuturesConfig::default(),
            causal: CausalDiscoveryConfig::default(),
            consistency: ConsistencyChunkingConfig::default(),
            plasticity: OnlinePlasticityConfig::default(),
            ontology: OntologyConfig::default(),
            physiology: PhysiologyConfig::default(),
            analysis: StreamingAnalysisConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingAnalysisConfig {
    pub enabled: bool,
    pub interval_batches: usize,
    pub max_tokens: usize,
    pub max_layers: usize,
}

impl Default for StreamingAnalysisConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_batches: 3,
            max_tokens: 256,
            max_layers: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingHypergraphConfig {
    pub enabled: bool,
    pub max_edges: usize,
    pub edge_decay: f64,
    pub edge_ttl_secs: f64,
    pub min_weight: f64,
    pub max_nodes_per_batch: usize,
}

impl Default for StreamingHypergraphConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_edges: 20_000,
            edge_decay: 0.98,
            edge_ttl_secs: 3600.0,
            min_weight: 0.02,
            max_nodes_per_batch: 48,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TemporalInferenceConfig {
    pub enabled: bool,
    pub layer_ema_alpha: f64,
    pub coherence_alpha: f64,
    pub event_decay_tau_secs: f64,
    pub base_rate_alpha: f64,
    pub dirichlet_prior: f64,
    pub intensity_floor: f64,
    pub calm_threshold: f64,
    pub surge_threshold: f64,
    pub excitation_boost: f64,
}

impl Default for TemporalInferenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            layer_ema_alpha: 0.2,
            coherence_alpha: 0.15,
            event_decay_tau_secs: 300.0,
            base_rate_alpha: 0.2,
            dirichlet_prior: 1.0,
            intensity_floor: 0.01,
            calm_threshold: 0.5,
            surge_threshold: 2.0,
            excitation_boost: 0.05,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingIngestConfig {
    pub people_video: bool,
    pub crowd_traffic: bool,
    pub public_topics: bool,
}

impl StreamingIngestConfig {
    pub fn enabled_source_count(&self) -> usize {
        self.people_video as usize + self.crowd_traffic as usize + self.public_topics as usize
    }
}

impl Default for StreamingIngestConfig {
    fn default() -> Self {
        Self {
            people_video: false,
            crowd_traffic: false,
            public_topics: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingLayerFlags {
    pub ultradian_enabled: bool,
    pub flow_enabled: bool,
    pub topic_event_enabled: bool,
    pub physiology_enabled: bool,
    pub behavior_enabled: bool,
}

impl Default for StreamingLayerFlags {
    fn default() -> Self {
        Self {
            ultradian_enabled: false,
            flow_enabled: true,
            topic_event_enabled: true,
            physiology_enabled: false,
            behavior_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct StreamingSpikeConfig {
    pub enabled: bool,
    pub max_neurons_per_pool: usize,
    pub max_inputs_per_pool: usize,
    pub embed_in_snapshot: bool,
    pub max_frames_per_snapshot: usize,
    pub threshold: f32,
    pub membrane_decay: f32,
    pub refractory_steps: u32,
    pub intensity_norm: f64,
    pub evidence_norm: f64,
    pub hypergraph_node_norm: f64,
    pub hypergraph_edge_norm: f64,
}

impl Default for StreamingSpikeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_neurons_per_pool: 256,
            max_inputs_per_pool: 256,
            embed_in_snapshot: false,
            max_frames_per_snapshot: 64,
            threshold: 1.0,
            membrane_decay: 0.95,
            refractory_steps: 2,
            intensity_norm: 5.0,
            evidence_norm: 10.0,
            hypergraph_node_norm: 200.0,
            hypergraph_edge_norm: 500.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BranchingFuturesConfig {
    pub enabled: bool,
    pub max_branches: usize,
    pub max_depth: usize,
    pub retrodiction_enabled: bool,
    pub retrodiction_min_intensity: f64,
    pub retrodiction_max: usize,
}

impl Default for BranchingFuturesConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_branches: 64,
            max_depth: 6,
            retrodiction_enabled: true,
            retrodiction_min_intensity: 0.8,
            retrodiction_max: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CausalDiscoveryConfig {
    pub enabled: bool,
    pub max_lag_secs: f64,
    pub intervention_enabled: bool,
    pub min_weight: f64,
    pub edge_decay: f64,
    pub max_edges: usize,
    pub max_nodes: usize,
}

impl Default for CausalDiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_lag_secs: 3600.0,
            intervention_enabled: false,
            min_weight: 0.05,
            edge_decay: 0.98,
            max_edges: 2048,
            max_nodes: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ConsistencyChunkingConfig {
    pub enabled: bool,
    pub novelty_threshold: f64,
    pub uncertainty_threshold: f64,
    pub min_support: usize,
}

impl Default for ConsistencyChunkingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            novelty_threshold: 0.6,
            uncertainty_threshold: 0.5,
            min_support: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OnlinePlasticityConfig {
    pub enabled: bool,
    pub surprise_threshold: f64,
    pub trust_region: f64,
    pub ema_teacher_alpha: f64,
    pub drift_threshold: f64,
    pub rollback_threshold: f64,
    pub reservoir_capacity: usize,
    pub horizon_initial_secs: f64,
    pub horizon_max_secs: f64,
    pub horizon_growth_factor: f64,
    pub horizon_improvement_steps: usize,
}

impl Default for OnlinePlasticityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            surprise_threshold: 0.7,
            trust_region: 0.1,
            ema_teacher_alpha: 0.98,
            drift_threshold: 0.4,
            rollback_threshold: 0.1,
            reservoir_capacity: 256,
            horizon_initial_secs: 60.0,
            horizon_max_secs: 3600.0,
            horizon_growth_factor: 1.25,
            horizon_improvement_steps: 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OntologyConfig {
    pub enabled: bool,
    pub window_minutes: usize,
    pub version_prefix: String,
}

impl Default for OntologyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            window_minutes: 60,
            version_prefix: "v".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysiologyConfig {
    pub enabled: bool,
    pub min_samples: usize,
    pub update_alpha: f64,
    pub covariance_floor: f64,
    pub max_deviation_update: f64,
    pub max_templates: usize,
    pub prior_strength: f64,
}

impl Default for PhysiologyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_samples: 5,
            update_alpha: 0.2,
            covariance_floor: 1e-3,
            max_deviation_update: 3.0,
            max_templates: 4096,
            prior_strength: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ComputeRoutingConfig {
    pub allow_gpu: bool,
    pub allow_quantum: bool,
    pub quantum_endpoints: Vec<QuantumEndpointConfig>,
}

impl Default for ComputeRoutingConfig {
    fn default() -> Self {
        Self {
            allow_gpu: true,
            allow_quantum: false,
            quantum_endpoints: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QuantumEndpointConfig {
    pub name: String,
    pub url: String,
    pub timeout_secs: u64,
    pub provider: String,
    pub priority: u8,
    pub auth_env: Option<String>,
    pub auth_header: String,
    pub auth_prefix: String,
}

impl Default for QuantumEndpointConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            url: String::new(),
            timeout_secs: 30,
            provider: "generic".to_string(),
            priority: 50,
            auth_env: None,
            auth_header: "Authorization".to_string(),
            auth_prefix: "Bearer ".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GovernanceConfig {
    pub disable_face_id: bool,
    pub disable_pii: bool,
    pub enforce_public_only: bool,
    pub immutable_audit_logs: bool,
    pub explainability_on: bool,
    pub disable_high_risk_interventions: bool,
    pub federated_learning_enabled: bool,
    pub dp_enabled: bool,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            disable_face_id: true,
            disable_pii: true,
            enforce_public_only: true,
            immutable_audit_logs: true,
            explainability_on: true,
            disable_high_risk_interventions: true,
            federated_learning_enabled: false,
            dp_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClusterConfig {
    pub enabled: bool,
    pub mode: String,
    pub min_nodes: usize,
    pub openstack_minimal: bool,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "local".to_string(),
            min_nodes: 1,
            openstack_minimal: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerConfig {
    pub enabled: bool,
    pub backend: String,
    pub endpoint: String,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: "none".to_string(),
            endpoint: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum NodeRole {
    Validator,
    Worker,
    Sensor,
}

impl Default for NodeRole {
    fn default() -> Self {
        NodeRole::Worker
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BridgeConfig {
    pub enabled: bool,
    pub challenge_window_secs: u64,
    pub max_proof_bytes: usize,
    pub chains: Vec<BridgeChainPolicy>,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            challenge_window_secs: 3600,
            max_proof_bytes: 262_144,
            chains: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BridgeChainPolicy {
    pub chain_id: String,
    #[serde(default)]
    pub chain_kind: ChainKind,
    #[serde(default)]
    pub verification: BridgeVerificationMode,
    pub min_confirmations: u32,
    pub relayer_quorum: u32,
    #[serde(default)]
    pub relayer_public_keys: Vec<String>,
    #[serde(default)]
    pub allowed_assets: Vec<String>,
    #[serde(default)]
    pub deposit_address: Option<String>,
    #[serde(default)]
    pub recipient_tag_template: Option<String>,
    pub max_deposit_amount: f64,
}

impl Default for BridgeChainPolicy {
    fn default() -> Self {
        Self {
            chain_id: String::new(),
            chain_kind: ChainKind::default(),
            verification: BridgeVerificationMode::default(),
            min_confirmations: 1,
            relayer_quorum: 2,
            relayer_public_keys: Vec::new(),
            allowed_assets: Vec::new(),
            deposit_address: None,
            recipient_tag_template: None,
            max_deposit_amount: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BlockchainConfig {
    pub enabled: bool,
    pub chain_id: String,
    pub consensus: String,
    pub bootstrap_peers: Vec<String>,
    #[serde(default)]
    pub node_role: NodeRole,
    #[serde(default)]
    pub reward_policy: RewardPolicyConfig,
    #[serde(default)]
    pub energy_efficiency: EnergyEfficiencyConfig,
    #[serde(default)]
    pub attestation: SensorAttestationConfig,
    #[serde(default)]
    pub require_sensor_attestation: bool,
    #[serde(default)]
    pub validator_policy: ValidatorPolicyConfig,
    #[serde(default)]
    pub fee_market: FeeMarketConfig,
    #[serde(default)]
    pub bridge: BridgeConfig,
}

impl Default for BlockchainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            chain_id: "w1z4rdv1510n-l1".to_string(),
            consensus: "poa".to_string(),
            bootstrap_peers: Vec::new(),
            node_role: NodeRole::default(),
            reward_policy: RewardPolicyConfig::default(),
            energy_efficiency: EnergyEfficiencyConfig::default(),
            attestation: SensorAttestationConfig::default(),
            require_sensor_attestation: false,
            validator_policy: ValidatorPolicyConfig::default(),
            fee_market: FeeMarketConfig::default(),
            bridge: BridgeConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RewardPolicyConfig {
    pub sensor_reward_weight: f64,
    pub compute_reward_weight: f64,
    pub energy_efficiency_weight: f64,
    pub uptime_reward_weight: f64,
}

impl Default for RewardPolicyConfig {
    fn default() -> Self {
        Self {
            sensor_reward_weight: 1.0,
            compute_reward_weight: 1.0,
            energy_efficiency_weight: 1.0,
            uptime_reward_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EnergyEfficiencyConfig {
    pub target_watts: f64,
    pub efficiency_baseline: f64,
}

impl Default for EnergyEfficiencyConfig {
    fn default() -> Self {
        Self {
            target_watts: 150.0,
            efficiency_baseline: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SensorAttestationConfig {
    pub endpoint: String,
    pub required: bool,
}

impl Default for SensorAttestationConfig {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            required: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ValidatorPolicyConfig {
    pub heartbeat_interval_secs: u64,
    pub max_missed_heartbeats: u32,
    pub jail_duration_secs: u64,
    pub downtime_penalty_score: f64,
}

impl Default for ValidatorPolicyConfig {
    fn default() -> Self {
        Self {
            heartbeat_interval_secs: 30,
            max_missed_heartbeats: 5,
            jail_duration_secs: 600,
            downtime_penalty_score: 1.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct FeeMarketConfig {
    pub enabled: bool,
    pub base_fee: f64,
    pub min_base_fee: f64,
    pub max_base_fee: f64,
    pub target_txs_per_window: u32,
    pub window_secs: u64,
    pub adjustment_rate: f64,
}

impl Default for FeeMarketConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            base_fee: 0.0,
            min_base_fee: 0.0,
            max_base_fee: 10.0,
            target_txs_per_window: 100,
            window_secs: 60,
            adjustment_rate: 0.125,
        }
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

#[cfg(test)]
mod tests {
    use super::RunConfig;

    #[test]
    fn quantum_remote_requires_endpoints() {
        let raw = r#"{
            "t_end": { "unix": 0 },
            "quantum": { "enabled": true, "remote_enabled": true },
            "compute": { "allow_quantum": true, "quantum_endpoints": [] }
        }"#;
        let config: RunConfig = serde_json::from_str(raw).expect("config");
        let err = config.validate().expect_err("should fail");
        assert!(
            err.to_string().contains("quantum.remote_enabled requires compute.quantum_endpoints")
        );
    }

    #[test]
    fn quantum_remote_accepts_endpoint() {
        let raw = r#"{
            "t_end": { "unix": 0 },
            "quantum": { "enabled": true, "remote_enabled": true },
            "compute": {
                "allow_quantum": true,
                "quantum_endpoints": [
                    { "name": "test", "url": "http://localhost:5050/quantum/submit" }
                ]
            }
        }"#;
        let config: RunConfig = serde_json::from_str(raw).expect("config");
        config.validate().expect("valid config");
    }
}
