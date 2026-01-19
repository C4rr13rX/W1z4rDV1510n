use crate::chain::ChainSpec;
use crate::config::NodeConfig;
use crate::data_mesh::{DataMeshEvent, DataMeshHandle, start_data_mesh};
use crate::ledger::LocalLedger;
use crate::openstack::OpenStackControlPlane;
use crate::paths::node_data_dir;
use crate::wallet::{WalletSigner, WalletStore, node_id_from_wallet};
use crate::p2p::NodeNetwork;
use anyhow::{Result, anyhow};
use std::fs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{info, warn};
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, EnergyEfficiencySample, NoopLedger, NodeCapabilities, NodeRegistration,
    ValidatorHeartbeat, WorkKind, WorkProof,
};
use w1z4rdv1510n::compute::{ComputeJobKind, ComputeRouter, ComputeTarget};
use w1z4rdv1510n::config::NodeRole;
use w1z4rdv1510n::hardware::HardwareProfile;
use w1z4rdv1510n::network::compute_payload_hash;
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::{NeuralFabricShare, StreamEnvelope, StreamingInference};
use w1z4rdv1510n::config::RunConfig;
use serde_json::Value;

pub struct NodeRuntime {
    config: NodeConfig,
    chain_spec: ChainSpec,
    compute_router: ComputeRouter,
    network: NodeNetwork,
    ledger: Arc<dyn BlockchainLedger>,
    profile: HardwareProfile,
    openstack: Option<OpenStackControlPlane>,
    wallet: Arc<WalletSigner>,
    data_mesh: Option<DataMeshHandle>,
}

impl NodeRuntime {
    pub fn new(mut config: NodeConfig) -> Result<Self> {
        let wallet = Arc::new(WalletStore::load_or_create_signer(&config.wallet)?);
        if config.network.bootstrap_peers.is_empty() && !config.blockchain.bootstrap_peers.is_empty()
        {
            config.network.bootstrap_peers = config.blockchain.bootstrap_peers.clone();
        }
        if config.node_id.trim().is_empty() || config.node_id == "node-001" {
            config.node_id = node_id_from_wallet(&wallet.wallet().address);
        }
        let chain_spec = ChainSpec::load(&config.chain_spec)?;
        let profile = HardwareProfile::detect();
        let compute_router = ComputeRouter::new(config.compute.clone(), config.cluster.clone());
        let network = NodeNetwork::new(config.network.clone());
        let ledger = build_ledger(&config)?;
        let openstack = if config.openstack.enabled {
            Some(OpenStackControlPlane::new(config.openstack.clone()))
        } else {
            None
        };
        Ok(Self {
            config,
            chain_spec,
            compute_router,
            network,
            ledger,
            profile,
            openstack,
            wallet,
            data_mesh: None,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.network.start(&self.config.node_id)?;
        self.network.connect_bootstrap()?;
        self.start_data_mesh()?;
        self.start_streaming_runtime()?;
        self.register_node()?;
        self.maybe_send_validator_heartbeat()?;
        self.start_validator_heartbeat_loop();
        let target = self.compute_router.route(ComputeJobKind::TensorHeavy);
        info!(
            target: "w1z4rdv1510n::node",
            node_id = self.config.node_id,
            peer_count = self.network.peer_count(),
            compute_target = ?target,
            chain_id = self.chain_spec.chain_id,
            wallet_address = self.wallet.wallet().address.as_str(),
            "node runtime started"
        );
        Ok(())
    }

    pub fn run_until_shutdown(mut self) -> Result<()> {
        self.start()?;
        wait_for_shutdown()?;
        Ok(())
    }

    fn start_data_mesh(&mut self) -> Result<()> {
        if !self.config.data.enabled {
            return Ok(());
        }
        let Some(publisher) = self.network.publisher() else {
            return Err(anyhow!("p2p network not started"));
        };
        let Some(network_rx) = self.network.take_message_receiver() else {
            return Err(anyhow!("p2p message receiver unavailable"));
        };
        let handle = start_data_mesh(
            self.config.data.clone(),
            self.config.node_id.clone(),
            self.wallet.clone(),
            self.ledger.clone(),
            publisher,
            network_rx,
        )?;
        self.data_mesh = Some(handle);
        Ok(())
    }

    fn start_streaming_runtime(&mut self) -> Result<()> {
        let streaming = self.config.streaming.clone();
        if !streaming.enabled {
            return Ok(());
        }
        if !streaming.ultradian_node {
            warn!(
                target: "w1z4rdv1510n::node",
                "streaming runtime disabled; ultradian_node flag is false"
            );
            return Ok(());
        }
        let Some(mesh) = self.data_mesh.clone() else {
            return Err(anyhow!("streaming runtime requires data mesh"));
        };
        let run_config_path = streaming.run_config_path.clone();
        let node_id = self.config.node_id.clone();
        let cpu_cores = self.profile.cpu_cores;
        let memory_gb = self.profile.total_memory_gb;
        let profile = self.profile.clone();
        let ledger = self.ledger.clone();
        let wallet = self.wallet.clone();
        let compute_config = self.config.compute.clone();
        let cluster_config = self.config.cluster.clone();
        let energy_reporting = self.config.energy_reporting.clone();
        let can_process_streams =
            cpu_cores >= streaming.min_cpu_cores && memory_gb >= streaming.min_memory_gb;
        let consume_streams = streaming.consume_streams && can_process_streams;
        let consume_shares = streaming.consume_shares;
        let publish_shares = streaming.publish_shares;
        if streaming.consume_streams && !can_process_streams {
            warn!(
                target: "w1z4rdv1510n::node",
                cpu_cores,
                memory_gb,
                "streaming runtime disabled stream processing due to resource limits"
            );
        }
        let stream_kind = streaming.stream_payload_kind.clone();
        let share_kind = streaming.share_payload_kind.clone();
        std::thread::spawn(move || {
            let raw = match fs::read_to_string(&run_config_path) {
                Ok(raw) => raw,
                Err(err) => {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        error = %err,
                        path = %run_config_path,
                        "failed to read streaming run config"
                    );
                    return;
                }
            };
            let run_config: RunConfig = match serde_json::from_str(&raw) {
                Ok(config) => config,
                Err(err) => {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        error = %err,
                        path = %run_config_path,
                        "failed to parse streaming run config"
                    );
                    return;
                }
            };
            if !run_config.streaming.enabled {
                warn!(
                    target: "w1z4rdv1510n::node",
                    path = %run_config_path,
                    "streaming.enabled must be true in run config"
                );
                return;
            }
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        error = %err,
                        "failed to start streaming runtime"
                    );
                    return;
                }
            };
            runtime.block_on(async move {
                let mut inference = StreamingInference::new(run_config);
                let compute_router = ComputeRouter::new(compute_config, cluster_config);
                let mut processed_queue: VecDeque<String> = VecDeque::new();
                let mut processed_set: HashSet<String> = HashSet::new();
                let max_processed = 4096usize;
                let mut events = mesh.subscribe();
                loop {
                    let event = match events.recv().await {
                        Ok(event) => event,
                        Err(_) => continue,
                    };
                    let DataMeshEvent::PayloadReady {
                        data_id,
                        payload_kind,
                        payload_hash,
                        node_id: origin_node_id,
                        sensor_id,
                        ..
                    } = event;
                    if payload_kind == stream_kind && consume_streams {
                        if processed_set.contains(&data_id) {
                            continue;
                        }
                        processed_set.insert(data_id.clone());
                        processed_queue.push_back(data_id.clone());
                        while processed_queue.len() > max_processed {
                            if let Some(old) = processed_queue.pop_front() {
                                processed_set.remove(&old);
                            }
                        }
                        let payload = match mesh.load_payload(data_id.clone()).await {
                            Ok(payload) => payload,
                            Err(_) => continue,
                        };
                        let envelope = match serde_json::from_slice::<StreamEnvelope>(&payload) {
                            Ok(envelope) => envelope,
                            Err(_) => continue,
                        };
                        let payload_bytes = payload.len();
                        let input_hash = if payload_hash.trim().is_empty() {
                            compute_payload_hash(&payload)
                        } else {
                            payload_hash
                        };
                        let start = Instant::now();
                        let outcome = match inference.handle_envelope(envelope) {
                            Ok(outcome) => outcome,
                            Err(_) => None,
                        };
                        if outcome.is_none() {
                            continue;
                        }
                        let elapsed_secs = start.elapsed().as_secs_f64().max(0.001);
                        let compute_target = compute_router.route(ComputeJobKind::TensorHeavy);
                        let watts = estimate_watts(&profile, compute_target);
                        let energy_joules = watts * elapsed_secs;
                        let throughput = payload_bytes as f64 / elapsed_secs;
                        if energy_reporting.enabled {
                            let sample = EnergyEfficiencySample {
                                node_id: node_id.clone(),
                                timestamp: now_timestamp(),
                                watts,
                                throughput,
                            };
                            if let Err(err) = ledger.submit_energy_sample(sample) {
                                warn!(
                                    target: "w1z4rdv1510n::node",
                                    error = %err,
                                    "energy sample rejected"
                                );
                            }
                        }
                        let mut output_hash = None;
                        if publish_shares {
                            if let Some(mut share) =
                                inference.take_last_fabric_share(node_id.clone())
                            {
                                share.metadata.insert(
                                    "source_data_id".to_string(),
                                    Value::String(data_id.clone()),
                                );
                                share.metadata.insert(
                                    "source_payload_hash".to_string(),
                                    Value::String(input_hash.clone()),
                                );
                                share.metadata.insert(
                                    "source_node_id".to_string(),
                                    Value::String(origin_node_id.clone()),
                                );
                                share.metadata.insert(
                                    "source_sensor_id".to_string(),
                                    Value::String(sensor_id.clone()),
                                );
                                share.metadata.insert(
                                    "compute_node_id".to_string(),
                                    Value::String(node_id.clone()),
                                );
                                share.metadata.insert(
                                    "compute_target".to_string(),
                                    Value::String(format!("{compute_target:?}")),
                                );
                                share.metadata.insert(
                                    "compute_elapsed_secs".to_string(),
                                    Value::from(elapsed_secs),
                                );
                                share.metadata.insert(
                                    "energy_joules".to_string(),
                                    Value::from(energy_joules),
                                );
                                share.metadata.insert(
                                    "energy_watts_est".to_string(),
                                    Value::from(watts),
                                );
                                share.metadata.insert(
                                    "throughput_bps".to_string(),
                                    Value::from(throughput),
                                );
                                share.metadata.insert(
                                    "energy_estimated".to_string(),
                                    Value::from(true),
                                );
                                if let Ok(serialized) = serde_json::to_vec(&share) {
                                    output_hash = Some(compute_payload_hash(&serialized));
                                }
                                let _ = mesh
                                    .ingest_fabric_share_with_kind(&share, &share_kind)
                                    .await;
                            }
                        }
                        let score = compute_reward_score(energy_joules, payload_bytes);
                        let completed_at = now_timestamp();
                        let work_id = work_id_for(
                            &node_id,
                            &data_id,
                            &input_hash,
                            output_hash.as_deref(),
                            completed_at,
                        );
                        let mut metrics = HashMap::new();
                        metrics.insert(
                            "stream_data_id".to_string(),
                            Value::String(data_id.clone()),
                        );
                        metrics.insert(
                            "input_payload_hash".to_string(),
                            Value::String(input_hash),
                        );
                        metrics.insert(
                            "input_payload_bytes".to_string(),
                            Value::from(payload_bytes as u64),
                        );
                        if let Some(hash) = output_hash {
                            metrics.insert(
                                "output_payload_hash".to_string(),
                                Value::String(hash),
                            );
                        }
                        metrics.insert(
                            "origin_node_id".to_string(),
                            Value::String(origin_node_id),
                        );
                        metrics.insert(
                            "origin_sensor_id".to_string(),
                            Value::String(sensor_id),
                        );
                        metrics.insert(
                            "compute_target".to_string(),
                            Value::String(format!("{compute_target:?}")),
                        );
                        metrics.insert(
                            "elapsed_secs".to_string(),
                            Value::from(elapsed_secs),
                        );
                        metrics.insert(
                            "energy_joules".to_string(),
                            Value::from(energy_joules),
                        );
                        metrics.insert(
                            "energy_watts_est".to_string(),
                            Value::from(watts),
                        );
                        metrics.insert(
                            "throughput_bps".to_string(),
                            Value::from(throughput),
                        );
                        metrics.insert(
                            "energy_estimated".to_string(),
                            Value::from(true),
                        );
                        metrics.insert(
                            "hardware_cpu_cores".to_string(),
                            Value::from(profile.cpu_cores as u64),
                        );
                        metrics.insert(
                            "hardware_memory_gb".to_string(),
                            Value::from(profile.total_memory_gb),
                        );
                        metrics.insert(
                            "hardware_has_gpu".to_string(),
                            Value::from(profile.has_gpu),
                        );
                        let proof = WorkProof {
                            work_id,
                            node_id: node_id.clone(),
                            kind: WorkKind::ComputeTask,
                            completed_at,
                            score,
                            fee_paid: 0.0,
                            metrics,
                            signature: String::new(),
                        };
                        let signed = wallet.sign_work_proof(proof);
                        if let Err(err) = ledger.submit_work_proof(signed) {
                            warn!(
                                target: "w1z4rdv1510n::node",
                                error = %err,
                                "work proof rejected"
                            );
                        }
                    } else if payload_kind == share_kind && consume_shares {
                        if let Ok(payload) = mesh.load_payload(data_id).await {
                            if let Ok(share) = serde_json::from_slice::<NeuralFabricShare>(&payload) {
                                if share.node_id == node_id {
                                    continue;
                                }
                                let _ = inference.handle_fabric_share(share);
                            }
                        }
                    }
                }
            });
        });
        info!(
            target: "w1z4rdv1510n::node",
            streaming_enabled = streaming.enabled,
            consume_streams,
            consume_shares,
            publish_shares,
            "streaming runtime started"
        );
        Ok(())
    }

    fn register_node(&self) -> Result<()> {
        if !self.config.blockchain.enabled {
            warn!(
                target: "w1z4rdv1510n::node",
                "blockchain disabled; skipping registration"
            );
            return Ok(());
        }
        let capabilities = NodeCapabilities {
            cpu_cores: self.profile.cpu_cores,
            memory_gb: self.profile.total_memory_gb,
            gpu_count: if self.profile.has_gpu { 1 } else { 0 },
        };
        let registration = NodeRegistration {
            node_id: self.config.node_id.clone(),
            role: self.config.node_role.clone(),
            capabilities,
            sensors: self.config.sensors.clone(),
            wallet_address: self.wallet.wallet().address.clone(),
            wallet_public_key: self.wallet.wallet().public_key.clone(),
            signature: String::new(),
        };
        let registration = self.wallet.sign_node_registration(registration);
        if let Err(err) = self.ledger.register_node(registration) {
            warn!(
                target: "w1z4rdv1510n::node",
                error = %err,
                "ledger registration failed; running in offline mode"
            );
        }
        Ok(())
    }

    fn maybe_send_validator_heartbeat(&self) -> Result<()> {
        if !self.config.blockchain.enabled {
            return Ok(());
        }
        if !matches!(self.config.node_role, NodeRole::Validator) {
            return Ok(());
        }
        let heartbeat = ValidatorHeartbeat {
            node_id: self.config.node_id.clone(),
            timestamp: now_timestamp(),
            fee_paid: 0.0,
            signature: String::new(),
        };
        let heartbeat = self.wallet.sign_validator_heartbeat(heartbeat);
        if let Err(err) = self.ledger.submit_validator_heartbeat(heartbeat) {
            warn!(
                target: "w1z4rdv1510n::node",
                error = %err,
                "validator heartbeat rejected"
            );
        }
        Ok(())
    }

    fn start_validator_heartbeat_loop(&self) {
        if !self.config.blockchain.enabled {
            return;
        }
        if !matches!(self.config.node_role, NodeRole::Validator) {
            return;
        }
        let interval = self
            .config
            .blockchain
            .validator_policy
            .heartbeat_interval_secs
            .max(1);
        let ledger = self.ledger.clone();
        let node_id = self.config.node_id.clone();
        let wallet = self.wallet.clone();
        std::thread::spawn(move || {
            let delay = Duration::from_secs(interval);
            loop {
                std::thread::sleep(delay);
                let heartbeat = ValidatorHeartbeat {
                    node_id: node_id.clone(),
                    timestamp: now_timestamp(),
                    fee_paid: 0.0,
                    signature: String::new(),
                };
                let heartbeat = wallet.sign_validator_heartbeat(heartbeat);
                if let Err(err) = ledger.submit_validator_heartbeat(heartbeat) {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        error = %err,
                        "validator heartbeat rejected"
                    );
                }
            }
        });
    }

    pub fn openstack_control_plane(&self) -> Option<&OpenStackControlPlane> {
        self.openstack.as_ref()
    }
}

fn wait_for_shutdown() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|err| anyhow!("build shutdown runtime: {err}"))?;
    runtime.block_on(async {
        tokio::signal::ctrl_c()
            .await
            .map_err(|err| anyhow!("shutdown signal error: {err}"))?;
        info!(
            target: "w1z4rdv1510n::node",
            "shutdown signal received"
        );
        Ok::<_, anyhow::Error>(())
    })?;
    Ok(())
}

fn build_ledger(config: &NodeConfig) -> Result<Arc<dyn BlockchainLedger>> {
    if !config.ledger.enabled {
        return Ok(Arc::new(NoopLedger::default()));
    }
    let backend = config.ledger.backend.trim().to_ascii_lowercase();
    match backend.as_str() {
        "local" | "file" => {
            let path = if config.ledger.endpoint.trim().is_empty() {
                node_data_dir().join("ledger.json")
            } else {
                config.ledger.endpoint.clone().into()
            };
            Ok(Arc::new(LocalLedger::load_or_create(
                path,
                config.blockchain.validator_policy.clone(),
                config.blockchain.fee_market.clone(),
                config.blockchain.bridge.clone(),
            )?))
        }
        "none" | "" => Ok(Arc::new(NoopLedger::default())),
        other => {
            warn!(
                target: "w1z4rdv1510n::node",
                backend = %other,
                "unknown ledger backend; falling back to noop ledger"
            );
            Ok(Arc::new(NoopLedger::default()))
        }
    }
}

fn estimate_watts(profile: &HardwareProfile, target: ComputeTarget) -> f64 {
    let cpu = profile.cpu_cores as f64;
    let mem = profile.total_memory_gb.max(0.5);
    let mut watts = 8.0 + mem * 0.3;
    match target {
        ComputeTarget::Cpu => {
            watts += cpu * 1.6;
        }
        ComputeTarget::Gpu => {
            watts += cpu * 1.0 + if profile.has_gpu { 60.0 } else { 0.0 };
        }
        ComputeTarget::Cluster => {
            watts += cpu * 1.2;
        }
        ComputeTarget::Quantum => {
            watts += 25.0;
        }
    }
    watts.clamp(5.0, 500.0)
}

fn compute_reward_score(energy_joules: f64, payload_bytes: usize) -> f64 {
    let size_factor = ((payload_bytes as f64) / 65_536.0).clamp(0.5, 4.0);
    let base = (energy_joules / 150.0) * size_factor;
    base.clamp(0.05, 100.0)
}

fn work_id_for(
    node_id: &str,
    data_id: &str,
    input_hash: &str,
    output_hash: Option<&str>,
    completed_at: Timestamp,
) -> String {
    let marker = format!(
        "work|{}|{}|{}|{}|{}",
        node_id,
        data_id,
        input_hash,
        output_hash.unwrap_or(""),
        completed_at.unix
    );
    compute_payload_hash(marker.as_bytes())
}

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}
