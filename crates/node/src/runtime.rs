use crate::chain::ChainSpec;
use crate::config::NodeConfig;
use crate::data_mesh::{DataMeshEvent, DataMeshHandle, start_data_mesh};
use crate::ledger::LocalLedger;
use crate::openstack::OpenStackControlPlane;
use crate::paths::node_data_dir;
use crate::wallet::{WalletSigner, WalletStore, node_id_from_wallet};
use crate::performance::{
    LocalPerformanceSample, NodeMetricsReport, NodePerformanceTracker, OffloadReason,
};
use crate::p2p::NodeNetwork;
use anyhow::{Result, anyhow};
use std::fs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
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

const OFFLOAD_MAX_HOPS: u64 = 2;

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
    performance: Arc<Mutex<NodePerformanceTracker>>,
}

impl NodeRuntime {
    pub fn new(mut config: NodeConfig) -> Result<Self> {
        let wallet = Arc::new(WalletStore::load_or_create_signer(&config.wallet)?);
        if !config.workload.enable_storage {
            config.data.host_storage = false;
        }
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
        let performance = Arc::new(Mutex::new(NodePerformanceTracker::new(
            config.node_id.clone(),
            config.peer_scoring.clone(),
            config.workload.clone(),
        )));
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
            performance,
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
        let workload = self.config.workload.clone();
        let peer_scoring = self.config.peer_scoring.clone();
        let performance = self.performance.clone();
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
        let consume_streams =
            streaming.consume_streams && can_process_streams && workload.enable_stream_processing;
        let consume_shares = streaming.consume_shares && workload.enable_share_consume;
        let publish_shares = streaming.publish_shares && workload.enable_share_publish;
        let publish_streams = streaming.publish_streams;
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
                let metacognition_log_interval = Duration::from_secs(5);
                let mut last_metacognition_log = Instant::now();
                let run_config_path = PathBuf::from(&run_config_path);
                let mut last_config_check = Instant::now();
                let mut last_run_config_mtime = config_modified_time(&run_config_path).ok();
                let mut events = mesh.subscribe();
                loop {
                    if last_config_check.elapsed() >= Duration::from_secs(2) {
                        last_config_check = Instant::now();
                        if let Ok(modified) = config_modified_time(&run_config_path) {
                            let should_reload = last_run_config_mtime
                                .map(|prev| modified > prev)
                                .unwrap_or(true);
                            if should_reload {
                                if let Ok(raw) = fs::read_to_string(&run_config_path) {
                                    if let Ok(updated) = serde_json::from_str::<RunConfig>(&raw) {
                                        inference.update_metacognition_config(
                                            updated.streaming.metacognition.clone(),
                                        );
                                        last_run_config_mtime = Some(modified);
                                        info!(
                                            target: "w1z4rdv1510n::node",
                                            path = %run_config_path.display(),
                                            "reloaded metacognition config"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    let event = match events.recv().await {
                        Ok(event) => event,
                        Err(_) => continue,
                    };
                    let (data_id, payload_kind, payload_hash, origin_node_id, sensor_id) = match event {
                        DataMeshEvent::PayloadReady {
                            data_id,
                            payload_kind,
                            payload_hash,
                            node_id: origin_node_id,
                            sensor_id,
                            ..
                        } => (data_id, payload_kind, payload_hash, origin_node_id, sensor_id),
                        DataMeshEvent::NodeMetrics { report } => {
                            if peer_scoring.enabled {
                                let now = now_timestamp();
                                if let Ok(mut tracker) = performance.lock() {
                                    tracker.ingest_peer_report(report, now);
                                }
                            }
                            continue;
                        }
                    };
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
                        let mut envelope = match serde_json::from_slice::<StreamEnvelope>(&payload) {
                            Ok(envelope) => envelope,
                            Err(_) => continue,
                        };
                        let offload_target = offload_target_from_metadata(&envelope.metadata);
                        if let Some(target) = offload_target.as_ref() {
                            if target != &node_id {
                                continue;
                            }
                        }
                        let offload_hops = offload_hops_from_metadata(&envelope.metadata);
                        let force_process = offload_target.as_deref() == Some(node_id.as_str());
                        let payload_bytes = payload.len();
                        let input_hash = if payload_hash.trim().is_empty() {
                            compute_payload_hash(&payload)
                        } else {
                            payload_hash
                        };
                        let now = now_timestamp();
                        let mut should_process = true;
                        let mut offload_reason = None;
                        let mut offload_candidate = None;
                        if peer_scoring.enabled {
                            match performance.lock() {
                                Ok(mut tracker) => {
                                    should_process = tracker.should_process_streams(now, true);
                                    if !should_process {
                                        offload_reason = tracker.offload_reason(now);
                                        offload_candidate =
                                            tracker.select_offload_peer(now, offload_reason);
                                    }
                                }
                                Err(_) => {
                                    should_process = true;
                                }
                            }
                        }
                        if force_process {
                            should_process = true;
                        }
                        if !should_process {
                            if publish_streams
                                && offload_target.is_none()
                                && offload_hops < OFFLOAD_MAX_HOPS
                            {
                                if let Some(candidate) = offload_candidate {
                                    let reason =
                                        offload_reason.unwrap_or(OffloadReason::Overloaded);
                                    apply_offload_metadata(
                                        &mut envelope.metadata,
                                        &node_id,
                                        &candidate.node_id,
                                        reason,
                                        &origin_node_id,
                                        &data_id,
                                        &input_hash,
                                        &sensor_id,
                                        offload_hops.saturating_add(1),
                                        now,
                                    );
                                    let offload_sensor =
                                        format!("offload:{}:{}", node_id, sensor_id);
                                    let _ = mesh
                                        .ingest_stream_envelope_with_kind(
                                            offload_sensor,
                                            &envelope,
                                            &stream_kind,
                                        )
                                        .await;
                                }
                            }
                            continue;
                        }
                        let offload_metrics = extract_offload_metrics(&envelope.metadata);
                        let offload_origin = envelope
                            .metadata
                            .get("offload_origin_node_id")
                            .and_then(|val| val.as_str())
                            .map(|val| val.to_string());
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
                        let mut accuracy = None;
                        if let Some(mut share) = inference.take_last_fabric_share(node_id.clone()) {
                            accuracy = accuracy_from_metadata(&share.metadata);
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
                            log_metacognition_summary(
                                &share.metadata,
                                now,
                                &mut last_metacognition_log,
                                metacognition_log_interval,
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
                            if publish_shares {
                                let _ = mesh
                                    .ingest_fabric_share_with_kind(&share, &share_kind)
                                    .await;
                            }
                        }
                        if peer_scoring.enabled {
                            let sample = LocalPerformanceSample {
                                timestamp: now_timestamp(),
                                payload_bytes,
                                elapsed_secs,
                                energy_joules,
                                accuracy,
                            };
                            let report = match performance.lock() {
                                Ok(mut tracker) => {
                                    let sample_ts = sample.timestamp;
                                    tracker.update_local(sample);
                                    tracker.take_report(sample_ts)
                                }
                                Err(_) => None,
                            };
                            if let Some(mut report) = report {
                                augment_node_metrics_tags(
                                    &mut report,
                                    &profile,
                                    &compute_router,
                                    &workload,
                                );
                                let _ = mesh.publish_node_metrics(report);
                            }
                        }
                        let accuracy_factor = reward_accuracy_factor(accuracy);
                        let offload_penalty = reward_offload_penalty(offload_hops);
                        let offload_bonus = reward_offload_bonus(offload_origin.as_deref(), &node_id);
                        let score = compute_reward_score(
                            energy_joules,
                            payload_bytes,
                            accuracy_factor,
                            offload_penalty,
                            offload_bonus,
                        );
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
                            "reward_accuracy_factor".to_string(),
                            Value::from(accuracy_factor),
                        );
                        metrics.insert(
                            "reward_offload_penalty".to_string(),
                            Value::from(offload_penalty),
                        );
                        metrics.insert(
                            "reward_offload_bonus".to_string(),
                            Value::from(offload_bonus),
                        );
                        metrics.insert(
                            "reward_base_score".to_string(),
                            Value::from(reward_score_base(energy_joules, payload_bytes)),
                        );
                        if let Some(accuracy) = accuracy {
                            metrics.insert("accuracy_score".to_string(), Value::from(accuracy));
                        }
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
                        for (key, value) in offload_metrics {
                            metrics.insert(key, value);
                        }
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

fn reward_score_base(energy_joules: f64, payload_bytes: usize) -> f64 {
    let size_factor = ((payload_bytes as f64) / 65_536.0).clamp(0.5, 4.0);
    let base = (energy_joules / 150.0) * size_factor;
    base.clamp(0.05, 100.0)
}

fn reward_accuracy_factor(accuracy: Option<f64>) -> f64 {
    match accuracy {
        Some(value) => (0.5 + 0.5 * value.clamp(0.0, 1.0)).clamp(0.5, 1.0),
        None => 1.0,
    }
}

fn reward_offload_penalty(hops: u64) -> f64 {
    let penalty = 1.0 - 0.05 * hops.min(6) as f64;
    penalty.clamp(0.7, 1.0)
}

fn reward_offload_bonus(offload_origin: Option<&str>, node_id: &str) -> f64 {
    if let Some(origin) = offload_origin {
        if !origin.trim().is_empty() && origin != node_id {
            return 1.05;
        }
    }
    1.0
}

fn compute_reward_score(
    energy_joules: f64,
    payload_bytes: usize,
    accuracy_factor: f64,
    offload_penalty: f64,
    offload_bonus: f64,
) -> f64 {
    let base = reward_score_base(energy_joules, payload_bytes);
    let score = base * accuracy_factor * offload_penalty * offload_bonus;
    score.clamp(0.05, 100.0)
}

fn accuracy_from_metadata(metadata: &HashMap<String, Value>) -> Option<f64> {
    if let Some(score) = metadata
        .get("plasticity_report")
        .and_then(|val| val.get("calibration_score"))
        .and_then(|val| val.as_f64())
    {
        return Some(score.clamp(0.0, 1.0));
    }
    metadata
        .get("analysis_report")
        .and_then(|val| val.get("token_confidence"))
        .and_then(|val| val.get("mean"))
        .and_then(|val| val.as_f64())
        .map(|score| score.clamp(0.0, 1.0))
}

fn log_metacognition_summary(
    metadata: &HashMap<String, Value>,
    now: Timestamp,
    last_log: &mut Instant,
    interval: Duration,
) {
    if last_log.elapsed() < interval {
        return;
    }
    let Some(report) = metadata.get("metacognition_report") else {
        return;
    };
    let depth = report.get("reflection_depth").and_then(|val| val.as_u64());
    let model_accuracy = report.get("model_accuracy").and_then(|val| val.as_f64());
    let baseline_accuracy = report.get("baseline_accuracy").and_then(|val| val.as_f64());
    let summary = report.get("summary").and_then(|val| val.as_str());
    let metadata = report.get("metadata");
    let uncertainty = metadata
        .and_then(|val| val.get("uncertainty"))
        .and_then(|val| val.as_f64());
    let reason = metadata
        .and_then(|val| val.get("depth_reason"))
        .and_then(|val| val.as_str())
        .unwrap_or("unknown");
    info!(
        target: "w1z4rdv1510n::node",
        timestamp = now.unix,
        reflection_depth = ?depth,
        model_accuracy = ?model_accuracy,
        baseline_accuracy = ?baseline_accuracy,
        uncertainty = ?uncertainty,
        depth_reason = %reason,
        summary = summary.unwrap_or("n/a"),
        "metacognition reflection update"
    );
    *last_log = Instant::now();
}

fn offload_target_from_metadata(metadata: &HashMap<String, Value>) -> Option<String> {
    metadata
        .get("offload_target_node_id")
        .and_then(|val| val.as_str())
        .map(|val| val.to_string())
}

fn offload_hops_from_metadata(metadata: &HashMap<String, Value>) -> u64 {
    metadata
        .get("offload_hops")
        .and_then(|val| val.as_u64())
        .unwrap_or(0)
}

fn apply_offload_metadata(
    metadata: &mut HashMap<String, Value>,
    origin_node_id: &str,
    target_node_id: &str,
    reason: OffloadReason,
    source_node_id: &str,
    source_data_id: &str,
    source_payload_hash: &str,
    source_sensor_id: &str,
    hops: u64,
    now: Timestamp,
) {
    metadata.insert(
        "offload_origin_node_id".to_string(),
        Value::String(origin_node_id.to_string()),
    );
    metadata.insert(
        "offload_target_node_id".to_string(),
        Value::String(target_node_id.to_string()),
    );
    metadata.insert(
        "offload_reason".to_string(),
        Value::String(reason.as_str().to_string()),
    );
    metadata.insert("offload_hops".to_string(), Value::from(hops));
    metadata.insert(
        "offload_requested_at".to_string(),
        Value::from(now.unix),
    );
    metadata.insert(
        "offload_source_node_id".to_string(),
        Value::String(source_node_id.to_string()),
    );
    metadata.insert(
        "offload_source_data_id".to_string(),
        Value::String(source_data_id.to_string()),
    );
    metadata.insert(
        "offload_source_payload_hash".to_string(),
        Value::String(source_payload_hash.to_string()),
    );
    metadata.insert(
        "offload_source_sensor_id".to_string(),
        Value::String(source_sensor_id.to_string()),
    );
}

fn extract_offload_metrics(metadata: &HashMap<String, Value>) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    for key in [
        "offload_origin_node_id",
        "offload_target_node_id",
        "offload_reason",
        "offload_hops",
        "offload_requested_at",
        "offload_source_node_id",
        "offload_source_data_id",
        "offload_source_payload_hash",
        "offload_source_sensor_id",
    ] {
        if let Some(value) = metadata.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    out
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

fn augment_node_metrics_tags(
    report: &mut NodeMetricsReport,
    profile: &HardwareProfile,
    compute_router: &ComputeRouter,
    workload: &crate::config::WorkloadProfileConfig,
) {
    let tags = &mut report.tags;
    tags.insert("cpu_cores".to_string(), profile.cpu_cores.to_string());
    tags.insert(
        "memory_gb".to_string(),
        format!("{:.1}", profile.total_memory_gb),
    );
    tags.insert("has_gpu".to_string(), profile.has_gpu.to_string());
    let supports_gpu = compute_router.route(ComputeJobKind::TensorHeavy) == ComputeTarget::Gpu;
    let supports_quantum =
        compute_router.route(ComputeJobKind::QuantumAnneal) == ComputeTarget::Quantum;
    let supports_cluster = compute_router.route(ComputeJobKind::Graph) == ComputeTarget::Cluster;
    tags.insert("supports_gpu".to_string(), supports_gpu.to_string());
    tags.insert("supports_quantum".to_string(), supports_quantum.to_string());
    tags.insert("supports_cluster".to_string(), supports_cluster.to_string());
    tags.insert(
        "workload_streams".to_string(),
        workload.enable_stream_processing.to_string(),
    );
    tags.insert(
        "workload_storage".to_string(),
        workload.enable_storage.to_string(),
    );
}

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}

fn config_modified_time(path: &PathBuf) -> Result<SystemTime> {
    let metadata = fs::metadata(path)?;
    metadata.modified().map_err(|err| anyhow!(err))
}
