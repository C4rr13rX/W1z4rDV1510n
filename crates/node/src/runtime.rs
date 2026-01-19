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
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};
use w1z4rdv1510n::blockchain::{
    BlockchainLedger, NoopLedger, NodeCapabilities, NodeRegistration, ValidatorHeartbeat,
};
use w1z4rdv1510n::compute::{ComputeJobKind, ComputeRouter};
use w1z4rdv1510n::config::NodeRole;
use w1z4rdv1510n::hardware::HardwareProfile;
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::{NeuralFabricShare, StreamEnvelope, StreamingInference};
use w1z4rdv1510n::config::RunConfig;

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
        let Some(mesh) = self.data_mesh.clone() else {
            return Err(anyhow!("streaming runtime requires data mesh"));
        };
        let run_config_path = streaming.run_config_path.clone();
        let node_id = self.config.node_id.clone();
        let cpu_cores = self.profile.cpu_cores;
        let memory_gb = self.profile.total_memory_gb;
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
                let mut events = mesh.subscribe();
                loop {
                    let event = match events.recv().await {
                        Ok(event) => event,
                        Err(_) => continue,
                    };
                    let DataMeshEvent::PayloadReady {
                        data_id,
                        payload_kind,
                        ..
                    } = event;
                    if payload_kind == stream_kind && consume_streams {
                        if let Ok(payload) = mesh.load_payload(data_id).await {
                            if let Ok(envelope) = serde_json::from_slice::<StreamEnvelope>(&payload) {
                                let _ = inference.handle_envelope(envelope);
                                if publish_shares {
                                    if let Some(share) =
                                        inference.take_last_fabric_share(node_id.clone())
                                    {
                                        let _ = mesh.ingest_fabric_share(&share).await;
                                    }
                                }
                            }
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

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    Timestamp { unix }
}
