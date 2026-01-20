use crate::config::DataMeshConfig;
use crate::performance::NodeMetricsReport;
use crate::p2p::NetworkPublisher;
use crate::wallet::{WalletSigner, address_from_public_key, node_id_from_wallet};
use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::future;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::time::Duration;
use tracing::{info, warn};
use w1z4rdv1510n::blockchain::{BlockchainLedger, SensorCommitment, WorkKind, WorkProof};
use w1z4rdv1510n::network::{NetworkEnvelope, compute_payload_hash};
use w1z4rdv1510n::schema::Timestamp;
use w1z4rdv1510n::streaming::{NeuralFabricShare, StreamEnvelope};

const DATA_MESSAGE_KIND: &str = "data.message";
const DATA_EVENT_BUFFER: usize = 1024;
const NEURAL_FABRIC_KIND: &str = "neural.fabric.v1";
const STREAM_ENVELOPE_KIND: &str = "stream.envelope.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManifest {
    pub data_id: String,
    pub node_id: String,
    pub sensor_id: String,
    pub timestamp: Timestamp,
    pub payload_kind: String,
    pub payload_hash: String,
    pub size_bytes: usize,
    pub chunk_count: u32,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataChunk {
    pub data_id: String,
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub payload_hex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationReceipt {
    pub data_id: String,
    pub replica_node_id: String,
    pub timestamp: Timestamp,
    pub payload_hash: String,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRequest {
    pub data_id: String,
    pub requester_node_id: String,
    pub timestamp: Timestamp,
    pub public_key: String,
    pub signature: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DataMessage {
    Manifest(DataManifest),
    Chunk(DataChunk),
    Receipt(ReplicationReceipt),
    Request(DataRequest),
    NodeMetrics(NodeMetricsReport),
}

#[derive(Debug, Clone)]
pub enum DataMeshEvent {
    PayloadReady {
        data_id: String,
        payload_kind: String,
        payload_hash: String,
        timestamp: Timestamp,
        node_id: String,
        sensor_id: String,
    },
    NodeMetrics {
        report: NodeMetricsReport,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIngestRequest {
    pub sensor_id: String,
    pub payload_kind: String,
    #[serde(default)]
    pub payload_hex: Option<String>,
    #[serde(default)]
    pub payload_text: Option<String>,
    #[serde(default)]
    pub payload_json: Option<Value>,
    #[serde(default)]
    pub timestamp_unix: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataIngestResponse {
    pub data_id: String,
    pub payload_hash: String,
    pub chunk_count: u32,
    pub stored: bool,
    pub receipt_count: usize,
    pub replication_factor: usize,
    pub receipt_quorum: usize,
    pub quorum_met: bool,
    pub replication_met: bool,
    pub pending_quorum: usize,
    pub pending_replicas: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStatusResponse {
    pub data_id: String,
    pub stored: bool,
    pub receipt_count: usize,
    pub replication_factor: usize,
    pub receipt_quorum: usize,
    pub quorum_met: bool,
    pub replication_met: bool,
    pub pending_quorum: usize,
    pub pending_replicas: usize,
    pub manifest: Option<DataManifest>,
}

#[derive(Clone)]
pub struct DataMeshHandle {
    command_tx: mpsc::UnboundedSender<DataMeshCommand>,
    event_tx: broadcast::Sender<DataMeshEvent>,
}

impl DataMeshHandle {
    pub async fn ingest(&self, request: DataIngestRequest) -> Result<DataIngestResponse> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.command_tx
            .send(DataMeshCommand::Ingest { request, respond_to: resp_tx })
            .map_err(|_| anyhow!("data mesh command channel closed"))?;
        resp_rx
            .await
            .map_err(|_| anyhow!("data mesh ingest response dropped"))?
    }

    pub async fn status(&self, data_id: String) -> Result<DataStatusResponse> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.command_tx
            .send(DataMeshCommand::Status { data_id, respond_to: resp_tx })
            .map_err(|_| anyhow!("data mesh command channel closed"))?;
        resp_rx
            .await
            .map_err(|_| anyhow!("data mesh status response dropped"))?
    }

    pub async fn load_payload(&self, data_id: String) -> Result<Vec<u8>> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.command_tx
            .send(DataMeshCommand::LoadPayload { data_id, respond_to: resp_tx })
            .map_err(|_| anyhow!("data mesh command channel closed"))?;
        resp_rx
            .await
            .map_err(|_| anyhow!("data mesh load response dropped"))?
    }

    pub async fn ingest_fabric_share(&self, share: &NeuralFabricShare) -> Result<DataIngestResponse> {
        self.ingest_fabric_share_with_kind(share, NEURAL_FABRIC_KIND)
            .await
    }

    pub async fn ingest_fabric_share_with_kind(
        &self,
        share: &NeuralFabricShare,
        payload_kind: &str,
    ) -> Result<DataIngestResponse> {
        let payload_json = serde_json::to_value(share)
            .map_err(|err| anyhow!("serialize neural fabric share: {err}"))?;
        let request = DataIngestRequest {
            sensor_id: format!("neural-fabric:{}", share.node_id),
            payload_kind: payload_kind.to_string(),
            payload_hex: None,
            payload_text: None,
            payload_json: Some(payload_json),
            timestamp_unix: Some(share.timestamp.unix),
        };
        self.ingest(request).await
    }

    pub async fn ingest_stream_envelope(
        &self,
        sensor_id: String,
        envelope: &StreamEnvelope,
    ) -> Result<DataIngestResponse> {
        self.ingest_stream_envelope_with_kind(sensor_id, envelope, STREAM_ENVELOPE_KIND)
            .await
    }

    pub async fn ingest_stream_envelope_with_kind(
        &self,
        sensor_id: String,
        envelope: &StreamEnvelope,
        payload_kind: &str,
    ) -> Result<DataIngestResponse> {
        let payload_json =
            serde_json::to_value(envelope).map_err(|err| anyhow!("serialize stream envelope: {err}"))?;
        let request = DataIngestRequest {
            sensor_id,
            payload_kind: payload_kind.to_string(),
            payload_hex: None,
            payload_text: None,
            payload_json: Some(payload_json),
            timestamp_unix: Some(envelope.timestamp.unix),
        };
        self.ingest(request).await
    }

    pub async fn next_fabric_share(
        &self,
        receiver: &mut broadcast::Receiver<DataMeshEvent>,
    ) -> Option<NeuralFabricShare> {
        loop {
            let event = match receiver.recv().await {
                Ok(event) => event,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            };
            let data_id = match event {
                DataMeshEvent::PayloadReady {
                    data_id,
                    payload_kind,
                    ..
                } => {
                    if payload_kind != NEURAL_FABRIC_KIND {
                        continue;
                    }
                    data_id
                }
                DataMeshEvent::NodeMetrics { .. } => continue,
            };
            let payload = self.load_payload(data_id).await.ok()?;
            if let Ok(share) = serde_json::from_slice::<NeuralFabricShare>(&payload) {
                return Some(share);
            }
        }
    }

    pub async fn next_stream_envelope(
        &self,
        receiver: &mut broadcast::Receiver<DataMeshEvent>,
    ) -> Option<StreamEnvelope> {
        loop {
            let event = match receiver.recv().await {
                Ok(event) => event,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            };
            let data_id = match event {
                DataMeshEvent::PayloadReady {
                    data_id,
                    payload_kind,
                    ..
                } => {
                    if payload_kind != STREAM_ENVELOPE_KIND {
                        continue;
                    }
                    data_id
                }
                DataMeshEvent::NodeMetrics { .. } => continue,
            };
            let payload = self.load_payload(data_id).await.ok()?;
            if let Ok(envelope) = serde_json::from_slice::<StreamEnvelope>(&payload) {
                return Some(envelope);
            }
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<DataMeshEvent> {
        self.event_tx.subscribe()
    }

    pub fn publish_node_metrics(&self, report: NodeMetricsReport) -> Result<()> {
        self.command_tx
            .send(DataMeshCommand::PublishMetrics { report })
            .map_err(|_| anyhow!("data mesh command channel closed"))
    }
}

enum DataMeshCommand {
    Ingest {
        request: DataIngestRequest,
        respond_to: oneshot::Sender<Result<DataIngestResponse>>,
    },
    Status {
        data_id: String,
        respond_to: oneshot::Sender<Result<DataStatusResponse>>,
    },
    LoadPayload {
        data_id: String,
        respond_to: oneshot::Sender<Result<Vec<u8>>>,
    },
    NetworkEnvelope(NetworkEnvelope),
    PublishMetrics {
        report: NodeMetricsReport,
    },
}

pub fn start_data_mesh(
    config: DataMeshConfig,
    node_id: String,
    wallet: Arc<WalletSigner>,
    ledger: Arc<dyn BlockchainLedger>,
    publisher: NetworkPublisher,
    mut network_rx: mpsc::UnboundedReceiver<NetworkEnvelope>,
) -> Result<DataMeshHandle> {
    let (command_tx, mut command_rx) = mpsc::unbounded_channel();
    let (event_tx, _) = broadcast::channel(DATA_EVENT_BUFFER);
    let mut store = DataStore::new(config.clone())?;
    let node_id_clone = node_id.clone();
    let event_tx_clone = event_tx.clone();
    std::thread::spawn(move || {
        let runtime = match tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime,
            Err(err) => {
                warn!(target: "w1z4rdv1510n::node", error = %err, "failed to start data mesh runtime");
                return;
            }
        };
        runtime.block_on(async move {
            let mut maintenance = if config.maintenance_enabled {
                Some(tokio::time::interval(Duration::from_secs(
                    config.maintenance_interval_secs.max(1),
                )))
            } else {
                None
            };
            loop {
                let maintenance_tick = async {
                    if let Some(interval) = maintenance.as_mut() {
                        interval.tick().await;
                    } else {
                        future::pending::<()>().await;
                    }
                };
                tokio::select! {
                    Some(command) = command_rx.recv() => {
                        handle_command(
                            command,
                            &config,
                            &node_id_clone,
                            &wallet,
                            &ledger,
                            &publisher,
                            &event_tx_clone,
                            &mut store,
                        );
                    }
                    Some(envelope) = network_rx.recv() => {
                        let command = DataMeshCommand::NetworkEnvelope(envelope);
                        handle_command(
                            command,
                            &config,
                            &node_id_clone,
                            &wallet,
                            &ledger,
                            &publisher,
                            &event_tx_clone,
                            &mut store,
                        );
                    }
                    _ = maintenance_tick => {
                        run_maintenance(&config, &node_id_clone, &wallet, &publisher, &mut store);
                    }
                }
            }
        });
    });
    Ok(DataMeshHandle { command_tx, event_tx })
}

fn handle_command(
    command: DataMeshCommand,
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    ledger: &Arc<dyn BlockchainLedger>,
    publisher: &NetworkPublisher,
    event_tx: &broadcast::Sender<DataMeshEvent>,
    store: &mut DataStore,
) {
    match command {
        DataMeshCommand::Ingest { request, respond_to } => {
            let response = ingest_local(
                request,
                config,
                node_id,
                wallet,
                ledger,
                publisher,
                event_tx,
                store,
            );
            let _ = respond_to.send(response);
        }
        DataMeshCommand::Status { data_id, respond_to } => {
            let response = store.status(&data_id);
            let _ = respond_to.send(response);
        }
        DataMeshCommand::LoadPayload { data_id, respond_to } => {
            let response = store.load_payload(&data_id);
            let _ = respond_to.send(response);
        }
        DataMeshCommand::NetworkEnvelope(envelope) => {
            if envelope.payload_kind != DATA_MESSAGE_KIND {
                return;
            }
            if let Ok(bytes) = envelope.payload_bytes() {
                if let Ok(message) = serde_json::from_slice::<DataMessage>(&bytes) {
                    handle_network_message(
                        message,
                        Some(&envelope.public_key),
                        config,
                        node_id,
                        wallet,
                        ledger,
                        publisher,
                        event_tx,
                        store,
                    );
                }
            }
        }
        DataMeshCommand::PublishMetrics { report } => {
            if !config.enabled {
                return;
            }
            let _ = publisher.publish(signed_envelope(DataMessage::NodeMetrics(report), wallet));
        }
    }
}

fn run_maintenance(
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    publisher: &NetworkPublisher,
    store: &mut DataStore,
) {
    if !config.maintenance_enabled {
        return;
    }
    if !config.host_storage {
        return;
    }
    let mut outcome = store.audit_and_gc(now_unix(), node_id);
    let mut sent = 0usize;
    for data_id in outcome.repair_requests.drain(..) {
        let request = build_request(&data_id, node_id, wallet);
        if publisher
            .publish(signed_envelope(DataMessage::Request(request), wallet))
            .is_ok()
        {
            sent = sent.saturating_add(1);
        }
    }
    if outcome.expired > 0
        || outcome.corrupted_blob > 0
        || outcome.missing_blob > 0
        || outcome.orphan_blobs > 0
        || outcome.orphan_receipts > 0
        || outcome.size_evicted > 0
        || sent > 0
    {
        info!(
            target: "w1z4rdv1510n::node",
            scanned = outcome.scanned,
            expired = outcome.expired,
            corrupted = outcome.corrupted_blob,
            missing = outcome.missing_blob,
            orphan_blobs = outcome.orphan_blobs,
            orphan_receipts = outcome.orphan_receipts,
            size_evicted = outcome.size_evicted,
            bytes_freed = outcome.bytes_freed,
            repair_requests = sent,
            "data mesh maintenance"
        );
    }
}

fn ingest_local(
    request: DataIngestRequest,
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    ledger: &Arc<dyn BlockchainLedger>,
    publisher: &NetworkPublisher,
    event_tx: &broadcast::Sender<DataMeshEvent>,
    store: &mut DataStore,
) -> Result<DataIngestResponse> {
    if !config.enabled {
        anyhow::bail!("data mesh disabled");
    }
    if request.sensor_id.trim().is_empty() {
        anyhow::bail!("sensor_id must be set");
    }
    if request.payload_kind.trim().is_empty() {
        anyhow::bail!("payload_kind must be set");
    }
    let payload = request_payload_bytes(&request)?;
    if payload.len() > config.max_payload_bytes {
        anyhow::bail!("payload exceeds max_payload_bytes");
    }
    let timestamp = Timestamp {
        unix: request.timestamp_unix.unwrap_or_else(now_unix),
    };
    let payload_hash = compute_payload_hash(&payload);
    let data_id = data_id_for(node_id, &request.sensor_id, timestamp, &payload_hash);
    let chunks = chunk_payload(&data_id, &payload, config.chunk_size_bytes.max(1));
    let chunk_count = chunks.len() as u32;
    let manifest = DataManifest {
        data_id: data_id.clone(),
        node_id: node_id.to_string(),
        sensor_id: request.sensor_id.clone(),
        timestamp,
        payload_kind: request.payload_kind.clone(),
        payload_hash: payload_hash.clone(),
        size_bytes: payload.len(),
        chunk_count,
        public_key: wallet.wallet().public_key.clone(),
        signature: String::new(),
    };
    let signed_manifest = sign_manifest(manifest, wallet);
    store.store_manifest(&signed_manifest)?;
    store.store_payload(&data_id, &payload)?;
    emit_payload_ready(event_tx, &signed_manifest);
    let receipt = build_receipt(&data_id, &payload_hash, node_id, wallet);
    let receipt_record = if config.host_storage {
        let record = store.record_receipt(receipt.clone())?;
        if record.stored {
            maybe_submit_storage_reward(
                config,
                ledger,
                wallet,
                node_id,
                &data_id,
                &payload_hash,
                payload.len(),
            );
        }
        record
    } else {
        ReceiptRecord {
            count: store.receipt_count(&data_id),
            stored: false,
        }
    };
    let _ = publisher.publish(signed_envelope(DataMessage::Manifest(signed_manifest), wallet));
    for chunk in chunks {
        let _ = publisher.publish(signed_envelope(DataMessage::Chunk(chunk), wallet));
    }
    if receipt_record.stored {
        let _ = publisher.publish(signed_envelope(DataMessage::Receipt(receipt), wallet));
    }
    let commitment = SensorCommitment {
        node_id: node_id.to_string(),
        sensor_id: request.sensor_id,
        timestamp,
        payload_hash: payload_hash.clone(),
        fee_paid: 0.0,
        signature: String::new(),
    };
    let signed_commitment = wallet.sign_sensor_commitment(commitment);
    if let Err(err) = ledger.submit_sensor_commitment(signed_commitment) {
        warn!(
            target: "w1z4rdv1510n::node",
            error = %err,
            "ledger sensor commitment rejected"
        );
    }
    let receipt_count = receipt_record.count;
    let stored = store.has_data(&data_id);
    let state = quorum_state(receipt_count, config);
    Ok(DataIngestResponse {
        data_id,
        payload_hash,
        chunk_count,
        stored,
        receipt_count,
        replication_factor: config.replication_factor,
        receipt_quorum: config.receipt_quorum,
        quorum_met: state.quorum_met,
        replication_met: state.replication_met,
        pending_quorum: state.pending_quorum,
        pending_replicas: state.pending_replicas,
    })
}

fn handle_network_message(
    message: DataMessage,
    envelope_public_key: Option<&str>,
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    ledger: &Arc<dyn BlockchainLedger>,
    publisher: &NetworkPublisher,
    event_tx: &broadcast::Sender<DataMeshEvent>,
    store: &mut DataStore,
) {
    match message {
        DataMessage::Manifest(manifest) => {
            if !config.host_storage {
                return;
            }
            if config.require_manifest_signature && verify_manifest(&manifest).is_err() {
                return;
            }
            if store.store_manifest(&manifest).is_ok() {
                if store.has_data(&manifest.data_id) {
                    let receipt = build_receipt(
                        &manifest.data_id,
                        &manifest.payload_hash,
                        node_id,
                        wallet,
                    );
                    if let Ok(record) = store.record_receipt(receipt.clone()) {
                        if record.stored {
                            let _ = publisher.publish(signed_envelope(DataMessage::Receipt(receipt), wallet));
                            maybe_submit_storage_reward(
                                config,
                                ledger,
                                wallet,
                                node_id,
                                &manifest.data_id,
                                &manifest.payload_hash,
                                manifest.size_bytes,
                            );
                        }
                    }
                } else {
                    let receipt_count = store.receipt_count(&manifest.data_id);
                    if receipt_count < config.replication_factor {
                        let request = build_request(&manifest.data_id, node_id, wallet);
                        let _ = publisher.publish(signed_envelope(DataMessage::Request(request), wallet));
                    }
                }
            }
        }
        DataMessage::Chunk(chunk) => {
            if !config.host_storage {
                return;
            }
            let manifest = store.load_manifest(&chunk.data_id).ok().flatten();
            if store.apply_chunk(chunk, manifest.as_ref()).is_ok() {
                if let Some(manifest) = manifest {
                    if store.has_data(&manifest.data_id) {
                        emit_payload_ready(event_tx, &manifest);
                        let receipt = build_receipt(
                            &manifest.data_id,
                            &manifest.payload_hash,
                            node_id,
                            wallet,
                        );
                        if let Ok(record) = store.record_receipt(receipt.clone()) {
                            if record.stored {
                                let _ = publisher.publish(signed_envelope(DataMessage::Receipt(receipt), wallet));
                                maybe_submit_storage_reward(
                                    config,
                                    ledger,
                                    wallet,
                                    node_id,
                                    &manifest.data_id,
                                    &manifest.payload_hash,
                                    manifest.size_bytes,
                                );
                            }
                        }
                    }
                }
            }
        }
        DataMessage::Receipt(receipt) => {
            if config.require_receipt_signature && verify_receipt(&receipt).is_err() {
                return;
            }
            let _ = store.record_receipt(receipt);
        }
        DataMessage::Request(request) => {
            if verify_request(&request).is_err() {
                return;
            }
            if let Ok(payload) = store.load_payload(&request.data_id) {
                let chunks = chunk_payload(&request.data_id, &payload, config.chunk_size_bytes.max(1));
                for chunk in chunks {
                    let _ = publisher.publish(signed_envelope(DataMessage::Chunk(chunk), wallet));
                }
            }
        }
        DataMessage::NodeMetrics(report) => {
            if let Some(public_key) = envelope_public_key {
                if verify_node_metrics(&report, public_key).is_err() {
                    return;
                }
            }
            emit_node_metrics(event_tx, report);
        }
    }
}

fn request_payload_bytes(request: &DataIngestRequest) -> Result<Vec<u8>> {
    let mut provided = 0;
    if request.payload_hex.is_some() {
        provided += 1;
    }
    if request.payload_text.is_some() {
        provided += 1;
    }
    if request.payload_json.is_some() {
        provided += 1;
    }
    if provided != 1 {
        anyhow::bail!("exactly one payload format must be provided");
    }
    if let Some(hex) = &request.payload_hex {
        return hex_decode(hex);
    }
    if let Some(text) = &request.payload_text {
        return Ok(text.as_bytes().to_vec());
    }
    if let Some(json) = &request.payload_json {
        return serde_json::to_vec(json).map_err(|err| anyhow!("serialize payload_json: {err}"));
    }
    anyhow::bail!("payload missing")
}

fn data_id_for(node_id: &str, sensor_id: &str, timestamp: Timestamp, payload_hash: &str) -> String {
    let payload = format!(
        "data|{}|{}|{}|{}",
        node_id, sensor_id, timestamp.unix, payload_hash
    );
    compute_payload_hash(payload.as_bytes())
}

fn sign_manifest(mut manifest: DataManifest, wallet: &WalletSigner) -> DataManifest {
    manifest.public_key = wallet.wallet().public_key.clone();
    let payload = manifest_payload(&manifest);
    manifest.signature = wallet.sign_payload(payload.as_bytes());
    manifest
}

fn manifest_payload(manifest: &DataManifest) -> String {
    format!(
        "manifest|{}|{}|{}|{}|{}|{}|{}",
        manifest.data_id,
        manifest.node_id,
        manifest.sensor_id,
        manifest.timestamp.unix,
        manifest.payload_kind,
        manifest.payload_hash,
        manifest.size_bytes
    )
}

fn receipt_payload(receipt: &ReplicationReceipt) -> String {
    format!(
        "receipt|{}|{}|{}|{}",
        receipt.data_id,
        receipt.replica_node_id,
        receipt.timestamp.unix,
        receipt.payload_hash
    )
}

fn request_payload(request: &DataRequest) -> String {
    format!(
        "request|{}|{}|{}",
        request.data_id, request.requester_node_id, request.timestamp.unix
    )
}

fn build_receipt(data_id: &str, payload_hash: &str, node_id: &str, wallet: &WalletSigner) -> ReplicationReceipt {
    let timestamp = Timestamp { unix: now_unix() };
    let mut receipt = ReplicationReceipt {
        data_id: data_id.to_string(),
        replica_node_id: node_id.to_string(),
        timestamp,
        payload_hash: payload_hash.to_string(),
        public_key: wallet.wallet().public_key.clone(),
        signature: String::new(),
    };
    let payload = receipt_payload(&receipt);
    receipt.signature = wallet.sign_payload(payload.as_bytes());
    receipt
}

fn build_request(data_id: &str, node_id: &str, wallet: &WalletSigner) -> DataRequest {
    let timestamp = Timestamp { unix: now_unix() };
    let mut request = DataRequest {
        data_id: data_id.to_string(),
        requester_node_id: node_id.to_string(),
        timestamp,
        public_key: wallet.wallet().public_key.clone(),
        signature: String::new(),
    };
    let payload = request_payload(&request);
    request.signature = wallet.sign_payload(payload.as_bytes());
    request
}

fn signed_envelope(message: DataMessage, wallet: &WalletSigner) -> NetworkEnvelope {
    let payload = serde_json::to_vec(&message).unwrap_or_default();
    let mut envelope = NetworkEnvelope::new(DATA_MESSAGE_KIND, &payload, now_unix());
    envelope.public_key = wallet.wallet().public_key.clone();
    envelope.message_id = envelope.expected_message_id();
    envelope.signature = wallet.sign_payload(envelope.signing_payload().as_bytes());
    envelope
}

fn emit_payload_ready(event_tx: &broadcast::Sender<DataMeshEvent>, manifest: &DataManifest) {
    let event = DataMeshEvent::PayloadReady {
        data_id: manifest.data_id.clone(),
        payload_kind: manifest.payload_kind.clone(),
        payload_hash: manifest.payload_hash.clone(),
        timestamp: manifest.timestamp,
        node_id: manifest.node_id.clone(),
        sensor_id: manifest.sensor_id.clone(),
    };
    let _ = event_tx.send(event);
}

fn emit_node_metrics(event_tx: &broadcast::Sender<DataMeshEvent>, report: NodeMetricsReport) {
    let _ = event_tx.send(DataMeshEvent::NodeMetrics { report });
}

fn verify_manifest(manifest: &DataManifest) -> Result<()> {
    let public_key = decode_public_key(&manifest.public_key)?;
    let signature = decode_signature(&manifest.signature)?;
    let payload = manifest_payload(manifest);
    public_key
        .verify(payload.as_bytes(), &signature)
        .map_err(|err| anyhow!("manifest signature invalid: {err}"))?;
    let expected_node_id = node_id_from_public_key(&public_key)?;
    if manifest.node_id != expected_node_id {
        anyhow::bail!("manifest node_id mismatch");
    }
    Ok(())
}

fn verify_receipt(receipt: &ReplicationReceipt) -> Result<()> {
    let public_key = decode_public_key(&receipt.public_key)?;
    let signature = decode_signature(&receipt.signature)?;
    let payload = receipt_payload(receipt);
    public_key
        .verify(payload.as_bytes(), &signature)
        .map_err(|err| anyhow!("receipt signature invalid: {err}"))?;
    let expected_node_id = node_id_from_public_key(&public_key)?;
    if receipt.replica_node_id != expected_node_id {
        anyhow::bail!("receipt node_id mismatch");
    }
    Ok(())
}

fn verify_request(request: &DataRequest) -> Result<()> {
    let public_key = decode_public_key(&request.public_key)?;
    let signature = decode_signature(&request.signature)?;
    let payload = request_payload(request);
    public_key
        .verify(payload.as_bytes(), &signature)
        .map_err(|err| anyhow!("request signature invalid: {err}"))?;
    let expected_node_id = node_id_from_public_key(&public_key)?;
    if request.requester_node_id != expected_node_id {
        anyhow::bail!("request node_id mismatch");
    }
    Ok(())
}

fn verify_node_metrics(report: &NodeMetricsReport, public_key_hex: &str) -> Result<()> {
    if public_key_hex.trim().is_empty() {
        return Ok(());
    }
    let public_key = decode_public_key(public_key_hex)?;
    let expected_node_id = node_id_from_public_key(&public_key)?;
    if report.node_id != expected_node_id {
        anyhow::bail!("metrics node_id mismatch");
    }
    Ok(())
}

fn node_id_from_public_key(public_key: &VerifyingKey) -> Result<String> {
    let bytes = public_key.to_bytes();
    let address = address_from_public_key(&bytes);
    Ok(node_id_from_wallet(&address))
}

fn chunk_payload(data_id: &str, payload: &[u8], chunk_size: usize) -> Vec<DataChunk> {
    let mut chunks = Vec::new();
    let total = ((payload.len() + chunk_size - 1) / chunk_size).max(1);
    for (idx, part) in payload.chunks(chunk_size).enumerate() {
        chunks.push(DataChunk {
            data_id: data_id.to_string(),
            chunk_index: idx as u32,
            total_chunks: total as u32,
            payload_hex: hex_encode(part),
        });
    }
    chunks
}

fn now_unix() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn maybe_submit_storage_reward(
    config: &DataMeshConfig,
    ledger: &Arc<dyn BlockchainLedger>,
    wallet: &WalletSigner,
    node_id: &str,
    data_id: &str,
    payload_hash: &str,
    size_bytes: usize,
) {
    if !config.storage_reward_enabled {
        return;
    }
    let score = storage_reward_score(size_bytes, config);
    if score <= 0.0 {
        return;
    }
    let completed_at = Timestamp { unix: now_unix() };
    let work_id = storage_work_id(node_id, data_id, payload_hash);
    let mut metrics = HashMap::new();
    metrics.insert("data_id".to_string(), Value::String(data_id.to_string()));
    metrics.insert(
        "payload_hash".to_string(),
        Value::String(payload_hash.to_string()),
    );
    metrics.insert("size_bytes".to_string(), Value::from(size_bytes as u64));
    metrics.insert(
        "replication_factor".to_string(),
        Value::from(config.replication_factor as u64),
    );
    metrics.insert(
        "receipt_quorum".to_string(),
        Value::from(config.receipt_quorum as u64),
    );
    metrics.insert(
        "storage_reward_score".to_string(),
        Value::from(score),
    );
    let proof = WorkProof {
        work_id,
        node_id: node_id.to_string(),
        kind: WorkKind::StorageContribution,
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
            "storage work proof rejected"
        );
    }
}

fn storage_reward_score(size_bytes: usize, config: &DataMeshConfig) -> f64 {
    let size_mb = (size_bytes as f64 / 1_048_576.0).max(0.01);
    let mut score = config.storage_reward_base + size_mb * config.storage_reward_per_mb;
    if config.replication_factor > 0 {
        score /= config.replication_factor as f64;
    }
    score.clamp(0.01, 100.0)
}

fn storage_work_id(node_id: &str, data_id: &str, payload_hash: &str) -> String {
    let payload = format!("storage|{}|{}|{}", node_id, data_id, payload_hash);
    compute_payload_hash(payload.as_bytes())
}

struct PendingAssembly {
    total_chunks: u32,
    received: HashSet<u32>,
}

struct DataStore {
    config: DataMeshConfig,
    root: PathBuf,
    blob_dir: PathBuf,
    manifest_dir: PathBuf,
    receipt_dir: PathBuf,
    staging_dir: PathBuf,
    pending: HashMap<String, PendingAssembly>,
    pending_order: VecDeque<String>,
}

impl DataStore {
    fn new(config: DataMeshConfig) -> Result<Self> {
        let root = PathBuf::from(&config.storage_path);
        let blob_dir = root.join("blobs");
        let manifest_dir = root.join("manifests");
        let receipt_dir = root.join("receipts");
        let staging_dir = root.join("staging");
        fs::create_dir_all(&blob_dir)?;
        fs::create_dir_all(&manifest_dir)?;
        fs::create_dir_all(&receipt_dir)?;
        fs::create_dir_all(&staging_dir)?;
        Ok(Self {
            config,
            root,
            blob_dir,
            manifest_dir,
            receipt_dir,
            staging_dir,
            pending: HashMap::new(),
            pending_order: VecDeque::new(),
        })
    }

    fn store_manifest(&mut self, manifest: &DataManifest) -> Result<()> {
        let path = self.manifest_path(&manifest.data_id);
        if path.exists() {
            return Ok(());
        }
        let payload = serde_json::to_string_pretty(manifest)?;
        fs::write(path, payload)?;
        Ok(())
    }

    fn load_manifest(&self, data_id: &str) -> Result<Option<DataManifest>> {
        let path = self.manifest_path(data_id);
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&raw)?;
        Ok(Some(manifest))
    }

    fn store_payload(&self, data_id: &str, payload: &[u8]) -> Result<()> {
        let path = self.blob_path(data_id);
        fs::write(path, payload)?;
        Ok(())
    }

    fn load_payload(&self, data_id: &str) -> Result<Vec<u8>> {
        let path = self.blob_path(data_id);
        fs::read(&path).map_err(|err| anyhow!("read payload {}: {err}", path.display()))
    }

    fn has_data(&self, data_id: &str) -> bool {
        self.blob_path(data_id).exists()
    }

    fn apply_chunk(&mut self, mut chunk: DataChunk, manifest: Option<&DataManifest>) -> Result<()> {
        if chunk.data_id.trim().is_empty() {
            if let Some(manifest) = manifest {
                chunk.data_id = manifest.data_id.clone();
            } else {
                anyhow::bail!("chunk missing data_id");
            }
        }
        if chunk.payload_hex.trim().is_empty() {
            anyhow::bail!("chunk payload missing");
        }
        if chunk.total_chunks == 0 {
            anyhow::bail!("chunk total_chunks must be > 0");
        }
        let chunk_bytes = hex_decode(&chunk.payload_hex)?;
        let dir = self.staging_dir.join(&chunk.data_id);
        fs::create_dir_all(&dir)?;
        let chunk_path = dir.join(format!("chunk-{}.bin", chunk.chunk_index));
        fs::write(&chunk_path, &chunk_bytes)?;
        let (total_chunks, received_len) = {
            let pending = self.pending.entry(chunk.data_id.clone()).or_insert_with(|| {
                self.pending_order.push_back(chunk.data_id.clone());
                PendingAssembly {
                    total_chunks: chunk.total_chunks,
                    received: HashSet::new(),
                }
            });
            pending.total_chunks = chunk.total_chunks;
            pending.received.insert(chunk.chunk_index);
            (pending.total_chunks, pending.received.len())
        };
        self.trim_pending();

        if received_len < total_chunks as usize {
            return Ok(());
        }
        let mut payload = Vec::new();
        for idx in 0..total_chunks {
            let path = dir.join(format!("chunk-{}.bin", idx));
            let part = fs::read(&path).with_context(|| format!("read chunk {}", idx))?;
            payload.extend_from_slice(&part);
        }
        if let Some(manifest) = manifest {
            let hash = compute_payload_hash(&payload);
            if hash != manifest.payload_hash {
                anyhow::bail!("payload hash mismatch");
            }
        }
        self.store_payload(&chunk.data_id, &payload)?;
        let _ = fs::remove_dir_all(&dir);
        self.pending.remove(&chunk.data_id);
        Ok(())
    }

    fn record_receipt(&mut self, receipt: ReplicationReceipt) -> Result<ReceiptRecord> {
        let path = self.receipt_path(&receipt.data_id);
        let mut file = if path.exists() {
            let raw = fs::read_to_string(&path)?;
            serde_json::from_str::<ReceiptFile>(&raw)?
        } else {
            ReceiptFile::default()
        };
        if file
            .receipts
            .iter()
            .any(|existing| existing.replica_node_id == receipt.replica_node_id)
        {
            return Ok(ReceiptRecord {
                count: file.receipts.len(),
                stored: false,
            });
        }
        let max_receipts = self.config.replication_factor.max(1);
        if file.receipts.len() >= max_receipts {
            return Ok(ReceiptRecord {
                count: file.receipts.len(),
                stored: false,
            });
        }
        file.receipts.push(receipt);
        fs::write(&path, serde_json::to_string_pretty(&file)?)?;
        Ok(ReceiptRecord {
            count: file.receipts.len(),
            stored: true,
        })
    }

    fn audit_and_gc(&mut self, now: i64, node_id: &str) -> MaintenanceOutcome {
        let mut outcome = MaintenanceOutcome {
            scanned: 0,
            expired: 0,
            missing_blob: 0,
            corrupted_blob: 0,
            orphan_blobs: 0,
            orphan_receipts: 0,
            size_evicted: 0,
            bytes_freed: 0,
            repair_requests: Vec::new(),
        };
        let retention_secs = if self.config.retention_days == 0 {
            None
        } else {
            Some(self.config.retention_days as i64 * 86_400)
        };
        let mut manifest_ids = HashSet::new();
        let mut size_entries = Vec::new();
        let mut repair_candidates = HashSet::new();
        let max_repairs = self.config.max_repair_requests_per_tick.max(1);

        if let Ok(entries) = fs::read_dir(&self.manifest_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }
                let Some(data_id) = data_id_from_path(&path) else {
                    continue;
                };
                let raw = match fs::read_to_string(&path) {
                    Ok(raw) => raw,
                    Err(err) => {
                        warn!(
                            target: "w1z4rdv1510n::node",
                            error = %err,
                            data_id = data_id.as_str(),
                            "failed to read manifest"
                        );
                        continue;
                    }
                };
                let manifest: DataManifest = match serde_json::from_str(&raw) {
                    Ok(manifest) => manifest,
                    Err(err) => {
                        warn!(
                            target: "w1z4rdv1510n::node",
                            error = %err,
                            data_id = data_id.as_str(),
                            "invalid manifest; removing"
                        );
                        outcome.bytes_freed = outcome.bytes_freed.saturating_add(
                            self.remove_data_entry(&data_id),
                        );
                        continue;
                    }
                };
                if self.config.require_manifest_signature && verify_manifest(&manifest).is_err() {
                    warn!(
                        target: "w1z4rdv1510n::node",
                        data_id = data_id.as_str(),
                        "manifest signature invalid; removing data"
                    );
                    outcome.bytes_freed = outcome.bytes_freed.saturating_add(
                        self.remove_data_entry(&data_id),
                    );
                    continue;
                }
                outcome.scanned = outcome.scanned.saturating_add(1);
                if let Some(retention) = retention_secs {
                    if now.saturating_sub(manifest.timestamp.unix) > retention {
                        outcome.expired = outcome.expired.saturating_add(1);
                        outcome.bytes_freed = outcome.bytes_freed.saturating_add(
                            self.remove_data_entry(&data_id),
                        );
                        continue;
                    }
                }
                manifest_ids.insert(data_id.clone());

                let blob_path = self.blob_path(&data_id);
                if !blob_path.exists() {
                    outcome.missing_blob = outcome.missing_blob.saturating_add(1);
                    let _ = self.prune_receipt(&data_id, node_id);
                    if self.receipt_count(&data_id) < self.config.replication_factor {
                        repair_candidates.insert(data_id.clone());
                    }
                    continue;
                }
                let blob_bytes = fs::metadata(&blob_path)
                    .map(|meta| meta.len())
                    .unwrap_or(0);
                if blob_bytes != manifest.size_bytes as u64 {
                    outcome.corrupted_blob = outcome.corrupted_blob.saturating_add(1);
                    outcome.bytes_freed = outcome
                        .bytes_freed
                        .saturating_add(self.remove_blob(&data_id));
                    let _ = self.prune_receipt(&data_id, node_id);
                    if self.receipt_count(&data_id) < self.config.replication_factor {
                        repair_candidates.insert(data_id.clone());
                    }
                    continue;
                }
                match fs::read(&blob_path) {
                    Ok(payload) => {
                        let hash = compute_payload_hash(&payload);
                        if hash != manifest.payload_hash {
                            outcome.corrupted_blob = outcome.corrupted_blob.saturating_add(1);
                            outcome.bytes_freed = outcome
                                .bytes_freed
                                .saturating_add(self.remove_blob(&data_id));
                            let _ = self.prune_receipt(&data_id, node_id);
                            if self.receipt_count(&data_id) < self.config.replication_factor {
                                repair_candidates.insert(data_id.clone());
                            }
                            continue;
                        }
                    }
                    Err(err) => {
                        warn!(
                            target: "w1z4rdv1510n::node",
                            error = %err,
                            data_id = data_id.as_str(),
                            "failed to read blob"
                        );
                        outcome.missing_blob = outcome.missing_blob.saturating_add(1);
                        let _ = self.prune_receipt(&data_id, node_id);
                        if self.receipt_count(&data_id) < self.config.replication_factor {
                            repair_candidates.insert(data_id.clone());
                        }
                        continue;
                    }
                }
                size_entries.push(ManifestSizeEntry {
                    data_id,
                    timestamp: manifest.timestamp.unix,
                    size_bytes: blob_bytes,
                });
            }
        }

        if self.config.max_storage_bytes > 0 {
            let mut total_bytes: u64 = size_entries.iter().map(|entry| entry.size_bytes).sum();
            if total_bytes > self.config.max_storage_bytes {
                size_entries.sort_by_key(|entry| entry.timestamp);
                for entry in size_entries {
                    if total_bytes <= self.config.max_storage_bytes {
                        break;
                    }
                    let freed = self.remove_data_entry(&entry.data_id);
                    total_bytes = total_bytes.saturating_sub(entry.size_bytes);
                    outcome.bytes_freed = outcome.bytes_freed.saturating_add(freed);
                    outcome.size_evicted = outcome.size_evicted.saturating_add(1);
                    manifest_ids.remove(&entry.data_id);
                }
            }
        }

        if let Ok(entries) = fs::read_dir(&self.blob_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("bin") {
                    continue;
                }
                let Some(data_id) = data_id_from_path(&path) else {
                    continue;
                };
                if manifest_ids.contains(&data_id) {
                    continue;
                }
                outcome.bytes_freed = outcome.bytes_freed.saturating_add(
                    self.remove_data_entry(&data_id),
                );
                outcome.orphan_blobs = outcome.orphan_blobs.saturating_add(1);
            }
        }

        if let Ok(entries) = fs::read_dir(&self.receipt_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                    continue;
                }
                let Some(data_id) = data_id_from_path(&path) else {
                    continue;
                };
                if manifest_ids.contains(&data_id) {
                    continue;
                }
                let _ = fs::remove_file(&path);
                outcome.orphan_receipts = outcome.orphan_receipts.saturating_add(1);
            }
        }

        for data_id in repair_candidates.into_iter().take(max_repairs) {
            outcome.repair_requests.push(data_id);
        }

        outcome
    }

    fn remove_data_entry(&mut self, data_id: &str) -> u64 {
        let bytes = self.remove_blob(data_id);
        let _ = fs::remove_file(self.manifest_path(data_id));
        let _ = fs::remove_file(self.receipt_path(data_id));
        self.pending.remove(data_id);
        bytes
    }

    fn remove_blob(&mut self, data_id: &str) -> u64 {
        let path = self.blob_path(data_id);
        let bytes = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir_all(self.staging_dir.join(data_id));
        bytes
    }

    fn prune_receipt(&mut self, data_id: &str, node_id: &str) -> Result<()> {
        let path = self.receipt_path(data_id);
        if !path.exists() {
            return Ok(());
        }
        let raw = fs::read_to_string(&path)?;
        let mut file: ReceiptFile = serde_json::from_str(&raw)?;
        let original_len = file.receipts.len();
        file.receipts.retain(|receipt| receipt.replica_node_id != node_id);
        if file.receipts.len() == original_len {
            return Ok(());
        }
        if file.receipts.is_empty() {
            let _ = fs::remove_file(&path);
            return Ok(());
        }
        fs::write(&path, serde_json::to_string_pretty(&file)?)?;
        Ok(())
    }

    fn receipt_count(&self, data_id: &str) -> usize {
        let path = self.receipt_path(data_id);
        if !path.exists() {
            return 0;
        }
        let raw = match fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(_) => return 0,
        };
        serde_json::from_str::<ReceiptFile>(&raw)
            .map(|file| file.receipts.len())
            .unwrap_or(0)
    }

    fn status(&self, data_id: &str) -> Result<DataStatusResponse> {
        let manifest = self.load_manifest(data_id)?;
        let receipt_count = self.receipt_count(data_id);
        let state = quorum_state(receipt_count, &self.config);
        Ok(DataStatusResponse {
            data_id: data_id.to_string(),
            stored: self.has_data(data_id),
            receipt_count,
            replication_factor: self.config.replication_factor,
            receipt_quorum: self.config.receipt_quorum,
            quorum_met: state.quorum_met,
            replication_met: state.replication_met,
            pending_quorum: state.pending_quorum,
            pending_replicas: state.pending_replicas,
            manifest,
        })
    }

    fn manifest_path(&self, data_id: &str) -> PathBuf {
        self.manifest_dir.join(format!("{data_id}.json"))
    }

    fn blob_path(&self, data_id: &str) -> PathBuf {
        self.blob_dir.join(format!("{data_id}.bin"))
    }

    fn receipt_path(&self, data_id: &str) -> PathBuf {
        self.receipt_dir.join(format!("{data_id}.json"))
    }

    fn trim_pending(&mut self) {
        let max_pending = self.config.max_pending_chunks.max(1);
        while self.pending_order.len() > max_pending {
            if let Some(data_id) = self.pending_order.pop_front() {
                self.pending.remove(&data_id);
                let _ = fs::remove_dir_all(self.staging_dir.join(&data_id));
            }
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ReceiptFile {
    #[serde(default)]
    receipts: Vec<ReplicationReceipt>,
}

struct ReceiptRecord {
    count: usize,
    stored: bool,
}

struct MaintenanceOutcome {
    scanned: usize,
    expired: usize,
    missing_blob: usize,
    corrupted_blob: usize,
    orphan_blobs: usize,
    orphan_receipts: usize,
    size_evicted: usize,
    bytes_freed: u64,
    repair_requests: Vec<String>,
}

struct ManifestSizeEntry {
    data_id: String,
    timestamp: i64,
    size_bytes: u64,
}

struct QuorumState {
    quorum_met: bool,
    replication_met: bool,
    pending_quorum: usize,
    pending_replicas: usize,
}

fn quorum_state(receipt_count: usize, config: &DataMeshConfig) -> QuorumState {
    let receipt_quorum = config.receipt_quorum.max(1);
    let replication_factor = config.replication_factor.max(receipt_quorum);
    let quorum_met = receipt_count >= receipt_quorum;
    let replication_met = receipt_count >= replication_factor;
    QuorumState {
        quorum_met,
        replication_met,
        pending_quorum: receipt_quorum.saturating_sub(receipt_count),
        pending_replicas: replication_factor.saturating_sub(receipt_count),
    }
}

fn data_id_from_path(path: &Path) -> Option<String> {
    path.file_stem()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
}

fn hex_encode(bytes: &[u8]) -> String {
    const LUT: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(LUT[(b >> 4) as usize] as char);
        out.push(LUT[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_decode(hex: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(hex.len() / 2);
    let mut iter = hex.as_bytes().iter().copied();
    while let Some(high) = iter.next() {
        let low = iter
            .next()
            .ok_or_else(|| anyhow!("hex string has odd length"))?;
        let high_val = hex_value(high)?;
        let low_val = hex_value(low)?;
        out.push((high_val << 4) | low_val);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => anyhow::bail!("invalid hex character"),
    }
}

fn decode_public_key(hex: &str) -> Result<VerifyingKey> {
    let bytes = hex_decode(hex)?;
    if bytes.len() != 32 {
        anyhow::bail!("public key must be 32 bytes");
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    VerifyingKey::from_bytes(&arr).map_err(|err| anyhow!("invalid public key: {err}"))
}

fn decode_signature(hex: &str) -> Result<Signature> {
    let bytes = hex_decode(hex)?;
    Signature::from_slice(&bytes).map_err(|err| anyhow!("invalid signature: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WalletConfig;
    use crate::p2p::test_publisher;
    use crate::wallet::WalletStore;
    use tempfile::tempdir;
    use tokio::time::timeout;
    use w1z4rdv1510n::blockchain::NoopLedger;

    fn test_wallet() -> WalletSigner {
        let dir = tempdir().expect("tmp dir");
        let path = dir.path().join("wallet.json");
        let config = WalletConfig {
            enabled: true,
            path: path.to_string_lossy().into_owned(),
            auto_create: true,
            encrypted: false,
            passphrase_env: String::new(),
            prompt_on_load: false,
        };
        WalletStore::load_or_create_signer(&config).expect("wallet signer")
    }

    #[test]
    fn store_roundtrip_manifest_and_payload() {
        let dir = tempdir().expect("tmp dir");
        let mut config = DataMeshConfig::default();
        config.storage_path = dir.path().to_string_lossy().into_owned();
        let mut store = DataStore::new(config).expect("store");
        let manifest = DataManifest {
            data_id: "d1".to_string(),
            node_id: "node-x".to_string(),
            sensor_id: "sensor-1".to_string(),
            timestamp: Timestamp { unix: 10 },
            payload_kind: "test".to_string(),
            payload_hash: "hash".to_string(),
            size_bytes: 4,
            chunk_count: 1,
            public_key: "pub".to_string(),
            signature: "sig".to_string(),
        };
        store.store_manifest(&manifest).expect("manifest write");
        let loaded = store.load_manifest("d1").expect("load").expect("manifest");
        assert_eq!(loaded.data_id, "d1");
        store.store_payload("d1", b"data").expect("payload");
        assert!(store.has_data("d1"));
    }

    #[tokio::test]
    async fn ingest_creates_receipt() {
        let dir = tempdir().expect("tmp dir");
        let mut config = DataMeshConfig::default();
        config.storage_path = dir.path().to_string_lossy().into_owned();
        config.replication_factor = 1;
        config.receipt_quorum = 1;
        config.maintenance_enabled = false;
        let wallet = Arc::new(test_wallet());
        let node_id = node_id_from_wallet(&wallet.wallet().address);
        let ledger: Arc<dyn BlockchainLedger> = Arc::new(NoopLedger::default());
        let publisher = test_publisher();
        let (net_tx, net_rx) = mpsc::unbounded_channel();
        let handle = start_data_mesh(
            config,
            node_id,
            wallet,
            ledger,
            publisher,
            net_rx,
        )
        .expect("mesh");
        let request = DataIngestRequest {
            sensor_id: "sensor-1".to_string(),
            payload_kind: "text".to_string(),
            payload_text: Some("hello".to_string()),
            payload_hex: None,
            payload_json: None,
            timestamp_unix: Some(10),
        };
        let response = handle.ingest(request).await.expect("ingest");
        assert!(response.receipt_count >= 1);
        assert!(response.quorum_met);
        assert!(response.replication_met);
        drop(dir);
        drop(net_tx);
    }

    #[tokio::test]
    async fn ingest_emits_payload_ready_event() {
        let dir = tempdir().expect("tmp dir");
        let mut config = DataMeshConfig::default();
        config.storage_path = dir.path().to_string_lossy().into_owned();
        config.replication_factor = 1;
        config.receipt_quorum = 1;
        config.maintenance_enabled = false;
        let wallet = Arc::new(test_wallet());
        let expected_node_id = node_id_from_wallet(&wallet.wallet().address);
        let ledger: Arc<dyn BlockchainLedger> = Arc::new(NoopLedger::default());
        let publisher = test_publisher();
        let (_net_tx, net_rx) = mpsc::unbounded_channel();
        let handle = start_data_mesh(
            config,
            expected_node_id.clone(),
            wallet,
            ledger,
            publisher,
            net_rx,
        )
        .expect("mesh");
        let mut events = handle.subscribe();
        let request = DataIngestRequest {
            sensor_id: "sensor-1".to_string(),
            payload_kind: "neural.fabric.v1".to_string(),
            payload_text: Some("hello".to_string()),
            payload_hex: None,
            payload_json: None,
            timestamp_unix: Some(20),
        };
        let response = handle.ingest(request).await.expect("ingest");
        let event = timeout(Duration::from_secs(2), events.recv())
            .await
            .expect("event ready")
            .expect("event recv");
        let data_id = match event {
            DataMeshEvent::PayloadReady {
                data_id,
                payload_kind,
                payload_hash,
                timestamp,
                node_id,
                sensor_id,
            } => {
                assert_eq!(payload_kind, "neural.fabric.v1");
                assert!(!payload_hash.is_empty());
                assert_eq!(timestamp.unix, 20);
                assert_eq!(node_id, expected_node_id);
                assert_eq!(sensor_id, "sensor-1");
                data_id
            }
            DataMeshEvent::NodeMetrics { .. } => {
                panic!("unexpected node metrics event in test")
            }
        };
        let payload = handle.load_payload(data_id).await.expect("payload");
        assert_eq!(payload, b"hello");
        assert_eq!(response.chunk_count, 1);
        drop(dir);
    }

    #[tokio::test]
    async fn next_fabric_share_reads_payload() {
        let dir = tempdir().expect("tmp dir");
        let mut config = DataMeshConfig::default();
        config.storage_path = dir.path().to_string_lossy().into_owned();
        config.replication_factor = 1;
        config.receipt_quorum = 1;
        config.maintenance_enabled = false;
        let wallet = Arc::new(test_wallet());
        let node_id = node_id_from_wallet(&wallet.wallet().address);
        let ledger: Arc<dyn BlockchainLedger> = Arc::new(NoopLedger::default());
        let publisher = test_publisher();
        let (_net_tx, net_rx) = mpsc::unbounded_channel();
        let handle = start_data_mesh(
            config,
            node_id.clone(),
            wallet,
            ledger,
            publisher,
            net_rx,
        )
        .expect("mesh");
        let share = NeuralFabricShare {
            node_id: node_id.clone(),
            timestamp: Timestamp { unix: 30 },
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: Vec::new(),
            metacognition: None,
            metadata: HashMap::new(),
        };
        let mut events = handle.subscribe();
        let response = handle
            .ingest_fabric_share(&share)
            .await
            .expect("share ingest");
        let received = timeout(Duration::from_secs(2), handle.next_fabric_share(&mut events))
            .await
            .expect("share ready")
            .expect("share payload");
        assert_eq!(received.node_id, node_id);
        assert_eq!(received.timestamp.unix, 30);
        assert_eq!(response.chunk_count, 1);
        drop(dir);
    }

    #[tokio::test]
    async fn next_stream_envelope_reads_payload() {
        let dir = tempdir().expect("tmp dir");
        let mut config = DataMeshConfig::default();
        config.storage_path = dir.path().to_string_lossy().into_owned();
        config.replication_factor = 1;
        config.receipt_quorum = 1;
        config.maintenance_enabled = false;
        let wallet = Arc::new(test_wallet());
        let node_id = node_id_from_wallet(&wallet.wallet().address);
        let ledger: Arc<dyn BlockchainLedger> = Arc::new(NoopLedger::default());
        let publisher = test_publisher();
        let (_net_tx, net_rx) = mpsc::unbounded_channel();
        let handle = start_data_mesh(
            config,
            node_id,
            wallet,
            ledger,
            publisher,
            net_rx,
        )
        .expect("mesh");
        let envelope = StreamEnvelope {
            source: w1z4rdv1510n::streaming::StreamSource::PeopleVideo,
            timestamp: Timestamp { unix: 40 },
            payload: w1z4rdv1510n::streaming::StreamPayload::Json {
                value: serde_json::json!({ "entity_id": "e1", "signal": 1.0 }),
            },
            metadata: HashMap::new(),
        };
        let mut events = handle.subscribe();
        let response = handle
            .ingest_stream_envelope("sensor-stream".to_string(), &envelope)
            .await
            .expect("stream ingest");
        let received = timeout(Duration::from_secs(2), handle.next_stream_envelope(&mut events))
            .await
            .expect("stream ready")
            .expect("stream payload");
        assert_eq!(received.timestamp.unix, 40);
        assert_eq!(response.chunk_count, 1);
        drop(dir);
    }
}
