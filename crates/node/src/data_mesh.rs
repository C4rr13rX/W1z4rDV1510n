use crate::config::DataMeshConfig;
use crate::p2p::NetworkPublisher;
use crate::wallet::{WalletSigner, address_from_public_key, node_id_from_wallet};
use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::warn;
use w1z4rdv1510n::blockchain::{BlockchainLedger, SensorCommitment};
use w1z4rdv1510n::network::{NetworkEnvelope, compute_payload_hash};
use w1z4rdv1510n::schema::Timestamp;

const DATA_MESSAGE_KIND: &str = "data.message";

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
    NetworkEnvelope(NetworkEnvelope),
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
    let mut store = DataStore::new(config.clone())?;
    let node_id_clone = node_id.clone();
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
            loop {
                tokio::select! {
                    Some(command) = command_rx.recv() => {
                        handle_command(
                            command,
                            &config,
                            &node_id_clone,
                            &wallet,
                            &ledger,
                            &publisher,
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
                            &mut store,
                        );
                    }
                }
            }
        });
    });
    Ok(DataMeshHandle { command_tx })
}

fn handle_command(
    command: DataMeshCommand,
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    ledger: &Arc<dyn BlockchainLedger>,
    publisher: &NetworkPublisher,
    store: &mut DataStore,
) {
    match command {
        DataMeshCommand::Ingest { request, respond_to } => {
            let response = ingest_local(request, config, node_id, wallet, ledger, publisher, store);
            let _ = respond_to.send(response);
        }
        DataMeshCommand::Status { data_id, respond_to } => {
            let response = store.status(&data_id);
            let _ = respond_to.send(response);
        }
        DataMeshCommand::NetworkEnvelope(envelope) => {
            if envelope.payload_kind != DATA_MESSAGE_KIND {
                return;
            }
            if let Ok(bytes) = envelope.payload_bytes() {
                if let Ok(message) = serde_json::from_slice::<DataMessage>(&bytes) {
                    handle_network_message(message, config, node_id, wallet, publisher, store);
                }
            }
        }
    }
}

fn ingest_local(
    request: DataIngestRequest,
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    ledger: &Arc<dyn BlockchainLedger>,
    publisher: &NetworkPublisher,
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
    let receipt = build_receipt(&data_id, &payload_hash, node_id, wallet);
    let receipt_record = store.record_receipt(receipt.clone())?;
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
    let state = quorum_state(receipt_count, config);
    Ok(DataIngestResponse {
        data_id,
        payload_hash,
        chunk_count,
        stored: true,
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
    config: &DataMeshConfig,
    node_id: &str,
    wallet: &WalletSigner,
    publisher: &NetworkPublisher,
    store: &mut DataStore,
) {
    match message {
        DataMessage::Manifest(manifest) => {
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
            let manifest = store.load_manifest(&chunk.data_id).ok().flatten();
            if store.apply_chunk(chunk, manifest.as_ref()).is_ok() {
                if let Some(manifest) = manifest {
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
        fs::read(path).map_err(|err| anyhow!("read payload: {err}"))
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
        let wallet = Arc::new(test_wallet());
        let ledger: Arc<dyn BlockchainLedger> = Arc::new(NoopLedger::default());
        let publisher = test_publisher();
        let (net_tx, net_rx) = mpsc::unbounded_channel();
        let handle = start_data_mesh(
            config,
            "node-test".to_string(),
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
        drop(net_tx);
    }
}
