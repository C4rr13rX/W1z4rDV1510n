use crate::config::NodeConfig;
use crate::data_mesh::DataManifest;
use anyhow::{Result, anyhow};
use std::fs;
use std::path::{Path, PathBuf};
use w1z4rdv1510n::streaming::{LabelQueueReport, NeuralFabricShare, SubnetworkReport, VisualLabelReport};

pub fn parse_label_queue(share: &NeuralFabricShare) -> Option<LabelQueueReport> {
    let value = share.metadata.get("label_queue")?;
    serde_json::from_value(value.clone()).ok()
}

pub fn parse_visual_label_queue(share: &NeuralFabricShare) -> Option<VisualLabelReport> {
    let value = share.metadata.get("visual_label_queue")?;
    serde_json::from_value(value.clone()).ok()
}

pub fn parse_subnet_report(share: &NeuralFabricShare) -> Option<SubnetworkReport> {
    let value = share.metadata.get("subnet_registry")?;
    serde_json::from_value(value.clone()).ok()
}

pub fn load_latest_fabric_share(config: &NodeConfig) -> Result<Option<NeuralFabricShare>> {
    if !config.data.enabled {
        anyhow::bail!("data mesh is disabled");
    }
    if !config.data.host_storage {
        anyhow::bail!("data mesh storage is disabled");
    }
    let storage_path = PathBuf::from(&config.data.storage_path);
    let manifest_dir = storage_path.join("manifests");
    let blob_dir = storage_path.join("blobs");
    load_latest_fabric_share_from_dirs(
        &manifest_dir,
        &blob_dir,
        &config.streaming.share_payload_kind,
    )
}

pub fn load_latest_subnet_report(config: &NodeConfig) -> Result<Option<SubnetworkReport>> {
    let share = load_latest_fabric_share(config)?;
    Ok(share.as_ref().and_then(parse_subnet_report))
}

fn load_latest_fabric_share_from_dirs(
    manifest_dir: &Path,
    blob_dir: &Path,
    share_kind: &str,
) -> Result<Option<NeuralFabricShare>> {
    if !manifest_dir.exists() {
        return Ok(None);
    }
    let manifest = match load_latest_manifest(manifest_dir, share_kind)? {
        Some(manifest) => manifest,
        None => return Ok(None),
    };
    let blob_path = blob_dir.join(format!("{}.bin", manifest.data_id));
    if !blob_path.exists() {
        return Ok(None);
    }
    let payload = fs::read(&blob_path)
        .map_err(|err| anyhow!("read payload {}: {err}", blob_path.display()))?;
    let share = serde_json::from_slice::<NeuralFabricShare>(&payload)
        .map_err(|err| anyhow!("decode neural fabric share: {err}"))?;
    Ok(Some(share))
}

fn load_latest_manifest(manifest_dir: &Path, share_kind: &str) -> Result<Option<DataManifest>> {
    let mut latest: Option<DataManifest> = None;
    let entries = match fs::read_dir(manifest_dir) {
        Ok(entries) => entries,
        Err(err) => return Err(anyhow!("read manifest dir {}: {err}", manifest_dir.display())),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let raw = match fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(_) => continue,
        };
        let manifest = match serde_json::from_str::<DataManifest>(&raw) {
            Ok(manifest) => manifest,
            Err(_) => continue,
        };
        if manifest.payload_kind != share_kind {
            continue;
        }
        let replace = match latest.as_ref() {
            Some(current) => manifest.timestamp.unix > current.timestamp.unix,
            None => true,
        };
        if replace {
            latest = Some(manifest);
        }
    }
    Ok(latest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::tempdir;
    use w1z4rdv1510n::blockchain::WorkKind;
    use w1z4rdv1510n::schema::Timestamp;
    use w1z4rdv1510n::streaming::{LabelCandidate, VisualLabelTask};

    #[test]
    fn parse_label_queue_from_metadata() {
        let candidate = LabelCandidate {
            id: "c1".to_string(),
            work_id: "w1".to_string(),
            work_kind: WorkKind::HumanAnnotation,
            timestamp: Timestamp { unix: 1 },
            summary: "label".to_string(),
            entity_id: None,
            feature: "token:BehavioralAtom:motion_energy".to_string(),
            priority: 0.5,
            reward_score: 0.5,
            source: "test".to_string(),
            stream_source: None,
            source_quality: None,
            evidence: HashMap::new(),
        };
        let report = LabelQueueReport {
            timestamp: Timestamp { unix: 1 },
            pending: vec![candidate],
            total_pending: 1,
        };
        let mut share = NeuralFabricShare {
            node_id: "node-a".to_string(),
            timestamp: Timestamp { unix: 1 },
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: Vec::new(),
            metacognition: None,
            metadata: HashMap::new(),
        };
        share
            .metadata
            .insert("label_queue".to_string(), serde_json::to_value(&report).unwrap());
        let parsed = parse_label_queue(&share).expect("parsed");
        assert_eq!(parsed.total_pending, 1);
    }

    #[test]
    fn parse_visual_label_queue_from_metadata() {
        let task = VisualLabelTask {
            id: "v1".to_string(),
            work_id: "w1".to_string(),
            work_kind: WorkKind::HumanAnnotation,
            timestamp: Timestamp { unix: 2 },
            summary: "label region".to_string(),
            entity_id: None,
            frame_id: Some("frame-1".to_string()),
            image_ref: None,
            bbox: serde_json::json!([0.0, 0.0, 10.0, 10.0]),
            label_hint: Some("hand".to_string()),
            priority: 0.7,
            reward_score: 0.7,
            evidence: HashMap::new(),
        };
        let report = VisualLabelReport {
            timestamp: Timestamp { unix: 2 },
            pending: vec![task],
            total_pending: 1,
        };
        let mut share = NeuralFabricShare {
            node_id: "node-b".to_string(),
            timestamp: Timestamp { unix: 2 },
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: Vec::new(),
            metacognition: None,
            metadata: HashMap::new(),
        };
        share
            .metadata
            .insert("visual_label_queue".to_string(), serde_json::to_value(&report).unwrap());
        let parsed = parse_visual_label_queue(&share).expect("parsed");
        assert_eq!(parsed.pending.len(), 1);
    }

    #[test]
    fn parse_subnet_report_from_metadata() {
        let report = w1z4rdv1510n::streaming::SubnetworkReport {
            timestamp: Timestamp { unix: 1 },
            total_subnets: 1,
            active_subnets: 1,
            snapshots: Vec::new(),
            coactivity: HashMap::new(),
        };
        let mut share = NeuralFabricShare {
            node_id: "node-c".to_string(),
            timestamp: Timestamp { unix: 1 },
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: Vec::new(),
            metacognition: None,
            metadata: HashMap::new(),
        };
        share
            .metadata
            .insert("subnet_registry".to_string(), serde_json::to_value(&report).unwrap());
        let parsed = parse_subnet_report(&share).expect("parsed");
        assert_eq!(parsed.total_subnets, 1);
    }

    #[test]
    fn load_latest_fabric_share_from_storage() {
        let dir = tempdir().expect("tempdir");
        let manifest_dir = dir.path().join("manifests");
        let blob_dir = dir.path().join("blobs");
        fs::create_dir_all(&manifest_dir).expect("manifest dir");
        fs::create_dir_all(&blob_dir).expect("blob dir");

        let share_kind = "neural.fabric.v1";
        let older = DataManifest {
            data_id: "old".to_string(),
            node_id: "node-a".to_string(),
            sensor_id: "sensor".to_string(),
            timestamp: Timestamp { unix: 10 },
            payload_kind: share_kind.to_string(),
            payload_hash: "hash-old".to_string(),
            size_bytes: 1,
            chunk_count: 1,
            public_key: "pk".to_string(),
            signature: "sig".to_string(),
        };
        let newer = DataManifest {
            data_id: "new".to_string(),
            node_id: "node-b".to_string(),
            sensor_id: "sensor".to_string(),
            timestamp: Timestamp { unix: 20 },
            payload_kind: share_kind.to_string(),
            payload_hash: "hash-new".to_string(),
            size_bytes: 1,
            chunk_count: 1,
            public_key: "pk".to_string(),
            signature: "sig".to_string(),
        };
        fs::write(
            manifest_dir.join("old.json"),
            serde_json::to_vec(&older).expect("serialize"),
        )
        .expect("write older");
        fs::write(
            manifest_dir.join("new.json"),
            serde_json::to_vec(&newer).expect("serialize"),
        )
        .expect("write newer");

        let share = NeuralFabricShare {
            node_id: "node-b".to_string(),
            timestamp: Timestamp { unix: 20 },
            tokens: Vec::new(),
            layers: Vec::new(),
            motifs: Vec::new(),
            motif_transitions: Vec::new(),
            network_patterns: Vec::new(),
            metacognition: None,
            metadata: HashMap::new(),
        };
        let payload = serde_json::to_vec(&share).expect("share payload");
        fs::write(blob_dir.join("new.bin"), payload).expect("write blob");

        let loaded =
            load_latest_fabric_share_from_dirs(&manifest_dir, &blob_dir, share_kind)
                .expect("load");
        let loaded = loaded.expect("some");
        assert_eq!(loaded.node_id, "node-b");
    }
}
