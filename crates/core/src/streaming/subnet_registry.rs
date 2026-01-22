use crate::config::SubnetworkRegistryConfig;
use crate::schema::Timestamp;
use crate::spike::{NeuronKind, SpikeConfig, SpikeFrame, SpikeInput, SpikePool};
use crate::streaming::neuro_bridge::SubstreamReport;
use crate::streaming::schema::TokenBatch;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use tracing::warn;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetworkSnapshot {
    pub key: String,
    pub stability: f64,
    pub last_seen: Timestamp,
    pub last_activity: f64,
    pub last_quality: f64,
    pub last_matches: u64,
    pub neuron_count: usize,
    pub spike_count: usize,
    pub origin: String,
    #[serde(default)]
    pub last_frame: Option<SpikeFrame>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetworkReport {
    pub timestamp: Timestamp,
    pub total_subnets: usize,
    pub active_subnets: usize,
    #[serde(default)]
    pub snapshots: Vec<SubnetworkSnapshot>,
    #[serde(default)]
    pub coactivity: HashMap<String, HashMap<String, f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubnetworkRegistryState {
    pub saved_at: Timestamp,
    pub report: SubnetworkReport,
}

#[derive(Debug, Clone)]
struct SubstreamSignal {
    signal: f64,
    quality: f64,
    matches: u64,
}

#[derive(Debug, Clone)]
struct SubnetSeed {
    stability: f64,
    last_seen: Timestamp,
    origin: String,
}

struct SubstreamSubnet {
    key: String,
    stability: f64,
    last_seen: Timestamp,
    last_activity: f64,
    last_quality: f64,
    last_matches: u64,
    origin: String,
    pool: SpikePool,
    input_neuron: u32,
    stability_neuron: u32,
    inhibitory_neuron: u32,
    last_frame: Option<SpikeFrame>,
}

impl SubstreamSubnet {
    fn new(key: String, stability: f64, now: Timestamp, config: &SubnetworkRegistryConfig, origin: String) -> Self {
        let spike_config = SpikeConfig {
            threshold: config.spike_threshold,
            membrane_decay: config.membrane_decay,
            refractory_steps: config.refractory_steps,
        };
        let mut pool = SpikePool::new(format!("subnet::{}", key), spike_config);
        let input_neuron = pool.add_neuron(NeuronKind::Excitatory);
        let stability_neuron = pool.add_neuron(NeuronKind::Excitatory);
        let inhibitory_neuron = pool.add_neuron(NeuronKind::Inhibitory);
        pool.connect(input_neuron, stability_neuron, 0.6);
        pool.connect(stability_neuron, input_neuron, 0.4);
        pool.connect(inhibitory_neuron, input_neuron, 0.5);
        Self {
            key,
            stability,
            last_seen: now,
            last_activity: 0.0,
            last_quality: 0.0,
            last_matches: 0,
            origin,
            pool,
            input_neuron,
            stability_neuron,
            inhibitory_neuron,
            last_frame: None,
        }
    }

    fn update(
        &mut self,
        now: Timestamp,
        stability: f64,
        signal: Option<&SubstreamSignal>,
        max_inputs: usize,
    ) {
        self.stability = stability;
        self.last_seen = now;
        let mut inputs = Vec::new();
        if let Some(signal) = signal {
            self.last_activity = signal.signal;
            self.last_quality = signal.quality;
            self.last_matches = signal.matches;
            let excitatory = (signal.signal * signal.quality).clamp(0.0, 4.0);
            let inhibitory = (1.0 - signal.quality).clamp(0.0, 1.0);
            inputs.push(SpikeInput {
                target: self.input_neuron,
                excitatory: excitatory as f32,
                inhibitory: 0.0,
            });
            inputs.push(SpikeInput {
                target: self.inhibitory_neuron,
                excitatory: 0.0,
                inhibitory: inhibitory as f32,
            });
        }
        let stability_drive = stability.clamp(0.0, 1.0) as f32;
        inputs.push(SpikeInput {
            target: self.stability_neuron,
            excitatory: stability_drive,
            inhibitory: (1.0 - stability_drive).clamp(0.0, 1.0),
        });
        if inputs.len() > max_inputs.max(1) {
            inputs.truncate(max_inputs.max(1));
        }
        self.pool.enqueue_inputs(inputs);
        self.last_frame = Some(self.pool.step(now));
    }
}

#[derive(Debug, Clone)]
struct CoactivityState {
    count: u64,
    last_seen: Timestamp,
}

pub struct SubnetworkRegistry {
    config: SubnetworkRegistryConfig,
    subnets: HashMap<String, SubstreamSubnet>,
    seeds: HashMap<String, SubnetSeed>,
    coactivity: HashMap<(String, String), CoactivityState>,
    state_path: Option<PathBuf>,
    last_persist_unix: Option<i64>,
}

impl SubnetworkRegistry {
    pub fn new(config: SubnetworkRegistryConfig) -> Self {
        let state_path = if config.persist_state && !config.state_path.trim().is_empty() {
            Some(PathBuf::from(config.state_path.clone()))
        } else {
            None
        };
        let registry = Self {
            config,
            subnets: HashMap::new(),
            seeds: HashMap::new(),
            coactivity: HashMap::new(),
            state_path,
            last_persist_unix: None,
        };
        registry
    }

    pub fn load_state(&mut self) {
        let Some(path) = self.state_path.as_ref() else {
            return;
        };
        if !path.exists() {
            return;
        }
        let raw = match fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::streaming",
                    error = %err,
                    path = %path.display(),
                    "failed to read subnetwork registry state"
                );
                return;
            }
        };
        let state: SubnetworkRegistryState = match serde_json::from_str(&raw) {
            Ok(state) => state,
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::streaming",
                    error = %err,
                    path = %path.display(),
                    "failed to decode subnetwork registry state"
                );
                return;
            }
        };
        self.ingest_shared(&state.report);
        self.last_persist_unix = Some(state.saved_at.unix);
    }

    pub fn ingest_shared(&mut self, report: &SubnetworkReport) {
        if !self.config.enabled {
            return;
        }
        for snapshot in &report.snapshots {
            let seed = SubnetSeed {
                stability: snapshot.stability,
                last_seen: report.timestamp,
                origin: "shared".to_string(),
            };
            self.seeds
                .entry(snapshot.key.clone())
                .and_modify(|existing| {
                    if report.timestamp.unix >= existing.last_seen.unix {
                        existing.last_seen = report.timestamp;
                    }
                    if snapshot.stability > existing.stability {
                        existing.stability = snapshot.stability;
                    }
                })
                .or_insert(seed);
        }
        for (a, links) in &report.coactivity {
            for (b, score) in links {
                let (left, right) = ordered_pair(a, b);
                let count = (*score * 10.0).round().max(1.0) as u64;
                let entry = self.coactivity.entry((left, right)).or_insert(CoactivityState {
                    count: 0,
                    last_seen: report.timestamp,
                });
                entry.count = entry.count.max(count);
                entry.last_seen = report.timestamp;
            }
        }
    }

    pub fn update(&mut self, batch: &TokenBatch, report: &SubstreamReport) -> Option<SubnetworkReport> {
        if !self.config.enabled {
            return None;
        }
        let now = report.timestamp;
        let signals = collect_substream_signals(batch);
        let mut active_keys = Vec::new();
        for item in &report.items {
            let stability = item.stability as f64;
            if stability < self.config.min_stability {
                continue;
            }
            let origin = match item.origin {
                crate::streaming::neuro_bridge::SubstreamOrigin::Neuro => "neuro",
                crate::streaming::neuro_bridge::SubstreamOrigin::Auto => "auto",
            };
            let subnet = self.subnets.entry(item.key.clone()).or_insert_with(|| {
                SubstreamSubnet::new(
                    item.key.clone(),
                    stability,
                    now,
                    &self.config,
                    origin.to_string(),
                )
            });
            let signal = signals.get(&item.key);
            subnet.update(now, stability, signal, self.config.max_inputs_per_subnet);
            active_keys.push(item.key.clone());
        }
        for (key, seed) in self.seeds.clone() {
            if self.subnets.contains_key(&key) {
                continue;
            }
            if seed.stability < self.config.min_stability {
                continue;
            }
            let subnet = SubstreamSubnet::new(
                key.clone(),
                seed.stability,
                now,
                &self.config,
                seed.origin.clone(),
            );
            self.subnets.insert(key, subnet);
        }
        self.prune(now);
        self.update_coactivity(&active_keys, now);
        let snapshots = self.build_snapshots();
        let report = SubnetworkReport {
            timestamp: now,
            total_subnets: self.subnets.len(),
            active_subnets: active_keys.len(),
            snapshots,
            coactivity: self.coactivity_report(),
        };
        self.maybe_persist(&report);
        Some(report)
    }

    fn prune(&mut self, now: Timestamp) {
        let ttl = self.config.idle_ttl_secs.max(1) as i64;
        self.subnets.retain(|_, subnet| {
            now.unix.saturating_sub(subnet.last_seen.unix) <= ttl
        });
        let seed_ttl = ttl * 2;
        self.seeds.retain(|_, seed| {
            now.unix.saturating_sub(seed.last_seen.unix) <= seed_ttl
        });
        let coactivity_ttl = self.config.coactivity_ttl_secs.max(1) as i64;
        self.coactivity.retain(|_, state| {
            now.unix.saturating_sub(state.last_seen.unix) <= coactivity_ttl
        });
        if self.subnets.len() > self.config.max_subnets.max(1) {
            let mut items = self
                .subnets
                .iter()
                .map(|(key, subnet)| (key.clone(), subnet.stability))
                .collect::<Vec<_>>();
            items.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            let keep: HashSet<String> = items
                .into_iter()
                .take(self.config.max_subnets.max(1))
                .map(|item| item.0)
                .collect();
            self.subnets.retain(|key, _| keep.contains(key));
        }
    }

    fn build_snapshots(&self) -> Vec<SubnetworkSnapshot> {
        let mut snapshots = Vec::new();
        for subnet in self.subnets.values() {
            let spike_count = subnet
                .last_frame
                .as_ref()
                .map(|frame| frame.spikes.len())
                .unwrap_or(0);
            snapshots.push(SubnetworkSnapshot {
                key: subnet.key.clone(),
                stability: subnet.stability,
                last_seen: subnet.last_seen,
                last_activity: subnet.last_activity,
                last_quality: subnet.last_quality,
                last_matches: subnet.last_matches,
                neuron_count: subnet.pool.neuron_count(),
                spike_count,
                origin: subnet.origin.clone(),
                last_frame: if self.config.embed_in_snapshot {
                    subnet.last_frame.clone()
                } else {
                    None
                },
            });
        }
        snapshots
    }

    fn update_coactivity(&mut self, active: &[String], now: Timestamp) {
        let mut active = active.to_vec();
        active.sort();
        for i in 0..active.len() {
            for j in (i + 1)..active.len() {
                let key = (active[i].clone(), active[j].clone());
                let entry = self.coactivity.entry(key).or_insert(CoactivityState {
                    count: 0,
                    last_seen: now,
                });
                entry.count = entry.count.saturating_add(1);
                entry.last_seen = now;
            }
        }
    }

    fn coactivity_report(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut max_count = 1u64;
        for state in self.coactivity.values() {
            if state.count > max_count {
                max_count = state.count;
            }
        }
        let mut output: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for ((a, b), state) in &self.coactivity {
            let score = state.count as f64 / max_count as f64;
            output
                .entry(a.clone())
                .or_default()
                .insert(b.clone(), score);
            output
                .entry(b.clone())
                .or_default()
                .insert(a.clone(), score);
        }
        output
    }

    fn maybe_persist(&mut self, report: &SubnetworkReport) {
        if !self.config.persist_state {
            return;
        }
        let Some(path) = self.state_path.as_ref() else {
            return;
        };
        let now = report.timestamp.unix;
        let interval = self.config.persist_interval_secs.max(1) as i64;
        if let Some(last) = self.last_persist_unix {
            if now.saturating_sub(last) < interval {
                return;
            }
        }
        let state = SubnetworkRegistryState {
            saved_at: Timestamp { unix: now },
            report: report.clone(),
        };
        let payload = match serde_json::to_string_pretty(&state) {
            Ok(payload) => payload,
            Err(err) => {
                warn!(
                    target: "w1z4rdv1510n::streaming",
                    error = %err,
                    "failed to serialize subnetwork registry state"
                );
                return;
            }
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                if let Err(err) = fs::create_dir_all(parent) {
                    warn!(
                        target: "w1z4rdv1510n::streaming",
                        error = %err,
                        path = %parent.display(),
                        "failed to create subnetwork registry state directory"
                    );
                    return;
                }
            }
        }
        let tmp_path = path.with_extension("tmp");
        if let Err(err) = fs::write(&tmp_path, payload.as_bytes()) {
            warn!(
                target: "w1z4rdv1510n::streaming",
                error = %err,
                path = %tmp_path.display(),
                "failed to write subnetwork registry state"
            );
            return;
        }
        if let Err(err) = fs::rename(&tmp_path, path) {
            warn!(
                target: "w1z4rdv1510n::streaming",
                error = %err,
                path = %path.display(),
                "failed to finalize subnetwork registry state"
            );
            return;
        }
        self.last_persist_unix = Some(now);
    }
}

fn ordered_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

fn collect_substream_signals(batch: &TokenBatch) -> HashMap<String, SubstreamSignal> {
    let mut map: HashMap<String, (f64, f64, u64)> = HashMap::new();
    for token in &batch.tokens {
        let Some(key) = token
            .attributes
            .get("substream_key")
            .and_then(|val| val.as_str())
        else {
            continue;
        };
        let signal = token
            .attributes
            .get("substream_signal")
            .and_then(|val| val.as_f64())
            .unwrap_or(0.0);
        let quality = token
            .attributes
            .get("substream_quality")
            .and_then(|val| val.as_f64())
            .unwrap_or(token.confidence)
            .clamp(0.0, 1.0);
        let matches = token
            .attributes
            .get("substream_matches")
            .and_then(|val| val.as_u64())
            .unwrap_or(1);
        let entry = map.entry(key.to_string()).or_insert((0.0, 0.0, 0));
        entry.0 += signal * quality;
        entry.1 += quality;
        entry.2 = entry.2.saturating_add(matches);
    }
    let mut out = HashMap::new();
    for (key, (signal_sum, quality_sum, matches)) in map {
        let quality = if quality_sum > 0.0 {
            (quality_sum / (matches.max(1) as f64)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let signal = if quality_sum > 0.0 {
            signal_sum / quality_sum
        } else {
            0.0
        };
        out.insert(
            key,
            SubstreamSignal {
                signal,
                quality,
                matches,
            },
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::neuro_bridge::{SubstreamOrigin, SubstreamReportItem};
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState};
    use tempfile::tempdir;
    use serde_json::Value;

    #[test]
    fn subnet_registry_spawns_and_reports() {
        let mut registry = SubnetworkRegistry::new(SubnetworkRegistryConfig::default());
        let report = SubstreamReport {
            timestamp: Timestamp { unix: 10 },
            total_substreams: 1,
            items: vec![SubstreamReportItem {
                key: "auto::attr::speed::fast".to_string(),
                origin: SubstreamOrigin::Auto,
                stability: 0.8,
                attrs: HashMap::new(),
                roles: Vec::new(),
                last_seen: Timestamp { unix: 10 },
                last_signal: 1.0,
                last_quality: 0.8,
                last_matches: 2,
                signal_ema: 0.6,
            }],
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralToken,
                onset: Timestamp { unix: 10 },
                duration_secs: 1.0,
                confidence: 0.8,
                attributes: HashMap::from([
                    ("substream_key".to_string(), Value::String("auto::attr::speed::fast".to_string())),
                    ("substream_signal".to_string(), Value::from(0.7)),
                    ("substream_quality".to_string(), Value::from(0.9)),
                ]),
                source: None,
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 10 },
                phase: 0.0,
                amplitude: 0.0,
                coherence: 0.0,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        let snapshot = registry.update(&batch, &report).expect("report");
        assert_eq!(snapshot.total_subnets, 1);
        assert_eq!(snapshot.snapshots.len(), 1);
    }

    #[test]
    fn subnet_registry_prunes_by_ttl() {
        let mut config = SubnetworkRegistryConfig::default();
        config.idle_ttl_secs = 1;
        let mut registry = SubnetworkRegistry::new(config);
        let report = SubstreamReport {
            timestamp: Timestamp { unix: 1 },
            total_substreams: 1,
            items: vec![SubstreamReportItem {
                key: "auto::attr::speed::slow".to_string(),
                origin: SubstreamOrigin::Auto,
                stability: 0.9,
                attrs: HashMap::new(),
                roles: Vec::new(),
                last_seen: Timestamp { unix: 1 },
                last_signal: 1.0,
                last_quality: 0.8,
                last_matches: 2,
                signal_ema: 0.6,
            }],
        };
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 1 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        registry.update(&batch, &report).expect("report");
        let report_late = SubstreamReport {
            timestamp: Timestamp { unix: 10 },
            total_substreams: 0,
            items: Vec::new(),
        };
        let batch_late = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let snapshot = registry.update(&batch_late, &report_late).expect("report");
        assert_eq!(snapshot.total_subnets, 0);
    }

    #[test]
    fn subnet_registry_persists_state() {
        let dir = tempdir().expect("tempdir");
        let state_path = dir.path().join("subnet_state.json");
        let mut config = SubnetworkRegistryConfig::default();
        config.persist_state = true;
        config.state_path = state_path.to_string_lossy().into_owned();
        let mut registry = SubnetworkRegistry::new(config);
        let report = SubnetworkReport {
            timestamp: Timestamp { unix: 5 },
            total_subnets: 1,
            active_subnets: 1,
            snapshots: vec![SubnetworkSnapshot {
                key: "auto::attr::speed::steady".to_string(),
                stability: 0.7,
                last_seen: Timestamp { unix: 5 },
                last_activity: 0.3,
                last_quality: 0.8,
                last_matches: 2,
                neuron_count: 3,
                spike_count: 1,
                origin: "auto".to_string(),
                last_frame: None,
            }],
            coactivity: HashMap::new(),
        };
        registry.maybe_persist(&report);
        let mut loaded = SubnetworkRegistry::new(registry.config.clone());
        loaded.load_state();
        assert!(loaded.seeds.contains_key("auto::attr::speed::steady"));
    }
}
