use crate::config::{PeerScoringConfig, WorkloadProfileConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use w1z4rdv1510n::schema::Timestamp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkProfileSnapshot {
    pub sensor_ingest: bool,
    pub stream_processing: bool,
    pub share_publish: bool,
    pub share_consume: bool,
    pub storage: bool,
}

impl From<&WorkloadProfileConfig> for WorkProfileSnapshot {
    fn from(profile: &WorkloadProfileConfig) -> Self {
        Self {
            sensor_ingest: profile.enable_sensor_ingest,
            stream_processing: profile.enable_stream_processing,
            share_publish: profile.enable_share_publish,
            share_consume: profile.enable_share_consume,
            storage: profile.enable_storage,
        }
    }
}

impl Default for WorkProfileSnapshot {
    fn default() -> Self {
        Self {
            sensor_ingest: false,
            stream_processing: false,
            share_publish: false,
            share_consume: false,
            storage: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetricsReport {
    pub node_id: String,
    pub timestamp: Timestamp,
    pub efficiency: f64,
    pub accuracy: f64,
    pub throughput_bps: f64,
    pub latency_ms: f64,
    pub capacity_ratio: f64,
    pub available_capacity: bool,
    #[serde(default)]
    pub efficiency_ratio: f64,
    #[serde(default)]
    pub accuracy_ratio: f64,
    #[serde(default)]
    pub work_profile: WorkProfileSnapshot,
    #[serde(default)]
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct LocalPerformanceSample {
    pub timestamp: Timestamp,
    pub payload_bytes: usize,
    pub elapsed_secs: f64,
    pub energy_joules: f64,
    pub accuracy: Option<f64>,
}

#[derive(Debug, Clone)]
struct PeerEntry {
    report: NodeMetricsReport,
    received_at: Timestamp,
}

#[derive(Debug, Clone)]
struct EmaMetrics {
    samples: u64,
    accuracy_samples: u64,
    efficiency: f64,
    accuracy: f64,
    throughput_bps: f64,
    latency_ms: f64,
    capacity_ratio: f64,
}

impl Default for EmaMetrics {
    fn default() -> Self {
        Self {
            samples: 0,
            accuracy_samples: 0,
            efficiency: 0.0,
            accuracy: 0.0,
            throughput_bps: 0.0,
            latency_ms: 0.0,
            capacity_ratio: 0.0,
        }
    }
}

pub struct NodePerformanceTracker {
    node_id: String,
    config: PeerScoringConfig,
    workload: WorkloadProfileConfig,
    metrics: EmaMetrics,
    peers: HashMap<String, PeerEntry>,
    last_publish: Timestamp,
}

impl NodePerformanceTracker {
    pub fn new(
        node_id: String,
        config: PeerScoringConfig,
        workload: WorkloadProfileConfig,
    ) -> Self {
        Self {
            node_id,
            config,
            workload,
            metrics: EmaMetrics::default(),
            peers: HashMap::new(),
            last_publish: Timestamp { unix: 0 },
        }
    }

    pub fn update_local(&mut self, sample: LocalPerformanceSample) {
        let elapsed = sample.elapsed_secs.max(1e-6);
        let throughput = sample.payload_bytes as f64 / elapsed;
        let efficiency = if sample.energy_joules > 0.0 {
            sample.payload_bytes as f64 / sample.energy_joules
        } else {
            throughput
        };
        let latency_ms = elapsed * 1000.0;
        let capacity_ratio = if latency_ms > 0.0 {
            (self.config.target_latency_ms / latency_ms).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let alpha = self.config.ema_alpha.clamp(0.0, 1.0);
        self.metrics.efficiency = update_metric(self.metrics.efficiency, efficiency, alpha, &mut self.metrics.samples);
        self.metrics.throughput_bps =
            update_metric(self.metrics.throughput_bps, throughput, alpha, &mut self.metrics.samples);
        self.metrics.latency_ms =
            update_metric(self.metrics.latency_ms, latency_ms, alpha, &mut self.metrics.samples);
        self.metrics.capacity_ratio =
            update_metric(self.metrics.capacity_ratio, capacity_ratio, alpha, &mut self.metrics.samples);
        if let Some(accuracy) = sample.accuracy {
            let acc = accuracy.clamp(0.0, 1.0);
            self.metrics.accuracy =
                update_metric(self.metrics.accuracy, acc, alpha, &mut self.metrics.accuracy_samples);
        }
        let _ = sample.timestamp;
    }

    pub fn ingest_peer_report(&mut self, report: NodeMetricsReport, now: Timestamp) {
        if report.node_id.trim().is_empty() || report.node_id == self.node_id {
            return;
        }
        self.prune(now);
        self.peers.insert(
            report.node_id.clone(),
            PeerEntry {
                report,
                received_at: now,
            },
        );
    }

    pub fn should_process_streams(&mut self, now: Timestamp, base_can_process: bool) -> bool {
        if !base_can_process {
            return false;
        }
        if !self.config.enabled {
            return true;
        }
        self.prune(now);
        if self.metrics.samples == 0 {
            return true;
        }
        let avg_eff = self.average_efficiency(now);
        let eff_ratio = avg_eff
            .map(|avg| ratio(self.metrics.efficiency, avg))
            .unwrap_or(1.0);
        let low_eff = eff_ratio < self.config.efficiency_offload_threshold;
        let overloaded = self.metrics.capacity_ratio < self.config.capacity_threshold;
        let low_accuracy = self.metrics.accuracy_samples > 0
            && self.metrics.accuracy < self.config.accuracy_threshold;
        if (low_eff || overloaded || low_accuracy) && self.has_available_peer(now) {
            return false;
        }
        true
    }

    pub fn take_report(&mut self, now: Timestamp) -> Option<NodeMetricsReport> {
        if !self.config.enabled || self.metrics.samples == 0 {
            return None;
        }
        if now.unix - self.last_publish.unix < self.config.publish_interval_secs as i64 {
            return None;
        }
        self.prune(now);
        let avg_eff = self.average_efficiency(now).unwrap_or(self.metrics.efficiency);
        let avg_acc = self.average_accuracy(now).unwrap_or(self.metrics.accuracy);
        let efficiency_ratio = ratio(self.metrics.efficiency, avg_eff);
        let accuracy_ratio = ratio(self.metrics.accuracy, avg_acc);
        let capacity_ratio = self.metrics.capacity_ratio.clamp(0.0, 1.0);
        let available_capacity = capacity_ratio >= self.config.capacity_threshold;
        let report = NodeMetricsReport {
            node_id: self.node_id.clone(),
            timestamp: now,
            efficiency: self.metrics.efficiency.max(0.0),
            accuracy: self.metrics.accuracy.clamp(0.0, 1.0),
            throughput_bps: self.metrics.throughput_bps.max(0.0),
            latency_ms: self.metrics.latency_ms.max(0.0),
            capacity_ratio,
            available_capacity,
            efficiency_ratio,
            accuracy_ratio,
            work_profile: WorkProfileSnapshot::from(&self.workload),
            tags: HashMap::new(),
        };
        self.last_publish = now;
        Some(report)
    }

    fn average_efficiency(&self, now: Timestamp) -> Option<f64> {
        if self.peers.len() < self.config.min_peer_reports {
            return if self.metrics.samples > 0 {
                Some(self.metrics.efficiency)
            } else {
                None
            };
        }
        average_metric(
            now,
            self.metrics.samples,
            self.metrics.efficiency,
            &self.peers,
            |report| report.efficiency,
        )
    }

    fn average_accuracy(&self, now: Timestamp) -> Option<f64> {
        if self.peers.len() < self.config.min_peer_reports {
            return if self.metrics.accuracy_samples > 0 {
                Some(self.metrics.accuracy)
            } else {
                None
            };
        }
        average_metric(
            now,
            self.metrics.accuracy_samples,
            self.metrics.accuracy,
            &self.peers,
            |report| report.accuracy,
        )
    }

    fn has_available_peer(&self, now: Timestamp) -> bool {
        let ttl = self.config.report_ttl_secs as i64;
        self.peers.values().any(|entry| {
            let age = now.unix - entry.received_at.unix;
            if age > ttl {
                return false;
            }
            let report = &entry.report;
            report.work_profile.stream_processing
                && report.available_capacity
                && report.capacity_ratio >= self.config.capacity_threshold
                && report.accuracy >= self.config.accuracy_threshold
        })
    }

    fn prune(&mut self, now: Timestamp) {
        let ttl = self.config.report_ttl_secs as i64;
        self.peers.retain(|_, entry| now.unix - entry.received_at.unix <= ttl);
    }
}

fn update_metric(current: f64, sample: f64, alpha: f64, samples: &mut u64) -> f64 {
    if !sample.is_finite() {
        return current;
    }
    if *samples == 0 {
        *samples = 1;
        return sample;
    }
    let next = alpha * sample + (1.0 - alpha) * current;
    *samples = samples.saturating_add(1);
    next
}

fn ratio(value: f64, mean: f64) -> f64 {
    if mean <= 0.0 {
        return 1.0;
    }
    (value / mean).clamp(0.0, 10.0)
}

fn average_metric<F>(
    now: Timestamp,
    self_samples: u64,
    self_value: f64,
    peers: &HashMap<String, PeerEntry>,
    selector: F,
) -> Option<f64>
where
    F: Fn(&NodeMetricsReport) -> f64,
{
    let mut total = 0.0;
    let mut count = 0.0;
    if self_samples > 0 && self_value.is_finite() {
        total += self_value;
        count += 1.0;
    }
    for entry in peers.values() {
        if now.unix < entry.received_at.unix {
            continue;
        }
        let value = selector(&entry.report);
        if value.is_finite() {
            total += value;
            count += 1.0;
        }
    }
    if count > 0.0 {
        Some(total / count)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{PeerScoringConfig, WorkloadProfileConfig};

    #[test]
    fn offloads_when_efficiency_is_low_and_peer_available() {
        let mut config = PeerScoringConfig::default();
        config.efficiency_offload_threshold = 0.25;
        config.capacity_threshold = 0.4;
        config.accuracy_threshold = 0.2;
        let workload = WorkloadProfileConfig::default();
        let mut tracker = NodePerformanceTracker::new("n1".to_string(), config.clone(), workload);
        tracker.update_local(LocalPerformanceSample {
            timestamp: Timestamp { unix: 10 },
            payload_bytes: 100,
            elapsed_secs: 1.0,
            energy_joules: 200.0,
            accuracy: Some(0.3),
        });
        let peer = NodeMetricsReport {
            node_id: "n2".to_string(),
            timestamp: Timestamp { unix: 10 },
            efficiency: 10.0,
            accuracy: 0.9,
            throughput_bps: 1000.0,
            latency_ms: 50.0,
            capacity_ratio: 0.9,
            available_capacity: true,
            efficiency_ratio: 1.0,
            accuracy_ratio: 1.0,
            work_profile: WorkProfileSnapshot {
                sensor_ingest: true,
                stream_processing: true,
                share_publish: true,
                share_consume: true,
                storage: true,
            },
            tags: HashMap::new(),
        };
        tracker.ingest_peer_report(peer, Timestamp { unix: 11 });
        assert!(!tracker.should_process_streams(Timestamp { unix: 12 }, true));
    }
}
