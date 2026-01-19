use crate::schema::Timestamp;
use crate::streaming::schema::{LayerState, TokenBatch};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionInfo {
    pub name: String,
    pub mean: f64,
    pub std: f64,
    pub count: usize,
    pub last_seen: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionReport {
    pub timestamp: Timestamp,
    pub dimensions: Vec<DimensionInfo>,
    pub emergent: Vec<DimensionInfo>,
    pub total_dimensions: usize,
}

#[derive(Debug, Clone)]
pub struct DimensionConfig {
    pub min_samples: usize,
    pub min_variance: f64,
    pub max_dimensions: usize,
    pub emit_every_secs: f64,
}

impl Default for DimensionConfig {
    fn default() -> Self {
        Self {
            min_samples: 12,
            min_variance: 1e-3,
            max_dimensions: 2048,
            emit_every_secs: 30.0,
        }
    }
}

#[derive(Debug, Clone)]
struct DimensionStats {
    mean: f64,
    m2: f64,
    count: usize,
    last_seen: Timestamp,
}

impl DimensionStats {
    fn new(ts: Timestamp) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            count: 0,
            last_seen: ts,
        }
    }

    fn update(&mut self, value: f64, ts: Timestamp) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.last_seen = ts;
    }

    fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count as f64 - 1.0)
        } else {
            0.0
        }
    }

    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

pub struct DimensionTracker {
    config: DimensionConfig,
    stats: HashMap<String, DimensionStats>,
    emitted: HashSet<String>,
    last_emit: Option<Timestamp>,
}

impl DimensionTracker {
    pub fn new(config: DimensionConfig) -> Self {
        Self {
            config,
            stats: HashMap::new(),
            emitted: HashSet::new(),
            last_emit: None,
        }
    }

    pub fn update(&mut self, batch: &TokenBatch) -> Option<DimensionReport> {
        let mut touched = HashSet::new();
        for token in &batch.tokens {
            let prefix = format!("token:{:?}", token.kind);
            for (key, value) in &token.attributes {
                if let Some(val) = numeric_value(value) {
                    let name = format!("{prefix}:{key}");
                    self.observe(&name, val, batch.timestamp);
                    touched.insert(name);
                }
            }
        }
        for layer in &batch.layers {
            self.observe_layer(layer, batch.timestamp, &mut touched);
        }

        self.prune();

        let mut emergent = Vec::new();
        let mut dimensions = Vec::new();
        for (name, stats) in &self.stats {
            let info = DimensionInfo {
                name: name.clone(),
                mean: stats.mean,
                std: stats.std(),
                count: stats.count,
                last_seen: stats.last_seen,
            };
            if stats.count >= self.config.min_samples
                && stats.variance() >= self.config.min_variance
                && !self.emitted.contains(name)
            {
                emergent.push(info.clone());
            }
            if touched.contains(name) {
                dimensions.push(info);
            }
        }
        for info in &emergent {
            self.emitted.insert(info.name.clone());
        }

        if emergent.is_empty() && !self.should_emit(batch.timestamp) {
            return None;
        }

        self.last_emit = Some(batch.timestamp);
        Some(DimensionReport {
            timestamp: batch.timestamp,
            dimensions,
            emergent,
            total_dimensions: self.stats.len(),
        })
    }

    fn observe(&mut self, name: &str, value: f64, ts: Timestamp) {
        if self.stats.len() >= self.config.max_dimensions && !self.stats.contains_key(name) {
            return;
        }
        let entry = self
            .stats
            .entry(name.to_string())
            .or_insert_with(|| DimensionStats::new(ts));
        entry.update(value, ts);
    }

    fn observe_layer(&mut self, layer: &LayerState, ts: Timestamp, touched: &mut HashSet<String>) {
        let prefix = format!("layer:{:?}", layer.kind);
        let metrics = [
            ("phase", layer.phase),
            ("amplitude", layer.amplitude),
            ("coherence", layer.coherence),
        ];
        for (key, val) in metrics {
            let name = format!("{prefix}:{key}");
            self.observe(&name, val, ts);
            touched.insert(name);
        }
        for (key, value) in &layer.attributes {
            if let Some(val) = numeric_value(value) {
                let name = format!("{prefix}:{key}");
                self.observe(&name, val, ts);
                touched.insert(name);
            }
        }
    }

    fn prune(&mut self) {
        let max = self.config.max_dimensions.max(1);
        if self.stats.len() <= max {
            return;
        }
        let mut entries: Vec<(String, DimensionStats)> = self.stats.drain().collect();
        entries.sort_by(|a, b| a.1.count.cmp(&b.1.count));
        entries.truncate(max);
        self.stats = entries.into_iter().collect();
    }

    fn should_emit(&self, ts: Timestamp) -> bool {
        let Some(last) = self.last_emit else {
            return true;
        };
        let elapsed = (ts.unix - last.unix).abs() as f64;
        elapsed >= self.config.emit_every_secs
    }
}

impl Default for DimensionTracker {
    fn default() -> Self {
        Self::new(DimensionConfig::default())
    }
}

fn numeric_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(num) => num.as_f64().filter(|v| v.is_finite()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource};
    use std::collections::HashMap;

    #[test]
    fn tracker_emits_emergent_dimension() {
        let mut tracker = DimensionTracker::default();
        let mut batch = TokenBatch {
            timestamp: Timestamp { unix: 10 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 10 },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::from([("feature_x".to_string(), Value::from(0.2))]),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 10 },
                phase: 0.2,
                amplitude: 0.4,
                coherence: 0.6,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        tracker.config.min_samples = 2;
        tracker.config.emit_every_secs = 0.0;
        assert!(tracker.update(&batch).is_some());
        batch.timestamp.unix += 1;
        batch.tokens[0]
            .attributes
            .insert("feature_x".to_string(), Value::from(0.6));
        let report = tracker.update(&batch).expect("report");
        assert!(!report.emergent.is_empty());
    }
}
