use crate::config::OnlinePlasticityConfig;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSignal {
    pub domain: String,
    pub context: String,
    pub surprise: f64,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone)]
pub struct ReplaySample {
    pub signal: OutcomeSignal,
    pub weight: f64,
}

#[derive(Debug, Default)]
pub struct ReplayReservoir {
    capacity: usize,
    buckets: HashMap<String, Vec<ReplaySample>>,
}

impl ReplayReservoir {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            buckets: HashMap::new(),
        }
    }

    pub fn push(&mut self, sample: ReplaySample) {
        let key = format!("{}|{}", sample.signal.domain, sample.signal.context);
        let bucket = self.buckets.entry(key).or_default();
        bucket.push(sample);
        if bucket.len() > self.capacity {
            let overflow = bucket.len() - self.capacity;
            bucket.drain(0..overflow);
        }
    }

    pub fn sample_bucket(&self, key: &str) -> Option<&[ReplaySample]> {
        self.buckets.get(key).map(|items| items.as_slice())
    }
}

#[derive(Debug, Clone)]
pub enum PlasticityDecision {
    NoUpdate { reason: String },
    QueueReplay { bucket: String },
}

pub struct OnlinePlasticity {
    config: OnlinePlasticityConfig,
    reservoir: ReplayReservoir,
}

impl OnlinePlasticity {
    pub fn new(config: OnlinePlasticityConfig, reservoir_capacity: usize) -> Self {
        Self {
            config,
            reservoir: ReplayReservoir::new(reservoir_capacity),
        }
    }

    pub fn observe_outcome(&mut self, signal: OutcomeSignal) -> PlasticityDecision {
        if !self.config.enabled {
            return PlasticityDecision::NoUpdate {
                reason: "plasticity_disabled".to_string(),
            };
        }
        if signal.surprise < self.config.surprise_threshold {
            return PlasticityDecision::NoUpdate {
                reason: "surprise_below_threshold".to_string(),
            };
        }
        let bucket = format!("{}|{}", signal.domain, signal.context);
        let weight = signal.surprise.min(5.0);
        self.reservoir.push(ReplaySample { signal, weight });
        PlasticityDecision::QueueReplay { bucket }
    }

    pub fn reservoir(&self) -> &ReplayReservoir {
        &self.reservoir
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plasticity_queues_replay_on_surprise() {
        let config = OnlinePlasticityConfig {
            enabled: true,
            surprise_threshold: 0.5,
            trust_region: 0.1,
            ema_teacher_alpha: 0.9,
        };
        let mut plasticity = OnlinePlasticity::new(config, 8);
        let signal = OutcomeSignal {
            domain: "crowd".to_string(),
            context: "zone_a".to_string(),
            surprise: 0.9,
            timestamp: Timestamp { unix: 0 },
        };
        let decision = plasticity.observe_outcome(signal);
        assert!(matches!(decision, PlasticityDecision::QueueReplay { .. }));
    }
}
