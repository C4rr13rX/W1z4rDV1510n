use crate::schema::Timestamp;
use crate::spike::{NeuronKind, SpikeConfig, SpikeInput, SpikeMessage, SpikeMessageBus, SpikePool};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for dynamic pool spawning and lifecycle management.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DynamicPoolConfig {
    pub enabled: bool,
    pub max_dynamic_pools: usize,
    pub max_neurons_per_pool: usize,
    pub max_inputs_per_pool: usize,
    pub spawn_support_threshold: usize,
    pub prune_idle_secs: i64,
    pub hebbian_window_secs: i64,
    pub spike_config: SpikeConfig,
}

impl Default for DynamicPoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_dynamic_pools: 128,
            max_neurons_per_pool: 64,
            max_inputs_per_pool: 128,
            spawn_support_threshold: 5,
            prune_idle_secs: 3600,
            hebbian_window_secs: 2,
            spike_config: SpikeConfig::default(),
        }
    }
}

/// Summary of a single dynamic pool's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSummary {
    pub category_id: String,
    pub neuron_count: usize,
    pub observation_count: u64,
    pub last_active: Timestamp,
    pub associated_categories: Vec<String>,
}

/// Report emitted by the dynamic pool registry describing current state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPoolReport {
    pub timestamp: Timestamp,
    pub total_pools: usize,
    pub newly_spawned: Vec<String>,
    pub pruned: Vec<String>,
    pub pool_summaries: Vec<PoolSummary>,
}

/// Internal tracker for a single category observed by the system.
struct CategoryTracker {
    category_id: String,
    observation_count: u64,
    last_observed: Timestamp,
    feature_accumulator: Vec<f64>,
    pool: Option<SpikePool>,
    neuron_labels: HashMap<String, u32>,
}

impl CategoryTracker {
    fn new(category_id: &str) -> Self {
        Self {
            category_id: category_id.to_string(),
            observation_count: 0,
            last_observed: Timestamp { unix: 0 },
            feature_accumulator: Vec::new(),
            pool: None,
            neuron_labels: HashMap::new(),
        }
    }

    /// Update the running mean of observed features.
    fn accumulate_features(&mut self, features: &[f64]) {
        if self.feature_accumulator.is_empty() {
            self.feature_accumulator = features.to_vec();
        } else {
            // Extend if new features have more dimensions than previously seen.
            if features.len() > self.feature_accumulator.len() {
                self.feature_accumulator.resize(features.len(), 0.0);
            }
            let n = self.observation_count as f64;
            for (i, &v) in features.iter().enumerate() {
                // Incremental mean: mean_new = mean_old + (v - mean_old) / n
                self.feature_accumulator[i] += (v - self.feature_accumulator[i]) / n;
            }
        }
    }

    /// Build a pool summary for reporting.
    fn summary(&self, associations: &[String]) -> PoolSummary {
        PoolSummary {
            category_id: self.category_id.clone(),
            neuron_count: self.pool.as_ref().map_or(0, |p| p.neuron_count()),
            observation_count: self.observation_count,
            last_active: self.last_observed,
            associated_categories: associations.to_vec(),
        }
    }
}

/// Registry that dynamically spawns and manages spike pools as new categories emerge.
pub struct DynamicPoolRegistry {
    config: DynamicPoolConfig,
    trackers: HashMap<String, CategoryTracker>,
    bus: SpikeMessageBus,
    /// Hebbian association weights between category pairs. Key is sorted pair (a, b) where a < b.
    associations: HashMap<(String, String), f64>,
    /// Tracks the most recent observation time per category for Hebbian windowing.
    recent_observations: Vec<(String, i64)>,
    /// Categories spawned since last report was generated.
    newly_spawned: Vec<String>,
}

impl DynamicPoolRegistry {
    pub fn new(config: DynamicPoolConfig) -> Self {
        Self {
            config,
            trackers: HashMap::new(),
            bus: SpikeMessageBus::default(),
            associations: HashMap::new(),
            recent_observations: Vec::new(),
            newly_spawned: Vec::new(),
        }
    }

    /// Observe a category with a set of feature values. When enough observations accumulate
    /// and no pool exists yet, a new spike pool is spawned with one neuron per feature dimension.
    pub fn observe_category(&mut self, category_id: &str, features: &[f64], timestamp: Timestamp) {
        if !self.config.enabled {
            return;
        }

        // Pre-compute active pool count before taking a mutable borrow on trackers.
        let current_pool_count = self.trackers.values().filter(|t| t.pool.is_some()).count();

        let tracker = self
            .trackers
            .entry(category_id.to_string())
            .or_insert_with(|| CategoryTracker::new(category_id));

        tracker.observation_count += 1;
        tracker.last_observed = timestamp;
        tracker.accumulate_features(features);

        // Spawn pool once we reach the support threshold.
        if tracker.pool.is_none()
            && tracker.observation_count >= self.config.spawn_support_threshold as u64
            && current_pool_count < self.config.max_dynamic_pools
        {
            let mut pool = SpikePool::new(
                format!("dyn_{}", category_id),
                self.config.spike_config.clone(),
            );

            let num_neurons = tracker
                .feature_accumulator
                .len()
                .min(self.config.max_neurons_per_pool);

            for i in 0..num_neurons {
                // Alternate excitatory/inhibitory based on whether the mean feature value
                // is positive or non-positive, giving the pool internal structure.
                let kind = if tracker.feature_accumulator.get(i).copied().unwrap_or(0.0) > 0.0 {
                    NeuronKind::Excitatory
                } else {
                    NeuronKind::Inhibitory
                };
                let nid = pool.add_neuron(kind);
                tracker
                    .neuron_labels
                    .insert(format!("feat_{}", i), nid);
            }

            // Wire adjacent neurons to create lateral connectivity.
            if num_neurons > 1 {
                for i in 0..(num_neurons - 1) {
                    let weight = tracker
                        .feature_accumulator
                        .get(i)
                        .copied()
                        .unwrap_or(0.5)
                        .abs() as f32;
                    pool.connect(i as u32, (i + 1) as u32, weight.max(0.1));
                }
            }

            tracker.pool = Some(pool);
            self.newly_spawned.push(category_id.to_string());
        }

        // Hebbian co-occurrence: strengthen associations with categories observed recently.
        self.update_hebbian(category_id, timestamp.unix);
    }

    /// Route spike inputs to the pool for the given category. Returns any resulting spike
    /// messages from stepping the pool, or `None` if no pool exists for that category.
    pub fn route_to_pool(
        &mut self,
        category_id: &str,
        inputs: Vec<SpikeInput>,
        timestamp: Timestamp,
    ) -> Option<Vec<SpikeMessage>> {
        let tracker = self.trackers.get_mut(category_id)?;
        let pool = tracker.pool.as_mut()?;

        tracker.last_observed = timestamp;

        // Respect max inputs cap.
        let capped: Vec<SpikeInput> = inputs
            .into_iter()
            .take(self.config.max_inputs_per_pool)
            .collect();

        pool.enqueue_inputs(capped);
        let frame = pool.step(timestamp);

        let msg = SpikeMessage {
            pool_id: pool.id.clone(),
            frame,
        };
        self.bus.publish(msg.clone());

        Some(vec![msg])
    }

    /// Remove pools that have been idle longer than `prune_idle_secs`. Returns the IDs of
    /// pruned categories.
    pub fn prune_idle(&mut self, now: Timestamp) -> Vec<String> {
        let threshold = self.config.prune_idle_secs;
        let mut pruned = Vec::new();

        for tracker in self.trackers.values_mut() {
            if tracker.pool.is_some() {
                let idle_duration = now.unix - tracker.last_observed.unix;
                if idle_duration >= threshold {
                    tracker.pool = None;
                    tracker.neuron_labels.clear();
                    pruned.push(tracker.category_id.clone());
                }
            }
        }

        // Clean up associations involving pruned categories.
        for pruned_id in &pruned {
            self.associations
                .retain(|(a, b), _| a != pruned_id && b != pruned_id);
        }

        pruned
    }

    /// Number of categories that currently have a live spike pool.
    pub fn pool_count(&self) -> usize {
        self.active_pool_count()
    }

    /// Generate a report describing the current state of all dynamic pools.
    pub fn report(&self, timestamp: Timestamp) -> DynamicPoolReport {
        let pool_summaries: Vec<PoolSummary> = self
            .trackers
            .values()
            .filter(|t| t.pool.is_some())
            .map(|t| {
                let assoc = self.associations_for(&t.category_id);
                t.summary(&assoc)
            })
            .collect();

        DynamicPoolReport {
            timestamp,
            total_pools: self.active_pool_count(),
            newly_spawned: self.newly_spawned.clone(),
            pruned: Vec::new(),
            pool_summaries,
        }
    }

    // ── internal helpers ──

    fn active_pool_count(&self) -> usize {
        self.trackers.values().filter(|t| t.pool.is_some()).count()
    }

    /// Update Hebbian associations: any category observed within `hebbian_window_secs` of
    /// another gets its association weight incremented.
    fn update_hebbian(&mut self, category_id: &str, unix: i64) {
        let window = self.config.hebbian_window_secs;

        // Strengthen associations with recently observed categories.
        for (other_id, other_time) in &self.recent_observations {
            if other_id == category_id {
                continue;
            }
            if (unix - other_time).abs() <= window {
                let key = Self::assoc_key(category_id, other_id);
                let weight = self.associations.entry(key).or_insert(0.0);
                *weight += 1.0;
            }
        }

        // Expire old entries and add the current observation.
        self.recent_observations
            .retain(|(_, t)| (unix - *t).abs() <= window * 2);
        self.recent_observations
            .push((category_id.to_string(), unix));
    }

    /// Build a canonical sorted key for a pair of categories.
    fn assoc_key(a: &str, b: &str) -> (String, String) {
        if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        }
    }

    /// Return all categories associated with `category_id`.
    fn associations_for(&self, category_id: &str) -> Vec<String> {
        self.associations
            .iter()
            .filter_map(|((a, b), weight)| {
                if *weight <= 0.0 {
                    return None;
                }
                if a == category_id {
                    Some(b.clone())
                } else if b == category_id {
                    Some(a.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> DynamicPoolConfig {
        DynamicPoolConfig {
            spawn_support_threshold: 5,
            prune_idle_secs: 100,
            ..DynamicPoolConfig::default()
        }
    }

    #[test]
    fn spawns_pool_after_threshold_observations() {
        let mut registry = DynamicPoolRegistry::new(default_config());

        // Observe 4 times — should not spawn yet.
        for i in 0..4 {
            registry.observe_category("motion_a", &[1.0, -0.5, 0.3], Timestamp { unix: i });
        }
        assert_eq!(registry.pool_count(), 0, "pool should not exist before threshold");

        // Fifth observation crosses the threshold.
        registry.observe_category("motion_a", &[1.0, -0.5, 0.3], Timestamp { unix: 5 });
        assert_eq!(registry.pool_count(), 1, "pool should exist after threshold");

        // Verify pool has correct neuron count (one per feature dimension).
        let report = registry.report(Timestamp { unix: 5 });
        assert_eq!(report.pool_summaries.len(), 1);
        assert_eq!(report.pool_summaries[0].neuron_count, 3);
        assert_eq!(report.pool_summaries[0].category_id, "motion_a");
        assert!(report.newly_spawned.contains(&"motion_a".to_string()));
    }

    #[test]
    fn prunes_idle_pools() {
        let mut registry = DynamicPoolRegistry::new(default_config());

        // Spawn a pool.
        for i in 0..5 {
            registry.observe_category("idle_cat", &[0.5, 0.5], Timestamp { unix: i });
        }
        assert_eq!(registry.pool_count(), 1);

        // Prune with a timestamp just under the threshold — pool should remain.
        let pruned = registry.prune_idle(Timestamp { unix: 4 + 99 });
        assert!(pruned.is_empty());
        assert_eq!(registry.pool_count(), 1);

        // Prune with a timestamp at/past the threshold — pool should be removed.
        let pruned = registry.prune_idle(Timestamp { unix: 4 + 100 });
        assert_eq!(pruned.len(), 1);
        assert_eq!(pruned[0], "idle_cat");
        assert_eq!(registry.pool_count(), 0);
    }

    #[test]
    fn routes_inputs_and_produces_messages() {
        let mut registry = DynamicPoolRegistry::new(default_config());

        // Spawn pool.
        for i in 0..5 {
            registry.observe_category("routed", &[1.0], Timestamp { unix: i });
        }

        let inputs = vec![SpikeInput {
            target: 0,
            excitatory: 2.0,
            inhibitory: 0.0,
        }];

        let msgs = registry
            .route_to_pool("routed", inputs, Timestamp { unix: 10 })
            .expect("pool should exist");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].pool_id, "dyn_routed");
    }

    #[test]
    fn returns_none_for_unknown_category() {
        let mut registry = DynamicPoolRegistry::new(default_config());
        let result = registry.route_to_pool("nonexistent", vec![], Timestamp { unix: 0 });
        assert!(result.is_none());
    }

    #[test]
    fn hebbian_associations_formed() {
        let mut config = default_config();
        config.hebbian_window_secs = 5;
        let mut registry = DynamicPoolRegistry::new(config);

        // Spawn two pools close in time.
        for i in 0..5 {
            registry.observe_category("cat_a", &[1.0], Timestamp { unix: i });
            registry.observe_category("cat_b", &[1.0], Timestamp { unix: i });
        }

        let report = registry.report(Timestamp { unix: 5 });
        let summary_a = report
            .pool_summaries
            .iter()
            .find(|s| s.category_id == "cat_a")
            .expect("cat_a summary");
        assert!(
            summary_a.associated_categories.contains(&"cat_b".to_string()),
            "cat_a should be associated with cat_b"
        );
    }

    #[test]
    fn respects_max_dynamic_pools() {
        let mut config = default_config();
        config.max_dynamic_pools = 2;
        config.spawn_support_threshold = 1;
        let mut registry = DynamicPoolRegistry::new(config);

        registry.observe_category("a", &[1.0], Timestamp { unix: 0 });
        registry.observe_category("b", &[1.0], Timestamp { unix: 0 });
        registry.observe_category("c", &[1.0], Timestamp { unix: 0 });

        assert_eq!(registry.pool_count(), 2, "should cap at max_dynamic_pools");
    }
}
