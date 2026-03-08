use crate::schema::Timestamp;
use crate::streaming::schema::StreamSource;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRegistryConfig {
    pub enabled: bool,
    pub max_sensors: usize,
    pub capability_ttl_secs: i64,
    pub auto_integrate: bool,
}

impl Default for SensorRegistryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_sensors: 64,
            capability_ttl_secs: 300,
            auto_integrate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Capability descriptor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorCapability {
    pub sensor_id: String,
    pub name: String,
    /// Freeform kind tag: "video", "audio", "imu", "physiological", "environmental", etc.
    pub kind: String,
    /// Data fields this sensor provides.
    pub fields: Vec<String>,
    pub sample_rate_hz: f64,
    /// Human-readable resolution string, e.g. "1920x1080" for video sensors.
    pub resolution: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorRegistryReport {
    pub timestamp: Timestamp,
    pub total_sensors: usize,
    pub active_sensors: Vec<String>,
    pub newly_registered: Vec<String>,
    pub expired: Vec<String>,
}

// ---------------------------------------------------------------------------
// Internal bookkeeping
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SensorEntry {
    capability: SensorCapability,
    last_seen: i64,
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub struct SensorRegistry {
    config: SensorRegistryConfig,
    sensors: HashMap<String, SensorEntry>,
    newly_registered: Vec<String>,
    recently_expired: Vec<String>,
}

impl SensorRegistry {
    pub fn new(config: SensorRegistryConfig) -> Self {
        Self {
            config,
            sensors: HashMap::new(),
            newly_registered: Vec::new(),
            recently_expired: Vec::new(),
        }
    }

    /// Register a sensor capability. Returns `true` when the sensor is genuinely
    /// new (not already tracked).
    pub fn register(&mut self, capability: SensorCapability, timestamp: Timestamp) -> bool {
        if !self.config.enabled {
            return false;
        }

        let id = capability.sensor_id.clone();

        if self.sensors.contains_key(&id) {
            // Already registered – just refresh the heartbeat.
            if let Some(entry) = self.sensors.get_mut(&id) {
                entry.last_seen = timestamp.unix;
                entry.capability = capability;
            }
            return false;
        }

        // Enforce capacity limit.
        if self.sensors.len() >= self.config.max_sensors {
            return false;
        }

        self.sensors.insert(
            id.clone(),
            SensorEntry {
                capability,
                last_seen: timestamp.unix,
            },
        );
        self.newly_registered.push(id);
        true
    }

    /// Refresh the last-seen timestamp for a sensor, keeping it alive.
    pub fn heartbeat(&mut self, sensor_id: &str, timestamp: Timestamp) {
        if let Some(entry) = self.sensors.get_mut(sensor_id) {
            entry.last_seen = timestamp.unix;
        }
    }

    /// Remove sensors that have not been seen within `capability_ttl_secs`.
    /// Returns the IDs of pruned sensors.
    pub fn prune_stale(&mut self, now: Timestamp) -> Vec<String> {
        let ttl = self.config.capability_ttl_secs;
        let stale_ids: Vec<String> = self
            .sensors
            .iter()
            .filter(|(_, entry)| now.unix - entry.last_seen > ttl)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &stale_ids {
            self.sensors.remove(id);
        }

        self.recently_expired = stale_ids.clone();
        stale_ids
    }

    /// Return references to all currently-active sensor capabilities.
    pub fn active_sensors(&self) -> Vec<&SensorCapability> {
        self.sensors.values().map(|e| &e.capability).collect()
    }

    /// Heuristically map a sensor's `kind` tag to the closest `StreamSource`.
    pub fn infer_stream_source(&self, sensor_id: &str) -> Option<StreamSource> {
        let entry = self.sensors.get(sensor_id)?;
        let kind_lower = entry.capability.kind.to_ascii_lowercase();

        if kind_lower.contains("video") || kind_lower.contains("camera") {
            Some(StreamSource::PeopleVideo)
        } else if kind_lower.contains("traffic") || kind_lower.contains("crowd") || kind_lower.contains("lidar") {
            Some(StreamSource::CrowdTraffic)
        } else if kind_lower.contains("text") || kind_lower.contains("ocr") || kind_lower.contains("annotation") {
            Some(StreamSource::TextAnnotations)
        } else if kind_lower.contains("topic") || kind_lower.contains("audio") || kind_lower.contains("microphone") {
            Some(StreamSource::PublicTopics)
        } else {
            None
        }
    }

    /// Build a snapshot report of the registry state.
    pub fn report(&self, timestamp: Timestamp) -> SensorRegistryReport {
        SensorRegistryReport {
            timestamp,
            total_sensors: self.sensors.len(),
            active_sensors: self.sensors.keys().cloned().collect(),
            newly_registered: self.newly_registered.clone(),
            expired: self.recently_expired.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(unix: i64) -> Timestamp {
        Timestamp { unix }
    }

    fn make_cap(id: &str, kind: &str) -> SensorCapability {
        SensorCapability {
            sensor_id: id.to_string(),
            name: format!("Test {}", id),
            kind: kind.to_string(),
            fields: vec!["x".into(), "y".into()],
            sample_rate_hz: 30.0,
            resolution: None,
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn register_returns_true_for_new_sensor() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig::default());
        let cap = make_cap("cam-0", "video");
        assert!(reg.register(cap, ts(100)));
        assert_eq!(reg.active_sensors().len(), 1);
    }

    #[test]
    fn duplicate_register_returns_false() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig::default());
        let cap = make_cap("cam-0", "video");
        assert!(reg.register(cap.clone(), ts(100)));
        assert!(!reg.register(cap, ts(101)));
        assert_eq!(reg.active_sensors().len(), 1);
    }

    #[test]
    fn prune_stale_removes_expired() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig {
            capability_ttl_secs: 60,
            ..SensorRegistryConfig::default()
        });
        reg.register(make_cap("a", "video"), ts(100));
        reg.register(make_cap("b", "audio"), ts(150));

        // At t=200, sensor "a" is 100s old (>60 ttl), sensor "b" is 50s old (<=60 ttl).
        let expired = reg.prune_stale(ts(200));
        assert_eq!(expired, vec!["a".to_string()]);
        assert_eq!(reg.active_sensors().len(), 1);
    }

    #[test]
    fn heartbeat_keeps_sensor_alive() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig {
            capability_ttl_secs: 60,
            ..SensorRegistryConfig::default()
        });
        reg.register(make_cap("a", "video"), ts(100));
        reg.heartbeat("a", ts(190));

        let expired = reg.prune_stale(ts(200));
        assert!(expired.is_empty());
    }

    #[test]
    fn max_sensors_enforced() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig {
            max_sensors: 2,
            ..SensorRegistryConfig::default()
        });
        assert!(reg.register(make_cap("a", "video"), ts(100)));
        assert!(reg.register(make_cap("b", "audio"), ts(100)));
        assert!(!reg.register(make_cap("c", "imu"), ts(100)));
    }

    #[test]
    fn infer_stream_source_maps_correctly() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig::default());
        reg.register(make_cap("cam", "video"), ts(1));
        reg.register(make_cap("mic", "microphone"), ts(1));
        reg.register(make_cap("lid", "lidar_traffic"), ts(1));
        reg.register(make_cap("txt", "text_ocr"), ts(1));
        reg.register(make_cap("env", "environmental"), ts(1));

        assert_eq!(reg.infer_stream_source("cam"), Some(StreamSource::PeopleVideo));
        assert_eq!(reg.infer_stream_source("mic"), Some(StreamSource::PublicTopics));
        assert_eq!(reg.infer_stream_source("lid"), Some(StreamSource::CrowdTraffic));
        assert_eq!(reg.infer_stream_source("txt"), Some(StreamSource::TextAnnotations));
        assert_eq!(reg.infer_stream_source("env"), None);
    }

    #[test]
    fn report_captures_state() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig::default());
        reg.register(make_cap("s1", "video"), ts(10));
        reg.register(make_cap("s2", "audio"), ts(10));

        let rpt = reg.report(ts(20));
        assert_eq!(rpt.total_sensors, 2);
        assert_eq!(rpt.newly_registered.len(), 2);
    }

    #[test]
    fn disabled_registry_rejects_all() {
        let mut reg = SensorRegistry::new(SensorRegistryConfig {
            enabled: false,
            ..SensorRegistryConfig::default()
        });
        assert!(!reg.register(make_cap("x", "video"), ts(1)));
        assert_eq!(reg.active_sensors().len(), 0);
    }
}
