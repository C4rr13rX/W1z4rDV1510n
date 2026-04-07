/// Sub-ultradian rhythm detector.
///
/// Extends the ultradian analysis stack downward in timescale to capture the
/// acute biological and behavioural responses that occur in seconds-to-minutes
/// ranges — the timescale of threat responses, startle reactions, and the
/// arousal buildup that precedes a violent event.
///
/// Timescale bands
/// ───────────────
///   Startle        0–10 s   freeze/orient, amygdala first-pass
///   AcuteArousal  10–60 s   sympathetic activation, full threat assessment
///   ThreatVigilance 1–5 min  sustained scan, proxemics monitoring
///   SustainedArousal 5–20 min HPA axis, cortisl ramp, hypervigilance
///
/// Observable proxies (no biosensors required)
/// ────────────────────────────────────────────
/// From video: micro-freeze, postural stiffening, gaze velocity spike, hand
///   position changes, scanning rate
/// From audio: voice fundamental frequency shift, fluency reduction
/// From radar/depth: breathing rate change, micro-tremor
///
/// The detector works purely from behavioral signal deviations — it infers
/// arousal state from what the sensor sees, not from direct physiological
/// measurement.

use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ─── Timescale bands ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubUltradianBand {
    /// 0–10 s: startle / freeze / orient response
    Startle,
    /// 10–60 s: acute sympathetic activation, fight-or-flight
    AcuteArousal,
    /// 1–5 min: threat vigilance maintenance
    ThreatVigilance,
    /// 5–20 min: sustained arousal / HPA axis engagement
    SustainedArousal,
}

impl SubUltradianBand {
    pub const ALL: [SubUltradianBand; 4] = [
        SubUltradianBand::Startle,
        SubUltradianBand::AcuteArousal,
        SubUltradianBand::ThreatVigilance,
        SubUltradianBand::SustainedArousal,
    ];

    /// Window length in seconds for this band's signal integration.
    pub fn window_secs(self) -> f64 {
        match self {
            SubUltradianBand::Startle          => 10.0,
            SubUltradianBand::AcuteArousal     => 60.0,
            SubUltradianBand::ThreatVigilance  => 300.0,
            SubUltradianBand::SustainedArousal => 1200.0,
        }
    }

    /// Disruption threshold: deviation (as fraction of normal) that triggers
    /// a meaningful arousal signal at this band.
    pub fn disruption_threshold(self) -> f32 {
        match self {
            SubUltradianBand::Startle          => 0.40, // high — startle is a spike
            SubUltradianBand::AcuteArousal     => 0.25,
            SubUltradianBand::ThreatVigilance  => 0.15,
            SubUltradianBand::SustainedArousal => 0.10,
        }
    }

    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            SubUltradianBand::Startle          => "startle",
            SubUltradianBand::AcuteArousal     => "acute_arousal",
            SubUltradianBand::ThreatVigilance  => "threat_vigilance",
            SubUltradianBand::SustainedArousal => "sustained_arousal",
        }
    }
}

// ─── Signal sample ────────────────────────────────────────────────────────────

/// A timestamped behavioural signal sample from any sensor proxy.
#[derive(Debug, Clone)]
pub struct BehavioralSample {
    pub timestamp: Timestamp,
    /// Normalised signal value.  0 = resting baseline, positive = elevated.
    pub value: f32,
    /// Confidence in this sample [0,1].
    pub confidence: f32,
    /// Which signal type this came from (e.g. "gaze_velocity", "postural_rigid").
    pub signal_type: String,
}

// ─── Per-band state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BandState {
    band: SubUltradianBand,
    /// Circular buffer of samples within the band window.
    samples: VecDeque<BehavioralSample>,
    /// Current smoothed amplitude (EMA).
    pub amplitude: f32,
    /// How much the rhythm has been disrupted vs normal.
    pub disruption: f32,
    /// Whether this band is currently in an arousal episode.
    pub aroused: bool,
}

impl BandState {
    fn new(band: SubUltradianBand) -> Self {
        Self {
            band,
            samples: VecDeque::new(),
            amplitude: 0.0,
            disruption: 0.0,
            aroused: false,
        }
    }

    fn push(&mut self, sample: BehavioralSample, now_unix: i64) {
        let window = self.band.window_secs() as i64;
        // Evict samples outside the window.
        while let Some(front) = self.samples.front() {
            if now_unix - front.timestamp.unix > window { self.samples.pop_front(); }
            else { break; }
        }
        self.samples.push_back(sample);
        self.recalculate();
    }

    fn recalculate(&mut self) {
        if self.samples.is_empty() {
            self.amplitude = 0.0;
            self.disruption = 0.0;
            self.aroused = false;
            return;
        }
        // Weighted mean — recent samples weight more (exponential decay).
        let n = self.samples.len();
        let mut sum = 0.0f32;
        let mut w_sum = 0.0f32;
        for (i, s) in self.samples.iter().enumerate() {
            let recency = (i + 1) as f32 / n as f32; // 0..1, recent = higher
            let w = recency * s.confidence;
            sum += s.value * w;
            w_sum += w;
        }
        self.amplitude = if w_sum > 1e-6 { (sum / w_sum).clamp(0.0, 1.0) } else { 0.0 };

        // Disruption = how far amplitude exceeds the threshold for this band.
        let thresh = self.band.disruption_threshold();
        self.disruption = ((self.amplitude - thresh) / (1.0 - thresh)).clamp(0.0, 1.0);
        self.aroused = self.disruption > 0.1;
    }
}

// ─── Sub-ultradian state snapshot ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubUltradianBandState {
    pub band: SubUltradianBand,
    pub amplitude: f32,
    pub disruption: f32,
    pub aroused: bool,
}

/// Full sub-ultradian state for one entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubUltradianState {
    pub entity_id: String,
    pub bands: Vec<SubUltradianBandState>,
    /// Combined arousal score across all bands [0,1].
    pub combined_arousal: f32,
    /// Which bands are in active arousal.
    pub active_bands: Vec<SubUltradianBand>,
    /// Contribution to HealthVector.TC (temporal coherence).
    /// High arousal = TC damage.
    pub tc_impact: f32,
    /// Contribution to HealthVector.AR (adaptive reserve).
    /// Sustained arousal depletes reserve.
    pub ar_drain: f32,
}

// ─── Detector ─────────────────────────────────────────────────────────────────

/// Per-entity sub-ultradian rhythm detector.
#[derive(Debug)]
pub struct SubUltradianDetector {
    entity_id: String,
    bands: [BandState; 4],
}

impl SubUltradianDetector {
    pub fn new(entity_id: impl Into<String>) -> Self {
        Self {
            entity_id: entity_id.into(),
            bands: [
                BandState::new(SubUltradianBand::Startle),
                BandState::new(SubUltradianBand::AcuteArousal),
                BandState::new(SubUltradianBand::ThreatVigilance),
                BandState::new(SubUltradianBand::SustainedArousal),
            ],
        }
    }

    /// Push a behavioural signal sample.  The detector fans it out to all bands.
    pub fn push_sample(&mut self, sample: BehavioralSample) {
        let t = sample.timestamp.unix;
        for band in &mut self.bands {
            band.push(sample.clone(), t);
        }
    }

    /// Snapshot the current sub-ultradian state.
    pub fn snapshot(&self) -> SubUltradianState {
        let band_states: Vec<SubUltradianBandState> = self.bands.iter().map(|b| {
            SubUltradianBandState {
                band: b.band,
                amplitude: b.amplitude,
                disruption: b.disruption,
                aroused: b.aroused,
            }
        }).collect();

        let active_bands: Vec<SubUltradianBand> = self.bands.iter()
            .filter(|b| b.aroused)
            .map(|b| b.band)
            .collect();

        // Combined arousal = weighted mean; faster bands (Startle) get more weight
        // for immediate threat, slower bands (SustainedArousal) indicate chronic stress.
        let weights = [0.4f32, 0.3, 0.2, 0.1];
        let combined_arousal = self.bands.iter().zip(weights.iter())
            .map(|(b, &w)| b.disruption * w)
            .sum::<f32>()
            .clamp(0.0, 1.0);

        // TC impact: rhythm disruption maps directly to temporal coherence damage.
        // Startle dominates here — it breaks the most immediate rhythms.
        let tc_impact = (self.bands[0].disruption * 0.5
            + self.bands[1].disruption * 0.3
            + self.bands[2].disruption * 0.2).clamp(0.0, 1.0);

        // AR drain: sustained arousal depletes adaptive reserve.
        let ar_drain = (self.bands[2].disruption * 0.4
            + self.bands[3].disruption * 0.6).clamp(0.0, 1.0);

        SubUltradianState {
            entity_id: self.entity_id.clone(),
            bands: band_states,
            combined_arousal,
            active_bands,
            tc_impact,
            ar_drain,
        }
    }
}

// ─── Registry ─────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct SubUltradianRegistry {
    detectors: HashMap<String, SubUltradianDetector>,
}

impl SubUltradianRegistry {
    pub fn push_sample(&mut self, entity_id: &str, sample: BehavioralSample) {
        self.detectors
            .entry(entity_id.to_string())
            .or_insert_with(|| SubUltradianDetector::new(entity_id))
            .push_sample(sample);
    }

    pub fn snapshot(&self, entity_id: &str) -> Option<SubUltradianState> {
        self.detectors.get(entity_id).map(|d| d.snapshot())
    }

    pub fn all_snapshots(&self) -> Vec<SubUltradianState> {
        self.detectors.values().map(|d| d.snapshot()).collect()
    }

    /// Infer behavioural samples from a set of token attributes.
    /// Used to feed the detector from the streaming pipeline without
    /// needing direct biosensor access.
    pub fn ingest_from_attributes(
        &mut self,
        entity_id: &str,
        timestamp: Timestamp,
        attrs: &std::collections::HashMap<String, serde_json::Value>,
    ) {
        let proxies: &[(&str, &str, f32)] = &[
            ("gaze_velocity",      "gaze_velocity",    1.0),
            ("postural_rigidity",  "postural_rigid",   1.0),
            ("micro_freeze",       "micro_freeze",     1.0),
            ("motion_energy",      "motion_energy",    0.5),
            ("scan_rate",          "scan_rate",        0.8),
            ("proximity_rate",     "proximity_rate",   0.7),
            ("voice_pitch_delta",  "voice_pitch",      0.9),
            ("breathing_rate",     "breathing_rate",   0.8),
        ];
        for (attr_key, signal_type, confidence_scale) in proxies {
            if let Some(val) = attrs.get(*attr_key).and_then(|v| v.as_f64()) {
                if val.is_finite() {
                    self.push_sample(entity_id, BehavioralSample {
                        timestamp,
                        value: (val as f32).clamp(0.0, 1.0),
                        confidence: *confidence_scale,
                        signal_type: signal_type.to_string(),
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startle_spike_triggers_arousal() {
        let mut det = SubUltradianDetector::new("entity_1");
        let ts = Timestamp { unix: 1000 };
        for _ in 0..5 {
            det.push_sample(BehavioralSample {
                timestamp: ts,
                value: 0.9,
                confidence: 1.0,
                signal_type: "gaze_velocity".to_string(),
            });
        }
        let snap = det.snapshot();
        assert!(snap.bands[0].aroused, "startle band should be aroused");
        assert!(snap.combined_arousal > 0.1);
        assert!(snap.tc_impact > 0.1);
    }

    #[test]
    fn no_samples_means_no_arousal() {
        let det = SubUltradianDetector::new("entity_2");
        let snap = det.snapshot();
        assert!(!snap.bands[0].aroused);
        assert_eq!(snap.combined_arousal, 0.0);
    }
}
