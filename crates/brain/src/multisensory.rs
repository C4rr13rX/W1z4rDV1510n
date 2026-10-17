//! Multisensory integration module for `w1z4rd-brain`.
//!
//! Provides the full sensory scope of human perception and allows an unlimited
//! number of physical or synthetic sensory streams to integrate simultaneously
//! through tick-aligned Hebbian co-activation.
//!
//! # Modern Understanding of Human Senses
//!
//! Human perception extends far beyond the classical five senses to include:
//! 1. **Visual**: Chromatic spectral wavelengths (color), luminance (intensity),
//!    spatial geometry (depth/stereopsis), and optical flow (motion vectors).
//! 2. **Auditory**: Acoustic frequency spectrum (tonotopic pitch), transients/dynamics,
//!    speech phonemes, and spatial azimuth (interaural time/phase differences).
//! 3. **Tactile/Somatosensory**: Mechanoreception (pressure, fine touch, surface texture,
//!    and vibration frequencies from Pacinian/Meissner corpuscles).
//! 4. **Proprioceptive**: Spatial awareness of embodiment, joint angles, limb kinematics,
//!    muscle tension (Golgi tendon organs), and self-motion.
//! 5. **Vestibular**: 3D angular velocity (semicircular canals) and linear acceleration /
//!    gravitational orientation vector (otolith organs: utricle and saccule).
//! 6. **Thermoceptive**: Thermal gradients, warmth, cold, and heat flux rates.
//! 7. **Nociceptive**: Mechanical, thermal, and chemical strain, hazard signals, and damage overload.
//! 8. **Olfactory**: Volatile organic chemical detection and airborne concentration profiles.
//! 9. **Gustatory**: Soluble chemical profiles (sweet, salty, sour, bitter, umami, nutrient/toxin detection).
//! 10. **Interoceptive**: Visceral internal state, metabolic/electrical energy reserves, operational strain,
//!     thermal load, and homeostatic balance.
//! 11. **Chronoceptive**: Perception of time intervals, elapsed durations, sequence pacing, and rhythm.
//! 12. **Field/Orientation**: 3D magnetic heading, electromagnetic field vectors, and polarization.
//!
//! Any synthetic or specialized sensor (e.g. LiDAR point clouds, multispectral cameras, soil moisture,
//! robotic current draw) can be mapped into an unlimited collection of specialized sensory pools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::neuron::PoolId;
use crate::pool::AtomEncoding;

/// Classification of human and physical sensory modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SenseModality {
    /// Visual perception: color, intensity, depth, optical motion vectors.
    Visual,
    /// Auditory perception: frequency spectrum, loudness, rhythm, spatial acoustics.
    Auditory,
    /// Tactile perception: surface texture, pressure, micro-vibration, haptics.
    Tactile,
    /// Proprioceptive perception: joint angles, kinematics, limb positions, spatial pose.
    Proprioceptive,
    /// Vestibular perception: 3D acceleration, tilt, angular velocity, gravity orientation.
    Vestibular,
    /// Thermoceptive perception: temperature gradients, heat flux.
    Thermoceptive,
    /// Nociceptive perception: damage overload, mechanical/chemical strain, hazard alert.
    Nociceptive,
    /// Olfactory perception: airborne volatile chemical compounds.
    Olfactory,
    /// Gustatory perception: soluble chemical compounds (nutrients, sweetness, salts, pH).
    Gustatory,
    /// Interoceptive perception: internal energy reserves, thermal load, physiological status.
    Interoceptive,
    /// Chronoceptive perception: duration, time intervals, sequence pacing, cadence.
    Chronoceptive,
    /// Orientation / Field perception: magnetic heading, electromagnetic vector.
    FieldOrientation,
    /// Custom or synthetic sensor modality (e.g. LiDAR, radar, soil moisture, pH sensor).
    Custom(u32),
}

impl SenseModality {
    /// Standard human-readable name for this sense modality.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Visual => "visual",
            Self::Auditory => "auditory",
            Self::Tactile => "tactile",
            Self::Proprioceptive => "proprioceptive",
            Self::Vestibular => "vestibular",
            Self::Thermoceptive => "thermoceptive",
            Self::Nociceptive => "nociceptive",
            Self::Olfactory => "olfactory",
            Self::Gustatory => "gustatory",
            Self::Interoceptive => "interoceptive",
            Self::Chronoceptive => "chronoceptive",
            Self::FieldOrientation => "field_orientation",
            Self::Custom(_) => "custom_sensor",
        }
    }

    /// Default prefix for atom namespacing in this modality.
    pub fn prefix(&self) -> String {
        match self {
            Self::Visual => "vis".to_string(),
            Self::Auditory => "aud".to_string(),
            Self::Tactile => "tac".to_string(),
            Self::Proprioceptive => "kin".to_string(),
            Self::Vestibular => "ves".to_string(),
            Self::Thermoceptive => "thm".to_string(),
            Self::Nociceptive => "noc".to_string(),
            Self::Olfactory => "olf".to_string(),
            Self::Gustatory => "gus".to_string(),
            Self::Interoceptive => "int".to_string(),
            Self::Chronoceptive => "chr".to_string(),
            Self::FieldOrientation => "mag".to_string(),
            Self::Custom(id) => format!("c{id}"),
        }
    }
}

/// A synchronized observation across an arbitrary collection of senses.
///
/// In biological cognition, seeing an object, hearing its sound, feeling its texture,
/// and sensing one's body orientation happen at the same time. The brain's Hebbian
/// fabric binds these co-active signals into cross-pool memories.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultisensoryExperience {
    /// Timestamp or tick offset of this sensory moment.
    pub timestamp_ms: u64,
    /// Modality inputs mapping pool IDs to raw sensory byte frames.
    pub channels: Vec<(PoolId, SenseModality, Vec<u8>)>,
    /// Optional contextual description or metadata.
    pub context: String,
}

impl MultisensoryExperience {
    pub fn new(timestamp_ms: u64) -> Self {
        Self {
            timestamp_ms,
            channels: Vec::new(),
            context: String::new(),
        }
    }

    /// Add a sensory observation channel to this experience.
    pub fn with_channel(
        mut self,
        pool_id: PoolId,
        modality: SenseModality,
        data: impl Into<Vec<u8>>,
    ) -> Self {
        self.channels.push((pool_id, modality, data.into()));
        self
    }

    /// Set contextual text for this multisensory experience.
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = context.into();
        self
    }
}

/// An AtomEncoding implementation tailored for multisensory vectors.
///
/// Converts continuous multidimensional physical sensor signals (spectrograms,
/// joint vectors, RGB color histograms, acceleration) into discrete, highly
/// discriminative atom sequences with minimal RAM overhead.
#[derive(Debug, Clone)]
pub struct MultisensoryAtomEncoding {
    pub modality: SenseModality,
    pub prefix: String,
}

impl MultisensoryAtomEncoding {
    pub fn new(modality: SenseModality) -> Self {
        let prefix = modality.prefix();
        Self { modality, prefix }
    }

    pub fn with_prefix(modality: SenseModality, prefix: impl Into<String>) -> Self {
        Self {
            modality,
            prefix: prefix.into(),
        }
    }
}

impl AtomEncoding for MultisensoryAtomEncoding {
    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        if frame.is_empty() {
            return Vec::new();
        }
        let mut atoms = Vec::with_capacity(frame.len() * 2);
        // Prefix with modality atom
        atoms.push(format!("{}:modality", self.prefix));

        // Group into sensory feature tokens
        for (idx, chunk) in frame.chunks(4).enumerate() {
            let mut val: u32 = 0;
            for &b in chunk {
                val = (val << 8) | (b as u32);
            }
            // Discretize into quantile bins
            let bin = (val % 256) as u8;
            atoms.push(format!("{}:f{idx:02x}:b{bin:02x}", self.prefix));
        }

        atoms
    }

    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (label, _strength) in active_atoms {
            if let Some(pos) = label.rfind(":b") {
                if let Ok(b) = u8::from_str_radix(&label[pos + 2..], 16) {
                    out.push(b);
                }
            }
        }
        out
    }

    fn name(&self) -> &'static str {
        "multisensory-vector"
    }
}

/// Registry and router for multisensory channels in a brain instance.
#[derive(Debug, Clone, Default)]
pub struct MultisensoryChannelRouter {
    pool_to_modality: HashMap<PoolId, SenseModality>,
    modality_to_pools: HashMap<SenseModality, Vec<PoolId>>,
}

impl MultisensoryChannelRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a pool as belonging to a specific sensory modality.
    pub fn register(&mut self, pool_id: PoolId, modality: SenseModality) {
        self.pool_to_modality.insert(pool_id, modality);
        self.modality_to_pools
            .entry(modality)
            .or_default()
            .push(pool_id);
    }

    /// Get the modality of a given pool ID.
    pub fn modality_of(&self, pool_id: PoolId) -> Option<SenseModality> {
        self.pool_to_modality.get(&pool_id).copied()
    }

    /// Get all pool IDs associated with a sensory modality.
    pub fn pools_for(&self, modality: SenseModality) -> &[PoolId] {
        self.modality_to_pools
            .get(&modality)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Total registered sensory pools.
    pub fn channel_count(&self) -> usize {
        self.pool_to_modality.len()
    }
}
