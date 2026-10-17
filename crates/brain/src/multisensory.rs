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

/// 16-bit Audio Quantizer and Spectral Feature Extractor.
///
/// Enforces 16-bit signed PCM integer sampling [-32768, 32767] as the minimum
/// functional resolution threshold for acoustic feature extraction (avoiding 8-bit degradation).
#[derive(Debug, Clone, Default)]
pub struct Audio16BitQuantizer {
    pub sample_rate: u32,
}

impl Audio16BitQuantizer {
    pub fn new(sample_rate: u32) -> Self {
        Self { sample_rate }
    }

    /// Quantizes continuous f32 audio samples [-1.0, 1.0] to signed 16-bit PCM integers.
    pub fn quantize_f32_to_pcm16(samples: &[f32]) -> Vec<i16> {
        samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                if clamped < 0.0 {
                    (clamped * 32768.0).round() as i16
                } else {
                    (clamped * 32767.0).round() as i16
                }
            })
            .collect()
    }

    /// Extracts 16 discrete spectral energy sub-bands from 16-bit PCM samples.
    pub fn extract_band_energies_16(pcm16: &[i16]) -> [i16; 16] {
        let mut bands = [0i16; 16];
        if pcm16.is_empty() {
            return bands;
        }
        let chunk_size = (pcm16.len() / 16).max(1);
        for (b, chunk) in pcm16.chunks(chunk_size).take(16).enumerate() {
            let mut sum_sq = 0.0f64;
            for &val in chunk {
                let norm = (val as f64) / 32768.0;
                sum_sq += norm * norm;
            }
            let rms = (sum_sq / (chunk.len() as f64)).sqrt();
            bands[b] = ((rms.min(1.0)) * 32767.0).round() as i16;
        }
        bands
    }

    /// Converts 16 spectral bands into discrete atom tokens.
    pub fn atomize_bands(bands: &[i16; 16]) -> Vec<String> {
        let mut atoms = Vec::with_capacity(17);
        atoms.push("aud:modality".to_string());
        for (idx, &energy) in bands.iter().enumerate() {
            let quantized_bin = ((energy.max(0) as u32 * 255) / 32767) as u8;
            atoms.push(format!("aud:b{idx:02x}:v{quantized_bin:02x}"));
        }
        atoms
    }
}

/// A sparse orientation edge token emulating Hubel & Wiesel primary visual cortex (V1) simple cells.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrientationEdge {
    pub x: usize,
    pub y: usize,
    pub angle_deg: u16, // 0, 45, 90, 135
    pub magnitude: u8,
}

/// Output of biological cortical visual processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorticalVisualFeatures {
    pub dominant_color_vga: u8,
    pub sparse_edges: Vec<OrientationEdge>,
    pub compression_ratio: f32,
    pub atoms: Vec<String>,
}

/// Cortical Visual Encoder combining VGA 256-color palette quantization with
/// biological orientation-selective edge filters (angles of light: 0°, 45°, 90°, 135°).
#[derive(Debug, Clone, Default)]
pub struct CorticalVisualEncoder;

impl CorticalVisualEncoder {
    /// Quantizes RGB to an 8-bit VGA 256-color palette index.
    pub fn rgb_to_vga256(r: u8, g: u8, b: u8) -> u8 {
        let diff = (r as i32 - g as i32).abs().max((r as i32 - b as i32).abs()).max((g as i32 - b as i32).abs());
        if diff < 12 {
            let avg = ((r as u32 + g as u32 + b as u32) / 3) as u8;
            let gray_idx = ((avg as u32 * 23) / 255).min(23) as u8;
            232 + gray_idx
        } else {
            let r6 = ((r as u32 * 5) / 255).min(5) as u8;
            let g6 = ((g as u32 * 5) / 255).min(5) as u8;
            let b6 = ((b as u32 * 5) / 255).min(5) as u8;
            16 + 36 * r6 + 6 * g6 + b6
        }
    }

    /// Processes a 2D image buffer into sparse Hubel-Wiesel orientation angles of light
    /// and VGA 256-color palette atoms, discarding flat uniform areas.
    pub fn process_frame(
        width: usize,
        height: usize,
        rgba_or_rgb: &[u8],
        channels: usize,
        contrast_threshold: f32,
    ) -> CorticalVisualFeatures {
        let total_pixels = width * height;
        if total_pixels == 0 || rgba_or_rgb.len() < total_pixels * channels {
            return CorticalVisualFeatures {
                dominant_color_vga: 0,
                sparse_edges: Vec::new(),
                compression_ratio: 1.0,
                atoms: Vec::new(),
            };
        }

        let mut luminance = vec![0.0f32; total_pixels];
        let mut color_histogram = [0u32; 256];

        for y in 0..height {
            for x in 0..width {
                let p_idx = y * width + x;
                let src_idx = p_idx * channels;
                let r = rgba_or_rgb[src_idx];
                let g = rgba_or_rgb[src_idx + 1];
                let b = rgba_or_rgb[src_idx + 2];

                luminance[p_idx] = 0.299 * (r as f32) + 0.587 * (g as f32) + 0.114 * (b as f32);
                let vga = Self::rgb_to_vga256(r, g, b);
                color_histogram[vga as usize] += 1;
            }
        }

        // Find dominant palette color
        let mut dominant_color = 0u8;
        let mut max_count = 0u32;
        for (c, &cnt) in color_histogram.iter().enumerate() {
            if cnt > max_count {
                max_count = cnt;
                dominant_color = c as u8;
            }
        }

        // Sobel edge filtering for orientation angles of light
        let mut sparse_edges = Vec::new();
        let mut atoms = Vec::new();
        atoms.push("vis:modality".to_string());
        atoms.push(format!("vis:pal256:{dominant_color:02x}"));

        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let gx = -luminance[(y - 1) * width + (x - 1)]
                    + luminance[(y - 1) * width + (x + 1)]
                    - 2.0 * luminance[y * width + (x - 1)]
                    + 2.0 * luminance[y * width + (x + 1)]
                    - luminance[(y + 1) * width + (x - 1)]
                    + luminance[(y + 1) * width + (x + 1)];

                let gy = -luminance[(y - 1) * width + (x - 1)]
                    - 2.0 * luminance[(y - 1) * width + x]
                    - luminance[(y - 1) * width + (x + 1)]
                    + luminance[(y + 1) * width + (x - 1)]
                    + 2.0 * luminance[(y + 1) * width + x]
                    + luminance[(y + 1) * width + (x + 1)];

                let mag = (gx * gx + gy * gy).sqrt();
                if mag >= contrast_threshold {
                    // Orientation of the light bar/edge is perpendicular to the spatial gradient
                    let mut edge_rad = gy.atan2(gx) + std::f32::consts::FRAC_PI_2;
                    while edge_rad < 0.0 {
                        edge_rad += std::f32::consts::PI;
                    }
                    while edge_rad >= std::f32::consts::PI {
                        edge_rad -= std::f32::consts::PI;
                    }
                    let deg = edge_rad.to_degrees();

                    let discrete_angle = if deg >= 22.5 && deg < 67.5 {
                        45
                    } else if deg >= 67.5 && deg < 112.5 {
                        90
                    } else if deg >= 112.5 && deg < 157.5 {
                        135
                    } else {
                        0
                    };

                    let mag_u8 = mag.min(255.0) as u8;
                    sparse_edges.push(OrientationEdge {
                        x,
                        y,
                        angle_deg: discrete_angle,
                        magnitude: mag_u8,
                    });

                    // Emit spatial orientation atom
                    let norm_x = ((x as f32 / width as f32) * 16.0) as u8;
                    let norm_y = ((y as f32 / height as f32) * 16.0) as u8;
                    atoms.push(format!("vis:e:a{discrete_angle:03}:x{norm_x:01x}y{norm_y:01x}"));
                }
            }
        }

        let raw_bytes = (total_pixels * channels) as f32;
        let compressed_bytes = (3 + sparse_edges.len() * 4).max(1) as f32;
        let compression_ratio = raw_bytes / compressed_bytes;

        CorticalVisualFeatures {
            dominant_color_vga: dominant_color,
            sparse_edges,
            compression_ratio,
            atoms,
        }
    }
}

/// Proprioceptive Kinematic Encoder for normalizing joint angles and torques into 16-bit seed representations.
#[derive(Debug, Clone, Default)]
pub struct KinematicProprioceptionEncoder;

impl KinematicProprioceptionEncoder {
    /// Normalizes joint angle [-PI, PI] and torque [-maxTorque, maxTorque] into 16-bit signed values.
    pub fn quantize(angle_rad: f32, torque_nm: f32, max_torque_nm: f32) -> (i16, i16) {
        let norm_angle = (angle_rad / std::f32::consts::PI).clamp(-1.0, 1.0);
        let q_angle = if norm_angle < 0.0 {
            (norm_angle * 32768.0).round() as i16
        } else {
            (norm_angle * 32767.0).round() as i16
        };

        let q_torque = if max_torque_nm > 0.0 {
            let norm_torque = (torque_nm / max_torque_nm).clamp(-1.0, 1.0);
            if norm_torque < 0.0 {
                (norm_torque * 32768.0).round() as i16
            } else {
                (norm_torque * 32767.0).round() as i16
            }
        } else {
            0
        };

        (q_angle, q_torque)
    }

    /// Converts normalized joint values into discrete kinematic atom tokens.
    pub fn atomize_joint(joint_id: &str, q_angle: i16, q_torque: i16) -> Vec<String> {
        let angle_bin = (((q_angle as i32 + 32768) * 255) / 65535) as u8;
        let torque_bin = (((q_torque as i32 + 32768) * 255) / 65535) as u8;
        vec![
            "kin:modality".to_string(),
            format!("kin:j{joint_id}:a{angle_bin:02x}:t{torque_bin:02x}"),
        ]
    }
}

