use crate::compute::{ComputeJobKind, QuantumJob, QuantumResult};
use crate::config::QuantumConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MAX_SCALE: f64 = 10.0;
const MIN_SCALE: f64 = 0.1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCalibrationSnapshot {
    pub trotter_slices: usize,
    pub driver_strength: f64,
    pub driver_final_strength: f64,
    pub worldline_mix_prob: f64,
    pub slice_temperature_scale: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCalibrationRequest {
    pub run_id: String,
    pub best_energy: f64,
    pub acceptance_ratio: Option<f64>,
    pub energy_trace: Vec<f64>,
    pub config: QuantumCalibrationSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuantumCalibrationAdjust {
    pub driver_strength_scale: Option<f64>,
    pub driver_final_strength_scale: Option<f64>,
    pub slice_temperature_scale: Option<f64>,
    pub worldline_mix_delta: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCalibrationResponse {
    pub adjustments: QuantumCalibrationAdjust,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCalibrationState {
    pub driver_strength_scale: f64,
    pub driver_final_strength_scale: f64,
    pub slice_temperature_scale: f64,
    pub worldline_mix_delta: f64,
    pub samples: usize,
}

impl Default for QuantumCalibrationState {
    fn default() -> Self {
        Self {
            driver_strength_scale: 1.0,
            driver_final_strength_scale: 1.0,
            slice_temperature_scale: 1.0,
            worldline_mix_delta: 0.0,
            samples: 0,
        }
    }
}

impl QuantumCalibrationState {
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read quantum calibration from {:?}", path))?;
        let parsed = serde_json::from_str(&raw)
            .with_context(|| format!("failed to parse quantum calibration from {:?}", path))?;
        Ok(parsed)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create calibration directory {:?}", parent)
                })?;
            }
        }
        let serialized = serde_json::to_string_pretty(self)?;
        fs::write(path, serialized)
            .with_context(|| format!("failed to write quantum calibration to {:?}", path))?;
        Ok(())
    }

    pub fn apply(&self, base: &QuantumConfig) -> QuantumConfig {
        let mut adjusted = base.clone();
        adjusted.driver_strength =
            (base.driver_strength * self.driver_strength_scale).max(0.0);
        adjusted.driver_final_strength =
            (base.driver_final_strength * self.driver_final_strength_scale).max(0.0);
        adjusted.slice_temperature_scale =
            (base.slice_temperature_scale * self.slice_temperature_scale).max(1e-6);
        adjusted.worldline_mix_prob = (base.worldline_mix_prob + self.worldline_mix_delta)
            .clamp(0.0, 1.0);
        adjusted
    }

    pub fn update_from_adjustment(&mut self, adjust: &QuantumCalibrationAdjust, alpha: f64) {
        let alpha = alpha.clamp(0.0, 1.0);
        if let Some(scale) = adjust.driver_strength_scale {
            self.driver_strength_scale = blend_scale(self.driver_strength_scale, scale, alpha);
        }
        if let Some(scale) = adjust.driver_final_strength_scale {
            self.driver_final_strength_scale =
                blend_scale(self.driver_final_strength_scale, scale, alpha);
        }
        if let Some(scale) = adjust.slice_temperature_scale {
            self.slice_temperature_scale = blend_scale(self.slice_temperature_scale, scale, alpha);
        }
        if let Some(delta) = adjust.worldline_mix_delta {
            let next = self.worldline_mix_delta * (1.0 - alpha) + delta * alpha;
            self.worldline_mix_delta = next.clamp(-1.0, 1.0);
        }
        self.samples = self.samples.saturating_add(1);
    }
}

fn blend_scale(current: f64, target: f64, alpha: f64) -> f64 {
    let target = target.clamp(MIN_SCALE, MAX_SCALE);
    let blended = current * (1.0 - alpha) + target * alpha;
    blended.clamp(MIN_SCALE, MAX_SCALE)
}

pub fn build_calibration_request(
    run_id: String,
    best_energy: f64,
    acceptance_ratio: Option<f64>,
    energy_trace: &[f64],
    config: &QuantumConfig,
    max_trace_samples: usize,
) -> QuantumCalibrationRequest {
    QuantumCalibrationRequest {
        run_id,
        best_energy,
        acceptance_ratio,
        energy_trace: sample_trace(energy_trace, max_trace_samples),
        config: QuantumCalibrationSnapshot {
            trotter_slices: config.trotter_slices,
            driver_strength: config.driver_strength,
            driver_final_strength: config.driver_final_strength,
            worldline_mix_prob: config.worldline_mix_prob,
            slice_temperature_scale: config.slice_temperature_scale,
        },
    }
}

fn sample_trace(trace: &[f64], max_samples: usize) -> Vec<f64> {
    if trace.is_empty() {
        return Vec::new();
    }
    if max_samples == 0 || trace.len() <= max_samples {
        return trace.to_vec();
    }
    let step = (trace.len() as f64 / max_samples as f64).max(1.0);
    let mut out = Vec::with_capacity(max_samples);
    let mut idx = 0.0;
    while out.len() < max_samples && (idx as usize) < trace.len() {
        out.push(trace[idx as usize]);
        idx += step;
    }
    out
}

pub fn build_calibration_job(
    request: &QuantumCalibrationRequest,
    timeout_secs: u64,
) -> Result<QuantumJob> {
    let payload = serde_json::to_vec(request)?;
    Ok(QuantumJob {
        kind: ComputeJobKind::QuantumCalibration,
        payload,
        timeout_secs,
    })
}

pub fn parse_calibration_response(result: QuantumResult) -> Result<QuantumCalibrationResponse> {
    let parsed: QuantumCalibrationResponse = serde_json::from_slice(&result.payload)
        .context("invalid quantum calibration response payload")?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_trace_downsamples_to_limit() {
        let trace: Vec<f64> = (0..100).map(|idx| idx as f64).collect();
        let sampled = sample_trace(&trace, 10);
        assert_eq!(sampled.len(), 10);
        assert_eq!(sampled.first().copied().unwrap_or(-1.0), 0.0);
    }

    #[test]
    fn calibration_state_blends_scales() {
        let mut state = QuantumCalibrationState::default();
        let adjust = QuantumCalibrationAdjust {
            driver_strength_scale: Some(2.0),
            driver_final_strength_scale: Some(0.5),
            slice_temperature_scale: Some(1.5),
            worldline_mix_delta: Some(0.2),
        };
        state.update_from_adjustment(&adjust, 0.5);
        assert!((state.driver_strength_scale - 1.5).abs() < 1e-6);
        assert!((state.driver_final_strength_scale - 0.75).abs() < 1e-6);
        assert!((state.slice_temperature_scale - 1.25).abs() < 1e-6);
        assert!((state.worldline_mix_delta - 0.1).abs() < 1e-6);
    }

    #[test]
    fn calibration_apply_clamps_mix_prob() {
        let mut state = QuantumCalibrationState::default();
        state.worldline_mix_delta = 5.0;
        let config = QuantumConfig::default();
        let adjusted = state.apply(&config);
        assert!((0.0..=1.0).contains(&adjusted.worldline_mix_prob));
    }
}
