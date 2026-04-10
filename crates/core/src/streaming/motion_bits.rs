//! Motion Data-Bits Encoder
//!
//! Converts a stream of 2D cursor/pointer positions over time into the label
//! format understood by NeuronPool. Shares the same spatial grid vocabulary
//! as `ImageBitsEncoder` so that screen zones referenced by image labels and
//! cursor positions reference the exact same neurons — no glue code needed.
//!
//! Label families emitted:
//!
//!  * `mov:zone_x{N}_y{N}`         — current grid zone (matches img: zones)
//!  * `mov:zone_x{N}_y{N}_t{T}`    — zone at time slot T (temporal sequence)
//!  * `mov:dx{D}_dy{D}`            — direction bucket (coarse velocity vector)
//!  * `mov:speed_{s}`              — speed bucket: still / slow / medium / fast
//!  * `act:click_zone_x{N}_y{N}`   — click event at zone (emitted separately)
//!  * `act:click`                  — click event without position

use serde::{Deserialize, Serialize};

// ── Input ─────────────────────────────────────────────────────────────────────

/// A single position sample in a motion trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionSample {
    /// Horizontal position as a fraction of screen width [0, 1).
    pub x: f32,
    /// Vertical position as a fraction of screen height [0, 1).
    pub y: f32,
    /// Time offset within this trajectory in seconds (0.0 = start).
    pub t_secs: f32,
    /// Whether a primary button click occurred at this sample.
    pub click: bool,
}

impl MotionSample {
    pub fn new(x: f32, y: f32, t_secs: f32) -> Self {
        Self { x, y, t_secs, click: false }
    }
    pub fn with_click(x: f32, y: f32, t_secs: f32) -> Self {
        Self { x, y, t_secs, click: true }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionBitsConfig {
    /// Spatial grid size — must match image_bits grid for zone vocab to align.
    pub grid_x: usize,
    pub grid_y: usize,
    /// Number of temporal slots for `_t{T}` labels.
    pub time_slots: usize,
    /// Number of direction buckets per axis (dx, dy each bucketed 0..dir_bins).
    pub dir_bins: usize,
    /// Number of speed buckets.
    pub speed_bins: usize,
    /// Label prefix for position labels.
    pub stream_tag: String,
    /// Label prefix for action labels (click, drag, etc.).
    pub action_tag: String,
}

impl Default for MotionBitsConfig {
    fn default() -> Self {
        Self {
            grid_x:     8,
            grid_y:     8,
            time_slots: 16,
            dir_bins:   5,   // -2, -1, 0, +1, +2 mapped to 0-4
            speed_bins: 4,
            stream_tag: "mov".to_string(),
            action_tag: "act".to_string(),
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionBitsOutput {
    pub labels: Vec<String>,
    pub samples_processed: usize,
    /// The final position zone — most useful for click association.
    pub endpoint_zone: (usize, usize),
    /// Whether any click was emitted.
    pub has_click: bool,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct MotionBitsEncoder {
    cfg: MotionBitsConfig,
}

impl MotionBitsEncoder {
    pub fn new(cfg: MotionBitsConfig) -> Self {
        Self { cfg }
    }

    /// Encode a trajectory (ordered time series of position samples).
    pub fn encode_trajectory(&self, samples: &[MotionSample]) -> MotionBitsOutput {
        let tag  = &self.cfg.stream_tag;
        let atag = &self.cfg.action_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut has_click = false;
        let mut endpoint_zone = (0, 0);

        if samples.is_empty() {
            return MotionBitsOutput { labels, samples_processed: 0, endpoint_zone, has_click };
        }

        let total = samples.len();
        let t_max = samples.last().map(|s| s.t_secs).unwrap_or(1.0).max(0.001);

        for (i, sample) in samples.iter().enumerate() {
            let zx = ((sample.x * self.cfg.grid_x as f32) as usize).min(self.cfg.grid_x - 1);
            let zy = ((sample.y * self.cfg.grid_y as f32) as usize).min(self.cfg.grid_y - 1);
            endpoint_zone = (zx, zy);

            // Zone label
            labels.push(format!("{tag}:zone_x{zx}_y{zy}"));

            // Temporal zone label
            let t_slot = ((sample.t_secs / t_max) * self.cfg.time_slots as f32) as usize;
            let t_slot = t_slot.min(self.cfg.time_slots - 1);
            labels.push(format!("{tag}:zone_x{zx}_y{zy}_t{t_slot}"));
            labels.push(format!("{tag}:t{t_slot}"));

            // Direction and speed (relative to previous sample)
            if i > 0 {
                let prev = &samples[i - 1];
                let dt = (sample.t_secs - prev.t_secs).max(0.001);
                let dx = sample.x - prev.x;
                let dy = sample.y - prev.y;
                let speed = (dx * dx + dy * dy).sqrt() / dt;

                // Direction bins: normalise to [-1, 1], bucket to [0, dir_bins)
                let dx_bin = dir_bin(dx, self.cfg.dir_bins);
                let dy_bin = dir_bin(dy, self.cfg.dir_bins);
                labels.push(format!("{tag}:dx{dx_bin}_dy{dy_bin}"));

                // Speed bin
                let speed_bin = (speed * self.cfg.speed_bins as f32 * 4.0) as usize;
                let speed_bin = speed_bin.min(self.cfg.speed_bins - 1);
                labels.push(format!("{tag}:speed_{speed_bin}"));
            }

            // Click event
            if sample.click {
                has_click = true;
                labels.push(format!("{atag}:click"));
                labels.push(format!("{atag}:click_zone_x{zx}_y{zy}"));
            }
        }

        // Endpoint emphasis — the destination matters most for goal association.
        let (ex, ey) = endpoint_zone;
        labels.push(format!("{tag}:endpoint_x{ex}_y{ey}"));

        labels.sort_unstable();
        labels.dedup();

        MotionBitsOutput {
            labels,
            samples_processed: total,
            endpoint_zone,
            has_click,
        }
    }

    /// Encode a single position snapshot (no trajectory context).
    pub fn encode_point(&self, x: f32, y: f32, click: bool) -> MotionBitsOutput {
        let sample = if click {
            MotionSample::with_click(x, y, 0.0)
        } else {
            MotionSample::new(x, y, 0.0)
        };
        self.encode_trajectory(&[sample])
    }

    /// Decode activated motion labels back to the most likely target zone.
    /// Returns (zx, zy, strength) of the highest-confidence endpoint zone.
    ///
    /// Priority order:
    ///  1. If any `mov:endpoint_*` labels are activated, pick the highest-scoring one.
    ///  2. Otherwise fall back to `act:click_zone_*` labels.
    ///  3. Otherwise use bare zone labels (`mov:zone_x*_y*` without `_t`).
    ///
    /// Trajectory labels (`_t*`) are intentionally ignored — they represent
    /// path pass-throughs, not targets, and accumulate across many zones,
    /// drowning the specific endpoint signal.
    pub fn decode_target(activated: &[(String, f32)]) -> Option<(usize, usize, f32)> {
        use std::collections::HashMap;

        // Priority 1: endpoint labels
        let mut endpoint_scores: HashMap<(usize, usize), f32> = HashMap::new();
        let mut click_scores: HashMap<(usize, usize), f32> = HashMap::new();
        let mut zone_scores: HashMap<(usize, usize), f32> = HashMap::new();

        for (label, strength) in activated {
            if label.starts_with("mov:endpoint_x") {
                if let Some(key) = parse_zone(label, "mov:endpoint_x", "_y") {
                    *endpoint_scores.entry(key).or_insert(0.0) += strength;
                }
            } else if label.starts_with("act:click_zone_x") {
                if let Some(key) = parse_zone(label, "act:click_zone_x", "_y") {
                    *click_scores.entry(key).or_insert(0.0) += strength;
                }
            } else if label.starts_with("mov:zone_x") && !label.contains("_t") {
                if let Some(key) = parse_zone(label, "mov:zone_x", "_y") {
                    *zone_scores.entry(key).or_insert(0.0) += strength;
                }
            }
            // Ignore mov:zone_x*_y*_t* — trajectory noise, not target signals
        }

        // Pick best from highest-priority group that has any candidates
        let best_map = if !endpoint_scores.is_empty() {
            &endpoint_scores
        } else if !click_scores.is_empty() {
            &click_scores
        } else {
            &zone_scores
        };

        best_map
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(&(zx, zy), &score)| (zx, zy, score))
    }

    /// Convert a zone (zx, zy) back to approximate screen-fraction coordinates.
    pub fn zone_to_frac(&self, zx: usize, zy: usize) -> (f32, f32) {
        let x = (zx as f32 + 0.5) / self.cfg.grid_x as f32;
        let y = (zy as f32 + 0.5) / self.cfg.grid_y as f32;
        (x, y)
    }

    pub fn config(&self) -> &MotionBitsConfig {
        &self.cfg
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn dir_bin(delta: f32, bins: usize) -> usize {
    // Map [-0.1..+0.1] to [0..bins), clamped.
    let norm = (delta * 10.0 + 1.0) / 2.0; // → [0,1]
    ((norm * bins as f32) as usize).min(bins - 1)
}

fn parse_zone(label: &str, prefix: &str, sep: &str) -> Option<(usize, usize)> {
    let rest = label.strip_prefix(prefix)?;
    let (xs, ys) = rest.split_once(sep)?;
    // ys may have trailing _t{N} — take only up to next '_' or end
    let ys_clean = ys.split('_').next()?;
    let zx = xs.parse::<usize>().ok()?;
    let zy = ys_clean.parse::<usize>().ok()?;
    Some((zx, zy))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_emits_zone_and_click_labels() {
        let enc = MotionBitsEncoder::new(MotionBitsConfig::default());
        let traj = vec![
            MotionSample::new(0.1, 0.1, 0.0),
            MotionSample::new(0.3, 0.2, 0.1),
            MotionSample::with_click(0.7, 0.5, 0.3),
        ];
        let out = enc.encode_trajectory(&traj);
        assert!(out.has_click);
        assert!(out.labels.iter().any(|l| l.starts_with("mov:zone_")));
        assert!(out.labels.iter().any(|l| l.starts_with("act:click_zone_")));
        assert!(out.labels.iter().any(|l| l.starts_with("mov:endpoint_")));
    }

    #[test]
    fn decode_target_finds_highest_endpoint() {
        let activated = vec![
            ("mov:endpoint_x5_y3".to_string(), 0.92),
            ("mov:zone_x3_y2".to_string(), 0.5),
        ];
        let result = MotionBitsEncoder::decode_target(&activated);
        assert_eq!(result, Some((5, 3, 0.92)));
    }

    #[test]
    fn zone_matches_image_bits_grid() {
        let enc = MotionBitsEncoder::new(MotionBitsConfig::default());
        // Position at 0.9, 0.9 → zone 7, 7 on 8×8 grid
        let out = enc.encode_point(0.9, 0.9, false);
        assert!(out.labels.iter().any(|l| l.contains("zone_x7_y7")));
    }
}
