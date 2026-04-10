//! Image Data-Bits Encoder
//!
//! Converts raw pixel arrays (RGB or grayscale) into the label-token format
//! understood by the NeuronPool. Each pixel's position and colour values
//! become discrete symbol labels — "data bits" — so the neural fabric can
//! learn spatial and chromatic associations without any hard-coded CNN.
//!
//! Design principles (brain-like, not rigid):
//!
//!  * **Spatial zones** — the image is divided into a configurable grid
//!    (default 16×16 cells). Each occupied cell emits a `img:zX_Y` label so
//!    the fabric learns where features tend to appear.
//!
//!  * **Colour bins** — hue, saturation, and value (HSV) are each bucketed
//!    into a configurable number of bins (default 16). Each bin emits a label
//!    `img:hNNN`, `img:sNNN`, `img:vNNN`. Colour alone fires independently of
//!    position so the fabric can learn "blue" before it learns "sky".
//!
//!  * **Edge primitives** — a 3×3 Sobel kernel detects horizontal and vertical
//!    gradients; strong edges emit `img:edgeH` / `img:edgeV` / `img:edgeD`
//!    labels tied to their zone. This is the only engineered feature — it
//!    mimics the orientation-selective cells in V1 cortex.
//!
//!  * **Salience gating** — pixels below a configurable brightness threshold
//!    are skipped, reducing noise labels (mirrors retinal ganglion cell gating).
//!
//!  * **Temporal context** — labels include a configurable `stream_tag` prefix
//!    so the fabric knows these bits came from an image stream (vs audio/text).
//!
//! All labels are plain strings. They feed directly into `NeuronPool::train()`
//! or `NeuronPool::propagate()` with no additional glue code.

use serde::{Deserialize, Serialize};

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBitsConfig {
    /// Width of the incoming image in pixels.
    pub width: usize,
    /// Height of the incoming image in pixels.
    pub height: usize,
    /// Number of spatial grid cells per axis (grid_x × grid_y zones).
    pub grid_x: usize,
    pub grid_y: usize,
    /// Number of hue bins (colour wheel divided evenly).
    pub hue_bins: usize,
    /// Number of saturation bins.
    pub sat_bins: usize,
    /// Number of value (brightness) bins.
    pub val_bins: usize,
    /// Brightness (0–255) below which a pixel is skipped.
    pub salience_threshold: u8,
    /// Fraction of pixels to sample per zone (1.0 = all, 0.1 = 10%).
    pub sample_rate: f32,
    /// Label prefix, e.g. "img" → "img:zX_Y", "img:h3".
    pub stream_tag: String,
    /// Emit edge labels in addition to colour/position.
    pub edge_detection: bool,
}

impl Default for ImageBitsConfig {
    fn default() -> Self {
        Self {
            width:              64,
            height:             64,
            grid_x:             8,
            grid_y:             8,
            hue_bins:           16,
            sat_bins:           8,
            val_bins:           8,
            salience_threshold: 20,
            sample_rate:        0.25,
            stream_tag:         "img".to_string(),
            edge_detection:     true,
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBitsOutput {
    /// Flat list of label tokens for this frame.
    pub labels: Vec<String>,
    /// Width / height of the scaled image that was processed.
    pub processed_width: usize,
    pub processed_height: usize,
    /// How many pixels were emitted (after salience / sampling gates).
    pub active_pixels: usize,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct ImageBitsEncoder {
    cfg: ImageBitsConfig,
}

impl ImageBitsEncoder {
    pub fn new(cfg: ImageBitsConfig) -> Self {
        Self { cfg }
    }

    /// Encode a raw RGB pixel buffer into label tokens.
    ///
    /// `pixels` must be laid out as `[R, G, B, R, G, B, …]` in row-major
    /// order. Length must equal `width * height * 3`.
    pub fn encode_rgb(&self, pixels: &[u8], width: usize, height: usize) -> ImageBitsOutput {
        let tag = &self.cfg.stream_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut active = 0usize;

        // Build a flat greyscale + gradient buffer for edge detection.
        let grey: Vec<f32> = pixels
            .chunks(3)
            .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
            .collect();

        let stride_x = (width  as f32 / self.cfg.grid_x as f32).ceil() as usize;
        let stride_y = (height as f32 / self.cfg.grid_y as f32).ceil() as usize;

        // Sample deterministically — every Nth pixel in each zone.
        let sample_every = ((1.0 / self.cfg.sample_rate).round() as usize).max(1);
        let mut counter = 0usize;

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                let r = pixels[idx * 3];
                let g = pixels[idx * 3 + 1];
                let b = pixels[idx * 3 + 2];

                // Salience gate
                let brightness = grey[idx] as u8;
                if brightness < self.cfg.salience_threshold {
                    continue;
                }

                // Sampling gate
                counter += 1;
                if counter % sample_every != 0 {
                    continue;
                }

                active += 1;

                // ── Spatial zone ──────────────────────────────────────────────
                let zx = (x / stride_x).min(self.cfg.grid_x - 1);
                let zy = (y / stride_y).min(self.cfg.grid_y - 1);
                labels.push(format!("{tag}:z{zx}_{zy}"));

                // ── Colour (HSV) ──────────────────────────────────────────────
                let (h_bin, s_bin, v_bin) = rgb_to_hsv_bins(
                    r, g, b,
                    self.cfg.hue_bins,
                    self.cfg.sat_bins,
                    self.cfg.val_bins,
                );
                labels.push(format!("{tag}:h{h_bin}"));
                labels.push(format!("{tag}:s{s_bin}"));
                labels.push(format!("{tag}:v{v_bin}"));

                // Colour+zone composite — lets the fabric learn "blue in top-left"
                labels.push(format!("{tag}:h{h_bin}_z{zx}_{zy}"));

                // ── Edge detection (Sobel 3×3) ─────────────────────────────────
                if self.cfg.edge_detection && x > 0 && x < width - 1 && y > 0 && y < height - 1 {
                    let gx = sobel_x(&grey, x, y, width);
                    let gy = sobel_y(&grey, x, y, width);
                    let mag = (gx * gx + gy * gy).sqrt();
                    if mag > 30.0 {
                        let edge_kind = edge_label(gx, gy);
                        labels.push(format!("{tag}:edge{edge_kind}_z{zx}_{zy}"));
                    }
                }
            }
        }

        // Deduplicate — many pixels in the same zone fire the same zone label.
        labels.sort_unstable();
        labels.dedup();

        ImageBitsOutput {
            labels,
            processed_width:  width,
            processed_height: height,
            active_pixels:    active,
        }
    }

    /// Convenience: encode a greyscale image (single channel).
    pub fn encode_grey(&self, pixels: &[u8], width: usize, height: usize) -> ImageBitsOutput {
        // Expand to RGB by tripling each byte, then call encode_rgb.
        let rgb: Vec<u8> = pixels.iter().flat_map(|&v| [v, v, v]).collect();
        self.encode_rgb(&rgb, width, height)
    }

    /// Encode a JPEG/PNG from raw compressed bytes.
    /// Returns None if the bytes can't be decoded as an image.
    #[cfg(feature = "image-decode")]
    pub fn encode_bytes(&self, bytes: &[u8]) -> Option<ImageBitsOutput> {
        use image::GenericImageView;
        let img = image::load_from_memory(bytes).ok()?;
        let img = img.resize_exact(
            self.cfg.width as u32,
            self.cfg.height as u32,
            image::imageops::FilterType::Nearest,
        );
        let rgb = img.to_rgb8();
        Some(self.encode_rgb(rgb.as_raw(), self.cfg.width, self.cfg.height))
    }

    pub fn config(&self) -> &ImageBitsConfig {
        &self.cfg
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert RGB u8 → HSV bins.
fn rgb_to_hsv_bins(r: u8, g: u8, b: u8, hue_bins: usize, sat_bins: usize, val_bins: usize)
    -> (usize, usize, usize)
{
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;

    let cmax = rf.max(gf).max(bf);
    let cmin = rf.min(gf).min(bf);
    let delta = cmax - cmin;

    // Hue [0, 360)
    let hue = if delta < 1e-6 {
        0.0f32
    } else if (cmax - rf).abs() < 1e-6 {
        60.0 * (((gf - bf) / delta) % 6.0)
    } else if (cmax - gf).abs() < 1e-6 {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    let hue = ((hue + 360.0) % 360.0) / 360.0; // normalise to [0,1)

    let sat = if cmax < 1e-6 { 0.0f32 } else { delta / cmax };
    let val = cmax;

    let h_bin = ((hue * hue_bins as f32) as usize).min(hue_bins - 1);
    let s_bin = ((sat * sat_bins as f32) as usize).min(sat_bins - 1);
    let v_bin = ((val * val_bins as f32) as usize).min(val_bins - 1);

    (h_bin, s_bin, v_bin)
}

fn sobel_x(grey: &[f32], x: usize, y: usize, w: usize) -> f32 {
    let p = |dx: i32, dy: i32| grey[((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize];
    -p(-1,-1) + p(1,-1) - 2.0*p(-1,0) + 2.0*p(1,0) - p(-1,1) + p(1,1)
}

fn sobel_y(grey: &[f32], x: usize, y: usize, w: usize) -> f32 {
    let p = |dx: i32, dy: i32| grey[((y as i32 + dy) as usize) * w + (x as i32 + dx) as usize];
    -p(-1,-1) - 2.0*p(0,-1) - p(1,-1) + p(-1,1) + 2.0*p(0,1) + p(1,1)
}

fn edge_label(gx: f32, gy: f32) -> &'static str {
    let angle = gy.atan2(gx).to_degrees().abs() % 180.0;
    if angle < 22.5 || angle >= 157.5      { "H" }  // horizontal
    else if angle < 67.5                   { "D1" } // diagonal /
    else if angle < 112.5                  { "V" }  // vertical
    else                                   { "D2" } // diagonal \
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_red_pixel_emits_labels() {
        let cfg = ImageBitsConfig {
            width: 4, height: 4,
            grid_x: 2, grid_y: 2,
            hue_bins: 8, sat_bins: 4, val_bins: 4,
            salience_threshold: 10,
            sample_rate: 1.0,
            stream_tag: "img".to_string(),
            edge_detection: false,
        };
        let enc = ImageBitsEncoder::new(cfg);
        // 4×4 solid red image
        let pixels = vec![255u8, 0, 0; 16];
        let out = enc.encode_rgb(&pixels, 4, 4);
        assert!(!out.labels.is_empty());
        assert!(out.labels.iter().any(|l| l.starts_with("img:h")));
        assert!(out.labels.iter().any(|l| l.starts_with("img:z")));
    }

    #[test]
    fn dark_pixels_below_threshold_skipped() {
        let cfg = ImageBitsConfig {
            width: 4, height: 4,
            grid_x: 2, grid_y: 2,
            hue_bins: 8, sat_bins: 4, val_bins: 4,
            salience_threshold: 200, // very high
            sample_rate: 1.0,
            stream_tag: "img".to_string(),
            edge_detection: false,
        };
        let enc = ImageBitsEncoder::new(cfg);
        let pixels = vec![10u8, 10, 10; 16]; // dark grey
        let out = enc.encode_rgb(&pixels, 4, 4);
        assert_eq!(out.active_pixels, 0);
    }
}
