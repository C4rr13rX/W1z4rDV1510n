//! Audio Data-Bits Encoder
//!
//! Converts raw PCM audio samples into the label format understood by the
//! NeuronPool. Each short-time window of audio becomes a set of discrete
//! symbol labels — "data bits" — so the neural fabric can learn acoustic
//! patterns (phonemes, tones, rhythms) the same way it learns visual ones.
//!
//! Design principles (brain-like, not rigid):
//!
//!  * **Frequency bands** — the window is FFT-analysed and mapped into
//!    configurable frequency bands (default 32). Each band above threshold
//!    emits `aud:freq{N}` — analogous to cochlear hair-cell tonotopy.
//!
//!  * **Amplitude bins** — the RMS energy of each band is bucketed into
//!    configurable amplitude levels (default 8). Emits `aud:amp{N}`.
//!
//!  * **Temporal ticks** — a coarse time-slot index (within the audio chunk)
//!    is appended as `aud:t{N}`, giving the fabric a sense of temporal
//!    sequence without any explicit recurrence.
//!
//!  * **Composite labels** — `aud:freq{F}_t{T}` lets the fabric learn
//!    "this frequency appears at this point in time" (onset detection).
//!
//!  * **Silence gating** — windows below a configurable RMS threshold are
//!    skipped entirely (mirrors auditory nerve silence suppression).
//!
//! Output labels feed directly into `NeuronPool::train()` or
//! `NeuronPool::propagate()` — the same pipeline as image or text labels.

use serde::{Deserialize, Serialize};

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBitsConfig {
    /// Sample rate of the incoming PCM stream (Hz).
    pub sample_rate: u32,
    /// Number of PCM samples per analysis window (power of 2 recommended).
    pub window_size: usize,
    /// How many samples to advance between windows (overlap = window_size - hop).
    pub hop_size: usize,
    /// Number of frequency bands to bin the spectrum into.
    pub freq_bins: usize,
    /// Number of amplitude levels per band.
    pub amp_bins: usize,
    /// Number of coarse temporal slots within a chunk (for `aud:t{N}`).
    pub time_slots: usize,
    /// RMS energy (0–1 normalised) below which a window is treated as silence.
    pub silence_threshold: f32,
    /// Label prefix, e.g. "aud" → "aud:freq3", "aud:amp2".
    pub stream_tag: String,
}

impl Default for AudioBitsConfig {
    fn default() -> Self {
        Self {
            sample_rate:       16_000,
            window_size:       512,
            hop_size:          256,
            freq_bins:         32,
            amp_bins:          8,
            time_slots:        16,
            silence_threshold: 0.005,
            stream_tag:        "aud".to_string(),
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioBitsOutput {
    /// Flat list of label tokens for this audio chunk.
    pub labels: Vec<String>,
    /// Number of analysis windows processed.
    pub windows_processed: usize,
    /// Number of windows that passed the silence gate.
    pub active_windows: usize,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct AudioBitsEncoder {
    cfg: AudioBitsConfig,
}

impl AudioBitsEncoder {
    pub fn new(cfg: AudioBitsConfig) -> Self {
        Self { cfg }
    }

    /// Encode a mono PCM chunk into label strings.
    ///
    /// `samples` — f32 normalised to [-1, 1]. Length can be any size;
    /// the encoder slides a window across it with `hop_size` stride.
    pub fn encode_mono(&self, samples: &[f32]) -> AudioBitsOutput {
        let tag = &self.cfg.stream_tag;
        let ws = self.cfg.window_size;
        let hop = self.cfg.hop_size;
        let mut labels: Vec<String> = Vec::new();
        let mut windows_processed = 0usize;
        let mut active_windows = 0usize;

        let total_windows = if samples.len() >= ws {
            (samples.len() - ws) / hop + 1
        } else {
            0
        };

        for win_idx in 0..total_windows {
            let start = win_idx * hop;
            let window = &samples[start..start + ws];

            windows_processed += 1;

            // Silence gate: skip if RMS is below threshold.
            let rms = (window.iter().map(|&s| s * s).sum::<f32>() / ws as f32).sqrt();
            if rms < self.cfg.silence_threshold {
                continue;
            }
            active_windows += 1;

            // Coarse temporal slot within this chunk.
            let t_slot = ((win_idx as f32 / total_windows.max(1) as f32)
                * self.cfg.time_slots as f32) as usize;
            let t_slot = t_slot.min(self.cfg.time_slots - 1);

            // DFT magnitude spectrum (real-input, only positive frequencies).
            let spectrum = dft_magnitude(window);

            // Map spectrum bins → freq_bins by summing energy in each band.
            let nyquist_bins = spectrum.len(); // ws/2 + 1
            for band in 0..self.cfg.freq_bins {
                let lo = (band * nyquist_bins) / self.cfg.freq_bins;
                let hi = ((band + 1) * nyquist_bins) / self.cfg.freq_bins;
                let hi = hi.min(nyquist_bins);
                if lo >= hi {
                    continue;
                }
                let energy: f32 = spectrum[lo..hi].iter().sum::<f32>() / (hi - lo) as f32;

                // Normalise to [0, 1] assuming max FFT bin ≈ window_size/2.
                let norm = (energy / (ws as f32 * 0.5)).min(1.0);
                let amp_bin = ((norm * self.cfg.amp_bins as f32) as usize)
                    .min(self.cfg.amp_bins - 1);

                // Only emit if band has meaningful energy.
                if norm < 0.005 {
                    continue;
                }

                labels.push(format!("{tag}:freq{band}"));
                labels.push(format!("{tag}:amp{amp_bin}"));
                labels.push(format!("{tag}:freq{band}_t{t_slot}"));
                labels.push(format!("{tag}:amp{amp_bin}_t{t_slot}"));
            }

            // Overall loudness label.
            let loudness_bin = ((rms * self.cfg.amp_bins as f32) as usize)
                .min(self.cfg.amp_bins - 1);
            labels.push(format!("{tag}:loud{loudness_bin}"));
            labels.push(format!("{tag}:t{t_slot}"));
        }

        labels.sort_unstable();
        labels.dedup();

        AudioBitsOutput {
            labels,
            windows_processed,
            active_windows,
        }
    }

    /// Encode interleaved stereo PCM by downmixing to mono first.
    pub fn encode_stereo(&self, samples: &[f32]) -> AudioBitsOutput {
        let mono: Vec<f32> = samples
            .chunks(2)
            .map(|ch| (ch[0] + ch.get(1).copied().unwrap_or(0.0)) * 0.5)
            .collect();
        self.encode_mono(&mono)
    }

    /// Encode i16 PCM (common raw format) by converting to f32 first.
    pub fn encode_i16_mono(&self, samples: &[i16]) -> AudioBitsOutput {
        let f: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
        self.encode_mono(&f)
    }

    /// Decode and encode a WAV file from raw bytes.
    /// Handles mono and stereo, 16-bit integer PCM (the most common WAV format).
    /// Returns None if the bytes are not valid WAV.
    pub fn encode_wav_bytes(&self, bytes: &[u8]) -> Option<AudioBitsOutput> {
        use std::io::Cursor;
        let cursor = Cursor::new(bytes);
        let mut reader = hound::WavReader::new(cursor).ok()?;
        let spec = reader.spec();
        let samples_i16: Vec<i16> = reader
            .samples::<i16>()
            .filter_map(|s| s.ok())
            .collect();
        if spec.channels == 1 {
            Some(self.encode_i16_mono(&samples_i16))
        } else {
            // Stereo interleaved
            let f32s: Vec<f32> = samples_i16.iter().map(|&s| s as f32 / 32768.0).collect();
            Some(self.encode_stereo(&f32s))
        }
    }

    pub fn config(&self) -> &AudioBitsConfig {
        &self.cfg
    }
}

// ── DFT ───────────────────────────────────────────────────────────────────────
//
// A minimal O(n²) real-input DFT — correct, portable, no external deps.
// For production high-throughput ingestion, replace with an FFT crate
// (e.g. `rustfft`) — the encoder interface is identical.

fn dft_magnitude(window: &[f32]) -> Vec<f32> {
    let n = window.len();
    let half = n / 2 + 1;
    let mut out = vec![0.0f32; half];

    // Apply a simple Hann window to reduce spectral leakage.
    let hann: Vec<f32> = (0..n)
        .map(|i| {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
            window[i] * w
        })
        .collect();

    for k in 0..half {
        let mut re = 0.0f32;
        let mut im = 0.0f32;
        let angle = -2.0 * std::f32::consts::PI * k as f32 / n as f32;
        for (i, &s) in hann.iter().enumerate() {
            re += s * (angle * i as f32).cos();
            im += s * (angle * i as f32).sin();
        }
        out[k] = (re * re + im * im).sqrt();
    }
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_tone_emits_freq_labels() {
        let cfg = AudioBitsConfig {
            sample_rate: 8000,
            window_size: 256,
            hop_size: 128,
            freq_bins: 16,
            amp_bins: 4,
            time_slots: 8,
            silence_threshold: 0.001,
            stream_tag: "aud".to_string(),
        };
        let enc = AudioBitsEncoder::new(cfg);
        // 440 Hz tone at 8 kHz sample rate
        let samples: Vec<f32> = (0..512)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 8000.0).sin())
            .collect();
        let out = enc.encode_mono(&samples);
        assert!(!out.labels.is_empty(), "should emit labels for a 440 Hz tone");
        assert!(out.labels.iter().any(|l| l.starts_with("aud:freq")));
        assert!(out.active_windows > 0);
    }

    #[test]
    fn silence_below_threshold_skipped() {
        let cfg = AudioBitsConfig::default();
        let enc = AudioBitsEncoder::new(cfg);
        let silence = vec![0.0f32; 4096];
        let out = enc.encode_mono(&silence);
        assert_eq!(out.active_windows, 0);
        assert!(out.labels.is_empty());
    }
}
