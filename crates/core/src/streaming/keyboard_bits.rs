//! Keyboard Data-Bits Encoder
//!
//! Converts keyboard events into the label format understood by NeuronPool.
//! Key names, modifier state, and temporal position all become labels so the
//! fabric can learn sequences like Ctrl+C or typed words alongside the screen
//! context and goal text that surrounded them during training.
//!
//! Label families emitted:
//!
//!  * `key:k_{name}`           — the key itself (e.g. key:k_enter, key:k_a)
//!  * `key:mod_ctrl`           — Ctrl modifier was held
//!  * `key:mod_shift`          — Shift modifier was held
//!  * `key:mod_alt`            — Alt modifier was held
//!  * `key:combo_{mod}_{key}`  — modifier+key composite (key:combo_ctrl_c)
//!  * `key:t{T}`               — temporal slot within this sequence
//!  * `key:text_{word}`        — if the key sequence spells a recognisable word,
//!                               emits a txt-style word label for cross-modal
//!                               linkage with the text pool

use serde::{Deserialize, Serialize};

// ── Input ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEvent {
    /// Key name — use standard web KeyboardEvent.key values:
    /// "a".."z", "0".."9", "Enter", "Backspace", "Tab", "Escape",
    /// "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "F1".."F12", etc.
    pub key: String,
    pub ctrl:  bool,
    pub shift: bool,
    pub alt:   bool,
    /// Time offset in seconds from start of this sequence (0.0 = first key).
    pub t_secs: f32,
}

impl KeyEvent {
    pub fn simple(key: &str, t_secs: f32) -> Self {
        Self { key: key.to_string(), ctrl: false, shift: false, alt: false, t_secs }
    }
    pub fn ctrl(key: &str, t_secs: f32) -> Self {
        Self { key: key.to_string(), ctrl: true, shift: false, alt: false, t_secs }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardBitsConfig {
    pub time_slots: usize,
    pub stream_tag: String,
    /// Also emit word labels (txt:word_{w}) when typed characters spell a word.
    pub emit_word_labels: bool,
}

impl Default for KeyboardBitsConfig {
    fn default() -> Self {
        Self {
            time_slots:       16,
            stream_tag:       "key".to_string(),
            emit_word_labels: true,
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardBitsOutput {
    pub labels: Vec<String>,
    pub keys_processed: usize,
    /// Reconstructed typed text (printable characters only).
    pub typed_text: String,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct KeyboardBitsEncoder {
    cfg: KeyboardBitsConfig,
}

impl KeyboardBitsEncoder {
    pub fn new(cfg: KeyboardBitsConfig) -> Self {
        Self { cfg }
    }

    pub fn encode_sequence(&self, events: &[KeyEvent]) -> KeyboardBitsOutput {
        let tag = &self.cfg.stream_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut typed = String::new();

        let t_max = events.last().map(|e| e.t_secs).unwrap_or(1.0).max(0.001);

        for event in events {
            let key_norm = normalise_key(&event.key);

            // Key label
            labels.push(format!("{tag}:k_{key_norm}"));

            // Modifier labels
            if event.ctrl  { labels.push(format!("{tag}:mod_ctrl")); }
            if event.shift { labels.push(format!("{tag}:mod_shift")); }
            if event.alt   { labels.push(format!("{tag}:mod_alt")); }

            // Combo label
            if event.ctrl || event.shift || event.alt {
                let mut mods = String::new();
                if event.ctrl  { mods.push_str("ctrl_"); }
                if event.shift { mods.push_str("shift_"); }
                if event.alt   { mods.push_str("alt_"); }
                labels.push(format!("{tag}:combo_{mods}{key_norm}"));
            }

            // Temporal slot
            let t_slot = ((event.t_secs / t_max) * self.cfg.time_slots as f32) as usize;
            let t_slot = t_slot.min(self.cfg.time_slots - 1);
            labels.push(format!("{tag}:t{t_slot}"));
            labels.push(format!("{tag}:k_{key_norm}_t{t_slot}"));

            // Accumulate typed text
            if is_printable(&event.key) && !event.ctrl && !event.alt {
                if event.shift {
                    typed.push_str(&event.key.to_uppercase());
                } else {
                    typed.push_str(&event.key);
                }
            } else if event.key == "Space" || event.key == " " {
                typed.push(' ');
            } else if event.key == "Backspace" && !typed.is_empty() {
                typed.pop();
            } else if event.key == "Enter" {
                typed.push('\n');
            }
        }

        // Word labels from typed text
        if self.cfg.emit_word_labels {
            for word in typed.split_whitespace() {
                let w = word.to_lowercase();
                let w: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
                if w.len() >= 2 {
                    labels.push(format!("txt:word_{w}"));
                }
            }
        }

        labels.sort_unstable();
        labels.dedup();

        KeyboardBitsOutput {
            labels,
            keys_processed: events.len(),
            typed_text: typed,
        }
    }

    /// Convenience: encode a single keyboard shortcut.
    pub fn encode_shortcut(&self, key: &str, ctrl: bool, shift: bool, alt: bool) -> KeyboardBitsOutput {
        let event = KeyEvent { key: key.to_string(), ctrl, shift, alt, t_secs: 0.0 };
        self.encode_sequence(&[event])
    }

    pub fn config(&self) -> &KeyboardBitsConfig {
        &self.cfg
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn normalise_key(key: &str) -> String {
    key.to_lowercase()
       .replace(' ', "space")
       .replace('/', "slash")
       .replace('\\', "backslash")
       .replace('.', "dot")
       .replace(',', "comma")
}

fn is_printable(key: &str) -> bool {
    key.len() == 1 && key.chars().all(|c| c.is_alphanumeric() || c.is_ascii_punctuation())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctrl_c_emits_combo_label() {
        let enc = KeyboardBitsEncoder::new(KeyboardBitsConfig::default());
        let out = enc.encode_shortcut("c", true, false, false);
        assert!(out.labels.iter().any(|l| l == "key:combo_ctrl_c"));
        assert!(out.labels.iter().any(|l| l == "key:mod_ctrl"));
    }

    #[test]
    fn typed_word_emits_txt_label() {
        let enc = KeyboardBitsEncoder::new(KeyboardBitsConfig::default());
        let events = vec![
            KeyEvent::simple("h", 0.0),
            KeyEvent::simple("i", 0.1),
        ];
        let out = enc.encode_sequence(&events);
        // "hi" should become txt:word_hi
        assert!(out.labels.iter().any(|l| l == "txt:word_hi"));
        assert_eq!(out.typed_text, "hi");
    }
}
