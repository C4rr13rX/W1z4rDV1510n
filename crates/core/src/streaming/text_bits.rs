//! Text Data-Bits Encoder — raw-codepoint only.
//!
//! Emits one neuron label per UTF-8 codepoint and nothing else.
//! Label format: `txt:<base64url-no-pad(utf8 bytes of the codepoint)>`.
//!
//! There are NO predefined category labels (no word_, char_, punct_, role_,
//! zone_, size_, emph_, indent_, seq_, phon_). Spaces, newlines, tabs,
//! punctuation, letters, digits — every codepoint is its own neuron with an
//! ID derived only from its raw bytes. Higher structures (morphemes, words,
//! phrases, sentences) emerge via Hebbian mini-column collapse in the neuron
//! object pool — they are not imposed by this encoder.
//!
//! The same principle applies to every other sensor stream: image_bits emits
//! pixel-level atoms, audio_bits emits sample-level atoms. Nothing in the
//! label after the source prefix is a human-assigned category.
//!
//! Layout metadata (`TextRole`, `TextSize`, `TextEmphasis`, positions) is
//! still accepted on the request surface so callers don't break, but it is
//! NOT used to generate labels. The raw text content is the only input that
//! shapes the fabric.

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};

// ── Layout metadata (retained as request-surface only — ignored in encoding) ──

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextRole {
    Heading,
    Subheading,
    Body,
    Caption,
    ListItem,
    Label,
    Code,
    Footnote,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSize {
    Large,
    Medium,
    Small,
    Tiny,
}

impl TextSize {
    pub fn from_ratio(ratio: f32) -> Self {
        if ratio >= 1.4      { TextSize::Large }
        else if ratio >= 1.1 { TextSize::Medium }
        else if ratio >= 0.8 { TextSize::Small }
        else                  { TextSize::Tiny }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextEmphasis {
    None,
    Bold,
    Italic,
    BoldItalic,
}

/// A single span of text. Layout fields are accepted but ignored by the
/// encoder — only `text` affects labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    pub text: String,
    pub role: TextRole,
    pub size_ratio: f32,
    pub emphasis: TextEmphasis,
    pub indent: usize,
    pub x_frac: f32,
    pub y_frac: f32,
    pub seq_index: usize,
    pub seq_total: usize,
}

impl TextSpan {
    pub fn plain(text: &str) -> Self {
        Self {
            text:      text.to_string(),
            role:      TextRole::Body,
            size_ratio: 1.0,
            emphasis:  TextEmphasis::None,
            indent:    0,
            x_frac:    0.0,
            y_frac:    0.0,
            seq_index: 0,
            seq_total: 1,
        }
    }

    pub fn positioned(
        text: &str,
        role: TextRole,
        size_ratio: f32,
        emphasis: TextEmphasis,
        indent: usize,
        x_frac: f32,
        y_frac: f32,
        seq_index: usize,
        seq_total: usize,
    ) -> Self {
        Self { text: text.to_string(), role, size_ratio, emphasis, indent, x_frac, y_frac, seq_index, seq_total }
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBitsConfig {
    /// Source-stream prefix applied to every label, e.g. `"txt"`.
    pub stream_tag: String,
}

impl Default for TextBitsConfig {
    fn default() -> Self {
        Self { stream_tag: "txt".to_string() }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBitsOutput {
    /// Deduplicated set of atom labels present in the input.
    pub labels: Vec<String>,
    /// One ordered sequence of atom labels per span, in text order, including
    /// whitespace and punctuation. Feed to `/media/train_sequence` so STDP
    /// builds directional edges through the character chain.
    pub char_sequences: Vec<Vec<String>>,
    pub spans_processed: usize,
    /// Count of non-whitespace atoms (retained for compatibility).
    pub word_tokens: usize,
}

// ── Encoder ───────────────────────────────────────────────────────────────────

pub struct TextBitsEncoder {
    cfg: TextBitsConfig,
}

impl TextBitsEncoder {
    pub fn new(cfg: TextBitsConfig) -> Self {
        Self { cfg }
    }

    /// Encode a slice of spans as raw codepoint atoms. Layout metadata on the
    /// span is ignored — only `span.text` contributes to the label set.
    pub fn encode_spans(&self, spans: &[TextSpan]) -> TextBitsOutput {
        let tag = &self.cfg.stream_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut char_sequences: Vec<Vec<String>> = Vec::new();
        let mut non_ws_atoms = 0usize;

        for span in spans {
            let mut seq: Vec<String> = Vec::with_capacity(span.text.chars().count());
            for ch in span.text.chars() {
                let label = char_label_with_tag(tag, ch);
                if !ch.is_whitespace() {
                    non_ws_atoms += 1;
                }
                seq.push(label.clone());
                labels.push(label);
            }
            if !seq.is_empty() {
                char_sequences.push(seq);
            }
        }

        labels.sort_unstable();
        labels.dedup();

        TextBitsOutput {
            labels,
            char_sequences,
            spans_processed: spans.len(),
            word_tokens: non_ws_atoms,
        }
    }

    pub fn encode_plain(&self, text: &str) -> TextBitsOutput {
        let span = TextSpan::plain(text);
        self.encode_spans(&[span])
    }

    pub fn encode_page(&self, spans: &[TextSpan]) -> TextBitsOutput {
        self.encode_spans(spans)
    }

    pub fn config(&self) -> &TextBitsConfig {
        &self.cfg
    }
}

// ── Atom helpers ──────────────────────────────────────────────────────────────

/// Build the atom label for a codepoint under a given source tag:
/// `{tag}:{base64url-no-pad(utf8 bytes of ch)}`.
pub fn char_label_with_tag(tag: &str, ch: char) -> String {
    let mut buf = [0u8; 4];
    let bytes = ch.encode_utf8(&mut buf).as_bytes();
    format!("{tag}:{}", URL_SAFE_NO_PAD.encode(bytes))
}

/// Default text atom label: `txt:<b64url(ch)>`.
pub fn char_label(ch: char) -> String {
    char_label_with_tag("txt", ch)
}

/// Inverse of [`char_label`] / [`char_label_with_tag`]. Returns the single
/// codepoint encoded in the payload portion of a `txt:<payload>` label, or
/// `None` if the label has no `txt:` prefix or the payload does not decode to
/// exactly one codepoint. (Concept neurons formed by mini-column collapse
/// have payloads that are concatenations of multiple atom payloads and will
/// return `None` here — decode them by walking their constituent atoms.)
pub fn label_to_char(label: &str) -> Option<char> {
    let payload = label.strip_prefix("txt:")?;
    let bytes = URL_SAFE_NO_PAD.decode(payload).ok()?;
    let s = std::str::from_utf8(&bytes).ok()?;
    let mut it = s.chars();
    let ch = it.next()?;
    if it.next().is_some() {
        return None;
    }
    Some(ch)
}

/// Decode an atom label to its single-codepoint string form. Returns `None`
/// for non-atom labels (concept-neuron labels whose ID is a concatenation of
/// atom payloads do NOT round-trip through base64 decode — their output is
/// assembled by walking their member atoms, not by decoding the composite
/// label).
pub fn label_to_string(label: &str) -> Option<String> {
    label_to_char(label).map(|c| c.to_string())
}

/// Legacy name retained so the decoder keeps compiling during the migration.
/// Always returns `None` in the raw-codepoint scheme because there are no
/// named punctuation labels any more.
pub fn name_to_char(_name: &str) -> Option<char> {
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_char_round_trips() {
        let label = char_label('y');
        assert!(label.starts_with("txt:"));
        assert_eq!(label_to_char(&label), Some('y'));
    }

    #[test]
    fn space_and_punctuation_are_atoms() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let out = enc.encode_plain("hi!");
        // One label per unique codepoint: h, i, !
        assert_eq!(out.labels.len(), 3);
        for l in &out.labels {
            assert!(l.starts_with("txt:"));
            assert!(label_to_char(l).is_some(), "{l} should decode to a single codepoint");
        }
    }

    #[test]
    fn char_sequence_preserves_order_and_whitespace() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let out = enc.encode_plain("a b");
        assert_eq!(out.char_sequences.len(), 1);
        let seq = &out.char_sequences[0];
        assert_eq!(seq.len(), 3);
        assert_eq!(label_to_char(&seq[0]), Some('a'));
        assert_eq!(label_to_char(&seq[1]), Some(' '));
        assert_eq!(label_to_char(&seq[2]), Some('b'));
    }

    #[test]
    fn no_predefined_category_labels() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let out = enc.encode_plain("Hello, world!");
        for l in &out.labels {
            assert!(!l.contains("word_"),  "forbidden prefix word_ in {l}");
            assert!(!l.contains("char_"),  "forbidden prefix char_ in {l}");
            assert!(!l.contains("punct_"), "forbidden prefix punct_ in {l}");
            assert!(!l.contains("role_"),  "forbidden prefix role_ in {l}");
            assert!(!l.contains("zone_"),  "forbidden prefix zone_ in {l}");
            assert!(!l.contains("size_"),  "forbidden prefix size_ in {l}");
            assert!(!l.contains("seq_"),   "forbidden prefix seq_ in {l}");
            assert!(!l.contains("phon_"),  "forbidden prefix phon_ in {l}");
        }
    }

    #[test]
    fn multibyte_codepoint_encodes_and_decodes() {
        let label = char_label('é'); // U+00E9 — two UTF-8 bytes
        assert_eq!(label_to_char(&label), Some('é'));
    }

    #[test]
    fn concept_label_ids_are_opaque_and_unique() {
        // Concept neurons formed by collapse concatenate member base64 payloads
        // raw (no delimiter). The result is a stable unique ID — it is NOT
        // itself a decodable base64 payload. Composite decoding happens via
        // member traversal inside the neuron pool, not by decoding this label.
        let y = char_label('y');
        let e = char_label('e');
        let s = char_label('s');
        let yes_concept = format!(
            "txt:{}{}{}",
            y.strip_prefix("txt:").unwrap(),
            e.strip_prefix("txt:").unwrap(),
            s.strip_prefix("txt:").unwrap(),
        );
        let no_concept = format!(
            "txt:{}{}",
            char_label('n').strip_prefix("txt:").unwrap(),
            char_label('o').strip_prefix("txt:").unwrap(),
        );
        assert_ne!(yes_concept, no_concept);
        // Composites do not decode via label_to_char — they have more than one codepoint.
        assert!(label_to_char(&yes_concept).is_none());
    }
}
