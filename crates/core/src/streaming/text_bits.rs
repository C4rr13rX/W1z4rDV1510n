//! Text Data-Bits Encoder
//!
//! Converts text — with all its layout metadata — into the same label format
//! used by `ImageBitsEncoder` and `AudioBitsEncoder`. The result feeds directly
//! into `NeuronPool::train_weighted()`.
//!
//! The key principle: **layout is data**. The same word "Introduction" means
//! something different when it appears in 24pt bold at the top of a zone versus
//! 11pt body text mid-paragraph. A human brain reading a page captures both;
//! this encoder does too.
//!
//! Label families emitted:
//!
//!  * `txt:word_{w}`          — the word itself (normalised, lowercased)
//!  * `txt:phon_{p}`          — phonetic character n-grams (bigrams by default)
//!                              approximate phoneme clusters without a dictionary
//!  * `txt:role_{r}`          — structural role: heading / subheading / body /
//!                              caption / list / label / code / footnote
//!  * `txt:size_{s}`          — visual weight: large / medium / small / tiny
//!  * `txt:emph_{e}`          — emphasis: bold / italic / bold_italic / none
//!  * `txt:indent_{n}`        — indentation level (0 = flush left)
//!  * `txt:zone_x{n}_y{n}`    — page-grid zone matching image_bits grid
//!                              so image and text labels share the same spatial vocab
//!  * `txt:seq_{n}`           — token sequence position bucket (coarse: 0..time_slots)
//!  * `txt:word_{w}_zone_x{n}_y{n}` — composite: word in spatial context
//!  * `txt:role_{r}_zone_x{n}_y{n}` — composite: structural role in spatial context
//!
//! When a PDF page is rendered as an image AND its text is extracted with
//! layout metadata, training both together causes the pool to learn:
//!   `img:z0_0 + img:edgeH + txt:role_heading + txt:word_introduction`
//! as a single co-activation. Motif discovery then lifts this pattern to a
//! reusable motif: "section header".

use serde::{Deserialize, Serialize};

// ── Layout metadata ───────────────────────────────────────────────────────────

/// The structural role of a span of text on the page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextRole {
    Heading,
    Subheading,
    Body,
    Caption,
    ListItem,
    Label,      // axis labels, table headers, key-value pairs
    Code,
    Footnote,
    Unknown,
}

impl TextRole {
    fn label_str(self) -> &'static str {
        match self {
            TextRole::Heading    => "heading",
            TextRole::Subheading => "subheading",
            TextRole::Body       => "body",
            TextRole::Caption    => "caption",
            TextRole::ListItem   => "list",
            TextRole::Label      => "label",
            TextRole::Code       => "code",
            TextRole::Footnote   => "footnote",
            TextRole::Unknown    => "unknown",
        }
    }
}

/// Visual weight derived from font size (relative to page body size).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextSize {
    Large,   // heading-sized (>= 1.4× body)
    Medium,  // slightly larger than body (1.1–1.4×)
    Small,   // body text (0.8–1.1×)
    Tiny,    // footnotes, captions, labels (< 0.8×)
}

impl TextSize {
    pub fn from_ratio(ratio: f32) -> Self {
        if ratio >= 1.4      { TextSize::Large }
        else if ratio >= 1.1 { TextSize::Medium }
        else if ratio >= 0.8 { TextSize::Small }
        else                  { TextSize::Tiny }
    }

    fn label_str(self) -> &'static str {
        match self {
            TextSize::Large  => "large",
            TextSize::Medium => "medium",
            TextSize::Small  => "small",
            TextSize::Tiny   => "tiny",
        }
    }
}

/// Typographic emphasis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextEmphasis {
    None,
    Bold,
    Italic,
    BoldItalic,
}

impl TextEmphasis {
    fn label_str(self) -> &'static str {
        match self {
            TextEmphasis::None       => "none",
            TextEmphasis::Bold       => "bold",
            TextEmphasis::Italic     => "italic",
            TextEmphasis::BoldItalic => "bold_italic",
        }
    }
}

/// A single span of text with its layout context.
///
/// When processing plain text files (no layout info), use `TextSpan::plain(text)`.
/// When processing PDFs or rich documents, fill all fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextSpan {
    /// The raw text content of this span.
    pub text: String,
    /// Structural role on the page.
    pub role: TextRole,
    /// Font size relative to the document's body text size (1.0 = body size).
    /// Set to 1.0 if unknown.
    pub size_ratio: f32,
    /// Typographic emphasis.
    pub emphasis: TextEmphasis,
    /// Indentation level (0 = flush left, 1 = one indent step, etc.).
    pub indent: usize,
    /// Horizontal position on page as a fraction [0, 1). Left = 0.
    pub x_frac: f32,
    /// Vertical position on page as a fraction [0, 1). Top = 0.
    pub y_frac: f32,
    /// Sequential index of this span within the page (for `txt:seq_{n}`).
    pub seq_index: usize,
    /// Total spans on this page (for normalising seq_index).
    pub seq_total: usize,
}

impl TextSpan {
    /// Construct a span with no layout metadata (plain text mode).
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

    /// Construct a span with explicit role and position (PDF extraction mode).
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
    /// Label prefix, e.g. "txt" → "txt:word_duck", "txt:role_heading".
    pub stream_tag: String,
    /// Grid divisions matching image_bits (so spatial labels are comparable).
    pub grid_x: usize,
    pub grid_y: usize,
    /// Number of coarse sequence-position buckets (for `txt:seq_{n}`).
    pub time_slots: usize,
    /// Emit phonetic n-gram labels (`txt:phon_{ng}`).
    pub emit_phonetic: bool,
    /// Length of character n-grams used for phonetic approximation.
    pub ngram_len: usize,
    /// Emit composite word+zone labels (`txt:word_{w}_zone_x{n}_y{n}`).
    pub emit_word_zone: bool,
    /// Minimum word length to emit a word label (filters stopwords by length).
    pub min_word_len: usize,
}

impl Default for TextBitsConfig {
    fn default() -> Self {
        Self {
            stream_tag:      "txt".to_string(),
            grid_x:          8,
            grid_y:          8,
            time_slots:      16,
            emit_phonetic:   true,
            ngram_len:       2,
            emit_word_zone:  true,
            min_word_len:    2,
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBitsOutput {
    /// Flat deduplicated list of labels for this page / document chunk.
    pub labels: Vec<String>,
    /// How many spans were processed.
    pub spans_processed: usize,
    /// How many word tokens were emitted (before dedup).
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

    /// Encode a slice of spans (one page, one paragraph, or a whole document)
    /// into a deduplicated label set.
    pub fn encode_spans(&self, spans: &[TextSpan]) -> TextBitsOutput {
        let tag = &self.cfg.stream_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut word_tokens = 0usize;

        for span in spans {
            // ── Spatial zone (matches image_bits grid vocabulary) ─────────────
            let zx = ((span.x_frac * self.cfg.grid_x as f32) as usize)
                .min(self.cfg.grid_x - 1);
            let zy = ((span.y_frac * self.cfg.grid_y as f32) as usize)
                .min(self.cfg.grid_y - 1);

            // ── Structural role ────────────────────────────────────────────────
            let role_str = span.role.label_str();
            labels.push(format!("{tag}:role_{role_str}"));
            labels.push(format!("{tag}:role_{role_str}_zone_x{zx}_y{zy}"));

            // ── Visual weight ──────────────────────────────────────────────────
            let size = TextSize::from_ratio(span.size_ratio);
            labels.push(format!("{tag}:size_{}", size.label_str()));

            // ── Emphasis ───────────────────────────────────────────────────────
            let emph_str = span.emphasis.label_str();
            if span.emphasis != TextEmphasis::None {
                labels.push(format!("{tag}:emph_{emph_str}"));
            }

            // ── Indentation ────────────────────────────────────────────────────
            if span.indent > 0 {
                labels.push(format!("{tag}:indent_{}", span.indent.min(8)));
            }

            // ── Sequence position ──────────────────────────────────────────────
            let seq_slot = if span.seq_total > 1 {
                ((span.seq_index as f32 / span.seq_total as f32)
                    * self.cfg.time_slots as f32) as usize
            } else {
                0
            };
            let seq_slot = seq_slot.min(self.cfg.time_slots - 1);
            labels.push(format!("{tag}:seq_{seq_slot}"));

            // ── Words ──────────────────────────────────────────────────────────
            for word in tokenize_words(&span.text) {
                if word.len() < self.cfg.min_word_len {
                    continue;
                }
                word_tokens += 1;

                labels.push(format!("{tag}:word_{word}"));

                if self.cfg.emit_word_zone {
                    labels.push(format!("{tag}:word_{word}_zone_x{zx}_y{zy}"));
                }

                // Role + word composite — "heading introduces the word X"
                labels.push(format!("{tag}:role_{role_str}_word_{word}"));

                // ── Phonetic n-grams ───────────────────────────────────────────
                if self.cfg.emit_phonetic {
                    for ng in char_ngrams(&word, self.cfg.ngram_len) {
                        labels.push(format!("{tag}:phon_{ng}"));
                    }
                }
            }
        }

        labels.sort_unstable();
        labels.dedup();

        TextBitsOutput {
            labels,
            spans_processed: spans.len(),
            word_tokens,
        }
    }

    /// Convenience: encode a plain string with no layout metadata.
    /// Use when you have raw text files, not rich documents.
    pub fn encode_plain(&self, text: &str) -> TextBitsOutput {
        let words: Vec<&str> = text.split_whitespace().collect();
        let total = words.len();
        let spans: Vec<TextSpan> = words
            .iter()
            .enumerate()
            .map(|(i, w)| {
                let mut s = TextSpan::plain(w);
                s.seq_index = i;
                s.seq_total = total;
                s
            })
            .collect();
        self.encode_spans(&spans)
    }

    /// Encode a whole page as a sequence of spans, auto-assigning sequence
    /// positions. Spans must be ordered top-to-bottom, left-to-right.
    pub fn encode_page(&self, spans: &[TextSpan]) -> TextBitsOutput {
        // Re-index seq positions relative to this page.
        let total = spans.len();
        let mut reindexed: Vec<TextSpan> = spans.to_vec();
        for (i, s) in reindexed.iter_mut().enumerate() {
            s.seq_index = i;
            s.seq_total = total;
        }
        self.encode_spans(&reindexed)
    }

    pub fn config(&self) -> &TextBitsConfig {
        &self.cfg
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Split text into lowercase word tokens, stripping punctuation.
fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Generate character n-grams of length `n` for a word.
/// "duck" with n=2 → ["du", "uc", "ck"]
fn char_ngrams(word: &str, n: usize) -> Vec<String> {
    if n == 0 || word.len() < n {
        return vec![word.to_string()];
    }
    let chars: Vec<char> = word.chars().collect();
    chars.windows(n).map(|w| w.iter().collect()).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_span_emits_role_and_word_labels() {
        let cfg = TextBitsConfig::default();
        let enc = TextBitsEncoder::new(cfg);
        let spans = vec![TextSpan::positioned(
            "Introduction to Biology",
            TextRole::Heading,
            1.8,
            TextEmphasis::Bold,
            0,
            0.1, 0.05,
            0, 1,
        )];
        let out = enc.encode_spans(&spans);
        assert!(out.labels.iter().any(|l| l == "txt:role_heading"));
        assert!(out.labels.iter().any(|l| l == "txt:size_large"));
        assert!(out.labels.iter().any(|l| l == "txt:emph_bold"));
        assert!(out.labels.iter().any(|l| l.contains("word_introduction")));
        assert!(out.labels.iter().any(|l| l.contains("word_biology")));
    }

    #[test]
    fn plain_text_encodes_without_layout() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let out = enc.encode_plain("The duck swam across the pond");
        assert!(out.labels.iter().any(|l| l.contains("word_duck")));
        assert!(out.labels.iter().any(|l| l.contains("word_pond")));
    }

    #[test]
    fn zone_matches_image_bits_grid() {
        // A span at x=0.9, y=0.9 (bottom-right) should land in zone 7,7 on default 8×8 grid.
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let span = TextSpan::positioned("corner", TextRole::Body, 1.0, TextEmphasis::None, 0, 0.9, 0.9, 0, 1);
        let out = enc.encode_spans(&[span]);
        assert!(out.labels.iter().any(|l| l.contains("zone_x7_y7")));
    }
}
