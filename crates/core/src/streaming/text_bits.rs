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
//! ## Bottom-up language architecture
//!
//! Language understanding is built from the ground up, not tokenized from the top down:
//!
//!   Letters  →  Morphemes  →  Words  →  Usage relationships  →  Motifs
//!
//! The encoder feeds this by emitting labels at every level simultaneously.
//! The neural pool's Hebbian co-occurrence, kWTA sparsification, and mini-column
//! promotion do the work of discovering morphemes and word families — the encoder
//! just makes sure all the raw signal is present.
//!
//! **Punctuation is signal, not noise.** "Let's eat, Grandma." and "Let's eat
//! Grandma." have different meanings because of a comma.  Stripping punctuation
//! destroys that signal.  Every punctuation mark is emitted as its own label so
//! the pool learns its syntactic and semantic role from co-occurrence with the
//! words around it.
//!
//! Label families emitted:
//!
//!  * `txt:word_{w}`              — the full word (lowercased, punctuation stripped)
//!  * `txt:char_{c}`              — individual character (when `emit_chars` = true)
//!  * `txt:char_{c}_pos{n}`       — character with position within word (0-indexed,
//!                                   capped at pos9 for rare long words)
//!  * `txt:phon_{p}`              — character bigrams: approximate morpheme seeds
//!  * `txt:punct_{name}`          — punctuation marks: comma, period, question,
//!                                   exclaim, apostrophe, hyphen, colon, semicolon,
//!                                   quote, paren_open, paren_close (when `emit_punct`)
//!  * `txt:role_{r}`              — structural role: heading / subheading / body /
//!                                   caption / list / label / code / footnote
//!  * `txt:size_{s}`              — visual weight: large / medium / small / tiny
//!  * `txt:emph_{e}`              — emphasis: bold / italic / bold_italic / none
//!  * `txt:indent_{n}`            — indentation level (0 = flush left)
//!  * `txt:zone_x{n}_y{n}`        — page-grid zone matching image_bits grid
//!  * `txt:seq_{n}`               — token sequence position bucket (coarse)
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
    /// Minimum word length to emit a word label.
    /// Set to 1 so single letters ("A", "I") are encoded — the pool learns
    /// their meaning from context, same as any other word.
    pub min_word_len: usize,
    /// Emit individual character labels (`txt:char_{c}` and `txt:char_{c}_pos{n}`).
    /// This is the foundation of bottom-up language learning: letters fire before
    /// morphemes emerge, morphemes before words, through Hebbian co-occurrence and
    /// mini-column promotion.  Without this, the pool never sees sub-word structure.
    pub emit_chars: bool,
    /// Emit punctuation as labels (`txt:punct_comma`, `txt:punct_period`, etc.).
    /// Punctuation carries meaning: "Let's eat, Grandma." ≠ "Let's eat Grandma."
    /// The comma is part of the signal and must be learned, not stripped.
    pub emit_punct: bool,
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
            min_word_len:    1,
            emit_chars:      true,
            emit_punct:      true,
        }
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextBitsOutput {
    /// Flat deduplicated list of labels for this page / document chunk.
    /// Includes word labels, character labels, punctuation labels, role/zone
    /// composites — everything the pool needs for co-occurrence learning.
    pub labels: Vec<String>,
    /// Per-word character sequences, in text order.
    /// Each inner Vec is the ordered character label sequence for one word,
    /// e.g. ["txt:char_a", "txt:char_p", "txt:char_p", "txt:char_l", "txt:char_e"]
    /// for "apple".  Feed these to `/media/train_sequence` so STDP builds
    /// directional (pre→post) connections through each word's character chain.
    /// The pool then discovers morphemes (recurring sub-sequences) through
    /// mini-column promotion without any hand-coded rules.
    pub char_sequences: Vec<Vec<String>>,
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
    /// into a deduplicated label set plus per-word character sequences.
    pub fn encode_spans(&self, spans: &[TextSpan]) -> TextBitsOutput {
        let tag = &self.cfg.stream_tag;
        let mut labels: Vec<String> = Vec::new();
        let mut char_sequences: Vec<Vec<String>> = Vec::new();
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

            // ── Tokens (words + punctuation) ───────────────────────────────────
            for token in tokenize_text(&span.text) {
                match token {
                    Token::Word(word) => {
                        if word.len() < self.cfg.min_word_len {
                            continue;
                        }
                        word_tokens += 1;

                        // Word-level labels
                        labels.push(format!("{tag}:word_{word}"));
                        if self.cfg.emit_word_zone {
                            labels.push(format!("{tag}:word_{word}_zone_x{zx}_y{zy}"));
                        }
                        labels.push(format!("{tag}:role_{role_str}_word_{word}"));

                        // Phonetic bigrams — morpheme seeds
                        if self.cfg.emit_phonetic {
                            for ng in char_ngrams(&word, self.cfg.ngram_len) {
                                labels.push(format!("{tag}:phon_{ng}"));
                            }
                        }

                        // Character-level labels — the foundation layer.
                        // txt:char_{c}         — letter fires in context of word
                        // txt:char_{c}_pos{n}  — positional: 'a' at position 0 of
                        //                        "apple" ≠ 'a' at position 4 of "sofa"
                        //                        (capped at pos9 for long words)
                        if self.cfg.emit_chars {
                            let mut seq: Vec<String> = Vec::with_capacity(word.len());
                            for (pos, ch) in word.chars().enumerate() {
                                let cl = format!("{tag}:char_{ch}");
                                let pl = format!("{tag}:char_{ch}_pos{}", pos.min(9));
                                labels.push(cl.clone());
                                labels.push(pl);
                                seq.push(cl);
                            }
                            if !seq.is_empty() {
                                char_sequences.push(seq);
                            }
                        }
                    }
                    Token::Punct(name) => {
                        if self.cfg.emit_punct {
                            labels.push(format!("{tag}:punct_{name}"));
                        }
                    }
                }
            }
        }

        labels.sort_unstable();
        labels.dedup();

        TextBitsOutput {
            labels,
            char_sequences,
            spans_processed: spans.len(),
            word_tokens,
        }
    }

    /// Convenience: encode a plain string with no layout metadata.
    /// Use when you have raw text files, not rich documents.
    pub fn encode_plain(&self, text: &str) -> TextBitsOutput {
        // Treat the whole text as a single body span — layout unknown.
        // Punctuation and characters are still encoded correctly.
        let span = TextSpan::plain(text);
        self.encode_spans(&[span])
    }

    /// Encode a whole page as a sequence of spans, auto-assigning sequence
    /// positions. Spans must be ordered top-to-bottom, left-to-right.
    pub fn encode_page(&self, spans: &[TextSpan]) -> TextBitsOutput {
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

/// A single item from the text stream — either a word or a punctuation mark.
enum Token {
    Word(String),
    Punct(String),  // punctuation name, e.g. "comma", "period", "apostrophe"
}

/// Split text into an ordered sequence of word and punctuation tokens.
///
/// Words are lowercased alphabetic/numeric runs.  Every punctuation character
/// between words is emitted as a named `Token::Punct` rather than silently
/// dropped.  This lets the pool learn "comma follows noun in address form"
/// vs "comma follows verb in list form" through co-occurrence alone.
///
/// Examples:
///   "Let's eat, Grandma."
///     → [Word("let"), Punct("apostrophe"), Word("s"), Word("eat"),
///        Punct("comma"), Word("grandma"), Punct("period")]
///
///   "pH = 7.4"
///     → [Word("ph"), Punct("equals"), Word("7"), Punct("period"), Word("4")]
fn tokenize_text(text: &str) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut word_buf = String::new();

    let flush_word = |buf: &mut String, out: &mut Vec<Token>| {
        if !buf.is_empty() {
            out.push(Token::Word(buf.clone()));
            buf.clear();
        }
    };

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            word_buf.push(ch.to_lowercase().next().unwrap_or(ch));
        } else {
            flush_word(&mut word_buf, &mut tokens);
            if let Some(name) = punct_name(ch) {
                tokens.push(Token::Punct(name.to_string()));
            }
            // Whitespace and other chars are silently consumed (they're just
            // separators; their absence vs presence isn't meaningful signal).
        }
    }
    flush_word(&mut word_buf, &mut tokens);
    tokens
}

/// Map a character to its label name for non-alphanumeric signal.
/// Spaces and tabs are neurons — word boundaries are meaningful structure,
/// not mere separators. Every character fires.
fn punct_name(ch: char) -> Option<&'static str> {
    match ch {
        ' ' | '\t' => Some("space"),
        ','  => Some("comma"),
        '.'  => Some("period"),
        '?'  => Some("question"),
        '!'  => Some("exclaim"),
        '\'' | '\u{2019}' | '\u{0060}' => Some("apostrophe"),  // ' ' `
        '-'  | '\u{2013}' | '\u{2014}' => Some("hyphen"),       // - – —
        ':'  => Some("colon"),
        ';'  => Some("semicolon"),
        '"'  | '\u{201C}' | '\u{201D}' => Some("quote"),        // " " "
        '('  | '['  | '{' => Some("paren_open"),
        ')'  | ']'  | '}' => Some("paren_close"),
        '/'  => Some("slash"),
        '@'  => Some("at"),
        '#'  => Some("hash"),
        '+'  => Some("plus"),
        '='  => Some("equals"),
        '<'  => Some("lt"),
        '>'  => Some("gt"),
        '*'  => Some("star"),
        '&'  => Some("amp"),
        '%'  => Some("percent"),
        '$'  => Some("dollar"),
        _    => None,
    }
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
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
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
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let span = TextSpan::positioned("corner", TextRole::Body, 1.0, TextEmphasis::None, 0, 0.9, 0.9, 0, 1);
        let out = enc.encode_spans(&[span]);
        assert!(out.labels.iter().any(|l| l.contains("zone_x7_y7")));
    }

    #[test]
    fn punctuation_is_preserved_not_stripped() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        // The comma is the difference between cannibalism and a dinner invitation.
        let out = enc.encode_plain("Let's eat, Grandma.");
        assert!(out.labels.iter().any(|l| l == "txt:punct_apostrophe"),
            "apostrophe in Let's must be a label, not stripped");
        assert!(out.labels.iter().any(|l| l == "txt:punct_comma"),
            "comma before Grandma must be a label, not stripped");
        assert!(out.labels.iter().any(|l| l == "txt:punct_period"),
            "period at end must be a label, not stripped");
        // Words themselves still present
        assert!(out.labels.iter().any(|l| l.contains("word_eat")));
        assert!(out.labels.iter().any(|l| l.contains("word_grandma")));
    }

    #[test]
    fn character_labels_emitted_for_words() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        let out = enc.encode_plain("cat");
        assert!(out.labels.iter().any(|l| l == "txt:char_c"), "char_c missing");
        assert!(out.labels.iter().any(|l| l == "txt:char_a"), "char_a missing");
        assert!(out.labels.iter().any(|l| l == "txt:char_t"), "char_t missing");
        assert!(out.labels.iter().any(|l| l == "txt:char_c_pos0"), "positional char missing");
        // Character sequence should also be present
        assert!(!out.char_sequences.is_empty(), "char_sequences must not be empty");
        assert_eq!(out.char_sequences[0], vec!["txt:char_c", "txt:char_a", "txt:char_t"]);
    }

    #[test]
    fn morpheme_roots_emerge_through_bigrams() {
        let enc = TextBitsEncoder::new(TextBitsConfig::default());
        // "transport" and "import" both contain bigram "po" and "or" and "rt"
        // from the "port" root — the pool will Hebbian-connect them through
        // shared phon_ labels without any hand-coded rule.
        let out1 = enc.encode_plain("transport");
        let out2 = enc.encode_plain("import");
        let bigrams1: Vec<&str> = out1.labels.iter()
            .filter(|l| l.starts_with("txt:phon_"))
            .map(|l| l.as_str()).collect();
        let bigrams2: Vec<&str> = out2.labels.iter()
            .filter(|l| l.starts_with("txt:phon_"))
            .map(|l| l.as_str()).collect();
        // Both share "po", "or", "rt" from the "port" subsequence
        assert!(bigrams1.contains(&"txt:phon_po"), "transport should have phon_po");
        assert!(bigrams2.contains(&"txt:phon_po"), "import should have phon_po");
    }

    #[test]
    fn no_punct_mode_omits_punct_labels() {
        let mut cfg = TextBitsConfig::default();
        cfg.emit_punct = false;
        let enc = TextBitsEncoder::new(cfg);
        let out = enc.encode_plain("Hello, world!");
        assert!(!out.labels.iter().any(|l| l.starts_with("txt:punct_")),
            "punct labels must be absent when emit_punct=false");
    }
}
