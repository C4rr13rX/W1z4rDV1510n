//! Q&A neural fabric runtime.
//!
//! Implements the core loop described by the user:
//!   "Send the input network a question → the output network gives a response.
//!    It's accurate and fast because we are causing a reaction in a **state**
//!    rather than making computations. The computations exist in the architecture."
//!
//! Architecture
//! ────────────
//! • Every unique question token maps to an **input neuron** (excitatory,
//!   in the question pool).
//! • Every unique answer string is stored as an **answer entry** with a sparse
//!   Hebbian weight vector over question neurons.
//! • **Ingestion**: tokenize question + answer → strengthen weights between all
//!   question-token neurons and the answer neuron using Hebb's rule
//!   (Δw ∝ pre × post × lr).
//! • **Query**: tokenize question → compute activation for every answer entry
//!   (dot product of active question neurons against each entry's weight row)
//!   → return top-k matches sorted by activation.
//!
//! The result is a CPU-RAM associative memory: after learning, matching a
//! question fires connections that were literally grown by Hebbian co-activation
//! during ingestion.  No matrix multiplication at inference — just sparse dot
//! products over the answer rows that actually have non-zero weight overlap.
//!
//! Neurogenesis hook
//! ─────────────────
//! When a new question token is seen for the first time, a new input neuron is
//! born (neurogenesis).  When a new answer text arrives, a new answer neuron is
//! born.  Neurons are never destroyed; they decay toward zero but remain
//! available if the same pattern returns, which is how the fabric keeps
//! previously learned associations alive across time.

use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Configuration ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct QaRuntimeConfig {
    /// Hebbian learning rate — how much each ingested Q→A pair strengthens the
    /// connection between question-token neurons and the answer neuron.
    pub hebbian_lr: f32,
    /// EMA decay applied to all weights each time `decay_weights` is called
    /// (typically once per major time step).  Keeps older patterns from
    /// dominating indefinitely while still preserving long-term memory.
    pub weight_decay: f32,
    /// Asymptotic ceiling for Hebbian weights.  Weights grow toward this
    /// value but never exceed it.  Higher ceiling preserves relative strength
    /// between lightly- and heavily-trained pairs — no hard saturation at 1.0.
    pub weight_ceiling: f32,
    /// Maximum number of question-vocabulary neurons (stops unbounded growth).
    pub max_question_neurons: usize,
    /// Maximum number of stored answer entries.
    pub max_answers: usize,
    /// Minimum activation to include an answer in query results.
    pub min_activation: f32,
    /// Number of top answers to return from a query.
    pub top_k: usize,
    /// If true the runtime is wired into the streaming pipeline.
    pub enabled: bool,
}

impl Default for QaRuntimeConfig {
    fn default() -> Self {
        Self {
            hebbian_lr: 0.05,
            weight_decay: 0.999,
            weight_ceiling: 5.0,
            max_question_neurons: 65_536,
            max_answers: 131_072,
            // IDF-weighted activation floor.  With IDF reweighting, a single
            // discriminative token match (weight=0.05, IDF≈3.7) produces ~0.18
            // activation.  Background noise from common tokens like "is"+"a"
            // (combined ≈0.09 at Stage 0 scale) is well below this threshold.
            // 0.10 provides a clean boundary that holds as the corpus grows
            // (common-token IDF asymptotes as corpus expands; specific-token IDF
            // grows logarithmically with N, widening the gap further over time).
            min_activation: 0.10,
            top_k: 5,
            enabled: true,
        }
    }
}

// ─── Core types ────────────────────────────────────────────────────────────

/// A single Q&A pair stored in the fabric.  Carries provenance so the labeling
/// pipeline can route it for human verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaPair {
    /// Stable deterministic ID derived from the question + answer text.
    pub qa_id: String,
    pub question: String,
    pub answer: String,
    /// Source book identifier (e.g. `college_physics_2e`).
    #[serde(default)]
    pub book_id: String,
    /// 1-based page index in the source book.
    #[serde(default)]
    pub page_index: usize,
    /// Confidence of the original extraction (0–1).
    pub confidence: f32,
    /// When this pair was ingested into the fabric.
    pub ingested_at: Timestamp,
    /// How many times this pair has been reinforced by re-ingestion.
    pub reinforcement_count: u32,
}

/// An answer entry stored in the fabric.  The `weights` map associates
/// question-token neuron IDs with learned Hebbian weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnswerEntry {
    answer_id: u32,
    text: String,
    /// Sparse weight vector: question_neuron_id → Hebbian weight.
    weights: HashMap<u32, f32>,
    /// Total activation across all reinforcements (for ranking ties).
    cumulative_activation: f64,
    /// Highest source confidence seen across all pairs that map to this answer.
    /// Used to boost high-confidence pairs (e.g. toddler 0.95) over low-confidence
    /// textbook extractions (0.72) when raw activation is otherwise equal.
    #[serde(default = "default_confidence")]
    max_source_confidence: f32,
}

fn default_confidence() -> f32 { 0.5 }

/// A single query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaQueryResult {
    pub answer: String,
    /// Raw Hebbian activation score for this result.
    pub activation: f32,
    /// Normalized confidence in [0, 1].
    pub confidence: f32,
    pub qa_id: String,
    pub book_id: String,
    pub page_index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaIngestReport {
    pub qa_id: String,
    pub question_tokens: usize,
    pub new_question_neurons: usize,
    pub reinforced: bool,
    pub total_pairs: u64,
    pub total_question_neurons: usize,
    pub total_answers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaQueryReport {
    pub question: String,
    pub active_question_neurons: usize,
    pub results: Vec<QaQueryResult>,
    pub timestamp: Timestamp,
    /// Tokens from this query whose IDF exceeds the median IDF of all active
    /// tokens — i.e., the specific/rare words the network considers
    /// discriminative.  Computed entirely from learned training statistics;
    /// no hardcoded stop-word lists.
    #[serde(default)]
    pub significant_tokens: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaRuntimeReport {
    pub timestamp: Timestamp,
    pub pairs_ingested: u64,
    pub question_neurons: usize,
    pub answer_entries: usize,
}

// ─── Runtime ───────────────────────────────────────────────────────────────

/// The Q&A neural fabric runtime.
///
/// State lives entirely in CPU RAM.  No locks needed for single-threaded
/// streaming — the fabric is owned by `StreamingInference` which processes
/// one batch at a time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaRuntime {
    #[serde(skip)]
    config: QaRuntimeConfig,

    /// question token string → input neuron ID
    question_vocab: HashMap<String, u32>,
    next_q_id: u32,

    /// answer text hash → index into `answers`
    answer_index: HashMap<String, usize>,
    /// All stored answer entries, indexed by their position.
    answers: Vec<AnswerEntry>,

    /// Full Q&A pair metadata, keyed by qa_id.
    pairs: HashMap<String, QaPair>,

    /// Count of pairs ingested.
    pairs_ingested: u64,

    /// Per-neuron document frequency: how many distinct Q&A pairs each question
    /// neuron has appeared in.  Used to compute IDF at query time so that tokens
    /// present in nearly every question (e.g. "what", "is", "?") contribute
    /// near-zero activation while rare, specific tokens dominate.
    /// No stop-word lists needed — the architecture handles discrimination.
    #[serde(default)]
    neuron_doc_freq: HashMap<u32, u64>,
}

impl QaRuntime {
    pub fn new(config: QaRuntimeConfig) -> Self {
        Self {
            config,
            question_vocab: HashMap::new(),
            next_q_id: 0,
            answer_index: HashMap::new(),
            answers: Vec::new(),
            pairs: HashMap::new(),
            pairs_ingested: 0,
            neuron_doc_freq: HashMap::new(),
        }
    }

    // ── Ingestion ───────────────────────────────────────────────────────────

    /// Ingest a single Q&A pair into the Hebbian fabric.
    ///
    /// 1. Tokenize question → get/create question-token neurons (neurogenesis).
    /// 2. Look up the answer; create a new answer entry if novel.
    /// 3. Apply Hebbian update: for every active question neuron → answer neuron:
    ///       weight += hebbian_lr   (clipped to [0, 1])
    pub fn ingest(
        &mut self,
        question: &str,
        answer: &str,
        book_id: &str,
        page_index: usize,
        confidence: f32,
        timestamp: Timestamp,
    ) -> QaIngestReport {
        let qa_id = compute_payload_hash(
            format!("qa|{}|{}", question, answer).as_bytes()
        );

        // ── Question tokenization + neurogenesis ─────────────────────────
        let tokens = tokenize(question);
        let mut new_q_neurons = 0usize;
        let mut active_q_ids: Vec<u32> = Vec::with_capacity(tokens.len());

        for token in &tokens {
            if self.question_vocab.len() >= self.config.max_question_neurons {
                break;
            }
            let next = self.next_q_id;
            let id = *self.question_vocab.entry(token.clone()).or_insert_with(|| {
                let id = next;
                new_q_neurons += 1;
                id
            });
            if new_q_neurons > 0 && id == next {
                self.next_q_id = next.saturating_add(1);
            }
            if !active_q_ids.contains(&id) {
                active_q_ids.push(id);
            }
        }

        // ── IDF tracking: increment document frequency for each unique token ─
        // This runs unconditionally so that tokens appearing in every Q&A pair
        // (articles, question words, punctuation) accumulate a high doc_freq and
        // thus a near-zero IDF weight at query time — no stop-word list needed.
        for &q_id in &active_q_ids {
            *self.neuron_doc_freq.entry(q_id).or_insert(0) += 1;
        }

        // ── Answer entry lookup / creation ───────────────────────────────
        let answer_key = compute_payload_hash(answer.as_bytes());
        let reinforced;

        if let Some(&entry_idx) = self.answer_index.get(&answer_key) {
            // Answer already exists — reinforce its weights.
            reinforced = true;
            let lr  = self.config.hebbian_lr;
            let max = self.config.weight_ceiling;
            let entry = &mut self.answers[entry_idx];
            for &q_id in &active_q_ids {
                let w = entry.weights.entry(q_id).or_insert(0.0);
                *w = (*w + lr * (1.0 - *w / max)).min(max);
            }
            // Keep the highest source confidence seen for this answer so the
            // scoring pass can boost high-quality pairs over noisy extractions.
            if confidence > entry.max_source_confidence {
                entry.max_source_confidence = confidence;
            }
            if let Some(pair) = self.pairs.get_mut(&qa_id) {
                pair.reinforcement_count += 1;
            }
        } else {
            // New answer — neurogenesis.
            reinforced = false;
            if self.answers.len() < self.config.max_answers {
                let answer_id = self.answers.len() as u32;
                let mut entry = AnswerEntry {
                    answer_id,
                    text: answer.to_string(),
                    weights: HashMap::new(),
                    cumulative_activation: 0.0,
                    max_source_confidence: confidence,
                };
                self.hebbian_update_entry(&mut entry, &active_q_ids);
                self.answer_index.insert(answer_key, self.answers.len());
                self.answers.push(entry);

                self.pairs.insert(qa_id.clone(), QaPair {
                    qa_id: qa_id.clone(),
                    question: question.to_string(),
                    answer: answer.to_string(),
                    book_id: book_id.to_string(),
                    page_index,
                    confidence: confidence.clamp(0.0, 1.0),
                    ingested_at: timestamp,
                    reinforcement_count: 0,
                });
            }
        }

        self.pairs_ingested += 1;

        let q_neurons_total = self.question_vocab.len();
        let answer_total = self.answers.len();

        QaIngestReport {
            qa_id,
            question_tokens: tokens.len(),
            new_question_neurons: new_q_neurons,
            reinforced,
            total_pairs: self.pairs_ingested,
            total_question_neurons: q_neurons_total,
            total_answers: answer_total,
        }
    }

    /// Bulk-ingest a slice of Q&A candidate records (as produced by
    /// `prepare_textbook_qa_dataset.py` or the textbook segmentation pipeline).
    pub fn ingest_candidates(&mut self, candidates: &[QaCandidateRecord], timestamp: Timestamp) {
        for rec in candidates {
            if rec.confidence < 0.1 {
                continue; // skip very low-confidence extractions
            }
            self.ingest(
                &rec.question,
                &rec.answer,
                &rec.book_id,
                rec.page_index,
                rec.confidence as f32,
                timestamp,
            );
        }
    }

    // ── Query ───────────────────────────────────────────────────────────────

    /// Query the fabric with a natural-language question.
    ///
    /// 1. Tokenize the question → map each known token to its neuron ID.
    /// 2. For each stored answer entry: activation = Σ weight[q_neuron] for
    ///    each active question neuron.  Unknown tokens contribute 0.
    /// 3. Normalize and return top-k results above `min_activation`.
    pub fn query(&mut self, question: &str, timestamp: Timestamp) -> QaQueryReport {
        let tokens = tokenize(question);

        // Resolve token → neuron ID for known tokens only.
        // Fallback: if the stemmed form isn't in vocab (vocab was built before
        // stemming was added), try the unstemmed plural form so existing stored
        // data stays usable without a full re-ingest.
        let active_q_ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| {
                self.question_vocab.get(t).copied().or_else(|| {
                    if !t.ends_with('s') {
                        self.question_vocab.get(&format!("{}s", t)).copied()
                    } else {
                        None
                    }
                })
            })
            .collect();

        if active_q_ids.is_empty() || self.answers.is_empty() {
            return QaQueryReport {
                question: question.to_string(),
                active_question_neurons: 0,
                results: vec![],
                significant_tokens: vec![],
                timestamp,
            };
        }

        // ── IDF weights for active tokens ────────────────────────────────
        // Compute per-token IDF: ln((N+1) / (df+1)).
        // Tokens present in every Q&A pair (articles, question words, punctuation)
        // have df ≈ N → IDF ≈ 0, contributing near-zero activation.
        // Rare, specific tokens have df ≪ N → IDF is large, dominating the score.
        // No stop-word lists required — the Hebbian statistics handle discrimination.
        let n = self.pairs_ingested.max(1) as f32;
        let idf_weights: Vec<f32> = active_q_ids.iter().map(|&q_id| {
            let df = self.neuron_doc_freq.get(&q_id).copied().unwrap_or(1) as f32;
            ((n + 1.0) / (df + 1.0)).ln().max(0.0)
        }).collect();

        // ── Compute activation for every answer entry ────────────────────
        // Score = IDF-weighted Hebbian activation × source confidence.
        // IDF weighting ensures common tokens (ubiquitous in all questions) add
        // near-zero signal while rare, specific tokens (appear in few questions)
        // dominate and point to the right answer.
        let mut scored: Vec<(f32, usize)> = self
            .answers
            .iter_mut()
            .enumerate()
            .map(|(idx, entry)| {
                let mut act = 0.0f32;
                for (i, &q_id) in active_q_ids.iter().enumerate() {
                    if let Some(&w) = entry.weights.get(&q_id) {
                        act += w * idf_weights[i];
                    }
                }
                entry.cumulative_activation += act as f64;
                let weighted = act * entry.max_source_confidence;
                (weighted, idx)
            })
            .filter(|(act, _)| *act >= self.config.min_activation)
            .collect();

        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.config.top_k);

        // ── Normalize scores to [0, 1] ───────────────────────────────────
        // The confidence-weighted scoring above already separates high-quality
        // pairs (toddler: 0.95) from low-quality textbook extractions (0.72).
        // Relative normalization keeps the top answer at 1.0; the synthesis
        // relative-gap filter (see synthesize_response) then drops secondary
        // results that don't score within 80% of the winner.
        let max_act = scored.first().map(|(a, _)| *a).unwrap_or(1.0).max(1e-6);

        let results: Vec<QaQueryResult> = scored
            .into_iter()
            .filter_map(|(act, idx)| {
                let entry = &self.answers[idx];
                // Pair metadata (book_id, page_index) from the pairs map.
                // We search by answer text hash to find one representative pair.
                let answer_hash = compute_payload_hash(entry.text.as_bytes());
                let pair = self.pairs.values().find(|p| {
                    compute_payload_hash(p.answer.as_bytes()) == answer_hash
                });
                Some(QaQueryResult {
                    answer: entry.text.clone(),
                    activation: act,
                    confidence: (act / max_act).clamp(0.0, 1.0),
                    qa_id: pair.map(|p| p.qa_id.clone()).unwrap_or_default(),
                    book_id: pair.map(|p| p.book_id.clone()).unwrap_or_default(),
                    page_index: pair.map(|p| p.page_index).unwrap_or(0),
                })
            })
            .collect();

        // ── Significant tokens: learned discrimination, no hardcoded stops ───
        // Tokens whose IDF exceeds the median IDF of all active tokens are the
        // network's own assessment of what is specific/rare vs. ubiquitous noise.
        // These are the tokens that carry the most discriminative power for this
        // query — derived purely from training statistics.
        let mut sorted_idf = idf_weights.clone();
        sorted_idf.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_idf = if sorted_idf.is_empty() {
            0.0f32
        } else {
            sorted_idf[sorted_idf.len() / 2]
        };
        // Collect the vocab token strings for discriminative neurons
        let id_to_token: std::collections::HashMap<u32, &str> = self.question_vocab
            .iter().map(|(tok, &id)| (id, tok.as_str())).collect();
        let significant_tokens: Vec<String> = active_q_ids.iter()
            .zip(idf_weights.iter())
            .filter(|(_, idf)| **idf > median_idf && **idf > 0.5)
            .filter_map(|(&q_id, _)| id_to_token.get(&q_id).map(|t| t.to_string()))
            .collect();

        QaQueryReport {
            question: question.to_string(),
            active_question_neurons: active_q_ids.len(),
            results,
            significant_tokens,
            timestamp,
        }
    }

    // ── Maintenance ─────────────────────────────────────────────────────────

    /// Apply weight decay across all answer entries.  Call this periodically
    /// (e.g. once per hour of wall time) to let older memories fade unless
    /// continuously reinforced — the same EMA mechanism as the Rust NeuronPool.
    pub fn decay_weights(&mut self) {
        let decay = self.config.weight_decay;
        for entry in &mut self.answers {
            entry.weights.retain(|_, w| {
                *w *= decay;
                *w > 1e-6 // prune effectively-zero weights
            });
        }
    }

    /// Runtime status report.
    pub fn report(&self, timestamp: Timestamp) -> QaRuntimeReport {
        QaRuntimeReport {
            timestamp,
            pairs_ingested: self.pairs_ingested,
            question_neurons: self.question_vocab.len(),
            answer_entries: self.answers.len(),
        }
    }

    pub fn pairs_ingested(&self) -> u64 {
        self.pairs_ingested
    }

    pub fn question_neuron_count(&self) -> usize {
        self.question_vocab.len()
    }

    pub fn answer_count(&self) -> usize {
        self.answers.len()
    }

    // ── Persistence ─────────────────────────────────────────────────────────

    /// Save the QA store to disk (atomic write via .tmp file).
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let tmp = path.with_extension("json.tmp");
        let f = std::fs::File::create(&tmp)?;
        let mut w = std::io::BufWriter::new(f);
        serde_json::to_writer(&mut w, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        w.flush()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load from disk, restoring the config that was skipped during serde.
    pub fn load(path: &std::path::Path, config: QaRuntimeConfig) -> std::io::Result<Self> {
        let f = std::fs::File::open(path)?;
        let mut rt: QaRuntime = serde_json::from_reader(std::io::BufReader::new(f))
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        rt.config = config;
        Ok(rt)
    }

    // ── Internal ────────────────────────────────────────────────────────────

    fn hebbian_update_entry(&self, entry: &mut AnswerEntry, active_q_ids: &[u32]) {
        let lr  = self.config.hebbian_lr;
        let max = self.config.weight_ceiling;
        for &q_id in active_q_ids {
            let w = entry.weights.entry(q_id).or_insert(0.0);
            // Asymptotic growth: fast at first, slows as weight approaches
            // ceiling.  Preserves relative strength between lightly- and
            // heavily-trained pairs even after thousands of reinforcements —
            // no hard saturation at 1.0 that erases cumulative evidence.
            *w = (*w + lr * (1.0 - *w / max)).min(max);
        }
    }
}

impl Default for QaRuntime {
    fn default() -> Self {
        Self::new(QaRuntimeConfig::default())
    }
}

// ─── Helper record (mirrors qa_candidates.jsonl schema) ────────────────────

/// Mirrors the schema emitted by `scripts/prepare_textbook_qa_dataset.py`.
/// Used for bulk ingestion from JSONL files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaCandidateRecord {
    pub qa_id: String,
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub book_id: String,
    #[serde(default)]
    pub page_index: usize,
    pub confidence: f64,
    #[serde(default)]
    pub evidence: String,
    #[serde(default)]
    pub review_status: String,
}

// ─── Tokenizer ─────────────────────────────────────────────────────────────

/// Question tokenizer for Hebbian Q&A lookup.
///
/// Lowercases and splits text into word tokens and punctuation tokens.
/// Punctuation is preserved as named tokens (e.g. `punct_comma`) because it
/// carries meaning: "When should I not eat?" ≠ "When should I eat?" — the
/// word "not" changes the answer, and "Let's eat, Grandma." ≠ "Let's eat
/// Grandma." — the comma changes the entire meaning.
///
/// Tokenization policy:
///   Every token in the input is preserved — words, punctuation, articles,
///   question words, everything.  Discrimination is handled architecturally
///   by IDF-weighted query activation (common tokens that appear in nearly every
///   Q&A pair contribute near-zero weight; rare, specific tokens dominate).
///   This mirrors biological perception: a reading human processes all symbols
///   and the network learns what is discriminative through co-occurrence statistics,
///   not through hard-coded exclusion lists.
///
/// Stemming:
///   Basic suffix stripping is preserved for now so existing vocabulary built
///   from non-character-level training still matches.  Once the pool has been
///   trained at character level, morphology emerges from Hebbian co-occurrence
///   and the stemming can be removed entirely.
fn tokenize(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut word_buf = String::new();

    let flush = |buf: &mut String, out: &mut Vec<String>| {
        if !buf.is_empty() {
            // Basic plural stemming: "trees"→"tree", "animals"→"animal".
            // Temporary until character-level training makes morphology emerge.
            let stemmed = if buf.len() > 3
                && buf.ends_with('s')
                && !buf.ends_with("ss")
            {
                buf[..buf.len() - 1].to_string()
            } else {
                buf.clone()
            };
            if !stemmed.is_empty() {
                out.push(stemmed);
            }
            buf.clear();
        }
    };

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            word_buf.push(ch.to_lowercase().next().unwrap_or(ch));
        } else {
            flush(&mut word_buf, &mut tokens);
            // Punctuation as named tokens — they are part of the query signal.
            if let Some(name) = punct_token(ch) {
                tokens.push(format!("punct_{name}"));
            }
        }
    }
    flush(&mut word_buf, &mut tokens);
    tokens
}

/// Map punctuation characters to token names for the QA tokenizer.
fn punct_token(ch: char) -> Option<&'static str> {
    match ch {
        ','  => Some("comma"),
        '.'  => Some("period"),
        '?'  => Some("question"),
        '!'  => Some("exclaim"),
        '\'' | '\u{2019}' | '\u{0060}' => Some("apostrophe"),
        '-'  | '\u{2013}' | '\u{2014}' => Some("hyphen"),
        ':'  => Some("colon"),
        ';'  => Some("semicolon"),
        _    => None,
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(unix: u64) -> Timestamp {
        Timestamp { unix: unix as i64 }
    }

    #[test]
    fn ingest_and_query_returns_correct_answer() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        let rep = rt.ingest(
            "What is photosynthesis?",
            "Photosynthesis is the process by which plants convert light into energy.",
            "biology_2e",
            42,
            0.9,
            ts(1),
        );
        assert!(rep.question_tokens >= 1); // includes "photosynthesis" plus common tokens; IDF will down-weight common ones
        assert!(!rep.reinforced);
        assert_eq!(rep.total_pairs, 1);

        let report = rt.query("What is photosynthesis?", ts(2));
        assert!(!report.results.is_empty(), "should find an answer");
        assert!(report.results[0].answer.contains("plants"));
    }

    #[test]
    fn reinforcement_increases_answer_weights() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        rt.ingest("What is osmosis?", "Osmosis is the diffusion of water across a membrane.", "biology_2e", 10, 0.8, ts(1));
        let r1 = rt.query("What is osmosis?", ts(2));
        let first_act = r1.results[0].activation;

        // Reingest same pair — weights should strengthen.
        rt.ingest("What is osmosis?", "Osmosis is the diffusion of water across a membrane.", "biology_2e", 10, 0.8, ts(3));
        let r2 = rt.query("What is osmosis?", ts(4));
        assert!(r2.results[0].activation >= first_act);
    }

    #[test]
    fn unknown_question_returns_empty() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        rt.ingest("What is mitosis?", "Cell division.", "biology_2e", 5, 0.7, ts(1));
        // Completely unrelated tokens — none in vocab.
        let report = rt.query("ZZZ qqq xxx", ts(2));
        assert!(report.results.is_empty());
    }

    #[test]
    fn multiple_answers_ranked_correctly() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        rt.ingest("What is Newton's first law?", "An object at rest stays at rest.", "physics", 1, 0.9, ts(1));
        rt.ingest("What is Newton's second law?", "Force equals mass times acceleration.", "physics", 2, 0.9, ts(2));
        rt.ingest("What is Newton's third law?", "For every action there is an equal and opposite reaction.", "physics", 3, 0.9, ts(3));

        let report = rt.query("What is Newton's second law?", ts(4));
        assert!(!report.results.is_empty());
        // Top result should contain "acceleration" or "force".
        let top = &report.results[0].answer;
        assert!(
            top.to_lowercase().contains("acceleration") || top.to_lowercase().contains("force"),
            "unexpected top answer: {top}"
        );
    }

    #[test]
    fn decay_reduces_weights_over_time() {
        let mut rt = QaRuntime::new(QaRuntimeConfig {
            weight_decay: 0.5, weight_ceiling: 5.0, // aggressive decay for test
            ..Default::default()
        });
        rt.ingest("What is gravity?", "Gravity is the force that attracts bodies.", "physics", 1, 0.9, ts(1));
        let before = rt.query("What is gravity?", ts(2)).results[0].activation;
        // Apply many decay steps.
        for _ in 0..20 {
            rt.decay_weights();
        }
        let after = rt.query("What is gravity?", ts(3));
        // After heavy decay, activation should be zero (weights pruned below 1e-6).
        assert!(after.results.is_empty() || after.results[0].activation < before);
    }

    #[test]
    fn bulk_ingest_from_candidates() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        let candidates = vec![
            QaCandidateRecord {
                qa_id: "q1".into(),
                question: "What is a cell?".into(),
                answer: "A cell is the basic unit of life.".into(),
                book_id: "biology".into(),
                page_index: 1,
                confidence: 0.85,
                evidence: String::new(),
                review_status: "PENDING".into(),
            },
            QaCandidateRecord {
                qa_id: "q2".into(),
                question: "What is DNA?".into(),
                answer: "DNA carries genetic information.".into(),
                book_id: "biology".into(),
                page_index: 2,
                confidence: 0.04, // below threshold — should be skipped
                evidence: String::new(),
                review_status: "PENDING".into(),
            },
        ];
        rt.ingest_candidates(&candidates, ts(1));
        assert_eq!(rt.pairs_ingested(), 1); // q2 skipped (low confidence)
        assert_eq!(rt.answer_count(), 1);
    }

    #[test]
    fn report_reflects_state() {
        let mut rt = QaRuntime::new(QaRuntimeConfig::default());
        rt.ingest("What is entropy?", "Entropy measures disorder.", "thermo", 3, 0.8, ts(1));
        let rep = rt.report(ts(2));
        assert_eq!(rep.pairs_ingested, 1);
        assert_eq!(rep.answer_entries, 1);
        assert!(rep.question_neurons > 0);
    }
}
