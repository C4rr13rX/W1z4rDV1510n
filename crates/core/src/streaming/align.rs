use crate::config::StreamingConfig;
use crate::schema::Timestamp;
use crate::streaming::schema::{EventToken, StreamSource, TokenBatch};
use std::collections::HashMap;

pub struct StreamingAligner {
    tolerance_secs: f64,
    confidence_gate: f64,
    min_pending_sources: usize,
    pending: HashMap<StreamSource, TokenBatch>,
}

impl StreamingAligner {
    pub fn new(config: &StreamingConfig) -> Self {
        let enabled_sources = config.ingest.enabled_source_count();
        Self {
            tolerance_secs: config.temporal_tolerance_secs.max(0.0),
            confidence_gate: config.confidence_gate.clamp(0.0, 1.0),
            min_pending_sources: if enabled_sources > 1 { 2 } else { 1 },
            pending: HashMap::new(),
        }
    }

    pub fn push(&mut self, batch: TokenBatch) -> Option<TokenBatch> {
        let Some(source) = source_for_batch(&batch) else {
            return Some(batch);
        };
        self.pending.insert(source, batch);
        let reference = latest_timestamp(self.pending.values());
        self.prune(reference);
        if let Some(flushed) = self.flush_stale(reference) {
            return Some(flushed);
        }
        if self.pending.len() < self.min_pending_sources {
            return None;
        }
        self.try_fuse(reference)
    }

    fn prune(&mut self, reference: Timestamp) {
        let ttl = self.tolerance_secs.max(1.0) * 2.0;
        self.pending.retain(|_, batch| {
            timestamp_diff_secs(batch.timestamp, reference) <= ttl
        });
    }

    fn try_fuse(&mut self, reference: Timestamp) -> Option<TokenBatch> {
        let mut fused = TokenBatch {
            timestamp: reference,
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let mut used = Vec::new();
        for (source, candidate) in &self.pending {
            if timestamp_diff_secs(candidate.timestamp, reference) > self.tolerance_secs {
                continue;
            }
            let confidence = confidence_for_source(candidate, *source);
            if confidence < self.confidence_gate {
                continue;
            }
            fused.tokens.extend(filter_tokens(candidate.tokens.iter(), Some(*source)));
            fused.layers.extend(candidate.layers.iter().cloned());
            fused
                .source_confidence
                .insert(*source, confidence);
            used.push(*source);
        }
        for source in used {
            self.pending.remove(&source);
        }
        if fused.tokens.is_empty() && fused.layers.is_empty() {
            return None;
        }
        Some(fused)
    }

    fn flush_stale(&mut self, reference: Timestamp) -> Option<TokenBatch> {
        let (source, candidate) = self
            .pending
            .iter()
            .min_by_key(|(_, batch)| batch.timestamp.unix)
            .map(|(source, batch)| (*source, batch.clone()))?;
        if timestamp_diff_secs(candidate.timestamp, reference) <= self.tolerance_secs {
            return None;
        }
        let confidence = confidence_for_source(&candidate, source);
        if confidence < self.confidence_gate {
            self.pending.remove(&source);
            return None;
        }
        let flushed = TokenBatch {
            timestamp: candidate.timestamp,
            tokens: filter_tokens(candidate.tokens.iter(), Some(source)),
            layers: candidate.layers.clone(),
            source_confidence: HashMap::from([(source, confidence)]),
        };
        self.pending.remove(&source);
        if flushed.tokens.is_empty() && flushed.layers.is_empty() {
            None
        } else {
            Some(flushed)
        }
    }
}

fn source_for_batch(batch: &TokenBatch) -> Option<StreamSource> {
    if !batch.source_confidence.is_empty() {
        return batch
            .source_confidence
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(source, _)| *source);
    }
    let mut counts: HashMap<StreamSource, usize> = HashMap::new();
    for token in &batch.tokens {
        if let Some(source) = token.source {
            *counts.entry(source).or_insert(0) += 1;
        }
    }
    counts
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(source, _)| source)
}

fn confidence_for_source(batch: &TokenBatch, source: StreamSource) -> f64 {
    batch
        .source_confidence
        .get(&source)
        .copied()
        .unwrap_or(1.0)
        .clamp(0.0, 1.0)
}

fn latest_timestamp<'a>(batches: impl IntoIterator<Item = &'a TokenBatch>) -> Timestamp {
    let mut latest = Timestamp { unix: 0 };
    for batch in batches {
        if batch.timestamp.unix > latest.unix {
            latest = batch.timestamp;
        }
    }
    latest
}

fn timestamp_diff_secs(a: Timestamp, b: Timestamp) -> f64 {
    (a.unix - b.unix).abs() as f64
}

fn filter_tokens<'a>(
    tokens: impl IntoIterator<Item = &'a EventToken>,
    source: Option<StreamSource>,
) -> Vec<EventToken> {
    tokens
        .into_iter()
        .filter(|token| {
            source.map(|src| token.source.unwrap_or(src) == src).unwrap_or(true)
        })
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::schema::{EventKind, LayerKind, LayerState};

    fn batch_with_source(source: StreamSource, ts: i64, confidence: f64) -> TokenBatch {
        TokenBatch {
            timestamp: Timestamp { unix: ts },
            tokens: vec![EventToken {
                id: "t".to_string(),
                kind: EventKind::BehavioralToken,
                onset: Timestamp { unix: ts },
                duration_secs: 1.0,
                confidence: 1.0,
                attributes: HashMap::new(),
                source: Some(source),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: ts },
                phase: 0.0,
                amplitude: 0.5,
                coherence: 0.2,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::from([(source, confidence)]),
        }
    }

    #[test]
    fn aligner_fuses_within_tolerance() {
        let mut config = StreamingConfig::default();
        config.ingest.people_video = true;
        config.ingest.crowd_traffic = true;
        config.temporal_tolerance_secs = 5.0;
        config.confidence_gate = 0.5;
        let mut aligner = StreamingAligner::new(&config);
        let batch_a = batch_with_source(StreamSource::PeopleVideo, 100, 0.9);
        let batch_b = batch_with_source(StreamSource::CrowdTraffic, 102, 0.9);
        assert!(aligner.push(batch_a).is_none());
        let fused = aligner.push(batch_b).expect("fused");
        assert_eq!(fused.source_confidence.len(), 2);
        assert_eq!(fused.tokens.len(), 2);
    }

    #[test]
    fn aligner_gates_low_confidence_source() {
        let mut config = StreamingConfig::default();
        config.ingest.people_video = true;
        config.ingest.crowd_traffic = true;
        config.temporal_tolerance_secs = 5.0;
        config.confidence_gate = 0.8;
        let mut aligner = StreamingAligner::new(&config);
        let batch_a = batch_with_source(StreamSource::PeopleVideo, 100, 0.6);
        let batch_b = batch_with_source(StreamSource::CrowdTraffic, 100, 0.9);
        let _ = aligner.push(batch_a);
        let fused = aligner.push(batch_b).expect("fused");
        assert_eq!(fused.source_confidence.len(), 1);
    }
}
