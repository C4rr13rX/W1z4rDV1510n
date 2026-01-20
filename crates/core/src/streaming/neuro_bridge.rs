use crate::neuro::{MinicolumnSnapshot, NeuroRuntimeHandle, NeuroSnapshot};
use crate::schema::Timestamp;
use crate::streaming::schema::{EventKind, EventToken, LayerState, StreamSource, TokenBatch};
use crate::streaming::symbolize::{token_batch_to_snapshot, SymbolizeConfig};
use crate::streaming::ultradian::{SignalSample, UltradianLayerExtractor};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

const DEFAULT_MAX_SYMBOLS: usize = 256;
const DEFAULT_SAMPLE_STRIDE: usize = 1;
const DEFAULT_MIN_CONFIDENCE: f64 = 0.1;

pub struct NeuroStreamBridge {
    neuro: NeuroRuntimeHandle,
    symbolizer: SymbolizeConfig,
    step: usize,
    sample_stride: usize,
    max_symbols: usize,
    min_confidence: f64,
}

impl NeuroStreamBridge {
    pub fn new(neuro: NeuroRuntimeHandle, symbolizer: SymbolizeConfig) -> Self {
        Self {
            neuro,
            symbolizer,
            step: 0,
            sample_stride: DEFAULT_SAMPLE_STRIDE,
            max_symbols: DEFAULT_MAX_SYMBOLS,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
        }
    }

    pub fn observe_batch(&mut self, batch: &TokenBatch) -> Option<NeuroSnapshot> {
        if self.sample_stride > 1 {
            self.step = self.step.wrapping_add(1);
            if self.step % self.sample_stride != 0 {
                return None;
            }
        }
        let filtered = select_for_neuro(batch, self.max_symbols, self.min_confidence);
        if filtered.tokens.is_empty() && filtered.layers.is_empty() {
            return None;
        }
        let snapshot = token_batch_to_snapshot(&filtered, &self.symbolizer);
        self.neuro.observe_snapshot(&snapshot);
        Some(self.neuro.snapshot())
    }
}

pub struct SubstreamOutput {
    pub tokens: Vec<EventToken>,
    pub layers: Vec<LayerState>,
}

pub struct SubstreamRuntime {
    max_substreams: usize,
    min_stability: f32,
    filters: Vec<SubstreamFilter>,
    substreams: HashMap<String, SubstreamState>,
}

impl Default for SubstreamRuntime {
    fn default() -> Self {
        Self {
            max_substreams: 64,
            min_stability: 0.65,
            filters: Vec::new(),
            substreams: HashMap::new(),
        }
    }
}

impl SubstreamRuntime {
    pub fn update_from_neuro(&mut self, snapshot: &NeuroSnapshot, now: Timestamp) {
        let mut candidates = snapshot
            .minicolumns
            .iter()
            .filter(|m| !m.collapsed && m.stability >= self.min_stability)
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            self.filters.clear();
            self.substreams.clear();
            return;
        }
        candidates.sort_by(|a, b| {
            b.stability
                .partial_cmp(&a.stability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut filters = Vec::new();
        let mut keep = HashSet::new();
        for entry in candidates.into_iter().take(self.max_substreams.max(1)) {
            if let Some(filter) = SubstreamFilter::from_minicolumn(entry) {
                keep.insert(filter.label.clone());
                filters.push(filter);
            }
        }
        self.substreams
            .retain(|label, _| keep.contains(label));
        for filter in &filters {
            self.substreams
                .entry(filter.label.clone())
                .or_insert_with(|| SubstreamState::new(now));
        }
        self.filters = filters;
    }

    pub fn ingest(&mut self, batch: &TokenBatch) -> SubstreamOutput {
        let mut output = SubstreamOutput {
            tokens: Vec::new(),
            layers: Vec::new(),
        };
        if self.filters.is_empty() {
            return output;
        }
        for filter in &self.filters {
            let state = match self.substreams.get_mut(&filter.label) {
                Some(state) => state,
                None => continue,
            };
            let mut signal_sum = 0.0;
            let mut quality_sum = 0.0;
            let mut matches = 0.0;
            for token in &batch.tokens {
                if !matches_filter(token, filter) {
                    continue;
                }
                if let Some(signal) = signal_from_token(token) {
                    let quality = quality_from_token(token);
                    signal_sum += signal * quality;
                    quality_sum += quality;
                    matches += 1.0;
                }
            }
            if matches <= 0.0 {
                continue;
            }
            let mean_signal = if quality_sum > 0.0 {
                signal_sum / quality_sum
            } else {
                signal_sum / matches
            };
            let mean_quality = if matches > 0.0 {
                (quality_sum / matches).clamp(0.0, 1.0)
            } else {
                0.0
            };
            state.update(batch.timestamp, mean_signal);
            state.extractor.push_sample(SignalSample {
                timestamp: batch.timestamp,
                value: mean_signal,
                quality: mean_quality,
            });
            let mut layers = state.extractor.extract_layers();
            for layer in &mut layers {
                layer.attributes.insert(
                    "substream_key".to_string(),
                    Value::String(filter.label.clone()),
                );
                layer.attributes.insert(
                    "substream_stability".to_string(),
                    Value::from(filter.stability as f64),
                );
                layer.attributes.insert(
                    "substream_matches".to_string(),
                    Value::from(matches as u64),
                );
            }
            output.layers.extend(layers);
            let mut attrs = HashMap::new();
            attrs.insert(
                "substream_key".to_string(),
                Value::String(filter.label.clone()),
            );
            attrs.insert("substream_signal".to_string(), Value::from(mean_signal));
            attrs.insert(
                "substream_quality".to_string(),
                Value::from(mean_quality),
            );
            attrs.insert(
                "substream_matches".to_string(),
                Value::from(matches as u64),
            );
            attrs.insert(
                "substream_stability".to_string(),
                Value::from(filter.stability as f64),
            );
            output.tokens.push(EventToken {
                id: String::new(),
                kind: EventKind::BehavioralToken,
                onset: batch.timestamp,
                duration_secs: 1.0,
                confidence: mean_quality,
                attributes: attrs,
                source: None,
            });
        }
        output
    }
}

struct SubstreamState {
    extractor: UltradianLayerExtractor,
    last_seen: Timestamp,
    signal_ema: f64,
}

impl SubstreamState {
    fn new(now: Timestamp) -> Self {
        Self {
            extractor: UltradianLayerExtractor::new(),
            last_seen: now,
            signal_ema: 0.0,
        }
    }

    fn update(&mut self, now: Timestamp, signal: f64) {
        let alpha = 0.4;
        self.signal_ema = alpha * signal + (1.0 - alpha) * self.signal_ema;
        self.last_seen = now;
    }
}

#[derive(Debug, Clone)]
struct SubstreamFilter {
    label: String,
    stability: f32,
    attrs: HashMap<String, String>,
    roles: HashSet<String>,
}

impl SubstreamFilter {
    fn from_minicolumn(entry: &MinicolumnSnapshot) -> Option<Self> {
        let label = entry.label.trim();
        let raw = label.strip_prefix("mini::").unwrap_or(label);
        if raw.is_empty() {
            return None;
        }
        let mut attrs = HashMap::new();
        let mut roles = HashSet::new();
        for part in raw.split('|') {
            if let Some(rest) = part.strip_prefix("attr::") {
                if let Some((key, value)) = rest.split_once("::") {
                    attrs.insert(normalize_token(key), normalize_token(value));
                }
            } else if let Some(rest) = part.strip_prefix("role::") {
                let role = normalize_token(rest);
                if !role.is_empty() {
                    roles.insert(role);
                }
            }
        }
        if attrs.is_empty() && roles.is_empty() {
            return None;
        }
        Some(Self {
            label: label.to_string(),
            stability: entry.stability,
            attrs,
            roles,
        })
    }
}

fn select_for_neuro(
    batch: &TokenBatch,
    max_symbols: usize,
    min_confidence: f64,
) -> TokenBatch {
    if max_symbols == 0 {
        return TokenBatch {
            timestamp: batch.timestamp,
            tokens: Vec::new(),
            layers: Vec::new(),
            source_confidence: batch.source_confidence.clone(),
        };
    }
    let max_tokens = (max_symbols / 2).max(1);
    let max_layers = max_symbols.saturating_sub(max_tokens).max(1);
    let mut tokens = Vec::new();
    for token in &batch.tokens {
        if token.confidence >= min_confidence {
            tokens.push(token.clone());
        }
        if tokens.len() >= max_tokens {
            break;
        }
    }
    let mut layers = Vec::new();
    for layer in &batch.layers {
        let score = (layer.amplitude + layer.coherence) * 0.5;
        if score >= min_confidence {
            layers.push(layer.clone());
        }
        if layers.len() >= max_layers {
            break;
        }
    }
    TokenBatch {
        timestamp: batch.timestamp,
        tokens,
        layers,
        source_confidence: batch.source_confidence.clone(),
    }
}

fn matches_filter(token: &EventToken, filter: &SubstreamFilter) -> bool {
    let attrs = token_attr_map(token);
    for (key, value) in &filter.attrs {
        let Some(candidate) = attrs.get(key) else {
            return false;
        };
        if candidate != value {
            return false;
        }
    }
    if !filter.roles.is_empty() {
        if let Some(role) = attrs.get("role") {
            if !filter.roles.contains(role) {
                return false;
            }
        }
    }
    true
}

fn token_attr_map(token: &EventToken) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for (key, value) in &token.attributes {
        if let Some(text) = value.as_str() {
            map.insert(normalize_token(key), normalize_token(text));
        }
        if let Some(inner) = value.as_object() {
            for (inner_key, inner_val) in inner {
                if let Some(text) = inner_val.as_str() {
                    map.insert(normalize_token(inner_key), normalize_token(text));
                }
            }
        }
    }
    if let Some(source) = token.source {
        map.insert("source".to_string(), normalize_token(&source_label(source)));
    }
    map
}

fn signal_from_token(token: &EventToken) -> Option<f64> {
    let attrs = &token.attributes;
    for key in [
        "motor_signal",
        "signal",
        "intensity",
        "density",
        "velocity",
        "motion_energy",
        "amplitude",
    ] {
        if let Some(val) = attrs.get(key).and_then(|v| v.as_f64()) {
            if val.is_finite() {
                return Some(val);
            }
        }
    }
    None
}

fn quality_from_token(token: &EventToken) -> f64 {
    let attrs = &token.attributes;
    if let Some(val) = attrs.get("signal_quality").and_then(|v| v.as_f64()) {
        return val.clamp(0.0, 1.0);
    }
    if let Some(val) = attrs.get("quality").and_then(|v| v.as_f64()) {
        return val.clamp(0.0, 1.0);
    }
    token.confidence.clamp(0.0, 1.0)
}

fn normalize_token(raw: &str) -> String {
    let mut out = String::new();
    let mut last_underscore = false;
    for ch in raw.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_underscore = false;
        } else if !last_underscore {
            out.push('_');
            last_underscore = true;
        }
    }
    while out.ends_with('_') {
        out.pop();
    }
    if out.len() > 40 {
        out.truncate(40);
    }
    out
}

fn source_label(source: StreamSource) -> String {
    match source {
        StreamSource::PeopleVideo => "PEOPLE_VIDEO".to_string(),
        StreamSource::CrowdTraffic => "CROWD_TRAFFIC".to_string(),
        StreamSource::PublicTopics => "PUBLIC_TOPICS".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Timestamp;
    use crate::streaming::schema::{EventKind, TokenBatch};

    #[test]
    fn substream_emits_token_for_matching_attrs() {
        let mut runtime = SubstreamRuntime::default();
        let minicolumn = MinicolumnSnapshot {
            label: "mini::role::custom|attr::make::hyundai|attr::model::sonata".to_string(),
            stability: 0.9,
            inhibition: 0.0,
            collapsed: false,
            members: 3,
            born_at: 1,
        };
        let snapshot = NeuroSnapshot {
            minicolumns: vec![minicolumn],
            ..NeuroSnapshot::default()
        };
        let now = Timestamp { unix: 10 };
        runtime.update_from_neuro(&snapshot, now);
        let token = EventToken {
            id: "t1".to_string(),
            kind: EventKind::BehavioralAtom,
            onset: now,
            duration_secs: 1.0,
            confidence: 0.8,
            attributes: HashMap::from([
                ("make".to_string(), Value::String("Hyundai".to_string())),
                ("model".to_string(), Value::String("Sonata".to_string())),
                ("motor_signal".to_string(), Value::from(0.6)),
            ]),
            source: Some(StreamSource::PeopleVideo),
        };
        let batch = TokenBatch {
            timestamp: now,
            tokens: vec![token],
            layers: Vec::new(),
            source_confidence: HashMap::new(),
        };
        let output = runtime.ingest(&batch);
        assert_eq!(output.tokens.len(), 1);
        assert!(output.tokens[0]
            .attributes
            .get("substream_key")
            .is_some());
    }
}
