use crate::network::compute_payload_hash;
use crate::schema::{EnvironmentSnapshot, Position, Symbol, SymbolType, Timestamp};
use crate::streaming::schema::{EventKind, EventToken, LayerKind, LayerState, StreamSource, TokenBatch};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub struct SymbolizeConfig {
    pub time_scale: f64,
    pub domain_spacing: f64,
    pub layer_spacing: f64,
    pub layer_base_y: f64,
}

impl Default for SymbolizeConfig {
    fn default() -> Self {
        Self {
            time_scale: 0.01,
            domain_spacing: 5.0,
            layer_spacing: 2.0,
            layer_base_y: 20.0,
        }
    }
}

pub fn token_batch_to_snapshot(batch: &TokenBatch, config: &SymbolizeConfig) -> EnvironmentSnapshot {
    let mut symbols = Vec::new();
    let mut min_x = 0.0;
    let mut max_x = 0.0;
    let mut min_y = 0.0;
    let mut max_y = 0.0;

    for (idx, token) in batch.tokens.iter().enumerate() {
        let (position, source_label) =
            token_position(token, batch.timestamp, idx, config);
        let id = token_symbol_id(token, idx);
        let mut properties = HashMap::new();
        properties.insert(
            "token_kind".to_string(),
            Value::String(event_kind_label(token.kind)),
        );
        properties.insert(
            "duration_secs".to_string(),
            Value::from(token.duration_secs),
        );
        properties.insert("confidence".to_string(), Value::from(token.confidence));
        if let Some(source) = source_label {
            properties.insert("source".to_string(), Value::String(source));
        }
        if !token.attributes.is_empty() {
            properties.insert(
                "attributes".to_string(),
                Value::Object(map_from_attributes(&token.attributes)),
            );
        }
        update_bounds(position, &mut min_x, &mut max_x, &mut min_y, &mut max_y);
        symbols.push(Symbol {
            id,
            symbol_type: SymbolType::Custom,
            position,
            properties,
        });
    }

    for (idx, layer) in batch.layers.iter().enumerate() {
        let position = layer_position(layer, batch.timestamp, idx, config);
        let id = layer_symbol_id(layer, idx);
        let mut properties = HashMap::new();
        properties.insert(
            "layer_kind".to_string(),
            Value::String(layer_kind_label(layer.kind)),
        );
        properties.insert("phase".to_string(), Value::from(layer.phase));
        properties.insert("amplitude".to_string(), Value::from(layer.amplitude));
        properties.insert("coherence".to_string(), Value::from(layer.coherence));
        if !layer.attributes.is_empty() {
            properties.insert(
                "attributes".to_string(),
                Value::Object(map_from_attributes(&layer.attributes)),
            );
        }
        update_bounds(position, &mut min_x, &mut max_x, &mut min_y, &mut max_y);
        symbols.push(Symbol {
            id,
            symbol_type: SymbolType::Custom,
            position,
            properties,
        });
    }

    let width = (max_x - min_x).abs().max(1.0) + 1.0;
    let height = (max_y - min_y).abs().max(1.0) + 1.0;
    let mut bounds = HashMap::new();
    bounds.insert("width".to_string(), width);
    bounds.insert("height".to_string(), height);

    let mut metadata = HashMap::new();
    if !batch.source_confidence.is_empty() {
        let mut map = Map::new();
        for (source, value) in &batch.source_confidence {
            map.insert(stream_source_label(*source), Value::from(*value));
        }
        metadata.insert("source_confidence".to_string(), Value::Object(map));
    }
    metadata.insert("token_count".to_string(), Value::from(batch.tokens.len() as u64));
    metadata.insert("layer_count".to_string(), Value::from(batch.layers.len() as u64));

    EnvironmentSnapshot {
        timestamp: batch.timestamp,
        bounds,
        symbols,
        metadata,
        stack_history: Vec::new(),
    }
}

fn token_position(
    token: &EventToken,
    batch_time: Timestamp,
    idx: usize,
    config: &SymbolizeConfig,
) -> (Position, Option<String>) {
    let offset_secs = (token.onset.unix - batch_time.unix) as f64;
    let x = offset_secs * config.time_scale;
    let (source_idx, label) = source_index(token.source);
    let y = source_idx as f64 * config.domain_spacing;
    (
        Position {
            x,
            y,
            z: idx as f64 * 0.01,
        },
        label,
    )
}

fn layer_position(
    layer: &LayerState,
    batch_time: Timestamp,
    idx: usize,
    config: &SymbolizeConfig,
) -> Position {
    let offset_secs = (layer.timestamp.unix - batch_time.unix) as f64;
    let x = offset_secs * config.time_scale;
    let y = config.layer_base_y + idx as f64 * config.layer_spacing;
    Position { x, y, z: 0.0 }
}

fn token_symbol_id(token: &EventToken, idx: usize) -> String {
    if !token.id.trim().is_empty() {
        return format!("token:{}", token.id.trim());
    }
    let payload = format!(
        "token|{}|{}|{:.6}|{}",
        event_kind_label(token.kind),
        token.onset.unix,
        token.duration_secs,
        idx
    );
    format!("token:{}", compute_payload_hash(payload.as_bytes()))
}

fn layer_symbol_id(layer: &LayerState, idx: usize) -> String {
    let payload = format!(
        "layer|{}|{}|{:.6}|{}",
        layer_kind_label(layer.kind),
        layer.timestamp.unix,
        layer.amplitude,
        idx
    );
    format!("layer:{}", compute_payload_hash(payload.as_bytes()))
}

fn source_index(source: Option<StreamSource>) -> (usize, Option<String>) {
    match source {
        Some(StreamSource::PeopleVideo) => (0, Some(stream_source_label(StreamSource::PeopleVideo))),
        Some(StreamSource::CrowdTraffic) => (1, Some(stream_source_label(StreamSource::CrowdTraffic))),
        Some(StreamSource::PublicTopics) => (2, Some(stream_source_label(StreamSource::PublicTopics))),
        None => (3, None),
    }
}

fn stream_source_label(source: StreamSource) -> String {
    match source {
        StreamSource::PeopleVideo => "PEOPLE_VIDEO".to_string(),
        StreamSource::CrowdTraffic => "CROWD_TRAFFIC".to_string(),
        StreamSource::PublicTopics => "PUBLIC_TOPICS".to_string(),
    }
}

fn event_kind_label(kind: EventKind) -> String {
    match kind {
        EventKind::BehavioralAtom => "BEHAVIORAL_ATOM".to_string(),
        EventKind::BehavioralToken => "BEHAVIORAL_TOKEN".to_string(),
        EventKind::CrowdToken => "CROWD_TOKEN".to_string(),
        EventKind::TrafficToken => "TRAFFIC_TOKEN".to_string(),
        EventKind::TopicEventToken => "TOPIC_EVENT_TOKEN".to_string(),
    }
}

fn layer_kind_label(kind: LayerKind) -> String {
    match kind {
        LayerKind::UltradianMicroArousal => "ULTRADIAN_MICRO_AROUSAL".to_string(),
        LayerKind::UltradianBrac => "ULTRADIAN_BRAC".to_string(),
        LayerKind::UltradianMeso => "ULTRADIAN_MESO".to_string(),
        LayerKind::FlowDensity => "FLOW_DENSITY".to_string(),
        LayerKind::FlowVelocity => "FLOW_VELOCITY".to_string(),
        LayerKind::FlowDirectionality => "FLOW_DIRECTIONALITY".to_string(),
        LayerKind::FlowStopGoWave => "FLOW_STOP_GO_WAVE".to_string(),
        LayerKind::FlowMotif => "FLOW_MOTIF".to_string(),
        LayerKind::FlowSeasonalDaily => "FLOW_SEASONAL_DAILY".to_string(),
        LayerKind::FlowSeasonalWeekly => "FLOW_SEASONAL_WEEKLY".to_string(),
        LayerKind::TopicBurst => "TOPIC_BURST".to_string(),
        LayerKind::TopicDecay => "TOPIC_DECAY".to_string(),
        LayerKind::TopicExcitation => "TOPIC_EXCITATION".to_string(),
        LayerKind::TopicLeadLag => "TOPIC_LEAD_LAG".to_string(),
        LayerKind::TopicPeriodicity => "TOPIC_PERIODICITY".to_string(),
    }
}

fn map_from_attributes(input: &HashMap<String, Value>) -> Map<String, Value> {
    let mut map = Map::new();
    for (key, value) in input {
        map.insert(key.clone(), value.clone());
    }
    map
}

fn update_bounds(pos: Position, min_x: &mut f64, max_x: &mut f64, min_y: &mut f64, max_y: &mut f64) {
    if pos.x < *min_x {
        *min_x = pos.x;
    }
    if pos.x > *max_x {
        *max_x = pos.x;
    }
    if pos.y < *min_y {
        *min_y = pos.y;
    }
    if pos.y > *max_y {
        *max_y = pos.y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::schema::{EventToken, LayerState, TokenBatch};

    #[test]
    fn symbolizes_tokens_and_layers() {
        let batch = TokenBatch {
            timestamp: Timestamp { unix: 100 },
            tokens: vec![EventToken {
                id: "t1".to_string(),
                kind: EventKind::BehavioralAtom,
                onset: Timestamp { unix: 100 },
                duration_secs: 5.0,
                confidence: 0.9,
                attributes: HashMap::new(),
                source: Some(StreamSource::PeopleVideo),
            }],
            layers: vec![LayerState {
                kind: LayerKind::UltradianMicroArousal,
                timestamp: Timestamp { unix: 100 },
                phase: 0.1,
                amplitude: 0.5,
                coherence: 0.2,
                attributes: HashMap::new(),
            }],
            source_confidence: HashMap::new(),
        };
        let snapshot = token_batch_to_snapshot(&batch, &SymbolizeConfig::default());
        assert_eq!(snapshot.symbols.len(), 2);
        assert_eq!(snapshot.timestamp.unix, 100);
    }
}
