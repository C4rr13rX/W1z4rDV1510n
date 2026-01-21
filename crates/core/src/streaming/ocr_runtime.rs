use crate::config::{OcrOutputFormat, StreamingOcrConfig};
use crate::network::compute_payload_hash;
use crate::schema::Timestamp;
use crate::streaming::motor::PoseFrame;
use anyhow::{Context, Result};
use base64::engine::general_purpose;
use base64::Engine as _;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone)]
pub struct OcrBlock {
    pub text: String,
    pub confidence: Option<f64>,
    pub bbox: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct OcrResult {
    pub text: String,
    pub blocks: Vec<OcrBlock>,
    pub engine: String,
}

pub struct FrameOcrRuntime {
    config: StreamingOcrConfig,
    cache: HashMap<String, Timestamp>,
}

impl FrameOcrRuntime {
    pub fn new(config: StreamingOcrConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    pub fn maybe_enrich(
        &mut self,
        frame: &mut PoseFrame,
        envelope_meta: &HashMap<String, Value>,
    ) -> Option<OcrResult> {
        if !self.config.enabled {
            return None;
        }
        if has_text_metadata(&frame.metadata) {
            return None;
        }
        let timestamp = frame.timestamp.unwrap_or(Timestamp { unix: 0 });
        self.prune_cache(timestamp);
        let confidence = meta_confidence(&frame.metadata)
            .or_else(|| meta_confidence(envelope_meta))
            .unwrap_or(1.0);
        if confidence < self.config.min_frame_confidence {
            return None;
        }
        let cache_key = cache_key(&frame.metadata);
        if let Some(key) = &cache_key {
            if self.cache.contains_key(key) {
                return None;
            }
        }
        let image = resolve_image_source(&frame.metadata, &self.config)?;
        let result = run_ocr_command(&self.config, &image)?;
        if let Some(key) = cache_key {
            self.cache.insert(key, timestamp);
        }
        apply_result(frame, &result, &self.config);
        Some(result)
    }

    fn prune_cache(&mut self, now: Timestamp) {
        let ttl = self.config.cache_ttl_secs.max(1) as i64;
        let mut expired = Vec::new();
        for (key, last_seen) in &self.cache {
            if now.unix.saturating_sub(last_seen.unix) > ttl {
                expired.push(key.clone());
            }
        }
        for key in expired {
            self.cache.remove(&key);
        }
    }
}

struct OcrImage {
    label: String,
    bytes: Vec<u8>,
}

fn resolve_image_source(
    metadata: &HashMap<String, Value>,
    config: &StreamingOcrConfig,
) -> Option<OcrImage> {
    if config.allow_base64 {
        if let Some(bytes) = bytes_from_metadata(metadata) {
            let label = metadata
                .get("image_ref")
                .and_then(|val| val.as_str())
                .unwrap_or("frame.bin")
                .to_string();
            return Some(OcrImage { label, bytes });
        }
    }
    if config.allow_image_ref {
        if let Some(path) = image_ref_from_metadata(metadata) {
            let path_ref = PathBuf::from(path.clone());
            let bytes = fs::read(&path_ref).ok()?;
            return Some(OcrImage {
                label: path,
                bytes,
            });
        }
    }
    None
}

fn bytes_from_metadata(metadata: &HashMap<String, Value>) -> Option<Vec<u8>> {
    for key in ["image_b64", "image_bytes", "frame_b64", "frame_bytes"] {
        if let Some(value) = metadata.get(key) {
            if let Some(text) = value.as_str() {
                if let Some(decoded) = decode_base64(text) {
                    return Some(decoded);
                }
            }
            if let Some(array) = value.as_array() {
                let mut bytes = Vec::with_capacity(array.len());
                for item in array {
                    let val = item.as_u64()? as u8;
                    bytes.push(val);
                }
                if !bytes.is_empty() {
                    return Some(bytes);
                }
            }
        }
    }
    None
}

fn image_ref_from_metadata(metadata: &HashMap<String, Value>) -> Option<String> {
    for key in ["image_ref", "frame_ref", "frame_path", "image_path"] {
        if let Some(text) = metadata.get(key).and_then(|value| value.as_str()) {
            if !text.trim().is_empty() && !text.contains("://") {
                return Some(text.to_string());
            }
        }
    }
    None
}

fn decode_base64(value: &str) -> Option<Vec<u8>> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let payload = trimmed
        .split(',')
        .last()
        .unwrap_or(trimmed)
        .trim();
    general_purpose::STANDARD.decode(payload).ok()
}

fn run_ocr_command(config: &StreamingOcrConfig, image: &OcrImage) -> Option<OcrResult> {
    if config.command.is_empty() {
        return None;
    }
    let image_path = materialize_image(&image.label, &image.bytes).ok()?;
    let (bin, args) = resolve_command(&config.command, &image_path);
    let output = Command::new(bin)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if stdout.is_empty() {
        return None;
    }
    let engine = config
        .engine_label
        .clone()
        .or_else(|| config.command.first().cloned())
        .unwrap_or_else(|| "ocr".to_string());
    parse_ocr_output(config.output_format, &stdout, engine)
}

fn parse_ocr_output(
    format: OcrOutputFormat,
    stdout: &str,
    engine: String,
) -> Option<OcrResult> {
    match format {
        OcrOutputFormat::Plain => {
            let text = stdout.trim().to_string();
            if text.is_empty() {
                None
            } else {
                Some(OcrResult {
                    text,
                    blocks: Vec::new(),
                    engine,
                })
            }
        }
        OcrOutputFormat::Json => parse_json_output(stdout, engine),
    }
}

fn parse_json_output(stdout: &str, engine: String) -> Option<OcrResult> {
    let value: Value = serde_json::from_str(stdout).ok()?;
    let (text, mut blocks) = extract_text_blocks(&value);
    if text.trim().is_empty() && blocks.is_empty() {
        return None;
    }
    if blocks.is_empty() && !text.trim().is_empty() {
        blocks.push(OcrBlock {
            text: text.clone(),
            confidence: None,
            bbox: None,
        });
    }
    Some(OcrResult { text, blocks, engine })
}

fn extract_text_blocks(value: &Value) -> (String, Vec<OcrBlock>) {
    if let Some(array) = value.as_array() {
        let mut blocks = Vec::new();
        for item in array {
            if let Some(block) = block_from_value(item) {
                blocks.push(block);
            }
        }
        let text = join_blocks(&blocks);
        return (text, blocks);
    }
    if let Some(obj) = value.as_object() {
        let text = obj
            .get("text")
            .or_else(|| obj.get("full_text"))
            .and_then(|val| val.as_str())
            .unwrap_or_default()
            .to_string();
        let mut blocks = Vec::new();
        if let Some(list) = obj
            .get("blocks")
            .or_else(|| obj.get("lines"))
            .or_else(|| obj.get("items"))
            .and_then(|val| val.as_array())
        {
            for item in list {
                if let Some(block) = block_from_value(item) {
                    blocks.push(block);
                }
            }
        }
        let merged = if text.trim().is_empty() {
            join_blocks(&blocks)
        } else {
            text
        };
        return (merged, blocks);
    }
    (String::new(), Vec::new())
}

fn block_from_value(value: &Value) -> Option<OcrBlock> {
    if let Some(obj) = value.as_object() {
        let text = obj.get("text").and_then(|val| val.as_str())?.to_string();
        if text.trim().is_empty() {
            return None;
        }
        let confidence = obj.get("confidence").and_then(|val| val.as_f64());
        let bbox = obj.get("bbox").or_else(|| obj.get("box")).cloned();
        return Some(OcrBlock {
            text,
            confidence,
            bbox,
        });
    }
    value.as_str().map(|text| OcrBlock {
        text: text.to_string(),
        confidence: None,
        bbox: None,
    })
}

fn join_blocks(blocks: &[OcrBlock]) -> String {
    let mut parts = Vec::new();
    for block in blocks {
        let text = block.text.trim();
        if !text.is_empty() {
            parts.push(text.to_string());
        }
    }
    parts.join("\n")
}

fn apply_result(frame: &mut PoseFrame, result: &OcrResult, config: &StreamingOcrConfig) {
    let text = truncate_text(&result.text, config.max_text_len);
    if !text.is_empty() {
        frame
            .metadata
            .insert("ocr_text".to_string(), Value::String(text));
    }
    let mut blocks = Vec::new();
    let mut conf_sum = 0.0;
    let mut conf_count = 0.0;
    for block in result.blocks.iter().take(config.max_blocks.max(1)) {
        if block.text.trim().is_empty() {
            continue;
        }
        let mut obj = serde_json::Map::new();
        obj.insert("text".to_string(), Value::String(block.text.clone()));
        if let Some(conf) = block.confidence {
            conf_sum += conf.clamp(0.0, 1.0);
            conf_count += 1.0;
            obj.insert("confidence".to_string(), Value::from(conf));
        }
        if let Some(bbox) = block.bbox.clone() {
            obj.insert("bbox".to_string(), bbox);
        }
        blocks.push(Value::Object(obj));
    }
    if !blocks.is_empty() {
        frame
            .metadata
            .insert("text_blocks".to_string(), Value::Array(blocks));
    }
    if conf_count > 0.0 {
        frame.metadata.insert(
            "ocr_confidence".to_string(),
            Value::from((conf_sum / conf_count).clamp(0.0, 1.0)),
        );
    }
    frame
        .metadata
        .insert("ocr_engine".to_string(), Value::String(result.engine.clone()));
}

fn truncate_text(text: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }
    let trimmed = text.trim();
    if trimmed.len() <= max_len {
        return trimmed.to_string();
    }
    let mut out = trimmed[..max_len].to_string();
    out.push_str("...");
    out
}

fn cache_key(metadata: &HashMap<String, Value>) -> Option<String> {
    if let Some(frame_id) = metadata.get("frame_id").and_then(|val| val.as_str()) {
        if !frame_id.trim().is_empty() {
            return Some(format!("frame::{frame_id}"));
        }
    }
    if let Some(path) = image_ref_from_metadata(metadata) {
        if !path.trim().is_empty() {
            return Some(format!("path::{path}"));
        }
    }
    None
}

fn has_text_metadata(metadata: &HashMap<String, Value>) -> bool {
    if let Some(text) = metadata.get("ocr_text").and_then(|val| val.as_str()) {
        return !text.trim().is_empty();
    }
    if let Some(text) = metadata.get("detected_text").and_then(|val| val.as_str()) {
        return !text.trim().is_empty();
    }
    if let Some(Value::Array(blocks)) = metadata.get("text_blocks") {
        return !blocks.is_empty();
    }
    false
}

fn meta_confidence(metadata: &HashMap<String, Value>) -> Option<f64> {
    metadata
        .get("quality")
        .or_else(|| metadata.get("confidence"))
        .and_then(|val| val.as_f64())
        .map(|val| val.clamp(0.0, 1.0))
}

fn materialize_image(image_ref: &str, bytes: &[u8]) -> Result<PathBuf> {
    let mut path = PathBuf::from(image_ref);
    if path.exists() {
        return Ok(path);
    }
    let hash = compute_payload_hash(bytes);
    path = std::env::temp_dir().join(format!("ocr_{hash}.bin"));
    if !path.exists() {
        fs::write(&path, bytes).context("failed to write OCR temp file")?;
    }
    Ok(path)
}

fn resolve_command(command: &[String], image_path: &Path) -> (String, Vec<String>) {
    let mut args = Vec::new();
    let image_str = image_path.to_string_lossy();
    let bin = command.first().cloned().unwrap_or_default();
    for arg in command.iter().skip(1) {
        args.push(arg.replace("{image}", &image_str));
    }
    (bin, args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_output() {
        let result = parse_ocr_output(OcrOutputFormat::Plain, "Hello", "ocr".to_string())
            .expect("result");
        assert_eq!(result.text, "Hello");
    }

    #[test]
    fn parses_json_output_blocks() {
        let json = r#"{
            "text": "STOP SIGN",
            "blocks": [
                { "text": "STOP", "confidence": 0.9, "bbox": [0,0,10,10] },
                { "text": "SIGN", "confidence": 0.8, "bbox": [10,0,20,10] }
            ]
        }"#;
        let result = parse_ocr_output(OcrOutputFormat::Json, json, "ocr".to_string())
            .expect("result");
        assert_eq!(result.blocks.len(), 2);
        assert!(result.text.contains("STOP"));
    }
}
