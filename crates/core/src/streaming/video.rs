use crate::schema::Timestamp;
use crate::streaming::ingest::StreamIngestor;
use crate::streaming::motor::PoseFrame;
use crate::streaming::schema::{StreamEnvelope, StreamPayload, StreamSource};
use anyhow::Context;
use serde_json::Value;
use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct PoseCommandConfig {
    pub command: Vec<String>,
    pub source: String,
    pub metadata: HashMap<String, Value>,
}

impl PoseCommandConfig {
    pub fn new(command: Vec<String>, source: impl Into<String>) -> Self {
        let source = source.into();
        let mut metadata = HashMap::new();
        metadata.insert("stream_source".to_string(), Value::String(source.clone()));
        Self {
            command,
            source,
            metadata,
        }
    }

    pub fn resolve_command(&self) -> anyhow::Result<(String, Vec<String>)> {
        anyhow::ensure!(!self.command.is_empty(), "pose command must be non-empty");
        let bin = substitute_source(&self.command[0], &self.source);
        let args = self.command[1..]
            .iter()
            .map(|arg| substitute_source(arg, &self.source))
            .collect::<Vec<_>>();
        Ok((bin, args))
    }
}

pub struct PoseCommandIngestor {
    config: PoseCommandConfig,
    child: Child,
    reader: BufReader<ChildStdout>,
    drained: bool,
}

impl PoseCommandIngestor {
    pub fn spawn(config: PoseCommandConfig) -> anyhow::Result<Self> {
        let (bin, args) = config.resolve_command()?;
        let mut cmd = Command::new(bin);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());
        let mut child = cmd.spawn().context("failed to spawn pose command")?;
        let stdout = child
            .stdout
            .take()
            .context("pose command stdout unavailable")?;
        Ok(Self {
            config,
            child,
            reader: BufReader::new(stdout),
            drained: false,
        })
    }
}

impl StreamIngestor for PoseCommandIngestor {
    fn poll(&mut self) -> anyhow::Result<Option<StreamEnvelope>> {
        if self.drained {
            return Ok(None);
        }
        loop {
            let mut line = String::new();
            let bytes = self.reader.read_line(&mut line)?;
            if bytes == 0 {
                self.drained = true;
                return Ok(None);
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut frame: PoseFrame = match serde_json::from_str(trimmed) {
                Ok(frame) => frame,
                Err(err) => {
                    tracing::warn!(
                        target: "w1z4rdv1510n::streaming",
                        error = %err,
                        "pose command emitted invalid JSON"
                    );
                    continue;
                }
            };
            if frame.timestamp.is_none() {
                frame.timestamp = Some(now_timestamp());
            }
            let timestamp = frame.timestamp.unwrap_or_else(now_timestamp);
            let payload = serde_json::to_value(&frame)?;
            let mut metadata = self.config.metadata.clone();
            metadata.insert(
                "ingestor".to_string(),
                Value::String("pose_command".to_string()),
            );
            for (key, value) in &frame.metadata {
                metadata.insert(key.clone(), value.clone());
            }
            let envelope = StreamEnvelope {
                source: StreamSource::PeopleVideo,
                timestamp,
                payload: StreamPayload::Json { value: payload },
                metadata,
            };
            return Ok(Some(envelope));
        }
    }

    fn is_drained(&self) -> bool {
        self.drained
    }
}

impl Drop for PoseCommandIngestor {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

fn substitute_source(value: &str, source: &str) -> String {
    value.replace("{source}", source)
}

fn now_timestamp() -> Timestamp {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Timestamp { unix }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_command_substitutes_source() {
        let config = PoseCommandConfig::new(
            vec![
                "python".to_string(),
                "bridge.py".to_string(),
                "--source".to_string(),
                "{source}".to_string(),
            ],
            "rtsp://cam".to_string(),
        );
        let (bin, args) = config.resolve_command().expect("command");
        assert_eq!(bin, "python");
        assert_eq!(args.last().unwrap(), "rtsp://cam");
    }
}
