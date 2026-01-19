use crate::compute::{QuantumExecutor, QuantumJob, QuantumResult};
use crate::config::QuantumEndpointConfig;
use anyhow::{Context, Result};
use base64::Engine;
use reqwest::blocking::Client;
use reqwest::header::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct QuantumRequest {
    kind: crate::compute::ComputeJobKind,
    payload_b64: String,
    timeout_secs: u64,
}

#[derive(Debug, Deserialize)]
struct QuantumResponse {
    payload_b64: String,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

pub struct QuantumHttpExecutor {
    endpoints: Vec<QuantumEndpointConfig>,
    client: Client,
}

impl QuantumHttpExecutor {
    pub fn new(mut endpoints: Vec<QuantumEndpointConfig>) -> Result<Self> {
        endpoints.sort_by_key(|endpoint| endpoint.priority);
        let client = Client::builder()
            .build()
            .context("failed to build quantum HTTP client")?;
        Ok(Self { endpoints, client })
    }

    fn endpoint_auth_header(
        endpoint: &QuantumEndpointConfig,
    ) -> Result<Option<(HeaderName, HeaderValue)>> {
        let Some(env_name) = endpoint
            .auth_env
            .as_ref()
            .map(|val| val.trim())
            .filter(|val| !val.is_empty())
        else {
            return Ok(None);
        };
        let header_name = endpoint.auth_header.trim();
        if header_name.is_empty() {
            return Ok(None);
        }
        let token = std::env::var(env_name)
            .with_context(|| format!("missing quantum auth env var {env_name}"))?;
        let header_value = format!("{}{}", endpoint.auth_prefix, token);
        let name = HeaderName::from_bytes(header_name.as_bytes())
            .context("invalid quantum auth header name")?;
        let value = HeaderValue::from_str(&header_value)
            .context("invalid quantum auth header value")?;
        Ok(Some((name, value)))
    }

    fn submit_to_endpoint(
        &self,
        endpoint: &QuantumEndpointConfig,
        job: &QuantumJob,
        payload_b64: &str,
    ) -> Result<QuantumResult> {
        let timeout_secs = job
            .timeout_secs
            .max(endpoint.timeout_secs)
            .max(1);
        let request = QuantumRequest {
            kind: job.kind,
            payload_b64: payload_b64.to_string(),
            timeout_secs,
        };
        let mut builder = self
            .client
            .post(&endpoint.url)
            .timeout(Duration::from_secs(timeout_secs))
            .json(&request);
        if let Some((name, value)) = Self::endpoint_auth_header(endpoint)? {
            builder = builder.header(name, value);
        }
        let response = builder
            .send()
            .with_context(|| format!("quantum endpoint {} request failed", endpoint.name))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .text()
                .unwrap_or_else(|_| "<failed to read response body>".to_string());
            anyhow::bail!(
                "quantum endpoint {} returned {}: {}",
                endpoint.name,
                status.as_u16(),
                body
            );
        }
        let mut parsed: QuantumResponse = response
            .json()
            .with_context(|| format!("quantum endpoint {} invalid response", endpoint.name))?;
        let payload = base64::engine::general_purpose::STANDARD
            .decode(parsed.payload_b64.as_bytes())
            .context("invalid quantum response payload")?;
        parsed
            .metadata
            .insert("quantum_endpoint".to_string(), endpoint.name.clone());
        if !endpoint.provider.trim().is_empty() {
            parsed
                .metadata
                .insert("quantum_provider".to_string(), endpoint.provider.clone());
        }
        Ok(QuantumResult {
            payload,
            metadata: parsed.metadata,
        })
    }
}

impl QuantumExecutor for QuantumHttpExecutor {
    fn submit(&self, job: QuantumJob) -> Result<QuantumResult> {
        if self.endpoints.is_empty() {
            anyhow::bail!("no quantum endpoints configured");
        }
        let payload_b64 = base64::engine::general_purpose::STANDARD.encode(&job.payload);
        let mut last_err: Option<anyhow::Error> = None;
        for endpoint in &self.endpoints {
            match self.submit_to_endpoint(endpoint, &job, &payload_b64) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("quantum executor failed")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_auth_header_skips_when_not_configured() {
        let endpoint = QuantumEndpointConfig::default();
        let header = QuantumHttpExecutor::endpoint_auth_header(&endpoint).expect("header");
        assert!(header.is_none());
    }

    #[test]
    fn endpoint_auth_header_builds_bearer_header() {
        let env_name = "W1Z4RDV1510N_TEST_QUANTUM_TOKEN";
        unsafe {
            std::env::set_var(env_name, "token123");
        }
        let mut endpoint = QuantumEndpointConfig::default();
        endpoint.auth_env = Some(env_name.to_string());
        let header = QuantumHttpExecutor::endpoint_auth_header(&endpoint)
            .expect("header")
            .expect("expected header");
        assert_eq!(header.0.as_str(), "authorization");
        assert_eq!(header.1.to_str().unwrap_or(""), "Bearer token123");
    }
}
