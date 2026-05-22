//! HTTP-based `RemoteTransport` impl for §18.12 step 3.
//!
//! Lives in the brain_server binary tree (not the brain crate) so the
//! brain crate stays HTTP-client-free.  The brain crate defines the
//! `RemoteTransport` trait (`crates/brain/src/store/neuron_store.rs`);
//! this module supplies the production impl using reqwest's blocking
//! client.
//!
//! # Wire format
//!
//! Mirrors the shard endpoints defined in brain_server.rs:
//! - Fetch:  `GET  http://{peer}/shard/neuron/{pool_id}/{neuron_id}`
//!           → `application/octet-stream` body = bincode(Neuron)
//! - Put:    `POST http://{peer}/shard/put_neuron`
//!           → JSON `{ pool_id, neuron_b64 }`
//!           → JSON `{ inserted: bool }`

use base64::Engine as _;
use std::sync::Arc;
use std::time::Duration;

use w1z4rd_brain::neuron::{Neuron, NeuronId, PoolId};
use w1z4rd_brain::store::RemoteTransport;

/// Sync HTTP-backed `RemoteTransport`.  Uses reqwest's blocking client
/// — appropriate because `NeuronStore` methods are sync; callers that
/// need async wrap in `tokio::task::spawn_blocking`.
pub struct HttpRemoteTransport {
    client:   reqwest::blocking::Client,
    base_url: String,
}

impl HttpRemoteTransport {
    /// Construct against a peer's base URL like
    /// `http://192.168.1.43:8095`.  Strips trailing slash for cleanliness.
    pub fn new(base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') { url.pop(); }
        let client = reqwest::blocking::Client::builder()
            // Per §18.13 honest-tradeoffs: hot-path RPCs should be fast.
            // 30s connect/read budget is a soft ceiling; the brain's
            // operation timing is governed by the substrate-level
            // /sleep/status mechanism rather than HTTP timeouts.
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest blocking client construction (no network)");
        Self { client, base_url: url }
    }

    /// The peer's base URL.  For diagnostics + logging.
    pub fn base_url(&self) -> &str { &self.base_url }
}

impl RemoteTransport for HttpRemoteTransport {
    fn fetch_neuron(&self, pool: PoolId, id: NeuronId) -> Option<Neuron> {
        let url = format!("{}/shard/neuron/{}/{}", self.base_url, pool, id);
        let resp = match self.client.get(&url).send() {
            Ok(r)  => r,
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::fetch_neuron({} {}) network error: {}",
                    pool, id, e,
                );
                return None;
            }
        };
        if !resp.status().is_success() {
            // 404 is expected for unknown ids; other statuses are warnings.
            if resp.status() != reqwest::StatusCode::NOT_FOUND {
                tracing::warn!(
                    "HttpRemoteTransport::fetch_neuron({} {}) status {}",
                    pool, id, resp.status(),
                );
            }
            return None;
        }
        let body = match resp.bytes() {
            Ok(b)  => b,
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::fetch_neuron({} {}) body read: {}",
                    pool, id, e,
                );
                return None;
            }
        };
        match bincode::deserialize::<Neuron>(&body) {
            Ok(n) => Some(n),
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::fetch_neuron({} {}) bincode: {}",
                    pool, id, e,
                );
                None
            }
        }
    }

    fn put_neuron(&self, pool: PoolId, neuron: Neuron) -> bool {
        let body_bytes = match bincode::serialize(&neuron) {
            Ok(b)  => b,
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::put_neuron({} {}) bincode: {}",
                    pool, neuron.id, e,
                );
                return false;
            }
        };
        let neuron_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(&body_bytes);
        let url = format!("{}/shard/put_neuron", self.base_url);
        #[derive(serde::Serialize)]
        struct Req { pool_id: PoolId, neuron_b64: String }
        let req = Req { pool_id: pool, neuron_b64 };
        let resp = match self.client.post(&url).json(&req).send() {
            Ok(r)  => r,
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::put_neuron({} {}) network: {}",
                    pool, neuron.id, e,
                );
                return false;
            }
        };
        if !resp.status().is_success() {
            tracing::warn!(
                "HttpRemoteTransport::put_neuron({} {}) status {}",
                pool, neuron.id, resp.status(),
            );
            return false;
        }
        #[derive(serde::Deserialize)]
        struct Ack { inserted: bool }
        match resp.json::<Ack>() {
            Ok(a)  => a.inserted,
            Err(e) => {
                tracing::warn!(
                    "HttpRemoteTransport::put_neuron({} {}) ack parse: {}",
                    pool, neuron.id, e,
                );
                false
            }
        }
    }
}

/// Convenience: wrap an `HttpRemoteTransport` in `Arc<dyn RemoteTransport>`
/// for use with `RemoteNodeStore::new`.
pub fn arc_transport(base_url: impl Into<String>) -> Arc<dyn RemoteTransport> {
    Arc::new(HttpRemoteTransport::new(base_url))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_trailing_slashes_are_stripped() {
        let t = HttpRemoteTransport::new("http://localhost:8095///");
        assert_eq!(t.base_url(), "http://localhost:8095");
    }

    // The HTTP round-trip is exercised by the integration test in
    // crates/brain/tests/cluster_primitives.rs and by the end-to-end
    // brain_server validation; this file's unit tests are limited to
    // URL handling + sanity.
}
