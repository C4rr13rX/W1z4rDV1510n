use crate::config::RunConfig;
use crate::hardware::HardwareBackendType;
use crate::random::RandomProviderDescriptor;
use crate::results::Results;
use crate::schema::EnvironmentSnapshot;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictRequest {
    pub config: RunConfig,
    pub snapshot: EnvironmentSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Telemetry {
    pub run_id: u64,
    pub random_provider: RandomProviderDescriptor,
    pub hardware_backend: HardwareBackendType,
    pub energy_trace: Vec<f64>,
    pub acceptance_ratio: Option<f64>,
    pub best_energy: f64,
    pub completed_at_unix: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictResponse {
    pub results: Results,
    pub telemetry: Telemetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}
