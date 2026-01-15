use crate::config::RunConfig;
use crate::hardware::HardwareBackendType;
use crate::random::RandomProviderDescriptor;
use crate::results::Results;
use crate::schema::EnvironmentSnapshot;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum JobStatus {
    Queued,
    Running,
    Retrying,
    Failed,
    Succeeded,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStatusResponse {
    pub job_id: u64,
    pub status: JobStatus,
    pub attempts: u32,
    pub max_attempts: u32,
    pub enqueued_at_unix: u64,
    pub updated_at_unix: u64,
    pub last_error: Option<String>,
    pub run_id: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobDetailResponse {
    pub job: JobStatusResponse,
    pub response: Option<PredictResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictJobResponse {
    pub job: JobStatusResponse,
    pub result: Option<PredictResponse>,
}
