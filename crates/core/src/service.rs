//! Neuro API service — embeddable axum router.
//!
//! Call [`run`] from any binary or tokio task; it binds the port and returns
//! a [`ServiceHandle`] the caller can use to read live stats for a dashboard.

use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::{Component, Path as FsPath};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{Duration, sleep, timeout};

use crate::hardware::HardwareBackendType;
use crate::neuro::{
    CrossStreamActivation, NeuroRuntimeConfig, NeuroRuntimeHandle, SequenceReconstruction,
};
use crate::orchestrator::{RunOutcome, run_with_snapshot};
use crate::schema::EnvironmentSnapshot;
use crate::service_api::{
    ErrorResponse, JobDetailResponse, JobStatus, JobStatusResponse, PredictJobResponse,
    PredictRequest, PredictResponse, Telemetry,
};
use crate::service_storage::{JobFile, RunRecord, ServiceStorage, StoredRun};
use crate::telemetry::{HealthReport, ServiceMetrics};

const MAX_REQUEST_BYTES: usize = 10 * 1024 * 1024;
const MAX_PARTICLES:     usize = 16_384;
const MAX_ITERATIONS:    usize = 200_000;
const MAX_STACK_HISTORY: usize = 4_096;
const ALLOWED_PATH_ROOT: &str  = "logs";
const MAX_CONCURRENT_RUNS: usize = 4;
const MAX_JOB_ATTEMPTS:  u32   = 3;
const INLINE_WAIT_SECS:  u64   = 2;
const MAX_RUN_DURATION_SECS: u64 = 600;

// ── Public handle returned to callers ─────────────────────────────────────────

/// Cheap clone — everything behind Arc.
#[derive(Clone)]
pub struct ServiceHandle {
    pub neuro:   NeuroRuntimeHandle,
    pub metrics: ServiceMetrics,
}

impl ServiceHandle {
    pub fn health(&self) -> HealthReport {
        self.metrics.report()
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Start the neuro API service on `addr` in a background tokio task.
/// Returns a [`ServiceHandle`] so callers can read live metrics / neuro state.
pub async fn run(addr: SocketAddr, storage_path: &str) -> anyhow::Result<ServiceHandle> {
    let storage   = ServiceStorage::new(storage_path)?;
    let app_state = Arc::new(AppState::new(storage));
    let handle    = ServiceHandle {
        neuro:   app_state.neuro.clone(),
        metrics: app_state.metrics.clone(),
    };

    let app = build_router(app_state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(target: "w1z4rdv1510n::service", %addr, "neuro API listening");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app.into_make_service()).await {
            tracing::error!(target: "w1z4rdv1510n::service", "server error: {e}");
        }
    });

    Ok(handle)
}

// ── Router ────────────────────────────────────────────────────────────────────

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/predict",           post(predict_handler))
        .route("/jobs",              get(list_jobs_handler))
        .route("/jobs/:job_id",      get(get_job_handler))
        .route("/runs",              get(list_runs_handler))
        .route("/runs/:run_id",      get(get_run_handler))
        .route("/healthz",           get(health_handler))
        .route("/readyz",            get(ready_handler))
        .route("/neuro/train",       post(neuro_train_handler))
        .route("/neuro/activate",    post(neuro_activate_handler))
        .route("/neuro/reconstruct", post(neuro_reconstruct_handler))
        .route("/neuro/snapshot",    get(neuro_snapshot_handler))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state)
}

// ── AppState ──────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    metrics: ServiceMetrics,
    storage: Arc<ServiceStorage>,
    limiter: Arc<Semaphore>,
    neuro:   NeuroRuntimeHandle,
}

impl AppState {
    fn new(storage: ServiceStorage) -> Self {
        let empty_snap = EnvironmentSnapshot {
            timestamp:     crate::schema::Timestamp { unix: 0 },
            bounds:        std::collections::HashMap::new(),
            symbols:       Vec::new(),
            metadata:      std::collections::HashMap::new(),
            stack_history: Vec::new(),
        };
        let cfg   = NeuroRuntimeConfig { enabled: true, min_activation: 0.05, ..Default::default() };
        let neuro = Arc::new(crate::neuro::NeuroRuntime::new(&empty_snap, cfg));
        Self {
            metrics: ServiceMetrics::new(),
            storage: Arc::new(storage),
            limiter: Arc::new(Semaphore::new(MAX_CONCURRENT_RUNS)),
            neuro,
        }
    }
}

// ── Request / response types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct CrossStreamRequest {
    input_labels:  Vec<String>,
    target_stream: String,
    #[serde(default = "default_hops")]
    hops: usize,
}

#[derive(Deserialize)]
struct ReconstructRequest {
    frames:        Vec<Vec<String>>,
    target_stream: String,
    #[serde(default = "default_hops")]
    hops:  usize,
    #[serde(default = "default_carry")]
    carry: f32,
}

#[derive(Deserialize)]
struct NeuroTrainRequest {
    snapshot: EnvironmentSnapshot,
}

#[derive(Serialize)]
struct CrossStreamResponse {
    activations: Vec<CrossStreamActivation>,
}

#[derive(Deserialize)]
struct ListRunsParams { limit: Option<usize> }

fn default_hops()  -> usize { 3 }
fn default_carry() -> f32   { 0.25 }

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthReport> {
    Json(state.metrics.report())
}

async fn ready_handler(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthReport>) {
    let r = state.metrics.report();
    if r.ready { (StatusCode::OK, Json(r)) } else { (StatusCode::SERVICE_UNAVAILABLE, Json(r)) }
}

async fn list_runs_handler(
    State(state): State<Arc<AppState>>,
    Query(p): Query<ListRunsParams>,
) -> Result<Json<Vec<RunRecord>>, (StatusCode, Json<ErrorResponse>)> {
    state.storage.list_runs(p.limit.unwrap_or(20)).map(Json).map_err(svc_err)
}

async fn get_run_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(run_id): AxumPath<u64>,
) -> Result<Json<StoredRun>, (StatusCode, Json<ErrorResponse>)> {
    state.storage.load_run(run_id).map(Json).map_err(not_found)
}

async fn list_jobs_handler(
    State(state): State<Arc<AppState>>,
    Query(p): Query<ListRunsParams>,
) -> Result<Json<Vec<JobStatusResponse>>, (StatusCode, Json<ErrorResponse>)> {
    state.storage.list_jobs(p.limit.unwrap_or(50))
        .map(|jobs| Json(jobs.iter().map(job_status_response).collect()))
        .map_err(svc_err)
}

async fn get_job_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(job_id): AxumPath<u64>,
) -> Result<Json<JobDetailResponse>, (StatusCode, Json<ErrorResponse>)> {
    let job = state.storage.load_job(job_id).map_err(not_found)?;
    Ok(Json(JobDetailResponse { job: job_status_response(&job), response: job.response.clone() }))
}

async fn neuro_train_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<NeuroTrainRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    state.neuro.observe_snapshot(&req.snapshot);
    (StatusCode::OK, Json(serde_json::json!({"status":"ok"})))
}

async fn neuro_activate_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CrossStreamRequest>,
) -> (StatusCode, Json<CrossStreamResponse>) {
    let activations = state.neuro.cross_stream_activate(&req.input_labels, &req.target_stream, req.hops);
    (StatusCode::OK, Json(CrossStreamResponse { activations }))
}

async fn neuro_reconstruct_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ReconstructRequest>,
) -> (StatusCode, Json<SequenceReconstruction>) {
    let result = state.neuro.reconstruct_sequence(&req.frames, &req.target_stream, req.hops, req.carry);
    (StatusCode::OK, Json(result))
}

async fn neuro_snapshot_handler(
    State(state): State<Arc<AppState>>,
) -> (StatusCode, Json<serde_json::Value>) {
    match serde_json::to_value(&state.neuro.snapshot()) {
        Ok(v)  => (StatusCode::OK, Json(v)),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": e.to_string()}))),
    }
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<(StatusCode, Json<PredictJobResponse>), (StatusCode, Json<ErrorResponse>)> {
    let mut guard = state.metrics.start_request();
    let run_id    = guard.run_id();
    tracing::info!(target: "w1z4rdv1510n::service", run_id, "predict request");

    if let Err(msg) = validate_request(&request) {
        guard.mark_error(msg.clone());
        return Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })));
    }
    let job = state.storage.enqueue_job(&request, MAX_JOB_ATTEMPTS).map_err(|e| {
        guard.mark_error(e.to_string());
        (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "failed to enqueue job".into() }))
    })?;
    spawn_job_worker(state.clone(), job.job_id);

    match wait_for_job_completion(&state.storage, job.job_id).await {
        Some(Ok(resp)) => {
            guard.mark_success(resp.telemetry.hardware_backend.clone(), resp.telemetry.best_energy, resp.telemetry.acceptance_ratio);
            let status = state.storage.load_job(job.job_id).map(|j| job_status_response(&j))
                .unwrap_or_else(|_| quick_status(job.job_id, JobStatus::Succeeded));
            Ok((StatusCode::OK, Json(PredictJobResponse { job: status, result: Some(resp) })))
        }
        Some(Err(msg)) => {
            guard.mark_error(msg.clone());
            Err((StatusCode::BAD_REQUEST, Json(ErrorResponse { error: msg })))
        }
        None => {
            guard.mark_success(HardwareBackendType::Cpu, 0.0, None);
            let status = state.storage.load_job(job.job_id).map(|j| job_status_response(&j))
                .unwrap_or_else(|_| quick_status(job.job_id, JobStatus::Queued));
            Ok((StatusCode::ACCEPTED, Json(PredictJobResponse { job: status, result: None })))
        }
    }
}

// ── Worker / helpers ──────────────────────────────────────────────────────────

fn spawn_job_worker(state: Arc<AppState>, job_id: u64) {
    tokio::spawn(async move {
        let permit = state.limiter.clone().acquire_owned().await.ok();
        let mut job = match state.storage.load_job(job_id) {
            Ok(j) => j, Err(_) => return,
        };
        if job.attempts >= job.max_attempts && !matches!(job.status, JobStatus::Succeeded) {
            job.status = JobStatus::Failed;
            job.last_error.get_or_insert_with(|| "max attempts reached".into());
            job.updated_at_unix = unix_now();
            let _ = state.storage.write_job(&job);
            return;
        }
        job.status = if job.attempts > 0 { JobStatus::Retrying } else { JobStatus::Running };
        job.attempts += 1;
        job.updated_at_unix = unix_now();
        job.last_error = None;
        if state.storage.write_job(&job).is_err() { return; }

        let req     = job.request.clone();
        let storage = state.storage.clone();
        let outcome = timeout(
            Duration::from_secs(MAX_RUN_DURATION_SECS),
            tokio::task::spawn_blocking(move || run_with_snapshot(req.snapshot.clone(), req.config.clone())),
        ).await;

        match outcome {
            Ok(Ok(Ok(RunOutcome { results, random_provider, hardware_backend, acceptance_ratio }))) => {
                let tel = Telemetry {
                    run_id: job.job_id, random_provider,
                    hardware_backend: hardware_backend.clone(),
                    energy_trace:     results.diagnostics.energy_trace.clone(),
                    acceptance_ratio,
                    best_energy:      results.best_energy,
                    completed_at_unix: unix_now(),
                };
                let resp = PredictResponse { results, telemetry: tel };
                storage.persist_run(job.job_id, &job.request, &resp);
                job.response = Some(resp);
                job.status   = JobStatus::Succeeded;
                job.run_id   = Some(job.job_id);
            }
            Ok(Ok(Err(e))) => {
                job.last_error = Some(e.to_string());
                job.status = if job.attempts >= job.max_attempts { JobStatus::Failed } else { JobStatus::Retrying };
            }
            Ok(Err(join_err)) => {
                job.last_error = Some(join_err.to_string());
                job.status = if job.attempts >= job.max_attempts { JobStatus::Failed } else { JobStatus::Retrying };
            }
            Err(_) => {
                job.last_error = Some("job timed out".into());
                job.status = JobStatus::Failed;
            }
        }
        job.updated_at_unix = unix_now();
        drop(permit);
        let _ = state.storage.write_job(&job);
    });
}

async fn wait_for_job_completion(storage: &Arc<ServiceStorage>, job_id: u64) -> Option<Result<PredictResponse, String>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(INLINE_WAIT_SECS);
    loop {
        match storage.load_job(job_id) {
            Ok(job) => match job.status {
                JobStatus::Succeeded => if let Some(r) = job.response.clone() { return Some(Ok(r)); }
                JobStatus::Failed    => return Some(Err(job.last_error.unwrap_or_else(|| "failed".into()))),
                _                    => {}
            },
            Err(e)  => return Some(Err(e.to_string())),
        }
        if std::time::Instant::now() >= deadline { return None; }
        sleep(Duration::from_millis(100)).await;
    }
}

fn validate_request(r: &PredictRequest) -> Result<(), String> {
    if r.config.n_particles > MAX_PARTICLES {
        return Err(format!("n_particles {} exceeds limit {MAX_PARTICLES}", r.config.n_particles));
    }
    if r.config.schedule.n_iterations > MAX_ITERATIONS {
        return Err(format!("n_iterations {} exceeds limit {MAX_ITERATIONS}", r.config.schedule.n_iterations));
    }
    if r.snapshot.stack_history.len() > MAX_STACK_HISTORY {
        return Err(format!("stack_history {} exceeds limit {MAX_STACK_HISTORY}", r.snapshot.stack_history.len()));
    }
    check_paths(&r.config).map_err(|e| e.to_string())?;
    r.config.validate().map_err(|e| e.to_string())
}

fn check_paths(cfg: &crate::config::RunConfig) -> anyhow::Result<()> {
    for path in [
        cfg.logging.log_path.as_ref(),
        cfg.logging.live_frame_path.as_ref(),
        cfg.logging.live_neuro_path.as_ref(),
        cfg.output.output_path.as_ref(),
        cfg.output.summary_path.as_ref(),
    ].into_iter().flatten() {
        ensure_safe_path("path", path)?;
    }
    Ok(())
}

fn ensure_safe_path(label: &str, path: &FsPath) -> anyhow::Result<()> {
    if path.is_absolute() { anyhow::bail!("{label} must be relative"); }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        anyhow::bail!("{label} must not contain '..'");
    }
    match path.components().next() {
        Some(Component::Normal(r)) if r == ALLOWED_PATH_ROOT => Ok(()),
        _ => anyhow::bail!("{label} must reside under {ALLOWED_PATH_ROOT}/"),
    }
}

fn job_status_response(job: &JobFile) -> JobStatusResponse {
    JobStatusResponse {
        job_id:          job.job_id,
        status:          job.status.clone(),
        attempts:        job.attempts,
        max_attempts:    job.max_attempts,
        enqueued_at_unix: job.enqueued_at_unix,
        updated_at_unix: job.updated_at_unix,
        last_error:      job.last_error.clone(),
        run_id:          job.run_id,
    }
}

fn quick_status(job_id: u64, status: JobStatus) -> JobStatusResponse {
    let t = unix_now();
    JobStatusResponse { job_id, status, attempts: 0, max_attempts: MAX_JOB_ATTEMPTS,
        enqueued_at_unix: t, updated_at_unix: t, last_error: None, run_id: None }
}

fn svc_err(e: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: e.to_string() }))
}

fn not_found(e: anyhow::Error) -> (StatusCode, Json<ErrorResponse>) {
    (StatusCode::NOT_FOUND, Json(ErrorResponse { error: e.to_string() }))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}
