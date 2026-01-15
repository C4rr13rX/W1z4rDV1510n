use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::{Component, Path as FsPath};
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::time::{Duration, sleep, timeout};
use w1z4rdv1510n::hardware::HardwareBackendType;
use w1z4rdv1510n::orchestrator::{RunOutcome, run_with_snapshot};
use w1z4rdv1510n::service_api::{
    ErrorResponse, JobDetailResponse, JobStatus, JobStatusResponse, PredictJobResponse,
    PredictRequest, PredictResponse, Telemetry,
};
use w1z4rdv1510n::service_storage::{JobFile, RunRecord, ServiceStorage, StoredRun};
use w1z4rdv1510n::telemetry::{HealthReport, ServiceMetrics};

const MAX_REQUEST_BYTES: usize = 10 * 1024 * 1024;
const MAX_PARTICLES: usize = 16_384;
const MAX_ITERATIONS: usize = 200_000;
const MAX_STACK_HISTORY: usize = 4_096;
const ALLOWED_PATH_ROOT: &str = "logs";
const MAX_CONCURRENT_RUNS: usize = 4;
const MAX_JOB_ATTEMPTS: u32 = 3;
const INLINE_WAIT_SECS: u64 = 2;
const MAX_RUN_DURATION_SECS: u64 = 600;

#[derive(Clone)]
struct AppState {
    metrics: ServiceMetrics,
    storage: Arc<ServiceStorage>,
    limiter: Arc<Semaphore>,
}

impl AppState {
    fn new(storage: ServiceStorage) -> Self {
        Self {
            metrics: ServiceMetrics::new(),
            storage: Arc::new(storage),
            limiter: Arc::new(Semaphore::new(MAX_CONCURRENT_RUNS)),
        }
    }
}

#[derive(Deserialize)]
struct ListRunsParams {
    limit: Option<usize>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr: SocketAddr = std::env::var("W1Z4RDV1510N_SERVICE_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()
        .expect("invalid W1Z4RDV1510N_SERVICE_ADDR");

    let storage_path = std::env::var("W1Z4RDV1510N_SERVICE_STORAGE")
        .unwrap_or_else(|_| "logs/service_runs".into());
    let storage = ServiceStorage::new(&storage_path)?;
    tracing::info!(
        target: "w1z4rdv1510n::service",
        storage_path,
        "service storage initialized"
    );
    let app_state = Arc::new(AppState::new(storage));
    let app = Router::new()
        .route("/predict", post(predict_handler))
        .route("/jobs", get(list_jobs_handler))
        .route("/jobs/:job_id", get(get_job_handler))
        .route("/runs", get(list_runs_handler))
        .route("/runs/:run_id", get(get_run_handler))
        .route("/healthz", get(health_handler))
        .route("/readyz", get(ready_handler))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(app_state);

    tracing::info!(target: "w1z4rdv1510n::service", %addr, "listening for requests");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<(StatusCode, Json<PredictJobResponse>), (StatusCode, Json<ErrorResponse>)> {
    let mut request_guard = state.metrics.start_request();
    let run_id = request_guard.run_id();
    tracing::info!(
        target: "w1z4rdv1510n::service",
        run_id,
        "predict request received"
    );

    if let Err(message) = validate_request(&request) {
        request_guard.mark_error(message.clone());
        tracing::warn!(
            target: "w1z4rdv1510n::service",
            run_id,
            error = %message,
            "predict request rejected during validation"
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse { error: message }),
        ));
    }

    let job = state
        .storage
        .enqueue_job(&request, MAX_JOB_ATTEMPTS)
        .map_err(|err| {
            request_guard.mark_error(err.to_string());
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: "failed to enqueue job".into(),
                }),
            )
        })?;
    spawn_job_worker(state.clone(), job.job_id);

    match wait_for_job_completion(&state.storage, job.job_id).await {
        Some(Ok(response)) => {
            let backend = response.telemetry.hardware_backend.clone();
            let best_energy = response.telemetry.best_energy;
            let acceptance = response.telemetry.acceptance_ratio;
            request_guard.mark_success(backend, best_energy, acceptance);
            let job_status = state
                .storage
                .load_job(job.job_id)
                .map(|j| job_status_response(&j))
                .unwrap_or_else(|_| JobStatusResponse {
                    job_id: job.job_id,
                    status: JobStatus::Succeeded,
                    attempts: 1,
                    max_attempts: MAX_JOB_ATTEMPTS,
                    enqueued_at_unix: current_unix(),
                    updated_at_unix: current_unix(),
                    last_error: None,
                    run_id: Some(job.job_id),
                });
            Ok((
                StatusCode::OK,
                Json(PredictJobResponse {
                    job: job_status,
                    result: Some(response),
                }),
            ))
        }
        Some(Err(message)) => {
            request_guard.mark_error(message.clone());
            Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: message }),
            ))
        }
        None => {
            request_guard.mark_success(HardwareBackendType::Cpu, 0.0, None);
            let job_status = state
                .storage
                .load_job(job.job_id)
                .map(|j| job_status_response(&j))
                .unwrap_or_else(|_| JobStatusResponse {
                    job_id: job.job_id,
                    status: JobStatus::Queued,
                    attempts: 0,
                    max_attempts: MAX_JOB_ATTEMPTS,
                    enqueued_at_unix: current_unix(),
                    updated_at_unix: current_unix(),
                    last_error: None,
                    run_id: None,
                });
            Ok((
                StatusCode::ACCEPTED,
                Json(PredictJobResponse {
                    job: job_status,
                    result: None,
                }),
            ))
        }
    }
}

async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthReport> {
    Json(state.metrics.report())
}

async fn ready_handler(State(state): State<Arc<AppState>>) -> (StatusCode, Json<HealthReport>) {
    let report = state.metrics.report();
    if report.ready {
        (StatusCode::OK, Json(report))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, Json(report))
    }
}

async fn list_runs_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListRunsParams>,
) -> Result<Json<Vec<RunRecord>>, (StatusCode, Json<ErrorResponse>)> {
    let limit = params.limit.unwrap_or(20);
    state.storage.list_runs(limit).map(Json).map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })
}

async fn get_run_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(run_id): AxumPath<u64>,
) -> Result<Json<StoredRun>, (StatusCode, Json<ErrorResponse>)> {
    state.storage.load_run(run_id).map(Json).map_err(|err| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })
}

async fn list_jobs_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListRunsParams>,
) -> Result<Json<Vec<JobStatusResponse>>, (StatusCode, Json<ErrorResponse>)> {
    let limit = params.limit.unwrap_or(50);
    state
        .storage
        .list_jobs(limit)
        .map(|jobs| {
            Json(
                jobs.into_iter()
                    .map(|job| job_status_response(&job))
                    .collect(),
            )
        })
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            )
        })
}

async fn get_job_handler(
    State(state): State<Arc<AppState>>,
    AxumPath(job_id): AxumPath<u64>,
) -> Result<Json<JobDetailResponse>, (StatusCode, Json<ErrorResponse>)> {
    let job = state.storage.load_job(job_id).map_err(|err| {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    })?;
    Ok(Json(JobDetailResponse {
        job: job_status_response(&job),
        response: job.response.clone(),
    }))
}

fn current_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn job_status_response(job: &JobFile) -> JobStatusResponse {
    JobStatusResponse {
        job_id: job.job_id,
        status: job.status.clone(),
        attempts: job.attempts,
        max_attempts: job.max_attempts,
        enqueued_at_unix: job.enqueued_at_unix,
        updated_at_unix: job.updated_at_unix,
        last_error: job.last_error.clone(),
        run_id: job.run_id,
    }
}

fn spawn_job_worker(state: Arc<AppState>, job_id: u64) {
    tokio::spawn(async move {
        let permit = match state.limiter.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => return,
        };
        let mut job = match state.storage.load_job(job_id) {
            Ok(job) => job,
            Err(err) => {
                tracing::error!(
                    target: "w1z4rdv1510n::service",
                    job_id,
                    error = %err,
                    "failed to load job for processing"
                );
                return;
            }
        };
        if job.attempts >= job.max_attempts && !matches!(job.status, JobStatus::Succeeded) {
            job.status = JobStatus::Failed;
            job.last_error
                .get_or_insert_with(|| "max attempts reached".into());
            job.updated_at_unix = current_unix();
            let _ = state.storage.write_job(&job);
            return;
        }
        job.status = if job.attempts > 0 {
            JobStatus::Retrying
        } else {
            JobStatus::Running
        };
        job.attempts += 1;
        job.updated_at_unix = current_unix();
        job.last_error = None;
        if let Err(err) = state.storage.write_job(&job) {
            tracing::error!(
                target: "w1z4rdv1510n::service",
                job_id,
                error = %err,
                "failed to mark job running"
            );
            return;
        }
        let request_for_run = job.request.clone();
        let storage = Arc::clone(&state.storage);
        let run_task = tokio::task::spawn_blocking(move || {
            run_with_snapshot(
                request_for_run.snapshot.clone(),
                request_for_run.config.clone(),
            )
        });
        let outcome = timeout(Duration::from_secs(MAX_RUN_DURATION_SECS), run_task).await;
        let mut error_message: Option<String> = None;
        match outcome {
            Ok(join_result) => match join_result {
                Ok(Ok(RunOutcome {
                    results,
                    random_provider,
                    hardware_backend,
                    acceptance_ratio,
                })) => {
                    let telemetry = Telemetry {
                        run_id: job.job_id,
                        random_provider,
                        hardware_backend: hardware_backend.clone(),
                        energy_trace: results.diagnostics.energy_trace.clone(),
                        acceptance_ratio,
                        best_energy: results.best_energy,
                        completed_at_unix: current_unix(),
                    };
                    let response = PredictResponse { results, telemetry };
                    storage.persist_run(job.job_id, &job.request, &response);
                    job.response = Some(response);
                    job.status = JobStatus::Succeeded;
                    job.run_id = Some(job.job_id);
                }
                Ok(Err(err)) => {
                    error_message = Some(err.to_string());
                }
                Err(join_err) => {
                    error_message = Some(join_err.to_string());
                }
            },
            Err(_) => {
                error_message = Some("job execution timed out".into());
            }
        }
        if let Some(err) = error_message {
            job.last_error = Some(err);
            if job.attempts >= job.max_attempts {
                job.status = JobStatus::Failed;
            } else {
                job.status = JobStatus::Retrying;
            }
        }
        job.updated_at_unix = current_unix();
        drop(permit);
        if let Err(err) = state.storage.write_job(&job) {
            tracing::error!(
                target: "w1z4rdv1510n::service",
                job_id,
                error = %err,
                "failed to persist job state after run"
            );
        }
    });
}

async fn wait_for_job_completion(
    storage: &Arc<ServiceStorage>,
    job_id: u64,
) -> Option<Result<PredictResponse, String>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(INLINE_WAIT_SECS);
    loop {
        match storage.load_job(job_id) {
            Ok(job) => match job.status {
                JobStatus::Succeeded => {
                    if let Some(resp) = job.response.clone() {
                        return Some(Ok(resp));
                    }
                }
                JobStatus::Failed => {
                    return Some(Err(job.last_error.unwrap_or_else(|| "job failed".into())));
                }
                _ => {}
            },
            Err(err) => return Some(Err(err.to_string())),
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        sleep(Duration::from_millis(100)).await;
    }
}

fn validate_request(request: &PredictRequest) -> Result<(), String> {
    let cfg = &request.config;
    if cfg.n_particles > MAX_PARTICLES {
        return Err(format!(
            "n_particles {} exceeds service limit ({MAX_PARTICLES})",
            cfg.n_particles
        ));
    }
    if cfg.schedule.n_iterations > MAX_ITERATIONS {
        return Err(format!(
            "schedule.n_iterations {} exceeds service limit ({MAX_ITERATIONS})",
            cfg.schedule.n_iterations
        ));
    }
    if request.snapshot.stack_history.len() > MAX_STACK_HISTORY {
        return Err(format!(
            "stack_history length {} exceeds service limit ({MAX_STACK_HISTORY})",
            request.snapshot.stack_history.len()
        ));
    }
    check_paths(cfg).map_err(|e| e.to_string())?;
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(())
}

fn check_paths(cfg: &w1z4rdv1510n::config::RunConfig) -> anyhow::Result<()> {
    if let Some(path) = cfg.logging.log_path.as_ref() {
        ensure_safe_path("logging.log_path", path)?;
    }
    if let Some(path) = cfg.logging.live_frame_path.as_ref() {
        ensure_safe_path("logging.live_frame_path", path)?;
    }
    if let Some(path) = cfg.logging.live_neuro_path.as_ref() {
        ensure_safe_path("logging.live_neuro_path", path)?;
    }
    if let Some(path) = cfg.output.output_path.as_ref() {
        ensure_safe_path("output.output_path", path)?;
    }
    if let Some(path) = cfg.output.summary_path.as_ref() {
        ensure_safe_path("output.summary_path", path)?;
    }
    Ok(())
}

fn ensure_safe_path(label: &str, path: &FsPath) -> anyhow::Result<()> {
    if path.is_absolute() {
        anyhow::bail!("{label} must be relative and scoped under {ALLOWED_PATH_ROOT}/");
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        anyhow::bail!("{label} must not contain parent directory segments");
    }
    let mut components = path.components();
    match components.next() {
        Some(Component::Normal(root)) if root == ALLOWED_PATH_ROOT => Ok(()),
        _ => anyhow::bail!("{label} must reside under {ALLOWED_PATH_ROOT}/"),
    }
}
