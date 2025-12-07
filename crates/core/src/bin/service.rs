use axum::{
    Json, Router,
    extract::{Path, Query, State},
    http::StatusCode,
    routing::{get, post},
};
use serde::Deserialize;
use std::net::SocketAddr;
use std::sync::Arc;
use w1z4rdv1510n::orchestrator::{RunOutcome, run_with_snapshot};
use w1z4rdv1510n::service_api::{ErrorResponse, PredictRequest, PredictResponse, Telemetry};
use w1z4rdv1510n::service_storage::{RunRecord, ServiceStorage, StoredRun};
use w1z4rdv1510n::telemetry::{HealthReport, ServiceMetrics};

#[derive(Clone)]
struct AppState {
    metrics: ServiceMetrics,
    storage: Arc<ServiceStorage>,
}

impl AppState {
    fn new(storage: ServiceStorage) -> Self {
        Self {
            metrics: ServiceMetrics::new(),
            storage: Arc::new(storage),
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
        .route("/runs", get(list_runs_handler))
        .route("/runs/:run_id", get(get_run_handler))
        .route("/healthz", get(health_handler))
        .route("/readyz", get(ready_handler))
        .with_state(app_state);

    tracing::info!(target: "w1z4rdv1510n::service", %addr, "listening for requests");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut request_guard = state.metrics.start_request();
    let run_id = request_guard.run_id();
    tracing::info!(
        target: "w1z4rdv1510n::service",
        run_id,
        "predict request received"
    );
    match run_with_snapshot(request.snapshot.clone(), request.config.clone()) {
        Ok(RunOutcome {
            results,
            random_provider,
            hardware_backend,
            acceptance_ratio,
        }) => {
            let backend = hardware_backend.clone();
            request_guard.mark_success(backend.clone(), results.best_energy, acceptance_ratio);
            tracing::info!(
                target: "w1z4rdv1510n::service",
                run_id,
                backend = ?backend,
                best_energy = results.best_energy,
                acceptance_ratio = acceptance_ratio,
                "predict request completed"
            );
            let telemetry = Telemetry {
                run_id,
                random_provider,
                hardware_backend: backend,
                energy_trace: results.diagnostics.energy_trace.clone(),
                acceptance_ratio,
                best_energy: results.best_energy,
                completed_at_unix: current_unix(),
            };
            let response = PredictResponse { results, telemetry };
            state.storage.persist_run(run_id, &request, &response);
            Ok(Json(response))
        }
        Err(err) => {
            let message = err.to_string();
            request_guard.mark_error(message.clone());
            tracing::error!(
                target: "w1z4rdv1510n::service",
                run_id,
                error = %message,
                "predict request failed"
            );
            Err((
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse { error: message }),
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
    Path(run_id): Path<u64>,
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

fn current_unix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
