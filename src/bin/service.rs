use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use w1z4rdv1510n::config::RunConfig;
use w1z4rdv1510n::hardware::HardwareBackendType;
use w1z4rdv1510n::orchestrator::{RunOutcome, run_with_snapshot};
use w1z4rdv1510n::random::RandomProviderDescriptor;
use w1z4rdv1510n::results::Results;
use w1z4rdv1510n::schema::EnvironmentSnapshot;

#[derive(Deserialize)]
struct PredictRequest {
    config: RunConfig,
    snapshot: EnvironmentSnapshot,
}

#[derive(Serialize)]
struct Telemetry {
    random_provider: RandomProviderDescriptor,
    hardware_backend: HardwareBackendType,
    energy_trace: Vec<f64>,
    acceptance_ratio: Option<f64>,
    best_energy: f64,
}

#[derive(Serialize)]
struct PredictResponse {
    results: Results,
    telemetry: Telemetry,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Clone, Default)]
struct AppState;

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

    let app_state = AppState;
    let app = Router::new()
        .route("/predict", post(predict_handler))
        .with_state(Arc::new(app_state));

    tracing::info!(target: "w1z4rdv1510n::service", %addr, "listening for requests");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn predict_handler(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
    match run_with_snapshot(request.snapshot, request.config) {
        Ok(RunOutcome {
            results,
            random_provider,
            hardware_backend,
            acceptance_ratio,
        }) => {
            let telemetry = Telemetry {
                random_provider,
                hardware_backend,
                energy_trace: results.diagnostics.energy_trace.clone(),
                acceptance_ratio,
                best_energy: results.best_energy,
            };
            Ok(Json(PredictResponse { results, telemetry }))
        }
        Err(err) => Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )),
    }
}
