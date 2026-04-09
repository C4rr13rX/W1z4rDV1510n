//! Standalone neuro API binary (kept for backwards compatibility).
//! For new deployments use `w1z4rd` which embeds this service.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("W1Z4RDV1510N_SERVICE_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".into())
        .parse()?;
    let storage_path = std::env::var("W1Z4RDV1510N_SERVICE_STORAGE")
        .unwrap_or_else(|_| "logs/service_runs".into());

    tracing::info!(target: "w1z4rdv1510n::service", %addr, "neuro API starting");
    w1z4rdv1510n::service::run(addr, &storage_path).await?;

    // Park the thread — the service runs in background tasks.
    tokio::signal::ctrl_c().await?;
    Ok(())
}
