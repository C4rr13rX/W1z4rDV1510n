use crate::config::LoggingConfig;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::OnceLock;
use tracing::info;
use tracing_appender::{non_blocking, non_blocking::WorkerGuard};
use tracing_subscriber::{EnvFilter, fmt};

static LOGGING_INIT: OnceLock<()> = OnceLock::new();
static LOG_GUARD: OnceLock<Option<WorkerGuard>> = OnceLock::new();

/// Initialize global tracing subscriber according to `LoggingConfig`.
/// Subsequent calls are no-ops so external callers can safely invoke this multiple times.
pub fn init_logging(config: &LoggingConfig) -> Result<()> {
    if LOGGING_INIT.get().is_some() {
        return Ok(());
    }

    let level_override = std::env::var("SIMFUTURES_LOG").ok();
    let level = level_override
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| config.log_level.clone());
    let env_filter = EnvFilter::try_new(level.clone()).unwrap_or_else(|_| EnvFilter::new("info"));

    let base_builder = fmt::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_level(true)
        .with_thread_ids(true)
        .with_thread_names(true);

    let mut emit_json = config.json;
    let mut guard: Option<WorkerGuard> = None;
    let init_result = if let Some(path) = &config.log_path {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        std::fs::create_dir_all(dir)
            .with_context(|| format!("failed to create log directory {:?}", dir))?;
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("failed to open log file {:?}", path))?;
        let (writer, file_guard) = non_blocking(file);
        guard = Some(file_guard);
        if !config.json {
            emit_json = true;
        }
        let builder = base_builder.with_writer(writer);
        if emit_json {
            builder.json().try_init()
        } else {
            builder.compact().try_init()
        }
    } else {
        if emit_json {
            base_builder.json().try_init()
        } else {
            base_builder.compact().try_init()
        }
    };
    init_result.map_err(|err| anyhow::anyhow!("failed to initialize logging: {err}"))?;
    LOGGING_INIT.set(()).ok();
    LOG_GUARD.set(guard).ok();

    info!(
        target: "w1z4rdv1510n::logging",
        level = level.as_str(),
        json = emit_json,
        log_path = ?config.log_path,
        "logging initialized"
    );
    Ok(())
}
