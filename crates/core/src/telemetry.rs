use crate::hardware::HardwareBackendType;
use parking_lot::Mutex;
use serde::Serialize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|dur| dur.as_secs())
        .unwrap_or(0)
}

#[derive(Clone, Serialize)]
pub struct RunSnapshot {
    pub run_id: u64,
    pub backend: Option<HardwareBackendType>,
    pub duration_ms: u128,
    pub finished_at_unix: u64,
    pub best_energy: Option<f64>,
    pub acceptance_ratio: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct ErrorSnapshot {
    pub run_id: u64,
    pub message: String,
    pub finished_at_unix: u64,
    pub duration_ms: u128,
}

#[derive(Clone, Serialize)]
pub struct HealthReport {
    pub status: String,
    pub ready: bool,
    pub uptime_secs: u64,
    pub total_requests: u64,
    pub completed_requests: u64,
    pub inflight_requests: u64,
    pub last_success: Option<RunSnapshot>,
    pub last_error: Option<ErrorSnapshot>,
}

struct ServiceStats {
    start_time: Instant,
    total_requests: u64,
    completed_requests: u64,
    inflight: u64,
    last_success: Option<RunSnapshot>,
    last_error: Option<ErrorSnapshot>,
}

#[derive(Clone)]
pub struct ServiceMetrics {
    inner: Arc<Mutex<ServiceStats>>,
    next_run_id: Arc<AtomicU64>,
}

impl ServiceMetrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ServiceStats {
                start_time: Instant::now(),
                total_requests: 0,
                completed_requests: 0,
                inflight: 0,
                last_success: None,
                last_error: None,
            })),
            next_run_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn start_request(&self) -> RequestHandle {
        let run_id = self.next_run_id.fetch_add(1, Ordering::Relaxed);
        {
            let mut stats = self.inner.lock();
            stats.total_requests += 1;
            stats.inflight += 1;
        }
        RequestHandle {
            metrics: self.clone(),
            run_id,
            started: Instant::now(),
            finished: false,
        }
    }

    fn finish_success(
        &self,
        run_id: u64,
        duration: Duration,
        backend: HardwareBackendType,
        best_energy: f64,
        acceptance_ratio: Option<f64>,
    ) {
        let mut stats = self.inner.lock();
        stats.completed_requests += 1;
        stats.inflight = stats.inflight.saturating_sub(1);
        stats.last_success = Some(RunSnapshot {
            run_id,
            backend: Some(backend),
            duration_ms: duration.as_millis(),
            finished_at_unix: now_unix_secs(),
            best_energy: Some(best_energy),
            acceptance_ratio,
        });
    }

    fn finish_error(&self, run_id: u64, duration: Duration, message: String) {
        let mut stats = self.inner.lock();
        stats.completed_requests += 1;
        stats.inflight = stats.inflight.saturating_sub(1);
        stats.last_error = Some(ErrorSnapshot {
            run_id,
            message,
            finished_at_unix: now_unix_secs(),
            duration_ms: duration.as_millis(),
        });
    }

    fn abandon_request(&self) {
        let mut stats = self.inner.lock();
        stats.inflight = stats.inflight.saturating_sub(1);
    }

    pub fn report(&self) -> HealthReport {
        let stats = self.inner.lock();
        let uptime_secs = stats.start_time.elapsed().as_secs();
        let ready = match (&stats.last_success, &stats.last_error) {
            (Some(success), Some(error)) => success.finished_at_unix >= error.finished_at_unix,
            (Some(_), None) => true,
            (None, _) => false,
        };
        let status = if ready {
            "ok"
        } else if stats.completed_requests == 0 {
            "starting"
        } else {
            "degraded"
        }
        .to_string();
        HealthReport {
            status,
            ready,
            uptime_secs,
            total_requests: stats.total_requests,
            completed_requests: stats.completed_requests,
            inflight_requests: stats.inflight,
            last_success: stats.last_success.clone(),
            last_error: stats.last_error.clone(),
        }
    }
}

pub struct RequestHandle {
    metrics: ServiceMetrics,
    run_id: u64,
    started: Instant,
    finished: bool,
}

impl RequestHandle {
    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn started_at(&self) -> Instant {
        self.started
    }

    pub fn mark_success(
        &mut self,
        backend: HardwareBackendType,
        best_energy: f64,
        acceptance_ratio: Option<f64>,
    ) {
        self.metrics.finish_success(
            self.run_id,
            self.started.elapsed(),
            backend,
            best_energy,
            acceptance_ratio,
        );
        self.finished = true;
    }

    pub fn mark_error(&mut self, message: String) {
        self.metrics
            .finish_error(self.run_id, self.started.elapsed(), message);
        self.finished = true;
    }
}

impl Drop for RequestHandle {
    fn drop(&mut self) {
        if !self.finished {
            self.metrics.abandon_request();
        }
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_requires_success() {
        let metrics = ServiceMetrics::new();
        let mut handle = metrics.start_request();
        assert!(!metrics.report().ready);
        handle.mark_success(HardwareBackendType::Cpu, -1.0, Some(0.5));
        let report = metrics.report();
        assert!(report.ready);
        assert_eq!(report.total_requests, 1);
        assert_eq!(report.completed_requests, 1);
        assert_eq!(report.inflight_requests, 0);
        assert!(report.last_success.is_some());
    }

    #[test]
    fn errors_recorded_and_block_readiness() {
        let metrics = ServiceMetrics::new();
        {
            let mut handle = metrics.start_request();
            handle.mark_error("failure".to_string());
        }
        let report = metrics.report();
        assert_eq!(report.total_requests, 1);
        assert_eq!(report.completed_requests, 1);
        assert!(!report.ready);
        assert!(report.last_error.is_some());

        {
            let mut handle = metrics.start_request();
            handle.mark_success(HardwareBackendType::Cpu, -2.0, None);
        }
        let report = metrics.report();
        assert!(report.ready);
        assert!(report.last_success.is_some());
        assert!(report.last_error.is_some());
    }
}
