use crate::hardware::HardwareBackendType;
use crate::service_api::{JobStatus, PredictRequest, PredictResponse};
use rand::random;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_id: u64,
    pub finished_at_unix: u64,
    pub best_energy: f64,
    pub hardware_backend: HardwareBackendType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredRun {
    pub run_id: u64,
    pub finished_at_unix: u64,
    pub request: PredictRequest,
    pub response: PredictResponse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobFile {
    pub job_id: u64,
    pub status: JobStatus,
    pub attempts: u32,
    pub max_attempts: u32,
    pub enqueued_at_unix: u64,
    pub updated_at_unix: u64,
    pub last_error: Option<String>,
    pub request: PredictRequest,
    pub response: Option<PredictResponse>,
    pub run_id: Option<u64>,
}

pub struct ServiceStorage {
    root: PathBuf,
    queue_root: PathBuf,
}

impl ServiceStorage {
    pub fn new<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        let queue_root = root.join("queue");
        fs::create_dir_all(&queue_root)?;
        let storage = Self { root, queue_root };
        storage.requeue_stale_running()?;
        Ok(storage)
    }

    pub fn persist_run(&self, run_id: u64, request: &PredictRequest, response: &PredictResponse) {
        let finished_at_unix = response.telemetry.completed_at_unix;
        let stored = StoredRun {
            run_id,
            finished_at_unix,
            request: request.clone(),
            response: response.clone(),
        };
        let path = self.run_path(run_id, finished_at_unix);
        match serde_json::to_vec_pretty(&stored) {
            Ok(data) => {
                if let Err(err) = fs::write(&path, data) {
                    tracing::warn!(
                        target: "w1z4rdv1510n::service_storage",
                        run_id,
                        error = %err,
                        "failed to write run file"
                    );
                }
            }
            Err(err) => {
                tracing::warn!(
                    target: "w1z4rdv1510n::service_storage",
                    run_id,
                    error = %err,
                    "failed to serialize run"
                );
            }
        }
    }

    pub fn list_runs(&self, limit: usize) -> anyhow::Result<Vec<RunRecord>> {
        let mut records = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                continue;
            }
            if !entry
                .path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
            {
                continue;
            }
            if let Ok(stored) = self.read_run_file(entry.path()) {
                records.push(RunRecord {
                    run_id: stored.run_id,
                    finished_at_unix: stored.finished_at_unix,
                    best_energy: stored.response.telemetry.best_energy,
                    hardware_backend: stored.response.telemetry.hardware_backend,
                });
            }
        }
        records.sort_by_key(|rec| std::cmp::Reverse(rec.finished_at_unix));
        if records.len() > limit && limit > 0 {
            records.truncate(limit);
        }
        Ok(records)
    }

    pub fn load_run(&self, run_id: u64) -> anyhow::Result<StoredRun> {
        let path = self.find_run_path(run_id)?;
        self.read_run_file(path)
    }

    pub fn enqueue_job(
        &self,
        request: &PredictRequest,
        max_attempts: u32,
    ) -> anyhow::Result<JobFile> {
        let now = now_unix();
        let job = JobFile {
            job_id: generate_job_id(),
            status: JobStatus::Queued,
            attempts: 0,
            max_attempts,
            enqueued_at_unix: now,
            updated_at_unix: now,
            last_error: None,
            request: request.clone(),
            response: None,
            run_id: None,
        };
        self.write_job(&job)?;
        Ok(job)
    }

    pub fn load_job(&self, job_id: u64) -> anyhow::Result<JobFile> {
        let path = self.find_job_path(job_id)?;
        self.read_job_file(path)
    }

    pub fn write_job(&self, job: &JobFile) -> anyhow::Result<()> {
        let path = self.queue_path(job.job_id);
        let data = serde_json::to_vec_pretty(job)?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn list_jobs(&self, limit: usize) -> anyhow::Result<Vec<JobFile>> {
        let mut jobs = Vec::new();
        for entry in fs::read_dir(&self.queue_root)? {
            let entry = entry?;
            if !entry
                .path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
            {
                continue;
            }
            if let Ok(job) = self.read_job_file(entry.path()) {
                jobs.push(job);
            }
        }
        jobs.sort_by_key(|job| std::cmp::Reverse(job.enqueued_at_unix));
        if limit > 0 && jobs.len() > limit {
            jobs.truncate(limit);
        }
        Ok(jobs)
    }

    pub fn claim_next_job(&self) -> anyhow::Result<Option<JobFile>> {
        let mut queued: BTreeMap<u64, JobFile> = BTreeMap::new();
        for entry in fs::read_dir(&self.queue_root)? {
            let entry = entry?;
            if !entry
                .path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
            {
                continue;
            }
            if let Ok(job) = self.read_job_file(entry.path()) {
                if matches!(job.status, JobStatus::Queued | JobStatus::Retrying) {
                    queued.insert(job.enqueued_at_unix, job);
                }
            }
        }
        Ok(queued.into_values().next())
    }

    pub fn requeue_stale_running(&self) -> anyhow::Result<()> {
        for entry in fs::read_dir(&self.queue_root)? {
            let entry = entry?;
            if !entry
                .path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
            {
                continue;
            }
            let mut job = match self.read_job_file(entry.path()) {
                Ok(job) => job,
                Err(_) => continue,
            };
            if matches!(job.status, JobStatus::Running) {
                job.status = JobStatus::Queued;
                job.last_error = Some("requeued after restart".into());
                job.updated_at_unix = now_unix();
                self.write_job(&job)?;
            }
        }
        Ok(())
    }

    fn read_run_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<StoredRun> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }

    fn run_path(&self, run_id: u64, finished_at_unix: u64) -> PathBuf {
        self.root
            .join(format!("run_{run_id}_{finished_at_unix}.json"))
    }

    fn find_run_path(&self, run_id: u64) -> anyhow::Result<PathBuf> {
        let mut latest: Option<(u64, PathBuf)> = None;
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                continue;
            }
            let path = entry.path();
            if !path.extension().map(|ext| ext == "json").unwrap_or(false) {
                continue;
            }
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();
            if name == format!("run_{run_id}.json") {
                return Ok(path);
            }
            if let Some(rest) = name.strip_prefix(&format!("run_{run_id}_")) {
                if let Some(ts_str) = rest.strip_suffix(".json") {
                    if let Ok(ts) = ts_str.parse::<u64>() {
                        match &latest {
                            Some((best_ts, _)) if *best_ts >= ts => {}
                            _ => latest = Some((ts, path.clone())),
                        }
                    }
                }
            }
        }
        if let Some((_, path)) = latest {
            Ok(path)
        } else {
            anyhow::bail!("run {run_id} not found")
        }
    }

    fn queue_path(&self, job_id: u64) -> PathBuf {
        self.queue_root.join(format!("job_{job_id}.json"))
    }

    fn find_job_path(&self, job_id: u64) -> anyhow::Result<PathBuf> {
        let path = self.queue_path(job_id);
        if path.exists() {
            return Ok(path);
        }
        anyhow::bail!("job {job_id} not found")
    }

    fn read_job_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<JobFile> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RunConfig;
    use crate::hardware::HardwareBackendType;
    use crate::random::RandomProviderDescriptor;
    use crate::random::RandomProviderType;
    use crate::results::{Diagnostics, Results};
    use crate::schema::{DynamicState, EnvironmentSnapshot, Timestamp};
    use crate::service_api::{PredictRequest, PredictResponse, Telemetry};
    use serde_json::Value as JsonValue;
    use serde_json::json;
    use tempfile::tempdir;

    fn sample_config() -> RunConfig {
        let config_json: JsonValue = json!({
            "snapshot_file": "data/example.json",
            "t_end": { "unix": 0 }
        });
        serde_json::from_value(config_json).unwrap()
    }

    fn sample_request() -> PredictRequest {
        PredictRequest {
            config: sample_config(),
            snapshot: EnvironmentSnapshot {
                timestamp: Timestamp { unix: 0 },
                bounds: Default::default(),
                symbols: vec![],
                metadata: Default::default(),
                stack_history: Vec::new(),
            },
        }
    }

    fn sample_response(run_id: u64) -> PredictResponse {
        PredictResponse {
            results: Results {
                best_state: DynamicState::default(),
                best_energy: -1.0,
                clusters: vec![],
                diagnostics: Diagnostics {
                    energy_trace: vec![1.0],
                    diversity_metric: 0.0,
                    best_state_breakdown: None,
                    path_report: None,
                },
            },
            telemetry: Telemetry {
                run_id,
                random_provider: RandomProviderDescriptor {
                    provider: RandomProviderType::OsEntropy,
                    deterministic: false,
                    seed: None,
                },
                hardware_backend: HardwareBackendType::Cpu,
                energy_trace: vec![1.0],
                acceptance_ratio: Some(0.5),
                best_energy: -1.0,
                completed_at_unix: 123,
            },
        }
    }

    #[test]
    fn persist_and_load_round_trip() {
        let dir = tempdir().unwrap();
        let storage = ServiceStorage::new(dir.path()).unwrap();
        let request = sample_request();
        let response = sample_response(1);
        storage.persist_run(1, &request, &response);

        let stored = storage.load_run(1).unwrap();
        assert_eq!(stored.run_id, 1);
        assert_eq!(stored.response.telemetry.best_energy, -1.0);

        let records = storage.list_runs(10).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].run_id, 1);
    }

    #[test]
    fn enqueue_and_requeue_job() {
        let dir = tempdir().unwrap();
        let storage = ServiceStorage::new(dir.path()).unwrap();
        let request = sample_request();
        let job = storage.enqueue_job(&request, 2).unwrap();
        let loaded = storage.load_job(job.job_id).unwrap();
        assert_eq!(loaded.status, JobStatus::Queued);
        let mut running = loaded.clone();
        running.status = JobStatus::Running;
        storage.write_job(&running).unwrap();
        storage.requeue_stale_running().unwrap();
        let reloaded = storage.load_job(job.job_id).unwrap();
        assert!(matches!(reloaded.status, JobStatus::Queued));
        assert!(reloaded.last_error.is_some());
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn generate_job_id() -> u64 {
    let now = now_unix();
    let rand: u32 = random();
    (now << 32) ^ rand as u64
}
