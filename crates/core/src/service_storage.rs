use crate::hardware::HardwareBackendType;
use crate::service_api::{PredictRequest, PredictResponse};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

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

pub struct ServiceStorage {
    root: PathBuf,
}

impl ServiceStorage {
    pub fn new<P: AsRef<Path>>(root: P) -> anyhow::Result<Self> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn persist_run(&self, run_id: u64, request: &PredictRequest, response: &PredictResponse) {
        let finished_at_unix = response.telemetry.completed_at_unix;
        let stored = StoredRun {
            run_id,
            finished_at_unix,
            request: request.clone(),
            response: response.clone(),
        };
        let path = self.run_path(run_id);
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
        let path = self.run_path(run_id);
        self.read_run_file(path)
    }

    fn read_run_file<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<StoredRun> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }

    fn run_path(&self, run_id: u64) -> PathBuf {
        self.root.join(format!("run_{run_id}.json"))
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
}
