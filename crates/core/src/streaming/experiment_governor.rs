use crate::compute::{ComputeJobKind, QuantumExecutor, QuantumJob};
use crate::config::{ExperimentGovernorConfig, ExperimentMode};
use crate::schema::Timestamp;
use crate::streaming::behavior::MotifAssignmentAmbiguity;
use crate::streaming::outcome_feedback::OutcomeFeedbackReport;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentGovernorReport {
    pub timestamp: Timestamp,
    pub mode: ExperimentMode,
    pub jobs_last_minute: usize,
    pub motif_ambiguities: usize,
    pub reward: Option<f64>,
    pub ran_motif_assignment: bool,
    pub used_remote: bool,
    pub reason: String,
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MotifAssignmentJobRequest {
    timestamp: Timestamp,
    assignments: Vec<MotifAssignmentJobItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MotifAssignmentJobItem {
    assignment_id: String,
    entity_id: String,
    dtw_threshold: f64,
    best_cost: f64,
    candidates: Vec<MotifAssignmentCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MotifAssignmentCandidate {
    motif_id: String,
    cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MotifAssignmentJobResponse {
    assignments: Vec<MotifAssignmentJobResult>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MotifAssignmentJobResult {
    assignment_id: String,
    probabilities: Vec<f64>,
}

pub struct ExperimentGovernorRuntime {
    config: ExperimentGovernorConfig,
    recent_jobs: VecDeque<i64>,
    last_job_unix: i64,
}

impl ExperimentGovernorRuntime {
    pub fn new(config: ExperimentGovernorConfig) -> Self {
        Self {
            config,
            recent_jobs: VecDeque::new(),
            last_job_unix: i64::MIN / 2,
        }
    }

    pub fn update_config(&mut self, config: ExperimentGovernorConfig) {
        self.config = config;
    }

    pub fn update(
        &mut self,
        timestamp: Timestamp,
        ambiguities: &[MotifAssignmentAmbiguity],
        feedback: Option<&OutcomeFeedbackReport>,
        executor: Option<&dyn QuantumExecutor>,
        timeout_secs: u64,
    ) -> Option<ExperimentGovernorReport> {
        if !self.config.enabled {
            return None;
        }

        self.trim_jobs(timestamp.unix);
        let jobs_last_minute = self.recent_jobs.len();
        let reward = feedback.map(|r| r.reward);

        let mut ran = false;
        let mut used_remote = false;
        let mut metadata: HashMap<String, Value> = HashMap::new();

        let can_run = self.config.mode != ExperimentMode::Off
            && self.within_budget()
            && (timestamp.unix - self.last_job_unix) >= self.config.cooldown_secs.max(0);

        let wants_motif_assignment = self.config.motif_assignment_enabled
            && ambiguities.len() >= self.config.motif_ambiguity_min.max(1)
            && reward
                .map(|val| val <= self.config.reward_trigger)
                .unwrap_or(false);

        let reason = if !can_run {
            "cooldown_or_budget".to_string()
        } else if !wants_motif_assignment {
            "no_trigger".to_string()
        } else if executor.is_none() {
            "no_quantum_executor".to_string()
        } else if self.config.mode == ExperimentMode::Shadow {
            "shadow_would_run_motif_assignment".to_string()
        } else {
            if let Some(exec) = executor {
                if let Ok(response) = submit_motif_assignment(exec, timestamp, ambiguities, timeout_secs) {
                    ran = true;
                    used_remote = true;
                    metadata.insert(
                        "motif_assignment".to_string(),
                        serde_json::to_value(&response).unwrap_or(Value::Null),
                    );
                }
            }
            if ran {
                self.note_job(timestamp.unix);
            }
            "motif_assignment".to_string()
        };


        Some(ExperimentGovernorReport {
            timestamp,
            mode: self.config.mode,
            jobs_last_minute,
            motif_ambiguities: ambiguities.len(),
            reward,
            ran_motif_assignment: ran,
            used_remote,
            reason,
            metadata,
        })
    }

    fn within_budget(&self) -> bool {
        self.recent_jobs.len() < self.config.max_jobs_per_minute as usize
    }

    fn note_job(&mut self, now_unix: i64) {
        self.last_job_unix = now_unix;
        self.recent_jobs.push_back(now_unix);
        self.trim_jobs(now_unix);
    }

    fn trim_jobs(&mut self, now_unix: i64) {
        let cutoff = now_unix.saturating_sub(60);
        while let Some(front) = self.recent_jobs.front().copied() {
            if front >= cutoff {
                break;
            }
            self.recent_jobs.pop_front();
        }
    }
}

fn submit_motif_assignment(
    executor: &dyn QuantumExecutor,
    timestamp: Timestamp,
    ambiguities: &[MotifAssignmentAmbiguity],
    timeout_secs: u64,
) -> anyhow::Result<MotifAssignmentJobResponse> {
    let assignments = ambiguities
        .iter()
        .map(|amb| MotifAssignmentJobItem {
            assignment_id: amb.assignment_id.clone(),
            entity_id: amb.entity_id.clone(),
            dtw_threshold: amb.dtw_threshold,
            best_cost: amb.best_cost,
            candidates: amb
                .candidates
                .iter()
                .map(|c| MotifAssignmentCandidate {
                    motif_id: c.motif_id.clone(),
                    cost: c.cost,
                })
                .collect(),
        })
        .collect();
    let request = MotifAssignmentJobRequest {
        timestamp,
        assignments,
    };
    let payload = serde_json::to_vec(&request)?;
    let job = QuantumJob {
        kind: ComputeJobKind::MotifAssignment,
        payload,
        timeout_secs,
    };
    let result = executor.submit(job).context("motif assignment quantum job failed")?;
    let mut response: MotifAssignmentJobResponse = serde_json::from_slice(&result.payload)
        .context("invalid motif assignment response")?;
    // Merge transport metadata.
    for (k, v) in result.metadata {
        response.metadata.entry(k).or_insert(v);
    }
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop;
    impl QuantumExecutor for Noop {
        fn submit(&self, _job: QuantumJob) -> anyhow::Result<crate::compute::QuantumResult> {
            anyhow::bail!("noop")
        }
    }

    #[test]
    fn governor_respects_budget_and_trigger() {
        let mut cfg = ExperimentGovernorConfig::default();
        cfg.max_jobs_per_minute = 1;
        cfg.cooldown_secs = 0;
        cfg.reward_trigger = 0.0;
        let mut rt = ExperimentGovernorRuntime::new(cfg);
        let now = Timestamp { unix: 100 };
        let amb = MotifAssignmentAmbiguity {
            assignment_id: "a".to_string(),
            entity_id: "e".to_string(),
            timestamp: now,
            dtw_threshold: 2.0,
            best_cost: 1.0,
            candidates: vec![],
        };
        let fb = OutcomeFeedbackReport {
            timestamp: now,
            resolved: 1,
            pending: 0,
            avg_log_loss: 1.0,
            avg_baseline_log_loss: 1.0,
            reward: -0.1,
            slices: vec![],
            metadata: HashMap::new(),
        };
        let ambs = vec![amb.clone(); 10];
        let rep = rt.update(now, &ambs, Some(&fb), Some(&Noop), 1).unwrap();
        assert_eq!(rep.motif_ambiguities, 10);
        // Noop executor fails => ran_motif_assignment remains false, but we still should produce a report.
        assert!(!rep.ran_motif_assignment);
    }
}

