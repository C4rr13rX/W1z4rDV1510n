use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use w1z4rdv1510n::blockchain::WorkKind;
use w1z4rdv1510n::hardware::HardwareProfile;

const NODE_BYTES_ESTIMATE: usize = 8 * 1024;

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub nodes: usize,
    pub ticks: usize,
    pub max_messages_per_tick: usize,
    pub work_rate: f64,
    pub validation_fail_rate: f64,
    pub fanout: usize,
    pub seed: u64,
    pub throttle_ms: u64,
    pub max_nodes: Option<usize>,
    pub max_queue_depth: usize,
}

#[derive(Debug, Serialize)]
pub struct SimReport {
    pub requested_nodes: usize,
    pub simulated_nodes: usize,
    pub ticks: usize,
    pub messages_enqueued: usize,
    pub messages_processed: usize,
    pub work_submitted: usize,
    pub work_accepted: usize,
    pub work_rejected: usize,
    pub max_queue_depth: usize,
    pub avg_queue_depth: f64,
    pub total_rewards: f64,
    pub limit_reason: Option<String>,
    pub work_kind_counts: HashMap<String, usize>,
}

#[derive(Clone)]
struct WorkSubmission {
    node_id: usize,
    kind: WorkKind,
    score: f64,
}

enum SimMessage {
    Work { to: usize, submission: WorkSubmission },
}

pub fn run_simulation(config: SimConfig) -> SimReport {
    let profile = HardwareProfile::detect();
    let (node_count, limit_reason) = clamp_nodes(config.nodes, &profile, config.max_nodes);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut queue: Vec<SimMessage> = Vec::new();
    let mut balances = vec![0.0_f64; node_count.max(1)];
    let mut work_kind_counts: HashMap<String, usize> = HashMap::new();
    let mut total_queue = 0usize;
    let mut max_queue = 0usize;
    let mut messages_enqueued = 0usize;
    let mut messages_processed = 0usize;
    let mut work_submitted = 0usize;
    let mut work_accepted = 0usize;
    let mut work_rejected = 0usize;

    let ticks = config.ticks.max(1);
    let max_messages = config.max_messages_per_tick.max(1);
    let work_rate = config.work_rate.clamp(0.0, 1.0);
    let fail_rate = config.validation_fail_rate.clamp(0.0, 1.0);
    let fanout = config.fanout.max(1);

    for _ in 0..ticks {
        if node_count > 0 {
            for node_id in 0..node_count {
                if rng.r#gen::<f64>() <= work_rate {
                    let kind = random_work_kind(&mut rng);
                    *work_kind_counts.entry(format!("{kind:?}")).or_insert(0) += 1;
                    let submission = WorkSubmission {
                        node_id,
                        kind,
                        score: rng.gen_range(0.1..1.0),
                    };
                    work_submitted += 1;
                    let added = route_submission(
                        &mut rng,
                        node_count,
                        fanout,
                        submission,
                        &mut queue,
                        config.max_queue_depth,
                    );
                    messages_enqueued += added;
                }
            }
        }

        let mut processed_this_tick = 0usize;
        while processed_this_tick < max_messages && !queue.is_empty() {
            let message = queue.swap_remove(0);
            processed_this_tick += 1;
            messages_processed += 1;
            let SimMessage::Work { submission, .. } = message;
            if rng.r#gen::<f64>() < fail_rate {
                work_rejected += 1;
                continue;
            }
            work_accepted += 1;
            balances[submission.node_id] += submission.score;
        }

        total_queue += queue.len();
        max_queue = max_queue.max(queue.len());
        if config.throttle_ms > 0 {
            thread::sleep(Duration::from_millis(config.throttle_ms));
        }
    }

    let total_rewards: f64 = balances.iter().sum();
    let avg_queue = if ticks > 0 {
        total_queue as f64 / ticks as f64
    } else {
        0.0
    };

    SimReport {
        requested_nodes: config.nodes,
        simulated_nodes: node_count,
        ticks,
        messages_enqueued,
        messages_processed,
        work_submitted,
        work_accepted,
        work_rejected,
        max_queue_depth: max_queue,
        avg_queue_depth: avg_queue,
        total_rewards,
        limit_reason,
        work_kind_counts,
    }
}

fn route_submission(
    rng: &mut StdRng,
    node_count: usize,
    fanout: usize,
    submission: WorkSubmission,
    queue: &mut Vec<SimMessage>,
    max_queue_depth: usize,
) -> usize {
    let mut sent = 0usize;
    while sent < fanout {
        if queue.len() >= max_queue_depth {
            break;
        }
        let to = if node_count <= 1 {
            0
        } else {
            let mut candidate = rng.gen_range(0..node_count);
            if candidate == submission.node_id {
                candidate = (candidate + 1) % node_count;
            }
            candidate
        };
        queue.push(SimMessage::Work { to, submission: submission.clone() });
        sent += 1;
    }
    sent
}

fn random_work_kind(rng: &mut StdRng) -> WorkKind {
    match rng.gen_range(0..6) {
        0 => WorkKind::SensorIngest,
        1 => WorkKind::ComputeTask,
        2 => WorkKind::ModelUpdate,
        3 => WorkKind::CausalDiscovery,
        4 => WorkKind::Forecasting,
        _ => WorkKind::HumanAnnotation,
    }
}

fn clamp_nodes(
    requested: usize,
    profile: &HardwareProfile,
    max_nodes: Option<usize>,
) -> (usize, Option<String>) {
    let cpu_cap = profile.cpu_cores.saturating_mul(2000).max(1);
    let mem_bytes = (profile.total_memory_gb * 1024.0 * 1024.0 * 1024.0) as usize;
    let mem_cap = (mem_bytes / NODE_BYTES_ESTIMATE).max(1);
    let mut cap = cpu_cap.min(mem_cap);
    let mut reason = format!("cpu_cap={cpu_cap}, mem_cap={mem_cap}");
    if let Some(limit) = max_nodes {
        cap = cap.min(limit.max(1));
        reason.push_str(&format!(", max_nodes={limit}"));
    }
    if requested > cap {
        (cap, Some(format!("requested {requested} > cap {cap} ({reason})")))
    } else {
        (requested, None)
    }
}
