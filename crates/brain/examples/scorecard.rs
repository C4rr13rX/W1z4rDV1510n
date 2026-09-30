//! Scorecard: does the brain stay small while it stays right?
//!
//! One scene world -- rooms full of objects, each object with properties --
//! trained at a given scale, then probed three ways:
//!
//! - recall:      trained facts ("r03 lamp color?" -> "red"). Must stay 100%.
//! - integration: facts never trained but true in the world, derivable by
//!                chaining two trained ones ("r03 lamp on" -> "desk", "r03
//!                desk material" -> "oak", so "r03 lamp on material?" ->
//!                "oak"). Must never drop.
//! - footprint:   neurons, terminals, the largest fan-out on any one neuron
//!                (the hub that cannot leave RAM), and an estimate of
//!                resident bytes. Must stay flat as the world grows.
//!
//! Prints ONE JSON line. Peak process memory is measured from outside by
//! tools/capped.py, which also enforces the cap -- see tools/scorecard.py.
//!
//! Run: `cargo run --release --example scorecard -p w1z4rd-brain -- --scale 4`

use std::time::Instant;
use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS_PER_SCALE: usize = 8;
const EPOCHS: usize = 2;

const OBJECTS: &[&str] = &["bed", "chair", "mirror", "desk", "lamp", "door", "window", "paper"];
const COLORS: &[&str] = &["red", "blue", "green", "white", "black", "grey"];
const MATERIALS: &[&str] = &["oak", "steel", "glass", "cloth", "pine", "brass"];
/// Objects that rest on another object, and what they rest on.
const RESTS_ON: &[(&str, &str)] = &[("lamp", "desk"), ("paper", "desk"), ("mirror", "door")];

/// A question and the answer the world says is true.
struct Probe {
    query: String,
    answer: String,
}

/// The scene world at one scale: trained facts and untrained-but-true probes.
struct SceneWorld {
    facts: Vec<Probe>,
    integration: Vec<Probe>,
}

impl SceneWorld {
    fn new(scale: usize) -> Self {
        let mut facts = Vec::new();
        let mut integration = Vec::new();
        for r in 0..ROOMS_PER_SCALE * scale {
            let room = format!("r{r:03}");
            // Deterministic but varied per room, so answers are not guessable
            // from the object name alone.
            let pick = |list: &[&'static str], salt: usize| list[(r * 7 + salt * 3) % list.len()];
            for (i, obj) in OBJECTS.iter().enumerate() {
                facts.push(Probe { query: format!("{room} {obj} color?"), answer: pick(COLORS, i).into() });
                facts.push(Probe { query: format!("{room} {obj} material?"), answer: pick(MATERIALS, i + 1).into() });
            }
            for (obj, base) in RESTS_ON {
                facts.push(Probe { query: format!("{room} {obj} on?"), answer: (*base).into() });
                let base_idx = OBJECTS.iter().position(|o| o == base).unwrap();
                integration.push(Probe {
                    query: format!("{room} {obj} on material?"),
                    answer: pick(MATERIALS, base_idx + 1).into(),
                });
            }
        }
        Self { facts, integration }
    }
}

/// The brain under test, built the way examples/lab.rs builds it.
struct Subject {
    brain: Brain,
}

impl Subject {
    fn new() -> Self {
        let mut cfg = BrainConfig::default();
        cfg.binding_emergence_threshold = 3;
        cfg.moment_history_window = 256;
        let mut brain = Brain::new(cfg);
        for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
            let mut pc = PoolConfig::defaults(name, id);
            pc.recent_atoms_window = 2048;
            pc.concept_emergence_threshold = 2;
            pc.max_concept_member_count = 64;
            pc.decay_rate = 0.0001;
            pc.prune_floor = 0.005;
            brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
        }
        Self { brain }
    }

    /// Trains the way the node's /brain/pretrain route does: one binding
    /// episode per fact, shuffled each epoch.
    fn train(&mut self, facts: &[Probe]) {
        for epoch in 0..EPOCHS {
            for i in shuffled(facts.len(), epoch) {
                self.brain.pretrain_binding_episode(&[
                    (QUERY_POOL, facts[i].query.as_bytes().to_vec()),
                    (ANSWER_POOL, facts[i].answer.as_bytes().to_vec()),
                ]);
            }
        }
    }

    /// Answers the way the node's authoritative QA path does
    /// (crates/node/src/brain_api.rs): the trained binding first, integrate()
    /// as the fallback, then release whatever the query paged in.
    fn recall(&mut self, query: &str) -> Vec<u8> {
        self.brain.observe(QUERY_POOL, query.as_bytes());
        let legacy = self.brain.integrate(QUERY_POOL, ANSWER_POOL);
        let answer = self.brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).or(legacy.answer);
        let _ = self.brain.finish_read_only_inference();
        answer.unwrap_or_default()
    }

    fn infer(&mut self, query: &str) -> Vec<u8> {
        self.brain.observe(QUERY_POOL, query.as_bytes());
        self.brain
            .integrate_autonomous(QUERY_POOL, ANSWER_POOL, 100.0, 3, 200)
            .answer
            .unwrap_or_default()
    }

    /// (hub fan-out, estimated resident bytes) over every pool.
    fn footprint(&self) -> (usize, usize) {
        let terminal = std::mem::size_of::<w1z4rd_brain::Terminal>();
        // A terminal also costs one terminal_idx entry (key + value + control byte).
        let per_terminal = terminal + std::mem::size_of::<(w1z4rd_brain::NeuronRef, usize)>() + 1;
        let per_member = std::mem::size_of::<w1z4rd_brain::NeuronRef>();
        let per_neuron = std::mem::size_of::<w1z4rd_brain::Neuron>();
        let (mut hub, mut bytes) = (0usize, 0usize);
        for pid in self.brain.fabric().pool_ids() {
            let Some(pool) = self.brain.fabric().pool(pid) else { continue };
            for n in pool.read().iter_neurons() {
                hub = hub.max(n.terminals.len());
                bytes += per_neuron + n.label.len() + n.members.len() * per_member
                    + n.terminals.len() * per_terminal;
            }
        }
        (hub, bytes)
    }
}

fn shuffled(n: usize, epoch: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s: u64 = 0xC0FF_EECA_FEBA_BE ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
    idx
}

fn score(probes: &[Probe], mut ask: impl FnMut(&str) -> Vec<u8>) -> (f64, f64) {
    let t0 = Instant::now();
    let hits = probes.iter().filter(|p| ask(&p.query) == p.answer.as_bytes()).count();
    let ms_per = t0.elapsed().as_secs_f64() * 1000.0 / probes.len().max(1) as f64;
    (100.0 * hits as f64 / probes.len().max(1) as f64, ms_per)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale = args
        .iter()
        .position(|a| a == "--scale")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);

    let world = SceneWorld::new(scale);
    let mut subject = Subject::new();
    let t0 = Instant::now();
    subject.train(&world.facts);
    let train_s = t0.elapsed().as_secs_f64();

    let (recall_pct, recall_ms) = score(&world.facts, |q| subject.recall(q));
    let (integration_pct, infer_ms) = score(&world.integration, |q| subject.infer(q));
    let (hub, bytes) = subject.footprint();
    let s = subject.brain.stats();

    println!(
        "{}",
        serde_json::json!({
            "scale": scale,
            "facts": world.facts.len(),
            "integration_probes": world.integration.len(),
            "recall_pct": recall_pct,
            "integration_pct": integration_pct,
            "train_s": train_s,
            "recall_ms": recall_ms,
            "infer_ms": infer_ms,
            "neurons": s.total_neurons,
            "concepts": s.total_concepts,
            "terminals": s.total_terminals,
            "hub_fanout": hub,
            "est_resident_mb": bytes as f64 / 1_048_576.0,
        })
    );
}
