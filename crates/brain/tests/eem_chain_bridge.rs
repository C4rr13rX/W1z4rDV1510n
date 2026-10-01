//! Does a POPULATED fact graph chain, and what are its edges made of?
//!
//! `ffda7ae9` says nothing populates the EEM grounded-fact graph from
//! training, and that is measured: `register_fact` has exactly one production
//! caller (`brain.rs`, inside the Tier-2 consolidated arm of
//! `register_fingerprint`), `pretrain_binding_episode` never reaches it, and
//! the Tier-1 arm says so in its own comment -- "creates a binding neuron in
//! the binding pool but no EEM fact and no gossip. Visible to /chat
//! retrieval; invisible to EEM chain exploration."
//!
//! The obvious repair is "register a fact on the pretrain path too". This file
//! exists because that repair was ALREADY TRIED in a previous pass and
//! measured: it populated the graph, cost 5.3 MB at scale 64, and integration
//! stayed at 0. So population is necessary and not sufficient, and shipping it
//! again would be the inert fix this repository keeps paying for.
//!
//! So this file populates the graph from OUTSIDE, through the public
//! `Brain::eem_mut`, with no change to `brain.rs` at all -- and then asks what
//! the walk does with it. A test-side population is the cheapest possible way
//! to find out whether the production change is worth making before making it.
//!
//! Every number here is PRINTED. The assertions are only invariants that hold
//! whatever the distribution is, because asserting a distribution would tune
//! the test to this one world.

use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

fn subject() -> Brain {
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
    brain
}

/// What is firing in `pool` right now, as the fact-member refs
/// `Eem::register_fact` takes.
fn firing(brain: &Brain, pool: u32) -> Vec<u32> {
    brain
        .fabric()
        .pool(pool)
        .map(|p| p.read().currently_firing().into_iter().collect::<Vec<u32>>())
        .unwrap_or_default()
}

/// The two chaining facts per room, exactly as `integration_arm_census.rs`
/// trains them: neither answers "what is the lamp's material" alone, and the
/// composition of the two does.
const WORLD: [(&str, &str); 2] = [("lamp on", "desk"), ("desk material", "oak")];

fn train(brain: &mut Brain, rooms: u32) {
    for room in 0..rooms {
        for (q, a) in WORLD {
            brain.pretrain_binding_episode(&[
                (QUERY_POOL, format!("r{room:03} {q}").into_bytes()),
                (ANSWER_POOL, a.as_bytes().to_vec()),
            ]);
        }
    }
}

/// Reconstruct the member set a Tier-2 registration WOULD have used for one
/// episode, by reading what the two frames light up. This is the fingerprint's
/// own content -- `MomentFingerprint::from_fabric_moment` is built from the
/// per-pool fired sequences -- reached through public read-only calls so that
/// nothing in `brain.rs` has to change to take this measurement.
fn episode_members(brain: &mut Brain, q: &str, a: &str) -> Vec<(u32, u32)> {
    let mut members = Vec::new();
    brain.observe_read_only(QUERY_POOL, q.as_bytes());
    members.extend(firing(brain, QUERY_POOL).into_iter().map(|n| (QUERY_POOL, n)));
    brain.observe_read_only(ANSWER_POOL, a.as_bytes());
    members.extend(firing(brain, ANSWER_POOL).into_iter().map(|n| (ANSWER_POOL, n)));
    members.sort();
    members.dedup();
    members
}

#[test]
fn a_populated_fact_graph_is_a_byte_hairball_not_a_chain() {
    const ROOMS: u32 = 16;
    let mut brain = subject();
    train(&mut brain, ROOMS);

    // (1) The item's own claim, as a number rather than a grep.
    let facts_after_training = brain.eem().fact_count();
    eprintln!("grounded facts after {ROOMS} rooms x 2 episodes of ordinary training: {facts_after_training}");

    // (2) Populate the graph from outside, the way the Tier-2 arm would.
    // `source_binding` is a synthetic id per episode: `register_fact` keys
    // dedup on it, and distinct ids are what a real per-binding registration
    // would give.
    let mut answer_refs_desk: Vec<(u32, u32)> = Vec::new();
    let mut answer_refs_oak: Vec<(u32, u32)> = Vec::new();
    let mut members_per_fact: Vec<usize> = Vec::new();
    let mut source = 0u32;
    for room in 0..ROOMS {
        for (q, a) in WORLD {
            let members = episode_members(&mut brain, &format!("r{room:03} {q}"), a);
            members_per_fact.push(members.len());
            let answer_side: Vec<(u32, u32)> =
                members.iter().copied().filter(|(p, _)| *p == ANSWER_POOL).collect();
            if a == "desk" {
                answer_refs_desk = answer_side;
            } else {
                answer_refs_oak = answer_side;
            }
            brain.eem_mut().register_fact(source, members);
            source += 1;
        }
    }
    let populated = brain.eem().fact_count();
    let mean_members = members_per_fact.iter().sum::<usize>() as f64 / members_per_fact.len() as f64;
    eprintln!("after hand-population: facts {populated}  (registered {source} episodes)");
    eprintln!("  members per fact: mean {mean_members:.1}  min {}  max {}",
        members_per_fact.iter().min().copied().unwrap_or(0),
        members_per_fact.iter().max().copied().unwrap_or(0));

    // (3) THE EDGE CENSUS. A fact-graph edge exists between two facts when
    // they SHARE a member. Atoms in this brain are single BYTES
    // (ARCHITECTURE.md), so if facts are registered over atom refs then every
    // fact sharing a space or an 'e' with every other fact is one edge, and
    // the walk cannot discriminate. This is the number that says whether
    // populating the graph on the production path is worth anything.
    let mut fan_out: ahash::AHashMap<(u32, u32), usize> = ahash::AHashMap::new();
    for fact in brain.eem().iter_facts() {
        for &m in &fact.members {
            *fan_out.entry(m).or_insert(0) += 1;
        }
    }
    let mut ranked: Vec<((u32, u32), usize)> = fan_out.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    let shared_by_all = ranked.iter().filter(|(_, c)| *c == populated).count();
    let shared_by_more_than_one = ranked.iter().filter(|(_, c)| *c > 1).count();
    eprintln!("  distinct members overall            {}", ranked.len());
    eprintln!("  members present in EVERY fact       {shared_by_all}");
    eprintln!("  members present in >1 fact          {shared_by_more_than_one}");
    eprintln!("  top member fan-outs (facts touching one member):");
    for ((p, n), c) in ranked.iter().take(8) {
        eprintln!("      pool {p} neuron {n}: {c} of {populated} facts");
    }

    // (4) What the walk does on a probe that was never trained.
    let probe = format!("r{:03} lamp on material", 0);
    brain.observe_read_only(QUERY_POOL, probe.as_bytes());
    let seed: Vec<(u32, u32)> = firing(&brain, QUERY_POOL)
        .into_iter()
        .map(|n| (QUERY_POOL, n))
        .collect();
    let chain = brain.eem().chain_explore(&seed, 3, 16);
    let reached_answer: Vec<((u32, u32), f32)> = chain
        .reached_members
        .iter()
        .filter(|((p, _), _)| *p == ANSWER_POOL)
        .map(|(k, v)| (*k, *v))
        .collect();
    let distinct_conf: std::collections::BTreeSet<String> = reached_answer
        .iter()
        .map(|(_, c)| format!("{c:.4}"))
        .collect();
    // Can the walk tell the RIGHT answer from the wrong one? `oak` is correct
    // for this probe and `desk` is the distractor, and both are answer-pool
    // refs the walk can reach.
    let reached_oak = answer_refs_oak
        .iter()
        .filter(|m| chain.reached_members.contains_key(m))
        .count();
    let reached_desk = answer_refs_desk
        .iter()
        .filter(|m| chain.reached_members.contains_key(m))
        .count();
    eprintln!("probe {probe:?} (never trained)");
    eprintln!("  seeds                               {}", seed.len());
    eprintln!("  visited facts (max_visit 16)        {}", chain.visited_facts.len());
    eprintln!("  reached members                     {}", chain.reached_members.len());
    eprintln!("  of those in the ANSWER pool         {}", reached_answer.len());
    eprintln!("  distinct confidences among them     {} -> {:?}", distinct_conf.len(), distinct_conf);
    eprintln!("  refs of the CORRECT answer 'oak' reached   {reached_oak} of {}", answer_refs_oak.len());
    eprintln!("  refs of the DISTRACTOR  'desk' reached     {reached_desk} of {}", answer_refs_desk.len());

    // Invariants only.
    assert!(
        populated >= facts_after_training,
        "hand-population removed facts: {facts_after_training} -> {populated}"
    );
    assert!(
        !seed.is_empty(),
        "nothing fired for the probe, so the empty-seed arm returns before \
         chain_explore and no conclusion about the walk is available"
    );
    // A reached member must be reachable: every answer-pool ref the walk
    // returned has to carry a confidence in (0, 1].
    for ((p, n), c) in &reached_answer {
        assert!(
            *c > 0.0 && *c <= 1.0,
            "reached member (pool {p}, neuron {n}) has confidence {c}, outside (0,1]"
        );
    }
}
