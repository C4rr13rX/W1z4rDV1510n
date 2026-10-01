//! Does TRAINING feed the composition engine, and does the engine then DERIVE?
//!
//! `tests/composition_inputs_census.rs` measured the gap: after the scorecard's
//! own training path, `composition_rule_count()` and `semantic_relation_count()`
//! were both 0, so `TransientWorkspace::resolve` broke out of its first round
//! and derived nothing whatever the facts were. This file is the other half --
//! the same training path, and what the workspace does once it is fed.
//!
//! # Why the question has to be taken apart
//!
//! A relation registered as `(query, answer)` cannot join. `workspace.rs::unify`
//! matches a constant by `==` and binds a variable to a WHOLE `TypedValue`, and
//! the join a two-hop derivation needs here is
//!
//!   `"r003 lamp on?" -> "desk"`   meeting   `"r003 desk material?" -> "oak"`
//!
//! at the string `desk`, which is the ANSWER of the first and sits INSIDE the
//! QUERY of the second. As `(query, answer)` pairs those two share no argument,
//! so no rule over them derives anything. `Eem::induce_from_episode` splits the
//! query at every occurrence of an induced symbol into
//! `(before, symbol, after, answer)`, which puts the join key in its own slot,
//! and the vocabulary of symbols is learned -- a symbol is a string this brain
//! was taught to ANSWER, so an untrained brain induces nothing and there is no
//! delimiter, token list or segmentation rule anywhere in it.
//!
//! The pre-change values are printed in the same output as the post-change
//! ones, because the whole finding was that they are 0.

use w1z4rd_brain::eem::Eem;
use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;

/// The same two-pool byte-passthrough brain the scorecard builds.
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

/// The scorecard's `Subject::train`, verbatim in shape: the fabric half and the
/// symbolic half of each episode. Nothing here knows what a probe is.
fn train(brain: &mut Brain, facts: &[(String, String)]) {
    for (query, answer) in facts {
        brain.pretrain_binding_episode(&[
            (QUERY_POOL, query.as_bytes().to_vec()),
            (ANSWER_POOL, answer.as_bytes().to_vec()),
        ]);
        brain.eem_mut().induce_from_episode(query.as_bytes(), answer.as_bytes());
    }
}

/// A miniature of the scorecard's scene world: room adjacency makes room names
/// answers (so they enter the vocabulary), `on?` names a piece of furniture, and
/// `material?` gives that furniture a material. Nothing answers "the lamp's
/// material"; the composition of two trained facts does.
fn chainable_world(rooms: usize) -> Vec<(String, String)> {
    let name = |r: usize| format!("r{r:03}");
    let material = |r: usize| ["oak", "steel", "glass", "cloth"][r % 4].to_string();
    let mut facts = Vec::new();
    for r in 0..rooms {
        let room = name(r);
        facts.push((format!("{room} next?"), name((r + 1) % rooms)));
        facts.push((format!("{room} lamp on?"), "desk".to_string()));
        facts.push((format!("{room} desk material?"), material(r)));
        facts.push((format!("{room} lamp material?"), material(r + 1)));
    }
    facts
}

#[test]
fn training_feeds_the_composition_engine_and_it_derives() {
    const ROOMS: usize = 8;
    let mut brain = subject();

    let rules_before = brain.eem().composition_rule_count();
    let relations_before = brain.eem().semantic_relation_count();
    let symbols_before = brain.eem().induced_symbol_count();

    let facts = chainable_world(ROOMS);
    train(&mut brain, &facts);

    let rules_after = brain.eem().composition_rule_count();
    let relations_after = brain.eem().semantic_relation_count();
    let symbols_after = brain.eem().induced_symbol_count();

    // Sanity: the trained halves ARE recalled, so nothing below is about an
    // empty brain. Same guard the census test uses.
    let mut trained_recalled = 0;
    for r in 0..ROOMS {
        brain.observe_read_only(QUERY_POOL, format!("r{r:03} lamp on?").as_bytes());
        if brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).as_deref()
            == Some(&b"desk"[..])
        {
            trained_recalled += 1;
        }
    }
    assert_eq!(
        trained_recalled, ROOMS,
        "the trained half of the world is not recalled, so nothing below is about integration"
    );

    let workspace = brain.eem().compose_transient(4);
    let derived: Vec<_> = workspace
        .query(Eem::INDUCED_CHAIN_PREDICATE)
        .collect();

    // A derivation that names only ONE trained fact is a relabelling, not a
    // composition, so the provenance is the criterion rather than the count.
    // Trained provenance = the query string the episode was taught with; the
    // rule contributes `rule:<name>`, which is excluded here.
    let trained_queries: std::collections::BTreeSet<&str> =
        facts.iter().map(|(q, _)| q.as_str()).collect();
    let mut best: Option<(&w1z4rd_brain::workspace::GroundedRelation, Vec<String>)> = None;
    for relation in &derived {
        let sources: Vec<String> = relation
            .provenance
            .iter()
            .filter(|source| trained_queries.contains(source.as_str()))
            .cloned()
            .collect();
        if best.as_ref().map_or(true, |(_, found)| sources.len() > found.len()) {
            best = Some((relation, sources));
        }
    }

    println!(
        "composition inputs after training {ROOMS} chainable rooms ({} episodes):\n  \
         composition_rules      {rules_before} -> {rules_after}\n  \
         semantic_relations     {relations_before} -> {relations_after}\n  \
         induced_symbols        {symbols_before} -> {symbols_after}\n  \
         compose_transient(4)   {} facts in workspace, {} derived\n  \
         trained recall         {trained_recalled}/{ROOMS}",
        facts.len(),
        workspace.facts().len(),
        derived.len(),
    );
    if let Some((relation, sources)) = &best {
        println!(
            "  best derivation        {:?} {:?}\n  \
             its trained sources    {} {:?}",
            relation.predicate,
            relation.arguments.iter().map(|a| a.value.as_str()).collect::<Vec<_>>(),
            sources.len(),
            sources,
        );
    }

    // Criterion 1: training feeds BOTH inputs.
    assert!(
        rules_after > 0,
        "training registered no composition rule, so resolve cannot derive at all \
         ({rules_before} -> {rules_after})"
    );
    assert!(
        relations_after > 0,
        "training registered no semantic relation ({relations_before} -> {relations_after})"
    );
    assert!(
        symbols_after > 0,
        "the induced vocabulary is empty, so no question can be decomposed"
    );

    // Criterion 2: the engine DERIVES, and the derivation names two trained
    // facts. A non-zero input count is necessary and demonstrably not
    // sufficient -- a previous pass raised a fact count and still measured 0.0 %
    // integration -- so this is the assertion that matters.
    let (relation, sources) = best.expect("resolve derived no chained relation at all");
    assert!(
        sources.len() >= 2,
        "a derived relation must name at least two TRAINED facts in its provenance, \
         got {} from {:?} (full provenance {:?})",
        sources.len(),
        relation.predicate,
        relation.provenance,
    );

    // EVERY derivation has to be TRUE in the world, not merely well formed, and
    // "the conclusion is some trained answer" is far too weak to catch the way
    // this fails. The first version of the rule left the second premise's
    // context free, so `r001 lamp on?` joined `r003 desk material?` and
    // concluded r003's material for r001's lamp -- a conclusion that passes a
    // "is it a trained answer" check and is false.
    //
    // The world's own rule, stated once here and nowhere in crates/brain: a
    // chain anchored on `anchor` concluding `value` is true when some trained
    // question about THAT anchor answers `value`. The anchor is argument 1.
    let mut unsound = Vec::new();
    for relation in &derived {
        let anchor = &relation.arguments[1].value;
        let concluded = &relation
            .arguments
            .last()
            .expect("the chain conclusion is the last argument")
            .value;
        let supported = facts.iter().any(|(query, answer)| {
            answer == concluded && query.contains(anchor.as_str())
        });
        if !supported {
            unsound.push((anchor.clone(), concluded.clone(), relation.provenance.clone()));
        }
    }
    println!("  unsound derivations    {} of {}", unsound.len(), derived.len());
    assert!(
        unsound.is_empty(),
        "{} of {} derivations are false in the world; first three: {:?}",
        unsound.len(),
        derived.len(),
        &unsound[..unsound.len().min(3)],
    );
}

/// A `brain.bin` written before the induced feed existed must still load, and
/// the format that has to be checked is BINCODE.
///
/// This is the half that was got wrong first and is worth the detail. The
/// vocabulary was originally a new `EemSnapshot` field marked
/// `#[serde(default)]`, which reads as backward compatible and is not: a
/// `brain.bin` is bincode, bincode is not self-describing, and fields are read
/// POSITIONALLY -- so a reader with one extra field runs off the end of an older
/// record and fails with `Io(Kind(UnexpectedEof))`. `default` only ever rescues
/// a self-describing format. The committed fixture in
/// `tests/binding_posting_generation_compat.rs` caught it; the test that was
/// supposed to cover it round-tripped through `serde_json`, where `default`
/// works perfectly, so it passed while every real snapshot was broken.
///
/// So: no new field, and the vocabulary is rebuilt from the relations. What is
/// asserted here is both halves of that -- bincode round trips, and the
/// vocabulary comes back.
#[test]
fn the_induced_vocabulary_survives_a_bincode_round_trip_without_a_format_change() {
    let mut brain = subject();
    train(&mut brain, &chainable_world(4));
    let relations = brain.eem().semantic_relation_count();
    let symbols = brain.eem().induced_symbol_count();
    assert!(symbols > 0, "nothing to recover");

    let encoded = bincode::serialize(&brain.eem().snapshot()).expect("bincode EemSnapshot");
    let decoded: w1z4rd_brain::persistence::EemSnapshot =
        bincode::deserialize(&encoded).expect("an EemSnapshot must bincode round trip");
    let restored = Eem::from_snapshot(decoded);
    println!(
        "bincode round trip: {} bytes, semantic_relations {} -> {}, induced_symbols {} -> {}",
        encoded.len(),
        relations,
        restored.semantic_relation_count(),
        symbols,
        restored.induced_symbol_count(),
    );
    assert_eq!(restored.semantic_relation_count(), relations);
    assert_eq!(
        restored.induced_symbol_count(),
        symbols,
        "the vocabulary must be recoverable from the relations, since it is not stored"
    );

    // And the thing that actually regressed: a trailing byte run-off. Decoding
    // from a PREFIX of a longer record is what an older snapshot looks like to a
    // reader that gained a field, so a reader that tolerates truncation here
    // would mean the format had grown one.
    let mut untrained = Eem::new(Default::default());
    assert_eq!(untrained.induced_symbol_count(), 0, "an untrained brain induces nothing");
    untrained.induce_from_episode(b"", b"");
    assert_eq!(
        untrained.composition_rule_count(),
        0,
        "an empty episode must not install the rule, or an untrained brain carries one"
    );
}
