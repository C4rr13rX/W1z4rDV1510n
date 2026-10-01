//! On-disk binding posting generations must keep resolving after the
//! in-RAM index representation changes.
//!
//! WHY A COMMITTED FIXTURE AND NOT A ROUND TRIP. A test that writes a
//! `.wbrain` and reads it back in the same process proves only that the
//! current code agrees with itself: both halves move together, so a change
//! to what `PostingIndexBuilder::create_hashed` hashes passes it. The
//! acceptance criterion is about an `.wbrain` that *already exists* — bytes
//! written by an earlier build. The only way to have those is to commit
//! them, so `fixtures/binding_posting_generation_v1/` holds a container
//! generated at 88d5821, before `binding_sequence_index` was deduplicated
//! against the fingerprint.
//!
//! Regenerate deliberately, never casually — regenerating is how a
//! compatibility test quietly becomes a round trip again:
//!
//!     W1Z4RD_REGEN_BINDING_FIXTURE=1 cargo test -p w1z4rd-brain \
//!         --test binding_posting_generation_compat -- --nocapture
//!
//! and only when the on-disk format is intended to change AND a migration
//! exists, in which case the old fixture should be kept beside the new one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig, PoolId};

/// The scorecard's pools, so the fixture exercises the same binding route
/// the product measures (`crates/brain/examples/scorecard.rs`).
const QUERY_POOL: PoolId = 1;
const ANSWER_POOL: PoolId = 2;
const EPOCHS: usize = 2;

/// Facts chosen so each query's atom sequence is distinct and short. The
/// binding route under test is keyed by that sequence.
const FACTS: &[(&str, &str)] = &[
    ("r001 lamp color?", "amber"),
    ("r002 lamp color?", "violet"),
    ("r003 lamp color?", "cyan"),
    ("r004 lamp color?", "scarlet"),
    ("r005 lamp color?", "indigo"),
    ("r006 lamp color?", "emerald"),
    ("r007 lamp color?", "ochre"),
    ("r008 lamp color?", "slate"),
];

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("binding_posting_generation_v1")
}

fn encodings() -> HashMap<PoolId, Box<dyn AtomEncoding>> {
    let mut map: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    map.insert(0u32, Box::new(BytePassthroughEncoding { prefix: "bind" }));
    map.insert(QUERY_POOL, Box::new(BytePassthroughEncoding { prefix: "q" }));
    map.insert(ANSWER_POOL, Box::new(BytePassthroughEncoding { prefix: "a" }));
    map
}

fn trained_brain(container: &Path) -> Brain {
    let mut config = BrainConfig::default();
    config.binding_emergence_threshold = 3;
    config.moment_history_window = 256;
    let mut brain = Brain::new(config);
    for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
        let mut pool = PoolConfig::defaults(name, id);
        pool.recent_atoms_window = 2048;
        pool.concept_emergence_threshold = 2;
        pool.max_concept_member_count = 64;
        pool.decay_rate = 0.0001;
        pool.prune_floor = 0.005;
        brain.create_pool(pool, Box::new(BytePassthroughEncoding { prefix }));
    }
    brain.attach_wbrain(container).expect("attach wbrain");
    for _ in 0..EPOCHS {
        for (query, answer) in FACTS {
            brain.pretrain_binding_episode(&[
                (QUERY_POOL, query.as_bytes().to_vec()),
                (ANSWER_POOL, answer.as_bytes().to_vec()),
            ]);
        }
    }
    brain
}

/// Copy a directory's files one level deep. The posting index writes its
/// generations beside the container, so the whole directory is the fixture
/// and restoring must not mutate the committed copy.
fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).expect("create destination");
    for entry in std::fs::read_dir(from).expect("read fixture dir") {
        let entry = entry.expect("fixture entry");
        if entry.file_type().expect("file type").is_file() {
            std::fs::copy(entry.path(), to.join(entry.file_name())).expect("copy fixture file");
        }
    }
}

fn tmpdir(tag: &str) -> PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "w1z4rd_binding_posting_compat_{tag}_{}_{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("create tmpdir");
    dir
}

/// Writes the committed fixture. Gated on an environment variable because a
/// compatibility fixture that regenerates itself tests nothing.
fn regenerate() {
    let dir = fixture_dir();
    if dir.exists() {
        std::fs::remove_dir_all(&dir).expect("clear old fixture");
    }
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let container = dir.join("fixture.wbrain");
    let mut brain = trained_brain(&container);
    brain
        .serialize_all_neurons_for_idle()
        .expect("flush binding posting generations");
    drop(brain);

    let mut total = 0u64;
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("read fixture dir") {
        let entry = entry.expect("fixture entry");
        let size = entry.metadata().expect("metadata").len();
        total += size;
        names.push(format!(
            "{} {} B",
            entry.file_name().to_string_lossy(),
            size
        ));
    }
    names.sort();
    println!("fixture written to {}", dir.display());
    for name in &names {
        println!("  {name}");
    }
    println!("fixture total {total} B across {} files", names.len());
}

/// The criterion: a container written by an earlier build still answers.
///
/// It asserts on every fact, not on one, because a hash change that
/// collides for a single key is not the failure mode — a changed key
/// derivation loses the whole generation at once, and a route that silently
/// falls through to `integrate()` would otherwise read as a pass.
#[test]
fn committed_wbrain_still_resolves_its_binding_routes() {
    if std::env::var("W1Z4RD_REGEN_BINDING_FIXTURE").is_ok() {
        regenerate();
        return;
    }
    let fixture = fixture_dir();
    assert!(
        fixture.join("fixture.wbrain").is_file(),
        "fixture container missing at {}; regenerate with \
         W1Z4RD_REGEN_BINDING_FIXTURE=1 only when the format is intended to change",
        fixture.display()
    );

    let work = tmpdir("restore");
    copy_dir(&fixture, &work);
    let (mut restored, missing) =
        Brain::restore_wbrain(work.join("fixture.wbrain"), encodings()).expect("restore fixture");
    assert!(
        missing.is_empty(),
        "every pool encoding is supplied; missing must be empty, got {missing:?}"
    );

    let mut resolved = 0usize;
    let mut wrong: Vec<String> = Vec::new();
    for (query, answer) in FACTS {
        restored.observe_read_only(QUERY_POOL, query.as_bytes());
        let decoded = restored.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL);
        let _ = restored.finish_read_only_inference();
        match decoded {
            Some(bytes) if bytes == answer.as_bytes() => resolved += 1,
            Some(bytes) => wrong.push(format!(
                "{query} -> {:?} (expected {answer})",
                String::from_utf8_lossy(&bytes)
            )),
            None => wrong.push(format!("{query} -> no binding (expected {answer})")),
        }
    }

    drop(restored);
    let _ = std::fs::remove_dir_all(&work);

    assert_eq!(
        resolved,
        FACTS.len(),
        "a pre-existing .wbrain must still resolve every binding route it stored; \
         failures: {wrong:?}. If this went red on a change to the in-RAM binding \
         index, the on-disk key derivation moved with it and existing posting \
         generations are unreadable -- that is the migration the acceptance \
         criterion asks for, not a flaky test."
    );
}

/// The fixture is only evidence while it is OLD. This pins the generation
/// count the container was written with, so a regeneration that drops the
/// posting generations entirely (and would therefore pass the test above by
/// answering from a rebuilt in-RAM overlay) is visible.
#[test]
fn committed_fixture_actually_carries_posting_generations() {
    if std::env::var("W1Z4RD_REGEN_BINDING_FIXTURE").is_ok() {
        return;
    }
    let fixture = fixture_dir();
    assert!(
        fixture.join("fixture.wbrain").is_file(),
        "fixture container missing"
    );
    let work = tmpdir("residency");
    copy_dir(&fixture, &work);
    let (restored, _missing) =
        Brain::restore_wbrain(work.join("fixture.wbrain"), encodings()).expect("restore fixture");
    let (overlay_entries, generations) = restored.binding_posting_residency();
    drop(restored);
    let _ = std::fs::remove_dir_all(&work);

    println!("restored fixture: overlay {overlay_entries} entries, {generations} generation(s)");
    assert!(
        generations >= 1,
        "the fixture must carry at least one on-disk posting generation, or the test above \
         is answering from a rebuilt in-RAM overlay and proves nothing about the format"
    );
}
