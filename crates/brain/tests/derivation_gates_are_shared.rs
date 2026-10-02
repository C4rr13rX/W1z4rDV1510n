//! The node's two answer paths must decide to DERIVE under the same condition.
//!
//! A derivation is the only route in this product that answers a question the
//! brain was never asked, so whatever suppresses it decides how much
//! integration the product has. `crates/node/src/api.rs` has two sites that
//! call `derived_by_substitution_reply`, and they disagreed:
//!
//!   `/brain/ask`                   `answer empty || outside_grounding || !recalled_exactly`
//!   hypothesis research loop       `answer empty`
//!
//! So on the research loop any non-empty answer from
//! `integrate_autonomous_tuned` suppressed the derivation entirely, however far
//! that answer's best binding sat from the prompt. That is the shadowing
//! backlog item `[988dd17c]` describes. It named the fabric arm in
//! `Brain::integrate_autonomous_tuned` as the cause, and measurement in
//! `tests/fabric_arm_shadows_derivation.rs` put it HERE instead: the fabric arm
//! cannot fire on that world (`integrate()` answers 0 of 16 at
//! `fabric_confidence 0.0000`, below the node's 0.10 and the scorecard's
//! 100.0), while this gate is on the live path for every queued hypothesis.
//!
//! # Why a source scan and not a behavioural test
//!
//! The defect is a MISSING TERM in an `if`, and a behavioural test for it needs
//! a prompt on which the two conditions disagree -- which needs a non-empty
//! answer whose binding recall is below 1.0, which is exactly the state
//! measured not to occur on the worlds this crate can build. A behavioural test
//! would therefore be green with the bug put back: a vacuous guard, which is
//! the failure this lab has shipped at least five times. The scan reds on the
//! absence of the term, which is the thing that is actually wrong, and it is
//! the same technique `answer_path_parameters_are_shared.rs` already uses on
//! the same files.
//!
//! NOT asserted: `crates/node/src/brain_api.rs`'s `/brain/chat`. Its arms are
//! an if/else chain whose first arm is an exact `trained_decode` and whose
//! second requires `composition_used.len() >= 2` and `!speculation_flag` -- a
//! different and stricter structure, not this defect, and not a file this pass
//! may edit. Read, reported, left alone.
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --release --test derivation_gates_are_shared -- --nocapture

/// `CARGO_MANIFEST_DIR` is `<root>/crates/brain`, so the workspace root is two
/// levels up. Derived rather than taken from the CWD, which differs between
/// `cargo test` and a test binary run directly.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR is <root>/crates/brain, so it has two parents")
        .to_path_buf()
}

const ANSWER_PATH_FILE: &str = "crates/node/src/api.rs";
const DERIVATION_CALL: &str = "derived_by_substitution_reply";

/// How far back from a call site to look for the condition that guards it.
///
/// Both sites carry a long comment between the `if` and the call, so a
/// two-or-three-line window would miss the condition and the test would pass
/// vacuously. Sized to span the longer of the two with room to spare, and the
/// count of sites found is asserted below so a window that drifts off a site
/// reds rather than reporting zero.
const GUARD_WINDOW_LINES: usize = 40;

/// Every line that CALLS the derivation, with its line number. Comments
/// mentioning it are excluded: `api.rs` has one, and it would otherwise be
/// counted as a call site whose guard is missing.
fn derivation_call_sites(src: &str) -> Vec<(usize, String)> {
    src.lines()
        .enumerate()
        .filter(|(_, line)| {
            let t = line.trim_start();
            !t.starts_with("//") && t.contains(DERIVATION_CALL)
        })
        .map(|(i, line)| (i + 1, line.trim().to_string()))
        .collect()
}

/// The code (comments stripped) in the window above a call site.
fn guard_text(src: &str, call_line: usize) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let start = call_line.saturating_sub(GUARD_WINDOW_LINES + 1);
    lines[start..call_line.min(lines.len())]
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.starts_with("//"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn every_answer_path_derivation_is_gated_on_binding_recall() {
    let path = workspace_root().join(ANSWER_PATH_FILE);
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let sites = derivation_call_sites(&src);
    eprintln!(
        "{ANSWER_PATH_FILE}: {} site(s) calling {DERIVATION_CALL}",
        sites.len()
    );

    // A scan that found nothing passes for free. Both sites are named in this
    // file's header, so the count is knowable and is asserted.
    assert_eq!(
        sites.len(),
        2,
        "expected the 2 answer-path derivation sites this file documents (/brain/ask and the \
         hypothesis research loop); found {} -- if a site was added or removed, update this \
         file rather than loosening the count, because a scan over zero sites is vacuous",
        sites.len()
    );

    let mut ungated = Vec::new();
    for (line, _) in &sites {
        let guard = guard_text(&src, *line);
        // The term, not a particular spelling of the condition: either route
        // may compute it inline or bind it first.
        let gated = guard.contains("recalled_exactly") || guard.contains(".recall >=");
        eprintln!(
            "  {ANSWER_PATH_FILE}:{line}  gated on binding recall: {}",
            if gated { "yes" } else { "NO" }
        );
        if !gated {
            ungated.push(*line);
        }
    }

    assert!(
        ungated.is_empty(),
        "{ANSWER_PATH_FILE} derives without checking whether the prompt was recalled EXACTLY at \
         line(s) {ungated:?}. A non-empty answer whose best binding is an inexact match shadows \
         the derivation, which is the only route that answers a question the brain was never \
         asked. Gate on binding RECALL (a composite prompt scores precision 1.0 against its own \
         sub-question, so a precision test is inert -- measured recall 0.7500 against precision \
         1.0000 in tests/fabric_arm_shadows_derivation.rs)"
    );
}
