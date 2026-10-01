//! The parameters every answer route and the scorecard must share.
//!
//! # What this file is for
//!
//! The operator rule (2026-09-30) is that a gain counts only when the node's
//! answer routes use the same path the scorecard measures, and that the
//! scorecard's way of asking never changes unless the node changes the same
//! way in the same commit. Nothing enforced it, and the three
//! `Brain::integrate_autonomous` parameters had drifted into four different
//! configurations. Measured 2026-10-01 by walking the four call sites and
//! collecting each call's paren-balanced argument list:
//!
//! ```text
//! site                                       fabric_conf  depth  visit
//! crates/brain/examples/scorecard.rs:410       100.0        3     200   <- the GATED number
//! crates/node/src/brain_api.rs:5221              0.0        4     200   <- /brain/chat
//! crates/node/src/bin/brain_server.rs:2398       0.0        4     200   <- /chat
//! crates/node/src/api.rs:8287 (_tuned)           0.10       3      16   <- hypothesis loop
//! ```
//!
//! The spread matters in the expensive direction. `fabric_confidence_threshold`
//! is compared against `GroundingReport::fabric_confidence`, and that field is
//! a 0..1 quantity rather than an open-ended score — measured in
//! `src/grounding.rs`, not assumed: it initialises to `0.0` (`grounding.rs:52`),
//! it sits beside `input_atom_coverage`, documented as a "fraction"
//! (`grounding.rs:15`), and the only consumer of a confidence in that file,
//! `ConfidenceTier::from_confidence` (`grounding.rs:75`), splits on `c >= 0.5`.
//! So `100.0` is not a permissive setting that happens to differ by a
//! constant — it is UNREACHABLE, and the scorecard therefore measures
//! integration with the fabric-confidence arm switched off while both deployed
//! chat routes run it wide open at `0.0`. Under the owner's PRIORITY ZERO (the
//! brain never hallucinates) that inverts what the gate is worth: the
//! scorecard's `wrong = 0.0 at every scale` is a true statement about a
//! strictly more abstemious brain than the one the node ships, and the gate
//! cannot see a wrong answer the fabric arm invents in production.
//!
//! # Why constants and not a config struct
//!
//! The fault is a VALUE drifting across four files, not a behaviour needing a
//! policy. A struct would give every call site a place to override, which is
//! the thing that went wrong. Constants plus
//! `tests/answer_path_parameters_are_shared.rs` — a source scan that reds when
//! a site passes a literal again — fix the class rather than today's four
//! values.
//!
//! # Why the scorecard's values win
//!
//! Two of the three are a straight choice between a measured configuration and
//! an unmeasured one, so the gated one wins by default: it is the only
//! configuration any number in `docs/scorecard-baseline.json` describes.
//! `ANSWER_FABRIC_CONFIDENCE_THRESHOLD` additionally wins on PRIORITY ZERO —
//! it is the tighter of the two, and tightening the node can only move an
//! answer from wrong to silent, never the other way.
//!
//! What this costs the node is measured rather than assumed, by
//! `tests/ask_route_integrate_parity.rs`, which answers one trained world
//! twice and prints answered/correct per route. If adopting these values
//! removes CORRECT node answers, that number is a decision for the owner and
//! is reported rather than absorbed — which is why these are three named
//! constants with the measurement beside them and not a silent edit to four
//! argument lists.

/// `fabric_confidence_threshold`: the floor a plain Hebbian-propagation answer
/// must clear before `integrate_autonomous` will return it.
///
/// Compared against a 0..1 confidence, so this value switches the
/// fabric-confidence arm OFF and leaves the trained-binding and chain/
/// derivation arms as the only ways to an answer. That is deliberate and it is
/// what every gated integration number has always measured: a propagation
/// score is not a grounded derivation, and PRIORITY ZERO prefers silence to a
/// plausible one.
pub const ANSWER_FABRIC_CONFIDENCE_THRESHOLD: f32 = 100.0;

/// `chain_max_depth`: how many grounded facts an EEM chain may compose.
///
/// 3 is what the gated scorecard measures, and the scorecard's own deepest
/// held-out family (`next_on_material`) is 3 hops, so this is the depth the
/// published integration percentages describe. The node ran 4, which bought
/// nothing any measurement names and cost a wider search per answer.
pub const ANSWER_CHAIN_MAX_DEPTH: usize = 3;

/// `chain_max_visit`: the node budget of the chain explorer's walk.
///
/// 200 at the three sites that ANSWER. The fourth, the unattended hypothesis
/// loop, keeps 16 — see [`RESEARCH_LOOP_CHAIN_MAX_VISIT`], which is a declared
/// divergence with a measurement behind it and not the drift this module
/// exists to stop.
pub const ANSWER_CHAIN_MAX_VISIT: usize = 200;

/// The second DECLARED divergence: the hypothesis-research loop walks 16 nodes,
/// not 200.
///
/// This was nearly "fixed" to 200 in the name of making the configurations
/// match, and that would have been a latency regression traded for tidiness.
/// `crates/node/src/api.rs` records the measurement that set it — the values
/// were "reduced from 6/64 … each lock-hold drops from seconds to tens of ms
/// on a fat brain" — and that loop holds the brain mutex for the whole walk,
/// so its budget is `/brain/observe`'s tail latency. Goal (4) is inference in
/// ~1 ms, flat whatever was trained; a 12.5x wider walk on the lock is the
/// wrong direction and nothing here measured what it would cost.
///
/// The rule this keeps: a site may differ when a MEASUREMENT says it should,
/// and then it differs HERE, by name, next to the value it differs from.
/// What is forbidden is a site differing in its own argument list, where
/// nothing compares it to anything.
pub const RESEARCH_LOOP_CHAIN_MAX_VISIT: usize = 16;

/// The one DECLARED divergence: the unattended hypothesis-research loop
/// (`crates/node/src/api.rs`) keeps a permissive fabric gate.
///
/// It is here rather than inline because a divergence that lives in an
/// argument list is indistinguishable from the drift this module exists to
/// stop — the scan in `tests/answer_path_parameters_are_shared.rs` requires
/// every site to take its value from this module, so declaring it is the only
/// way to keep it.
///
/// Why it is allowed to differ: that loop does not RETURN an answer. It queues
/// a hypothesis for later confirmation, behind its own 0.5 confidence floor,
/// at most one per 30 s cycle, and is skipped entirely while the foreground is
/// training. PRIORITY ZERO is about what the brain tells the owner, and a
/// queued hypothesis is not that. The two routes that do answer the owner —
/// `/chat` and `/brain/chat` — take
/// [`ANSWER_FABRIC_CONFIDENCE_THRESHOLD`] with no local override.
///
/// What is NOT claimed: that 0.10 is the right value. It is the value that
/// site has always passed, commented there as "anything > random", and no
/// measurement here names a better one. Changing it needs a measurement of the
/// hypothesis queue, which nothing in this repository takes yet.
pub const RESEARCH_LOOP_FABRIC_CONFIDENCE_THRESHOLD: f32 = 0.10;

/// `binding_match_threshold` for `integrate_autonomous_tuned`: the OOV gate.
///
/// `integrate_autonomous` hardcodes 0.70 and `crates/node/src/api.rs` passes
/// 0.70 explicitly, so this one had NOT drifted. It is named here anyway, so
/// that the next caller takes it from the same place as the other three
/// instead of re-deriving which literal was right.
pub const ANSWER_BINDING_MATCH_THRESHOLD: f32 = 0.70;
