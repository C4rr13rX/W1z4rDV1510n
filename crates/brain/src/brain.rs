//! The top-level Brain struct per [`ARCHITECTURE.md`] §3 and §4.
//!
//! Owns a [`Fabric`] of pools and tracks the multi-pool firing history
//! needed for binding-concept emergence (spec §4.A) and for grounding-
//! report production at integration time (spec §2 / §4.D).
//!
//! Phase 1 supplied the substrate primitives.  This file ties them
//! together into something you can `create`, `observe` into, and
//! `integrate` from, with every output carrying a [`GroundingReport`].
//! That's the answer contract.

use ahash::AHashMap;
use std::collections::VecDeque;

use crate::action::{ActionEvent, ActionId};
use crate::annealer::{Annealer, AnnealerConfig};
use crate::eem::{Eem, EemConfig};
use crate::fabric::{Fabric, FabricConfig};
use crate::grounding::{AnswerWithGrounding, ConfidenceTier, GroundingReport};
use crate::identity::{
    BrainDeploymentSpec, BrainIdentitySpec, FeedbackLoopSpec, IdentityBuildError,
    PoolKind, PoolPrototypeRegistry,
};
use crate::network::{
    BrainId, GossipEquation, GossipMotif, NetworkState, PeerAccuracy,
    PeerContribution,
};
use crate::neuron::{NeuronId, NeuronKind, NeuronRef, PoolId, Neuron};
use crate::pool::{AtomEncoding, BytePassthroughEncoding, Pool, PoolConfig};

/// Which tier of substrate a [`BindingMatch`] succeeded at.  Stage 11
/// (concept-tier OOV) audits #1, #4 and #8 in the design log: the
/// tier tag travels alongside precision/recall so the single OOV gate
/// in `integrate_autonomous` can reason about WHAT kind of match it
/// passed (or rejected), without re-deriving it from the precision
/// number alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MatchTier {
    /// No binding produced any non-zero intersection at either tier.
    None,
    /// Match succeeded only at the atom level (raw bytes).  This is
    /// the Stage 9 baseline behavior.
    Atom,
    /// Match succeeded at the concept level (composite neurons that
    /// emerged via `recent_atoms` co-firing) and passed the coverage
    /// gate.  This is the Stage 11 add — for multi-character prompts
    /// like "photosynthesis" once a single concept neuron has emerged
    /// for the whole word, the binding match jumps from atom-precision
    /// ~0.4 to concept-precision 1.0.
    Concept,
}

/// Result of [`Brain::best_binding_match_v2`].  Carries the
/// precision/recall pair plus the tier at which the match was found.
#[derive(Debug, Clone, Copy)]
pub struct BindingMatch {
    pub precision: f32,
    pub recall:    f32,
    pub tier:      MatchTier,
}

/// Stage 13A — one neuron in a single pool's decoded extrusion.
#[derive(Debug, Clone)]
pub struct DecodedConcept {
    pub neuron:     NeuronRef,
    pub activation: f32,
    pub label:      String,
    pub bytes:      Vec<u8>,
}

/// Stage 13A — one pool's top-K decoded concepts from a settled state.
#[derive(Debug, Clone)]
pub struct PoolExtrusion {
    pub pool:    PoolId,
    pub decoded: Vec<DecodedConcept>,
}

/// Stage 13A — full multi-pool extrusion produced by
/// [`Brain::integrate_resonant`].  This is the substrate-level
/// support for the "epoxy mould" creation model: one input pattern
/// fires the constraint into the substrate, settling iterates the
/// resonance, and `pools` carries the coherent multi-pool state
/// that satisfied the constraint.
#[derive(Debug, Clone)]
pub struct ResonantExtrusion {
    pub iterations_run: usize,
    pub converged:      bool,
    pub pools:          Vec<PoolExtrusion>,
}

/// Stage 17.4 full — knobs for `Brain::run_eviction_pass`.  Per
/// [`ARCHITECTURE.md`] §17.4: low-salience, stale concept neurons get
/// paged out to cold tier in batches.  Atoms + binding-pool neurons
/// are never evicted by this policy.
#[derive(Debug, Clone, Copy)]
pub struct EvictionParams {
    /// Evict only concepts whose `salience_ema` is **below** this
    /// threshold.  Default `0.1` — well below the typical EMA of a
    /// concept that's been touched by any successful decode.
    pub max_salience_ema:    f32,
    /// Evict only concepts whose `last_fired_tick` is at least this
    /// many ticks in the past.  Default `1000`.
    pub min_stale_ticks:     u64,
    /// Cap evictions per pool per pass.  Default `1024` — bounded so a
    /// single pass doesn't lock the brain for an arbitrary amount of
    /// time.  Caller invokes the pass repeatedly to drain large brains.
    pub target_per_pool:     usize,
}

impl Default for EvictionParams {
    fn default() -> Self {
        Self {
            max_salience_ema: 0.1,
            min_stale_ticks:  1000,
            target_per_pool:  1024,
        }
    }
}

/// Stage 17.4 full — outcome of one `run_eviction_pass`.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct EvictionStats {
    pub pools_visited:    usize,
    pub neurons_evicted:  usize,
    pub errors:           usize,
    pub wall_time_ms:     u64,
}

impl BindingMatch {
    pub const NONE: Self = Self {
        precision: 0.0, recall: 0.0, tier: MatchTier::None,
    };
    /// Composite score used to pick the strongest match across tiers.
    #[inline]
    pub fn score(&self) -> f32 { self.precision * self.recall }
}

/// Sorted (pool, neuron) signature of a single tick's multi-pool
/// firing.  Used to detect recurring binding patterns: when the same
/// multi-pool firing-set has been observed `binding_emergence_threshold`
/// times within the history window, the brain births a binding concept
/// in the binding pool whose members reference every neuron in the
/// signature.
#[derive(Debug, Clone)]
struct MomentFingerprint {
    /// Sorted (pool, neuron) pairs used as the dedup key for
    /// recurring-fingerprint emergence detection.
    pairs: Vec<(PoolId, NeuronId)>,
    /// Original firing order (per-pool sequence).  Preserved
    /// separately from `pairs` so binding-pool concepts retain the
    /// temporal order in which atoms fired — which is what
    /// `decode_concept_members` walks to reconstruct the trained
    /// answer bytes.  Without this, decoded bindings appear in
    /// NeuronId-sorted order (e.g. 'animal' decodes as 'aaniml').
    /// Not part of equality/hash — same atom set in different
    /// firing orders is still the same binding for emergence
    /// purposes.
    ordered_per_pool: Vec<(PoolId, Vec<NeuronId>)>,
}

impl MomentFingerprint {
    fn from_fabric_moment(fired: &AHashMap<PoolId, Vec<NeuronId>>) -> Option<Self> {
        let mut pairs: Vec<(PoolId, NeuronId)> = fired.iter()
            .flat_map(|(&pid, ns)| ns.iter().map(move |&nid| (pid, nid)))
            .collect();
        if pairs.is_empty() { return None; }
        let pools_represented: std::collections::HashSet<PoolId> =
            pairs.iter().map(|(p, _)| *p).collect();
        // Binding candidates require ≥2 pools — single-pool firing is
        // a within-pool concept-emergence concern, not a binding one.
        if pools_represented.len() < 2 { return None; }
        // Capture firing order per pool BEFORE we sort `pairs`.
        let mut ordered_per_pool: Vec<(PoolId, Vec<NeuronId>)> = fired.iter()
            .map(|(&pid, ns)| (pid, ns.clone()))
            .collect();
        ordered_per_pool.sort_by_key(|(p, _)| *p);
        pairs.sort();
        Some(Self { pairs, ordered_per_pool })
    }
}

// A binding is a temporal episode, not merely a bag of fired neurons.
// `pairs` keeps the canonical set/multiset signature while
// `ordered_per_pool` distinguishes anagrams and reordered source/code
// sequences that carry different meaning.
impl std::cmp::PartialEq for MomentFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.pairs == other.pairs && self.ordered_per_pool == other.ordered_per_pool
    }
}
impl std::cmp::Eq for MomentFingerprint {}
impl std::hash::Hash for MomentFingerprint {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.pairs.hash(state);
        self.ordered_per_pool.hash(state);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BrainConfig {
    pub fabric: FabricConfig,
    /// "Consolidated" tier emergence threshold.  How many recurrences
    /// of the same multi-pool firing fingerprint promote a *full*
    /// binding — the kind that auto-populates the EEM with a grounded
    /// fact and gets gossiped to peers.  Defaults to 3.
    pub binding_emergence_threshold: u32,
    /// "Tentative" tier emergence threshold.  A lower bar that
    /// produces a binding neuron in the binding pool *without*
    /// registering an EEM fact or emitting gossip.  This is what /chat
    /// retrieval reads — so sparsely-trained pairs become recallable
    /// after `tentative_emergence_threshold` co-firings instead of
    /// having to clear the (much higher) consolidated bar.  Defaults
    /// to 1 — every cross-pool co-firing becomes a tentative binding.
    ///
    /// Set to `u32::MAX` to disable the tentative tier (legacy
    /// single-tier behavior).
    #[serde(default = "default_tentative_emergence_threshold")]
    pub tentative_emergence_threshold: u32,
    /// Sliding window over which fingerprint recurrence is counted.
    /// Older moments age out so only "recently sustained" co-firings
    /// produce bindings.
    pub moment_history_window: usize,
    /// Minimum precision×recall accepted by single-pool trained-binding
    /// retrieval. Kept per brain identity so a broad knowledge brain can be
    /// OOV-strict without imposing that policy on every deployment.
    #[serde(default = "default_min_atom_score")]
    pub min_atom_score: f32,
    /// Pressure-feedback band on the binding density signal
    /// (`bindings_per_observation`).  When the signal sits below
    /// `pressure_band_low` for `pressure_observation_grace` consecutive
    /// observations, the *consolidated* threshold ratchets down by 1
    /// (floor: 1).  When the signal climbs above `pressure_band_high`,
    /// the threshold ratchets up by 1 (ceiling: `pressure_threshold_max`).
    /// Hysteresis prevents oscillation.
    ///
    /// Set `pressure_adjust_enabled = false` to lock the threshold at
    /// its configured value (legacy static behavior).
    #[serde(default = "default_pressure_band_low")]
    pub pressure_band_low: f32,
    #[serde(default = "default_pressure_band_high")]
    pub pressure_band_high: f32,
    #[serde(default = "default_pressure_threshold_max")]
    pub pressure_threshold_max: u32,
    #[serde(default = "default_pressure_observation_grace")]
    pub pressure_observation_grace: u64,
    #[serde(default = "default_pressure_adjust_enabled")]
    pub pressure_adjust_enabled: bool,
    /// Per-pool config for the auto-created "binding pool" that hosts
    /// binding concepts.  Binding pool atoms are never used for sensor
    /// input — it only holds composite concepts whose members live in
    /// other pools.
    pub binding_pool_config: PoolConfig,
    /// EEM (Environmental Equation Matrix) config.  Spec §4.B.  An
    /// empty EEM is created on `Brain::new`; the caller seeds it via
    /// `brain.eem_mut().register_equation(...)`.
    pub eem: EemConfig,
    /// Temporal-prediction annealer config.  Spec §4.C.  An empty
    /// annealer is created on `Brain::new`; the brain automatically
    /// captures one frame per pool per `advance_tick`.
    pub annealer: AnnealerConfig,
}

fn default_tentative_emergence_threshold() -> u32 { 1 }
fn default_min_atom_score() -> f32 { 0.50 }
fn default_pressure_band_low() -> f32 { 0.001 }
fn default_pressure_band_high() -> f32 { 0.05 }
fn default_pressure_threshold_max() -> u32 { 10 }
fn default_pressure_observation_grace() -> u64 { 256 }
fn default_pressure_adjust_enabled() -> bool { true }

impl Default for BrainConfig {
    fn default() -> Self {
        let mut binding_pool_config = PoolConfig::defaults("binding", 0);
        // Binding pool doesn't need atom-sequence emergence — it only
        // hosts cross-pool members.  Disable by setting the threshold
        // unreachably high.
        binding_pool_config.concept_emergence_threshold = u32::MAX;
        Self {
            fabric: FabricConfig::default(),
            binding_emergence_threshold: 3,
            tentative_emergence_threshold: default_tentative_emergence_threshold(),
            moment_history_window: 64,
            min_atom_score: default_min_atom_score(),
            pressure_band_low: default_pressure_band_low(),
            pressure_band_high: default_pressure_band_high(),
            pressure_threshold_max: default_pressure_threshold_max(),
            pressure_observation_grace: default_pressure_observation_grace(),
            pressure_adjust_enabled: default_pressure_adjust_enabled(),
            binding_pool_config,
            eem: EemConfig::default(),
            annealer: AnnealerConfig::default(),
        }
    }
}

/// Stats surfaced via [`Brain::stats`].
#[derive(Debug, Clone, Default)]
pub struct BrainStats {
    pub tick:                u64,
    pub pool_count:          usize,
    pub total_neurons:       usize,
    pub total_concepts:      usize,
    pub total_binding:       usize,
    pub total_terminals:     usize,
    pub binding_pool_id:     PoolId,
    pub fingerprints_window: usize,
    /// Bindings that crossed `tentative_emergence_threshold` but not
    /// (yet) `binding_emergence_threshold`.  Visible to /chat retrieval;
    /// invisible to EEM chain exploration.
    pub tentative_bindings:  usize,
    /// Bindings that crossed `binding_emergence_threshold` and carry
    /// an EEM grounded fact.
    pub consolidated_bindings: usize,
    /// Current effective consolidated threshold (may differ from
    /// config if pressure-feedback adjusted it).
    pub current_threshold:   u32,
    /// Total observations (advance_tick calls with non-empty fingerprint)
    /// since brain construction or last snapshot restore.
    pub total_observations:  u64,
    /// `(tentative+consolidated)/total_observations` — drives the
    /// pressure feedback loop.
    pub binding_pressure:    f32,
}

/// One captured prompt→response moment, used by the self-test loop to
/// score recall without any externally-supplied evaluation set.  The
/// brain auto-captures these whenever a frame is observed in a pool
/// while another pool has a recent (within 2 ticks) frame — i.e. the
/// normal "ask in one pool, answer in another" interaction pattern.
#[derive(Debug, Clone)]
pub struct QaPair {
    pub prompt_pool:   PoolId,
    pub prompt:        Vec<u8>,
    pub response_pool: PoolId,
    pub response:      Vec<u8>,
    pub observed_tick: u64,
}

/// Bounded ring-buffer of recently observed QA pairs.  The brain uses it
/// as its self-supervised evaluation set: `Brain::self_test` samples
/// pairs from this buffer, replays the prompt, and scores how close the
/// decoded response is to the captured response.  Capacity-bounded so
/// memory stays finite across long training runs.
#[derive(Debug)]
pub struct QaDatabase {
    pairs:    VecDeque<QaPair>,
    capacity: usize,
}

impl QaDatabase {
    pub fn new(capacity: usize) -> Self {
        Self { pairs: VecDeque::with_capacity(capacity), capacity }
    }
    pub fn push(&mut self, p: QaPair) {
        if self.pairs.len() >= self.capacity { self.pairs.pop_front(); }
        self.pairs.push_back(p);
    }
    pub fn len(&self) -> usize { self.pairs.len() }
    pub fn is_empty(&self) -> bool { self.pairs.is_empty() }
    pub fn iter(&self) -> impl Iterator<Item = &QaPair> { self.pairs.iter() }
    /// Stride-sample up to `n` pairs spanning the buffer for deterministic
    /// coverage across recent + historic captures.
    pub fn sample(&self, n: usize) -> Vec<&QaPair> {
        if n == 0 || self.pairs.is_empty() { return Vec::new(); }
        if n >= self.pairs.len() { return self.pairs.iter().collect(); }
        let stride = (self.pairs.len() / n).max(1);
        self.pairs.iter().step_by(stride).take(n).collect()
    }
}

/// Self-tuning controller state — Phase D.  Hill-climbs the global
/// `decay_rate` knob using the self-test mean_byte_match as the
/// gradient signal.  Direction flips when recall worsens; magnitude
/// is multiplicative so the brain converges geometrically on a stable
/// fixpoint without ever being told what "good" decay looks like.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TuningState {
    /// Most recent `self_test` mean_byte_match used as the gradient.
    pub last_recall:     f32,
    /// Decay rate at the time `last_recall` was sampled.
    pub last_decay_rate: f32,
    /// Sign of the next nudge: +1.0 grows decay, -1.0 shrinks it.
    /// Flips whenever a step yields lower recall than the previous.
    pub direction:       f32,
    /// Best recall observed since the controller started, and the
    /// decay rate that produced it.  Surfaces in `/tuning_state` so
    /// the operator can see the substrate's discovered optimum.
    pub best_recall:     f32,
    pub best_decay_rate: f32,
    /// How many retune steps have run.
    pub steps:           u32,
    /// Condition-keyed memory: under what (concept_count_bucket,
    /// locked_count_bucket) condition was the best decay rate
    /// discovered.  The next retune at a similar condition uses this
    /// as a starting hypothesis rather than re-discovering from
    /// scratch.  Bucketed (log2 of counts) so similar-scale brains
    /// share entries.
    #[serde(serialize_with = "serialize_condition_best")]
    pub condition_best:  std::collections::HashMap<(u8, u8), (f32, f32)>,
}

fn serialize_condition_best<S>(
    map: &std::collections::HashMap<(u8, u8), (f32, f32)>,
    s: S,
) -> Result<S::Ok, S::Error>
where S: serde::Serializer {
    use serde::ser::SerializeSeq;
    let mut seq = s.serialize_seq(Some(map.len()))?;
    for (&(cbucket, lbucket), &(decay, recall)) in map {
        seq.serialize_element(&serde_json::json!({
            "concept_bucket": cbucket,
            "locked_bucket":  lbucket,
            "best_decay":     decay,
            "best_recall":    recall,
        }))?;
    }
    seq.end()
}

impl Default for TuningState {
    fn default() -> Self {
        Self {
            last_recall:     0.0,
            last_decay_rate: 2e-5,
            direction:       -1.0,  // shrink first — defaults are usually too aggressive
            best_recall:     0.0,
            best_decay_rate: 2e-5,
            steps:           0,
            condition_best:  std::collections::HashMap::new(),
        }
    }
}

/// What a single retune step did.  Reported back via `/retune`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TuningReport {
    pub recall_before:    f32,
    pub recall_after:     f32,
    pub decay_before:     f32,
    pub decay_after:      f32,
    pub direction_after:  f32,
    pub best_recall:      f32,
    pub best_decay_rate:  f32,
    pub concept_bucket:   u8,
    pub locked_bucket:    u8,
}

/// Per-pair recall outcome from a self-test pass.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SelfTestRecall {
    pub prompt:           String,
    pub expected:         String,
    pub decoded:          String,
    pub byte_match_ratio: f32,
    pub exact:            bool,
}

/// Aggregate self-test report — what the brain produces when it grades
/// its own recall against its captured QA buffer.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SelfTestReport {
    pub sampled:        usize,
    pub exact_recall:   usize,
    pub mean_byte_match: f32,
    pub per_pair:       Vec<SelfTestRecall>,
}

pub struct Brain {
    fabric:                       Fabric,
    config:                       BrainConfig,
    binding_pool_id:              PoolId,
    /// Fingerprint history.  Bounded by `moment_history_window`.
    moment_history:               VecDeque<MomentFingerprint>,
    /// Active-count of each fingerprint within the window.  Decays
    /// when an old fingerprint scrolls out of `moment_history`.
    binding_recurrences:          AHashMap<MomentFingerprint, u32>,
    /// Lifetime co-firing count per fingerprint — *never* decays.
    /// Sparsely-trained patterns (round-robin training where the
    /// same pair recurs far outside `moment_history_window`) never
    /// accumulate in `binding_recurrences`, so this is the fallback
    /// signal for promotion under sparse schedules.
    lifetime_recurrences:         AHashMap<MomentFingerprint, u32>,
    /// "Tentative" tier promotion — bindings that have a neuron in
    /// the binding pool but no EEM fact yet.  Visible to /chat
    /// retrieval (Stage 7 binding-pool routing reads ALL binding-pool
    /// concepts regardless of tier).  Invisible to EEM chain
    /// exploration.  Upgraded to `promoted_fingerprints` when the
    /// count crosses `binding_emergence_threshold`.
    tentative_promoted:           AHashMap<MomentFingerprint, NeuronId>,
    /// "Consolidated" tier promotion — these have an EEM grounded
    /// fact and have been gossiped to peers.
    promoted_fingerprints:        AHashMap<MomentFingerprint, NeuronId>,
    /// Count of advance_tick calls that produced a non-empty
    /// fingerprint.  Drives the pressure-feedback loop along with
    /// `len(tentative_promoted) + len(promoted_fingerprints)`.
    total_observations:           u64,
    /// Current pressure-adjusted *consolidated* threshold.  Tracks
    /// `config.binding_emergence_threshold` at construction, then
    /// drifts under the pressure feedback loop within
    /// `[1, config.pressure_threshold_max]`.
    current_threshold:            u32,
    /// Tick of the last pressure-adjust attempt; we only adjust once
    /// per `config.pressure_observation_grace` observations to avoid
    /// reacting to single-observation noise.
    last_pressure_check_obs:      u64,
    /// Phase 7: action layer state.  `None` until the caller
    /// designates an action pool via `designate_action_pool`.
    action_pool_id:               Option<PoolId>,
    /// Already-emitted action events waiting on outcome feedback.
    /// `feed_outcome` looks up the action_id and reinforces (or
    /// weakens) the source→action_neuron terminals per the outcome
    /// score.  Bounded by `action_history_max` to keep memory finite.
    pending_actions:              AHashMap<ActionId, ActionEvent>,
    next_action_id:               ActionId,
    /// Tracks which action ids have already been emitted this tick to
    /// prevent firing the same action neuron more than once per tick.
    emitted_this_tick:            ahash::AHashSet<NeuronRef>,
    /// Phase 5: Environmental Equation Matrix.  Owned by the brain so
    /// integration can consult equations and report their confidence
    /// alongside the fabric's.  Caller seeds equations directly via
    /// `eem_mut().register_equation(...)`.
    eem:                          Eem,
    /// Phase 6: Temporal-prediction annealer.  Captures one frame per
    /// pool per tick; consulted by `integrate_with_prediction`.
    annealer:                     Annealer,
    /// Phase 8: distributed-network state.  Pending outbound motif/
    /// equation gossip, received peer motifs, peer accuracy track
    /// record.  Transport (cluster) lives outside this crate; this
    /// state is the brain-side data model.
    network:                      NetworkState,
    /// Auto-captured prompt→response pairs for self-supervised recall
    /// scoring.  Phase A of the dynamical-feedback architecture: the
    /// brain grades itself against pairs it has actually observed
    /// during training, with no external answer key.
    qa_db:                        QaDatabase,
    /// Most recent frame observed in each pool plus the tick it landed
    /// on.  Used to recognise the cross-pool prompt→response pattern
    /// during `observe` so the QA buffer auto-populates.
    recent_frames:                AHashMap<PoolId, (Vec<u8>, u64)>,
    /// Self-tuning hill-climber state.  The brain feeds `self_test`
    /// recall back into pool decay_rate via this controller — Phase D.
    tuning:                       TuningState,
    /// Deployment-defined online feedback wiring. This state is rebuilt from
    /// the deployment spec after restore; delayed events are intentionally
    /// ephemeral because their source activations are tick-local.
    feedback_loops:               Vec<RuntimeFeedbackLoop>,
    delayed_feedback:             Vec<ScheduledFeedback>,
    feedback_events_emitted:      u64,
}

#[derive(Debug, Clone)]
struct RuntimeFeedbackLoop {
    source_pool: PoolId,
    target_pool: PoolId,
    spec: FeedbackLoopSpec,
    /// Integrates fractional dynamic gain into biologically meaningful
    /// firing frequency: gain 0.25 produces one feedback spike per four ticks.
    phase: f32,
}

#[derive(Debug, Clone)]
struct ScheduledFeedback {
    due_tick: u64,
    target_pool: PoolId,
    frame: Vec<u8>,
}

/// Cosine similarity between two sparse co-firing signatures.  A
/// signature is a `NeuronRef -> weight` map representing which OTHER
/// neurons this concept co-fires with (and how strongly).  Two
/// concepts are structurally analogous if their co-firing fingerprints
/// overlap — this is the architecture's "X is like Y" similarity
/// metric, not label-string matching.
fn cofiring_cosine(
    a: &ahash::AHashMap<NeuronRef, f32>,
    b: &ahash::AHashMap<NeuronRef, f32>,
) -> f32 {
    if a.is_empty() || b.is_empty() { return 0.0; }
    let mut dot = 0.0f32;
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    for (k, &va) in small {
        if let Some(&vb) = large.get(k) { dot += va * vb; }
    }
    let na: f32 = a.values().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.values().map(|v| v * v).sum::<f32>().sqrt();
    if na <= 0.0 || nb <= 0.0 { 0.0 } else { dot / (na * nb) }
}

/// Position-weighted byte match between two byte sequences.  1.0 means
/// identical-length identical-bytes; 0.0 means no positional matches.
/// Used by `Brain::self_test` to score decoded responses against
/// captured ground-truth.
fn byte_match_ratio_local(decoded: &[u8], expected: &[u8]) -> f32 {
    if decoded.is_empty() && expected.is_empty() { return 1.0; }
    let len = decoded.len().max(expected.len());
    if len == 0 { return 0.0; }
    let mut hits = 0usize;
    for i in 0..decoded.len().min(expected.len()) {
        if decoded[i] == expected[i] { hits += 1; }
    }
    hits as f32 / len as f32
}

impl Brain {
    /// Construct a fresh brain with no sensor pools yet.  The binding
    /// pool is auto-created at pool_id = `binding_pool_config.id`.
    /// Sensor pools get created by the caller via `create_pool`.
    pub fn new(config: BrainConfig) -> Self {
        let mut fabric = Fabric::new(config.fabric.clone());
        let binding_pool_id = config.binding_pool_config.id;
        let binding_pool = Pool::new(
            config.binding_pool_config.clone(),
            Box::new(BytePassthroughEncoding { prefix: "bind" }),
        );
        fabric.register_pool(binding_pool);
        let window = config.moment_history_window;
        let eem = Eem::new(config.eem.clone());
        let annealer = Annealer::new(config.annealer.clone());
        let initial_threshold = config.binding_emergence_threshold.max(1);
        Self {
            fabric,
            config,
            binding_pool_id,
            moment_history:          VecDeque::with_capacity(window),
            binding_recurrences:     AHashMap::new(),
            lifetime_recurrences:    AHashMap::new(),
            tentative_promoted:      AHashMap::new(),
            promoted_fingerprints:   AHashMap::new(),
            total_observations:      0,
            current_threshold:       initial_threshold,
            last_pressure_check_obs: 0,
            action_pool_id:        None,
            pending_actions:       AHashMap::new(),
            next_action_id:        1,
            emitted_this_tick:     ahash::AHashSet::new(),
            eem,
            annealer,
            network:               NetworkState::new(""),
            qa_db:                 QaDatabase::new(4096),
            recent_frames:         AHashMap::new(),
            tuning:                TuningState::default(),
            feedback_loops:        Vec::new(),
            delayed_feedback:      Vec::new(),
            feedback_events_emitted: 0,
        }
    }

    /// Resolve and install deployment feedback wiring for this brain.
    pub fn configure_feedback_loops(
        &mut self,
        identity: &BrainIdentitySpec,
        deployment: &BrainDeploymentSpec,
    ) -> Result<(), crate::identity::DeploymentValidationError> {
        deployment.validate(identity)?;
        let ids: std::collections::HashMap<&str, PoolId> = identity.pools.iter()
            .map(|pool| (pool.name.as_str(), pool.id)).collect();
        self.feedback_loops = deployment.feedback_loops.iter().map(|spec| RuntimeFeedbackLoop {
            source_pool: ids[spec.source_pool.as_str()],
            target_pool: ids[spec.target_pool.as_str()],
            spec: spec.clone(),
            phase: 0.0,
        }).collect();
        self.delayed_feedback.clear();
        Ok(())
    }

    pub fn feedback_loop_count(&self) -> usize { self.feedback_loops.len() }
    pub fn feedback_events_emitted(&self) -> u64 { self.feedback_events_emitted }

    /// Read-only access to the auto-captured QA buffer.
    pub fn qa_db(&self) -> &QaDatabase { &self.qa_db }

    /// Read-only access to the self-tuning controller state.
    pub fn tuning_state(&self) -> &TuningState { &self.tuning }

    /// One self-tuning step: run a self_test, hill-climb on decay_rate
    /// using the recall delta as the gradient signal.  Stores the
    /// best-seen (decay_rate, recall) pair indexed by current condition
    /// (concept_count_bucket, locked_count_bucket) so future runs in
    /// the same substrate regime warm-start from known-good values.
    ///
    /// No setpoint — the brain discovers its own decay optimum.
    pub fn retune(&mut self, sample_count: usize) -> TuningReport {
        let decay_before = self.tuning.last_decay_rate;
        let report = self.self_test(sample_count);
        let recall_now = report.mean_byte_match;
        let recall_before = self.tuning.last_recall;

        // Update best-ever recall first — we want to capture the
        // optimum even if the next nudge regresses.
        if recall_now > self.tuning.best_recall {
            self.tuning.best_recall = recall_now;
            self.tuning.best_decay_rate = decay_before;
        }

        // Bucket the current condition for the condition→value memory.
        // log2 floors give a coarse grid that handles brains spanning
        // many orders of magnitude in size.
        let total_concepts: usize = self.fabric.pool_ids().into_iter()
            .filter_map(|pid| self.fabric.pool(pid))
            .map(|p| {
                let r = p.read();
                r.iter_neurons().filter(|n| !n.is_atom()).count()
            })
            .sum();
        let locked = self.locked_terminal_count();
        let cbucket = (total_concepts as f64).max(1.0).log2().floor() as u8;
        let lbucket = (locked as f64).max(1.0).log2().floor() as u8;
        let cond = (cbucket, lbucket);

        // Remember the best decay we've seen at this condition.
        let cond_entry = self.tuning.condition_best.entry(cond)
            .or_insert((decay_before, recall_now));
        if recall_now > cond_entry.1 {
            *cond_entry = (decay_before, recall_now);
        }

        // Hill-climb step: if recall improved, keep direction; if it
        // worsened, reverse.
        if self.tuning.steps > 0 {
            if recall_now < recall_before { self.tuning.direction *= -1.0; }
        }

        // Step magnitude scales with the recall signal — strong
        // gradient → big step, no signal → tiny step.  This stops the
        // controller from drifting unboundedly when recall plateaus
        // (e.g. when the consolidation lock has saturated and decay
        // genuinely doesn't matter any more).
        let delta_mag = (recall_now - recall_before).abs();
        let step_pct = (0.01 + delta_mag * 0.5).clamp(0.005, 0.2);
        let step_factor = 1.0 + step_pct * self.tuning.direction;
        let mut decay_after = (decay_before * step_factor).clamp(1e-7, 0.01);

        // Condition-memory bias: if we've seen a better decay at this
        // condition before, blend toward it (20%).  Gives the brain a
        // long-horizon attractor toward the best-known config for its
        // current scale.
        if let Some(&(known_decay, known_recall)) = self.tuning.condition_best.get(&cond) {
            if known_recall > recall_now {
                decay_after = decay_after * 0.8 + known_decay * 0.2;
            }
        }

        // Apply to every pool's decay_rate.
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                let mut pw = pool.write();
                pw.config.decay_rate = decay_after;
            }
        }

        self.tuning.last_decay_rate = decay_after;
        self.tuning.last_recall = recall_now;
        self.tuning.steps += 1;

        TuningReport {
            recall_before,
            recall_after:    recall_now,
            decay_before,
            decay_after,
            direction_after: self.tuning.direction,
            best_recall:     self.tuning.best_recall,
            best_decay_rate: self.tuning.best_decay_rate,
            concept_bucket:  cond.0,
            locked_bucket:   cond.1,
        }
    }

    /// Total locked-terminal count across every pool.  These terminals
    /// are decay-exempt by the consolidation lock and form the brain's
    /// permanent recall floor.
    pub fn locked_terminal_count(&self) -> usize {
        self.fabric.pool_ids().into_iter()
            .filter_map(|pid| self.fabric.pool(pid))
            .map(|pool| {
                let p = pool.read();
                p.iter_neurons().map(|n| n.locked_terminal_count()).sum::<usize>()
            })
            .sum()
    }

    /// Chain integration — feed the integrate() answer back as a new
    /// query, looping until the answer stops changing or `max_hops` is
    /// reached.  Tests "answers that exist through the integration of
    /// training" when A→B and B→C are trained but A→C never was.
    ///
    /// Returns the chain of (query_bytes, answer_bytes) pairs in
    /// order so the caller can audit each hop's contribution.  The
    /// final tuple's answer is the integrated result.
    pub fn integrate_chain(
        &mut self,
        query_pool: PoolId,
        target_pool: PoolId,
        seed:       &[u8],
        max_hops:   usize,
    ) -> Vec<(Vec<u8>, Option<Vec<u8>>)> {
        let mut trail: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::with_capacity(max_hops);
        let mut current: Vec<u8> = seed.to_vec();
        let mut seen: ahash::AHashSet<Vec<u8>> = ahash::AHashSet::new();
        for _ in 0..max_hops {
            if !seen.insert(current.clone()) {
                // Already visited — converged onto a cycle.
                break;
            }
            // Fire the current query into query_pool, then feed it
            // ALSO into target_pool so the next hop can read it as a
            // "B is the new query" context.  The architecture is
            // pool-agnostic about chain direction: integrating from
            // query→target with current bytes injected into both pools
            // makes the next-hop binding match light up if it exists.
            self.fabric.observe(query_pool, &current);
            let ans = self.integrate(query_pool, target_pool);
            let decoded = ans.answer.clone();
            trail.push((current.clone(), decoded.clone()));
            match decoded {
                Some(bytes) if !bytes.is_empty() && bytes != current => {
                    current = bytes;
                }
                _ => break,  // null answer or fixpoint → stop chain
            }
        }
        trail
    }

    /// Run a self-supervised recall test: sample QA pairs from the
    /// auto-captured buffer, fire each prompt into its prompt pool,
    /// integrate into the response pool, and score the decoded answer
    /// against the captured response.  No external evaluation set
    /// required.
    ///
    /// Brain state is barely perturbed (one extra co-firing per probe
    /// at lr × cross_domain_scale for cross-pool atoms).  The
    /// consolidation lock prevents trained terminals from drifting.
    /// This is the brain's own dynamical-feedback signal: a falling
    /// `mean_byte_match` is the trigger Phase B will hook into to
    /// retune decay / emergence params.
    pub fn self_test(&mut self, sample_count: usize) -> SelfTestReport {
        let pairs: Vec<QaPair> = self.qa_db.sample(sample_count)
            .into_iter().cloned().collect();
        let mut report = SelfTestReport {
            sampled:         0,
            exact_recall:    0,
            mean_byte_match: 0.0,
            per_pair:        Vec::with_capacity(pairs.len()),
        };
        if pairs.is_empty() { return report; }
        let mut byte_sum = 0.0f32;
        for qp in &pairs {
            self.fabric.observe(qp.prompt_pool, &qp.prompt);
            let ans = self.integrate(qp.prompt_pool, qp.response_pool);
            let decoded = ans.answer.clone().unwrap_or_default();
            let bm = byte_match_ratio_local(&decoded, &qp.response);
            let exact = decoded == qp.response;
            byte_sum += bm;
            if exact { report.exact_recall += 1; }
            report.per_pair.push(SelfTestRecall {
                prompt:           String::from_utf8_lossy(&qp.prompt).to_string(),
                expected:         String::from_utf8_lossy(&qp.response).to_string(),
                decoded:          String::from_utf8_lossy(&decoded).to_string(),
                byte_match_ratio: bm,
                exact,
            });
        }
        report.sampled = pairs.len();
        report.mean_byte_match = byte_sum / (pairs.len() as f32);
        report
    }

    pub fn eem(&self) -> &Eem { &self.eem }
    pub fn eem_mut(&mut self) -> &mut Eem { &mut self.eem }
    pub fn annealer(&self) -> &Annealer { &self.annealer }
    pub fn annealer_mut(&mut self) -> &mut Annealer { &mut self.annealer }

    pub fn fabric(&self) -> &Fabric { &self.fabric }
    pub fn fabric_mut(&mut self) -> &mut Fabric { &mut self.fabric }
    pub fn binding_pool_id(&self) -> PoolId { self.binding_pool_id }
    pub fn min_atom_score(&self) -> f32 { self.config.min_atom_score }
    pub fn set_min_atom_score(&mut self, value: f32) {
        self.config.min_atom_score = value.clamp(0.0, 1.0);
    }

    /// Register a sensor pool.  The caller supplies the atomization
    /// contract via [`AtomEncoding`].  Returns the assigned pool id.
    pub fn create_pool(
        &mut self,
        config:   PoolConfig,
        encoding: Box<dyn AtomEncoding>,
    ) -> PoolId {
        let pool = Pool::new(config, encoding);
        self.fabric.register_pool(pool)
    }

    /// Observe a sensor frame into `pool_id`.  Fires atoms; returns
    /// the IDs that fired.  Multi-pool observation in the same tick
    /// is the normal mode — call `advance_tick` only when ready to
    /// close the moment.
    pub fn observe(&mut self, pool_id: PoolId, frame: &[u8]) -> Vec<NeuronId> {
        let qa_t0 = std::time::Instant::now();
        let now = self.fabric.current_tick();
        // QA capture: if some OTHER pool received a frame within the
        // last 2 ticks, this frame is its response.  Snapshot a
        // candidate then mutate qa_db to avoid an iter-and-mutate
        // borrow conflict.
        let captured: Option<(PoolId, Vec<u8>)> = self.recent_frames.iter()
            .filter(|(op, (_, t))| **op != pool_id && now.saturating_sub(*t) <= 2)
            .max_by_key(|(_, (_, t))| *t)
            .map(|(op, (f, _))| (*op, f.clone()));
        if let Some((prompt_pool, prompt_bytes)) = captured {
            if !frame.is_empty() && !prompt_bytes.is_empty() {
                self.qa_db.push(QaPair {
                    prompt_pool,
                    prompt:        prompt_bytes,
                    response_pool: pool_id,
                    response:      frame.to_vec(),
                    observed_tick: now,
                });
            }
        }
        self.recent_frames.insert(pool_id, (frame.to_vec(), now));
        let qa_capture_ns = qa_t0.elapsed().as_nanos() as u64;
        self.fabric.observe_profile.qa_capture_ns
            .fetch_add(qa_capture_ns, std::sync::atomic::Ordering::Relaxed);
        self.fabric.observe(pool_id, frame)
    }

    /// Install a transient, read-only query activation. This does not enter
    /// recent_frames or the Fabric moment and therefore cannot be learned.
    pub fn activate_for_prediction(&mut self, pool_id: PoolId, frame: &[u8]) -> Vec<NeuronId> {
        self.fabric.activate_for_prediction(pool_id, frame)
    }

    pub fn clear_prediction_activation(&mut self) {
        self.fabric.clear_prediction_activation();
    }

    /// Close the current tick.  Performs:
    /// 1. Cross-pool axon wiring for any co-fired pairs (Fabric does this).
    /// 2. Binding-concept emergence: if the current moment's multi-pool
    ///    firing fingerprint has recurred ≥ threshold times within the
    ///    history window, promote it.
    /// 3. Per-pool housekeeping (decay + prune).
    pub fn advance_tick(&mut self) {
        self.execute_feedback_loops();
        // Snapshot the current moment's firing BEFORE the fabric
        // advances (which clears it).  Build a fingerprint from
        // multi-pool firing for binding-concept tracking.  Fabric moments
        // intentionally contain atoms only to keep cross-pool Hebbian fanout
        // bounded.  Bindings, however, also need the already-collapsed
        // concepts or concept-tier inference has nothing to match against.
        // Enrich this private episodic snapshot with deterministically sorted
        // firing concepts without exposing them to cross-pool wiring.
        let fingerprint = {
            let moment = self.fabric.current_moment();
            let mut episodic_fired = moment.fired.clone();
            for (&pid, sequence) in episodic_fired.iter_mut() {
                if pid == self.binding_pool_id { continue; }
                if let Some(pool) = self.fabric.pool(pid) {
                    let pool = pool.read();
                    let mut concepts: Vec<NeuronId> = pool.currently_firing()
                        .filter(|nid| pool.get(*nid).is_some_and(|n| !n.is_atom()))
                        .collect();
                    concepts.sort_unstable();
                    sequence.extend(concepts);
                }
            }
            MomentFingerprint::from_fabric_moment(&episodic_fired)
        };

        // Capture per-pool activation frames into the annealer's
        // history.  One frame per pool per tick; empty frames are
        // skipped by Annealer::record_frame.  Done BEFORE
        // advance_tick because the fabric's housekeeping doesn't
        // clear pool activation, but the next tick's observes will.
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                let p = pool.read();
                let frame: AHashMap<NeuronId, f32> = p.currently_firing()
                    .map(|nid| (nid, p.activation(nid)))
                    .collect();
                drop(p);
                if !frame.is_empty() {
                    self.annealer.record_frame(pid, frame);
                }
            }
        }

        self.fabric.advance_tick();
        self.emitted_this_tick.clear();

        if let Some(fp) = fingerprint {
            self.register_fingerprint(fp);
        }
    }

    fn execute_feedback_loops(&mut self) {
        let now = self.fabric.current_tick();
        let mut newly_scheduled = Vec::new();

        for feedback in &mut self.feedback_loops {
            let Some(source) = self.fabric.pool(feedback.source_pool) else { continue; };
            let source = source.read();
            let mut labels: Vec<String> = source.currently_firing()
                .filter_map(|id| source.get(id).map(|n| n.label.clone()))
                .collect();
            if labels.is_empty() { continue; }
            labels.sort();
            labels.dedup();
            let control = source.control_state();
            let gain = feedback.spec.gain_mode.as_ref()
                .map(|mode| mode.evaluate(&control))
                .unwrap_or(feedback.spec.gain)
                .max(0.0);
            drop(source);

            feedback.phase += gain;
            let spikes = feedback.phase.floor() as usize;
            feedback.phase -= spikes as f32;
            if spikes == 0 { continue; }

            // Meta-pools learn a stable pattern-of-patterns fingerprint, not a
            // recursive copy of every lower-pool label.  This bounds feedback
            // frame size while preserving exact equality for recurring firing
            // sets and clean separation for different sets.
            let joined = labels.join("|");
            let digest = blake3::hash(joined.as_bytes());
            let frame = format!("feedback:{}:{}", feedback.spec.signal, digest.to_hex()).into_bytes();
            for _ in 0..spikes {
                newly_scheduled.push(ScheduledFeedback {
                    due_tick: now.saturating_add(feedback.spec.delay_ticks as u64),
                    target_pool: feedback.target_pool,
                    frame: frame.clone(),
                });
            }
        }

        self.delayed_feedback.extend(newly_scheduled);
        let mut pending = Vec::with_capacity(self.delayed_feedback.len());
        for event in self.delayed_feedback.drain(..) {
            if event.due_tick <= now {
                self.fabric.observe(event.target_pool, &event.frame);
                self.feedback_events_emitted = self.feedback_events_emitted.saturating_add(1);
            } else {
                pending.push(event);
            }
        }
        self.delayed_feedback = pending;
    }

    fn register_fingerprint(&mut self, fp: MomentFingerprint) {
        // Decay the oldest entry out of the window.  Lifetime count
        // does NOT decay — it's the sparse-schedule fallback.
        if self.moment_history.len() >= self.config.moment_history_window {
            if let Some(old) = self.moment_history.pop_front() {
                if let Some(c) = self.binding_recurrences.get_mut(&old) {
                    *c = c.saturating_sub(1);
                    if *c == 0 { self.binding_recurrences.remove(&old); }
                }
            }
        }
        self.moment_history.push_back(fp.clone());
        let win_count = self.binding_recurrences.entry(fp.clone()).or_insert(0);
        *win_count = win_count.saturating_add(1);
        let windowed = *win_count;

        let life_count = self.lifetime_recurrences.entry(fp.clone()).or_insert(0);
        *life_count = life_count.saturating_add(1);
        let lifetime = *life_count;

        self.total_observations = self.total_observations.saturating_add(1);

        // Hebbian frequency tracking: when an EXISTING promoted binding's
        // fingerprint recurs, bump its use_count so the decoder can weight
        // it.  This is what makes frequently-trained bindings (toddler
        // 'cat→animal' trained 8 times in toddler corpus + 3 times in
        // categorical_unified = 11) dominate competing bindings from
        // conflicting category entries ('cat→vehicle' trained only 3
        // times).  Without this, all bindings score by atom precision
        // (uniformly 1.0 for full overlap) and the decoder's smaller-
        // target-count tiebreak arbitrarily picks shorter category names.
        let existing_bid: Option<NeuronId> = self.promoted_fingerprints.get(&fp)
            .copied()
            .or_else(|| self.tentative_promoted.get(&fp).copied());
        if let Some(bid) = existing_bid {
            let now = self.fabric.current_tick();
            if let Some(bp) = self.fabric.pool(self.binding_pool_id) {
                let mut bp = bp.write();
                if let Some(n) = bp.get_mut(bid) {
                    n.use_count = n.use_count.saturating_add(1);
                    n.last_fired_tick = now;
                }
            }
        }

        // Promotion is driven by whichever signal is stronger: either
        // a dense recent burst (windowed) or sustained lifetime
        // co-occurrence (lifetime).  Sparse round-robin training
        // schedules — where the same pair recurs once every several
        // thousand ticks — never accumulate windowed count but DO
        // accumulate lifetime count, so this is the fix for the K-12
        // failure mode where 22K observations produced zero new
        // bindings under the old single-signal rule.
        let effective_count = windowed.max(lifetime);

        let tentative_thr = self.config.tentative_emergence_threshold;
        let consolidated_thr = self.current_threshold;
        let already_tentative = self.tentative_promoted.contains_key(&fp);
        let already_consolidated = self.promoted_fingerprints.contains_key(&fp);

        // Tier 1: tentative promotion.  Cheap — creates a binding
        // neuron in the binding pool but no EEM fact and no gossip.
        // Visible to /chat retrieval; invisible to EEM chain
        // exploration.
        if !already_tentative
            && !already_consolidated
            && effective_count >= tentative_thr
            && tentative_thr < u32::MAX
        {
            if let Some(id) = self.promote_binding_concept(&fp) {
                self.tentative_promoted.insert(fp.clone(), id);
            }
        }

        // Tier 2: consolidated promotion.  Registers an EEM grounded
        // fact and gossips the motif.  Upgrades the tentative-tier
        // binding (reuses its neuron id) when one exists.
        if !already_consolidated
            && effective_count >= consolidated_thr
        {
            let upgrade_id = self.tentative_promoted.remove(&fp);
            let id = match upgrade_id {
                Some(id) => Some(id),
                None     => self.promote_binding_concept(&fp),
            };
            if let Some(id) = id {
                self.network.pending_motif_out.push(GossipMotif {
                    source_brain:      self.network.brain_id.clone(),
                    fingerprint:       fp.pairs.clone(),
                    observation_count: effective_count,
                    local_confidence:  (effective_count as f32
                        / (consolidated_thr.max(1) as f32))
                        .min(1.0),
                    observed_at_tick:  self.fabric.current_tick(),
                });
                self.eem.register_fact(id, fp.pairs.clone());
                self.promoted_fingerprints.insert(fp, id);
            }
        }

        // Pressure feedback: every `pressure_observation_grace`
        // observations, nudge `current_threshold` up or down based
        // on the binding density signal.  Hysteresis: only adjust
        // when the signal is *outside* the [low, high] band.
        if self.config.pressure_adjust_enabled
            && self.total_observations
                .saturating_sub(self.last_pressure_check_obs)
                >= self.config.pressure_observation_grace
        {
            self.adjust_threshold_by_pressure();
            self.last_pressure_check_obs = self.total_observations;
        }
    }

    /// Bindings-per-observation ratio — the input to the pressure
    /// feedback loop.  Counts both tiers because both are valid
    /// bindings; the EEM-fact gate is what separates them.
    pub fn binding_pressure(&self) -> f32 {
        if self.total_observations == 0 {
            return 0.0;
        }
        let total_bindings = self.tentative_promoted.len()
            + self.promoted_fingerprints.len();
        (total_bindings as f32) / (self.total_observations as f32)
    }

    /// One step of the pressure feedback loop.  Public so callers
    /// (e.g. the supervisor) can force a recheck outside the
    /// observation-grace schedule.  Returns the *new* threshold.
    pub fn adjust_threshold_by_pressure(&mut self) -> u32 {
        let pressure = self.binding_pressure();
        let low = self.config.pressure_band_low;
        let high = self.config.pressure_band_high;
        let max = self.config.pressure_threshold_max.max(1);

        if pressure < low && self.current_threshold > 1 {
            self.current_threshold -= 1;
        } else if pressure > high && self.current_threshold < max {
            self.current_threshold += 1;
        }
        self.current_threshold
    }

    /// Failure-feedback hook: force-promote every fingerprint whose
    /// lifetime recurrence count is at least `min_count`, regardless
    /// of whether the windowed signal has decayed it.  Used by
    /// `/chat` when an OOV miss is reported — the system says "I
    /// don't know" precisely because the would-be binding never
    /// crossed threshold; this lets the next retrieval find it.
    ///
    /// Returns the list of fingerprint neuron ids that were newly
    /// tentative-promoted by this call.  Already-promoted
    /// fingerprints are not re-promoted (idempotent).
    pub fn force_promote_tentative(&mut self, min_count: u32) -> Vec<NeuronId> {
        let candidates: Vec<MomentFingerprint> = self.lifetime_recurrences.iter()
            .filter(|&(_, c)| *c >= min_count)
            .filter(|&(fp, _)|
                !self.tentative_promoted.contains_key(fp)
                && !self.promoted_fingerprints.contains_key(fp))
            .map(|(fp, _)| fp.clone())
            .collect();
        let mut out = Vec::with_capacity(candidates.len());
        for fp in candidates {
            if let Some(id) = self.promote_binding_concept(&fp) {
                self.tentative_promoted.insert(fp, id);
                out.push(id);
            }
        }
        out
    }

    /// Number of bindings in the tentative tier (binding-pool neuron
    /// exists, no EEM fact registered yet).
    pub fn tentative_binding_count(&self) -> usize {
        self.tentative_promoted.len()
    }

    /// Number of bindings in the consolidated tier (EEM fact registered).
    pub fn consolidated_binding_count(&self) -> usize {
        self.promoted_fingerprints.len()
    }

    /// Current pressure-adjusted consolidated emergence threshold.
    pub fn current_emergence_threshold(&self) -> u32 {
        self.current_threshold
    }

    /// Total observations (advance_tick calls with non-empty
    /// fingerprint) since construction or last snapshot restore.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    fn promote_binding_concept(&mut self, fp: &MomentFingerprint) -> Option<NeuronId> {
        let binding_pool = self.fabric.pool(self.binding_pool_id)?;
        let mut binding = binding_pool.write();
        // Composite label = sorted member references, joined.  Stable
        // and unique per fingerprint (used for dedup).
        let label: String = fp.pairs.iter()
            .map(|(p, n)| format!("p{}n{}", p, n))
            .collect::<Vec<_>>()
            .join("|");
        if binding.label_to_id(&label).is_some() {
            return None;  // already exists, idempotent.
        }
        // Members stored in FIRING ORDER (per-pool sequence as
        // observed at training time), NOT NeuronId-sorted order.
        // This preserves the temporal order required for clean
        // decoding via decode_concept_members — without it,
        // 'animal' atoms (a,n,i,m,a,l) decode as 'aaniml' because
        // sorting by NeuronId interleaves the duplicates.
        let members: Vec<NeuronRef> = fp.ordered_per_pool.iter()
            .flat_map(|(pid, ns)| ns.iter().map(|&nid| NeuronRef::new(*pid, nid)))
            .collect();
        let max_w = binding.config.max_weight;
        let now = self.fabric.current_tick();
        // Create the binding concept directly with cross-pool members.
        let id = binding.neuron_count() as NeuronId;
        let mut neuron = Neuron::new_concept(
            id, label.clone(), NeuronKind::Excitatory, members.clone(), now,
        );
        // Wire concept → member terminals top-down so activating the
        // binding fires its constituent neurons in all pools.
        let mut added: usize = 0;
        for m in &members {
            if neuron.reinforce_terminal(*m, 0.5, now, max_w) { added += 1; }
        }
        // append_neuron pushes the new concept (with its `added` terminals
        // already attached) onto the binding pool, so the counter grows
        // by exactly `added`.
        binding.total_terminals += added;
        binding.append_neuron(neuron, label.clone());
        drop(binding);

        // Wire member → binding terminals bottom-up so co-firing all
        // members activates the binding.
        let binding_ref = NeuronRef::new(self.binding_pool_id, id);
        for m in &members {
            if let Some(p) = self.fabric.pool(m.pool) {
                let mut pp = p.write();
                let mxw = pp.config.max_weight;
                let was_added = if let Some(n) = pp.get_mut(m.neuron) {
                    n.reinforce_terminal(binding_ref, 0.5, now, mxw)
                } else { false };
                if was_added { pp.total_terminals += 1; }
            }
        }
        Some(id)
    }

    /// Raw read of a pool's current activation map.  No interpretation,
    /// no grounding — just the substrate state.  Spec §1.6.
    pub fn read_activation(&self, pool_id: PoolId) -> AHashMap<NeuronId, f32> {
        let pool = match self.fabric.pool(pool_id) {
            Some(p) => p,
            None    => return AHashMap::new(),
        };
        let p = pool.read();
        let mut out = AHashMap::new();
        for nid in p.currently_firing() {
            out.insert(nid, p.activation(nid));
        }
        out
    }

    /// Produce an [`AnswerWithGrounding`] by propagating from
    /// `query_pool`'s currently-firing state and reading the resulting
    /// activation in `target_pool`.  Every output carries a
    /// Recursively compute the depth of a concept neuron: atoms are
    /// depth 0; concepts whose members are all atoms are depth 1;
    /// concepts whose members include concepts of depth N are depth
    /// N+1.  Used by [`Self::integrate_concept_first`] to pick the
    /// DEEPEST firing concept — the substrate's "highest layer that
    /// matches the input" per ARCHITECTURE.md §4.D.1.
    pub fn concept_depth(&self, pool_id: PoolId, nid: NeuronId) -> usize {
        let p = match self.fabric.pool(pool_id) {
            Some(p) => p,
            None    => return 0,
        };
        let p_read = p.read();
        let n = match p_read.get(nid) {
            Some(n) => n,
            None    => return 0,
        };
        if n.is_atom() { return 0; }
        let mut max_d = 0;
        for m in &n.members {
            if m.pool != pool_id { continue; }
            let d = self.concept_depth_inner(&p_read, m.neuron, 0, 16);
            if d > max_d { max_d = d; }
        }
        max_d + 1
    }

    fn concept_depth_inner(&self, p: &crate::pool::Pool, nid: NeuronId,
                            current: usize, cap: usize) -> usize {
        if current >= cap { return current; }
        let n = match p.get(nid) {
            Some(n) => n,
            None    => return current,
        };
        if n.is_atom() { return current; }
        let mut max_d = current;
        for m in &n.members {
            if m.pool != p.id() { continue; }
            let d = self.concept_depth_inner(p, m.neuron, current + 1, cap);
            if d > max_d { max_d = d; }
        }
        max_d
    }

    /// Concept-first retrieval — the user's "always look at the
    /// deepest level we see" inference contract from ARCHITECTURE.md
    /// §4.D.1.
    ///
    /// 1. Propagate from query_pool across the fabric using the
    ///    substrate's existing axon/dendrite wiring (no scoring
    ///    formula — just Hebbian terminals doing their job).
    /// 2. In target_pool, find the DEEPEST emerged concept neuron
    ///    with non-trivial activation.  That concept's hierarchical
    ///    depth tells us it represents the highest-layer match the
    ///    substrate has crystallised for this input.
    /// 3. Decode it via decode_concept_members and return.
    ///
    /// Falls back to atom-level reassembly only when NO concept in
    /// target_pool reaches the activation floor.  Trained-input
    /// retrieval should hit deterministic 100% via this path because
    /// the cross-pool axon between trained query-concept and trained
    /// target-concept lights up the target deterministically.
    ///
    /// Parallel to [`Self::integrate`] — does NOT touch its scoring
    /// formula or downstream consumers.
    pub fn integrate_concept_first(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> Option<Vec<u8>> {
        let propagated = self.fabric.propagate(query_pool);
        let target_acts = propagated.get(&target_pool)?;
        if target_acts.is_empty() { return None; }

        let pool_handle = self.fabric.pool(target_pool)?;
        let p = pool_handle.read();

        const ACTIVATION_FLOOR: f32 = 0.001;
        // Skip runaway-emergence concepts.  The substrate's emergence
        // can produce concepts whose decode runs many characters by
        // tiling a shorter pattern (e.g., "animalanimalanimal" or
        // "bodybodybody" emerging under dense-burst training).
        // Trained category responses are short (≤ 18 bytes for
        // "musical_instrument", ≤ 6 for most).  A 24-byte cap keeps
        // legitimate trained answers while filtering runaways.
        const MAX_REASONABLE_DECODE: usize = 24;

        // Score concepts by AVG-MEMBER-ACTIVATION (the same signal
        // /integrate's Stage 6 selection uses): the concept whose
        // members ALL fire strongly is the one whose pattern was
        // actually triggered by the cross-pool propagation.  A
        // concept whose members partly fire (because it includes
        // extra atoms from unrelated words) gets a lower score.
        //
        // Tiebreak by concept depth (deeper hierarchy wins).
        //
        // Sanity filters reject runaway-emergence concepts:
        //   - decode > 24 bytes (longer than any trained categorical
        //     response; runaway tilings exceed this)
        //   - low unique-byte ratio for decodes >= 8 bytes (catches
        //     'bodybody' / 'animalanim' style partial repeats)
        let mut best: Option<(NeuronId, Vec<u8>, f32, usize)> = None;
        for (nid, _act) in target_acts.iter() {
            let n = match p.get(*nid) {
                Some(n) => n,
                None    => continue,
            };
            if n.is_atom() { continue; }
            // Compute avg member activation in target pool.
            let in_pool_members: Vec<NeuronId> = n.members.iter()
                .filter(|m| m.pool == target_pool)
                .map(|m| m.neuron)
                .collect();
            if in_pool_members.is_empty() { continue; }
            let member_sum: f32 = in_pool_members.iter()
                .map(|mid| target_acts.get(mid).copied().unwrap_or(0.0))
                .sum();
            let avg_member_act = member_sum / in_pool_members.len() as f32;
            if avg_member_act < ACTIVATION_FLOOR { continue; }
            // Decode + sanity filters.
            let decoded = p.decode_concept_members(&n.members);
            if decoded.is_empty() || decoded.len() > MAX_REASONABLE_DECODE {
                continue;
            }
            if decoded.len() >= 8 {
                let unique: ahash::AHashSet<u8> = decoded.iter().copied().collect();
                let ratio = unique.len() as f32 / decoded.len() as f32;
                if ratio < 0.6 {
                    continue;
                }
            }
            let depth = {
                let mut max_d = 0;
                for m in &n.members {
                    if m.pool != target_pool { continue; }
                    let d = self.concept_depth_inner(&p, m.neuron, 0, 16);
                    if d > max_d { max_d = d; }
                }
                max_d + 1
            };
            // Score = avg_member_activation × sqrt(member_count).
            // The √len factor rewards longer concepts when their
            // member activation pattern is well-covered.  Pure short
            // morphemes ('ala', 'oo') get a small √2-√3 boost, but
            // a fully-activated word-level concept ('animal' = √6)
            // wins decisively over morpheme fragments.
            let length_factor = (in_pool_members.len() as f32).sqrt();
            let score = avg_member_act * length_factor;
            match &best {
                None => best = Some((*nid, decoded, score, depth)),
                Some((_, _, prev_score, prev_depth)) => {
                    if score > *prev_score
                       || (score == *prev_score && depth > *prev_depth) {
                        best = Some((*nid, decoded, score, depth));
                    }
                }
            }
        }

        match best {
            Some((_nid, decoded, _score, _depth)) => {
                Some(decoded)
            }
            None => {
                // No concept fired — fall back to atom-level
                // reassembly.  Walk firing target atoms in
                // descending activation order and emit their bytes.
                let mut atom_acts: Vec<(NeuronId, f32)> = target_acts.iter()
                    .filter(|(nid, _)| p.get(**nid).map_or(false, |n| n.is_atom()))
                    .filter(|(_, a)| **a >= ACTIVATION_FLOOR)
                    .map(|(k, v)| (*k, *v))
                    .collect();
                if atom_acts.is_empty() { return None; }
                atom_acts.sort_by(|a, b|
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut out = Vec::new();
                for (nid, _) in atom_acts.iter().take(32) {
                    if let Some(n) = p.get(*nid) {
                        out.extend(p.encoding_reassemble(&[(n.label.as_str(), 1.0)]));
                    }
                }
                if out.is_empty() { None } else { Some(out) }
            }
        }
    }

    /// [`GroundingReport`] per spec §2.7.
    ///
    /// The brain doesn't decide whether to surface uncertainty —
    /// uncertainty is always surfaced.  Caller reads the report.
    pub fn integrate(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> AnswerWithGrounding {
        // 1. Propagate from query pool across the fabric.
        let propagated = self.fabric.propagate(query_pool);

        // 2. Input atom coverage in the query pool: how many of the
        // currently-firing neurons there map to known atoms.  At this
        // phase every fired neuron is by construction in `label_to_id`,
        // so coverage is 1.0 if anything fired and 0 otherwise.  Once
        // sensor adapters route unknown bytes through atom birth this
        // remains 1.0 for known input, with adaptive plasticity per
        // spec §2.5 handling the novelty signal at the concept layer.
        let query_pool_handle = match self.fabric.pool(query_pool) {
            Some(p) => p,
            None    => return AnswerWithGrounding::unknown(
                format!("unknown pool id {}", query_pool), target_pool),
        };
        let fired_in_query = query_pool_handle.read()
            .currently_firing().collect::<Vec<_>>();
        if fired_in_query.is_empty() {
            return AnswerWithGrounding::unknown(
                "no input observed in query pool — cannot integrate", target_pool);
        }
        let input_atom_coverage = 1.0_f32;

        // 3. Read target pool activation from propagation results.
        let empty: AHashMap<NeuronId, f32> = AHashMap::new();
        let target_activation = propagated.get(&target_pool).unwrap_or(&empty);

        // 4. Compute Jaccard between the strongest target concept's
        // member-atom set (within the same target pool only) and the
        // input atoms.  Internal use only — spec §1.6 says no Jaccard
        // ranker exposed at the API surface; here it's a measurement
        // for the grounding report, not a selection mechanism.
        let target_pool_handle = match self.fabric.pool(target_pool) {
            Some(p) => p,
            None    => return AnswerWithGrounding::unknown(
                format!("unknown target pool id {}", target_pool), target_pool),
        };
        let target = target_pool_handle.read();

        // Stage 3b (selection-side): build a set of target-pool
        // concepts that are DIRECTLY targeted by a query-pool concept
        // via Stage 3's concept→concept cross-pool terminal.  These
        // are the specific binding partners the substrate identified
        // during training; they get a score boost in selection so the
        // density-based picker doesn't get drowned out by short,
        // shared atom-level concepts ("00" beating "C000" etc.).
        let directly_targeted: ahash::AHashSet<NeuronId> = {
            let q = query_pool_handle.read();
            let mut set = ahash::AHashSet::new();
            for nid in q.currently_firing() {
                if let Some(qn) = q.get(nid) {
                    if qn.is_atom() { continue; }
                    for t in &qn.terminals {
                        if t.target.pool != target_pool { continue; }
                        if let Some(tn) = target.get(t.target.neuron) {
                            if !tn.is_atom() {
                                set.insert(t.target.neuron);
                            }
                        }
                    }
                }
            }
            set
        };

        // Phase B: the LOCKED variant of directly_targeted — target
        // concepts the query reaches through high-consolidation
        // (decay-exempt) cross-pool terminals.  These are the canonical
        // trained bindings; their score must dominate atom-soup
        // byproducts (concepts that share some member atoms but were
        // never trained as the actual response).
        //
        // The signal is structural: a terminal whose consolidation has
        // crossed CONSOLIDATION_LOCK has been reinforced on at least
        // that many distinct ticks — the brain's own evidence that
        // this is a trained pairing, not a random co-firing.  Selection
        // boost is 16× so a single locked path slams the canonical
        // answer into first place over any number of atom-overlap
        // concepts at equivalent avg_member_act.
        let locked_targeted: ahash::AHashSet<NeuronId> = {
            let q = query_pool_handle.read();
            let mut set = ahash::AHashSet::new();
            for nid in q.currently_firing() {
                if let Some(qn) = q.get(nid) {
                    for t in &qn.terminals {
                        if t.target.pool != target_pool { continue; }
                        if t.consolidation < crate::neuron::Neuron::CONSOLIDATION_LOCK { continue; }
                        if let Some(tn) = target.get(t.target.neuron) {
                            if !tn.is_atom() {
                                set.insert(t.target.neuron);
                            }
                        }
                    }
                }
            }
            set
        };

        // Stage 7 (binding-pool routing): find the binding concept
        // whose QUERY-POOL member atoms BEST MATCH the currently-firing
        // query atom set.  Its target-pool member atoms then form a
        // strong hint for which action concept the substrate associated
        // with this prompt during training.
        //
        // We rank bindings by precision × recall over query atoms
        // rather than raw activation — raw activation favors bindings
        // with more total atoms (more activation paths in propagation),
        // which is the wrong signal.  We want the binding whose
        // text-pool member set IS the query's firing set.
        // Phase B v2 — also remember the binding NEURON id and its
        // match score, not just the atom set, so we can read the
        // canonical response sequence directly from the binding's
        // target-pool members (in firing order) when the match is
        // exact and the binding is trained.
        let (binding_id_opt, binding_score, binding_target_atoms): (
            Option<NeuronId>, f32, ahash::AHashSet<NeuronId>
        ) = {
            let bpid = self.binding_pool_id;
            if bpid == target_pool || bpid == query_pool {
                (None, 0.0, ahash::AHashSet::new())
            } else {
                let q_atoms: ahash::AHashSet<NeuronId> = {
                    let q = query_pool_handle.read();
                    q.currently_firing()
                        .filter(|nid| q.get(*nid).map_or(false, |n| n.is_atom()))
                        .collect()
                };
                if q_atoms.is_empty() {
                    (None, 0.0, ahash::AHashSet::new())
                } else if let Some(bp) = self.fabric.pool(bpid) {
                    let bp_read = bp.read();
                    let mut best: Option<(NeuronId, f32)> = None;
                    for n in bp_read.iter_neurons() {
                        if n.is_atom() { continue; }
                        let bind_query: ahash::AHashSet<NeuronId> = n.members.iter()
                            .filter(|m| m.pool == query_pool)
                            .map(|m| m.neuron)
                            .collect();
                        let bind_target_has = n.members.iter()
                            .any(|m| m.pool == target_pool);
                        if bind_query.is_empty() || !bind_target_has { continue; }
                        let intersect = bind_query.iter()
                            .filter(|a| q_atoms.contains(a))
                            .count() as f32;
                        let precision = intersect / bind_query.len() as f32;
                        let recall = intersect / q_atoms.len() as f32;
                        let score = precision * recall;
                        if score > 0.0 && best.map_or(true, |(_, s)| score > s) {
                            best = Some((n.id, score));
                        }
                    }
                    match best {
                        Some((bnid, sc)) => {
                            let atoms = if let Some(n) = bp_read.get(bnid) {
                                n.members.iter()
                                    .filter(|m| m.pool == target_pool)
                                    .map(|m| m.neuron)
                                    .collect()
                            } else { ahash::AHashSet::new() };
                            (Some(bnid), sc, atoms)
                        }
                        None => (None, 0.0, ahash::AHashSet::new()),
                    }
                } else { (None, 0.0, ahash::AHashSet::new()) }
            }
        };

        // Phase B v2 — exact-binding shortcut: if a binding neuron
        // matches the query at near-perfect precision×recall (>= 0.95)
        // AND has been used at least twice (use_count > 1, i.e. the
        // trained pathway), emit ITS target-pool member subsequence
        // directly in FIRING ORDER.  This bypasses the target-concept
        // selector entirely — the binding IS the canonical
        // prompt→response record from training, and reading its
        // ordered target members reproduces the trained answer
        // bit-for-bit.
        //
        // The score gate (0.95) lets near-anagram matches (e.g.
        // adding a single trailing space) still hit; the use_count
        // gate ensures we don't read a one-off binding that hasn't
        // been validated by repeated co-firing.  The "answer" wraps
        // the binding's ordered atom-leaf decode so callers don't
        // need to know about the bypass — it's still an
        // AnswerWithGrounding.
        if binding_score >= 0.95 {
            if let Some(bnid) = binding_id_opt {
                if let Some(bp) = self.fabric.pool(self.binding_pool_id) {
                    let bp_read = bp.read();
                    let (use_count, target_members): (u64, Vec<NeuronRef>) =
                        match bp_read.get(bnid) {
                            Some(n) => (
                                n.use_count,
                                n.members.iter()
                                    .filter(|m| m.pool == target_pool)
                                    .copied()
                                    .collect(),
                            ),
                            None => (0, Vec::new()),
                        };
                    drop(bp_read);
                    if use_count >= 2 && !target_members.is_empty() {
                        let decoded = target.decode_concept_members(&target_members);
                        if !decoded.is_empty() {
                            return AnswerWithGrounding {
                                answer: Some(decoded),
                                grounding: GroundingReport {
                                    input_atom_coverage,
                                    strongest_match: Some(NeuronRef::new(
                                        self.binding_pool_id, bnid)),
                                    strongest_match_jaccard: binding_score,
                                    composition_used: Vec::new(),
                                    fabric_confidence: binding_score,
                                    eem_confidence: None,
                                    annealer_confidence: None,
                                    integrated_confidence: binding_score,
                                    outside_grounding: false,
                                    // Cross-pool composition is by
                                    // definition speculative (spec
                                    // §2.3): the answer came from a
                                    // binding-pool concept whose
                                    // members span more than one
                                    // pool, not from a direct retrieve
                                    // of a single trained pair.
                                    speculation_flag: query_pool != target_pool,
                                    peer_contributions: Vec::new(),
                                },
                                confidence_tier: ConfidenceTier::from_confidence(
                                    binding_score, false, false),
                                next_steps_if_ungrounded: Vec::new(),
                            };
                        }
                    }
                }
            }
        }

        // Selection (Stage 6 — coverage-aware density):
        //
        //   score = avg_member_activation
        //         × pathway_boost
        //         × sqrt(direct_member_count)
        //         × unique_ratio
        //
        // where:
        //   avg_member_activation = mean of target_activation across the
        //                           concept's in-pool member neurons.
        //                           High when ALL members of the concept
        //                           are firing strongly — i.e., the
        //                           concept's full pattern is present.
        //   sqrt(direct_member_count) — rewards longer concepts when
        //                           coverage is good (so "color" wins
        //                           over "co" when all of c,o,l,r are
        //                           active), without letting concepts
        //                           with 20+ repetitions dominate (sqrt
        //                           grows slowly).
        //   unique_ratio = unique_member_ids / direct_member_count.
        //                  Penalizes concepts with REPEATED members —
        //                  the long-pattern concepts from Experiment A
        //                  (4× "x+y}" pattern) have unique_ratio 0.25
        //                  while clean concepts have 1.0.
        //   pathway_boost — same 4× factor as before for concepts that
        //                  a query-pool concept directly targets via
        //                  Stage 3 concept→concept cross-pool wiring.
        //
        // This replaces the earlier `(act × boost) / expanded_size`
        // density rule, which favored short concepts so aggressively
        // that trained full-word responses lost to 2-byte prefix
        // concepts during chat retrieval.
        let pathway_boost = 4.0_f32;
        let mut strongest: Option<(NeuronRef, f32)> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        let mut all_active_concepts: Vec<NeuronRef> = Vec::new();
        for (&nid, &act) in target_activation.iter() {
            if act < 0.001 { continue; }
            if let Some(n) = target.get(nid) {
                if !n.is_atom() {
                    let r = NeuronRef::new(target_pool, nid);
                    all_active_concepts.push(r);

                    let in_pool_members: Vec<NeuronId> = n.members.iter()
                        .filter(|m| m.pool == target_pool)
                        .map(|m| m.neuron)
                        .collect();
                    let member_count = in_pool_members.len().max(1);
                    let unique_count: usize = in_pool_members.iter()
                        .collect::<ahash::AHashSet<_>>().len().max(1);
                    let member_activation_sum: f32 = in_pool_members.iter()
                        .map(|nid| target_activation.get(nid).copied().unwrap_or(0.0))
                        .sum();
                    let avg_member_act = member_activation_sum / member_count as f32;

                    let boost = if directly_targeted.contains(&nid) { pathway_boost } else { 1.0 };
                    let length_factor = (member_count as f32).sqrt();

                    // Information factor: rewards unique-atom count
                    // while penalizing pure repetition.  Linear ratio
                    // was too punitive on small repetitions like
                    // "animal" (one repeated 'a') vs "nimal" (no
                    // repetition).  This form gives "animal" a slight
                    // edge over "nimal" because of its extra atom,
                    // while still strongly penalizing concepts that
                    // are mostly repetition (4× "x+y}" loses to
                    // "x+y}").
                    //
                    //   info = unique_count - 0.3 × (member_count - unique_count)
                    //
                    // For 4-unique-of-4: info = 4
                    // For 4-unique-of-8: info = 4 - 0.3*4 = 2.8
                    // For 4-unique-of-16: info = 4 - 0.3*12 = 0.4
                    // For 6-unique-of-7 (vehicle): info = 6 - 0.3 = 5.7
                    // For 5-unique-of-6 (animal): info = 5 - 0.3 = 4.7
                    // For 5-unique-of-5 (nimal): info = 5
                    let repetition = (member_count - unique_count) as f32;
                    let info_factor = (unique_count as f32 - 0.3 * repetition).max(0.1);

                    // Stage 7 binding-pool routing boost.  If the
                    // strongest active binding's target-pool atoms are
                    // a subset of (or close match to) this candidate
                    // concept's member-atom set, boost it.  This routes
                    // selection through the pair-specific binding
                    // structure the substrate built during training.
                    let binding_boost = if !binding_target_atoms.is_empty() {
                        let in_pool_member_set: ahash::AHashSet<NeuronId> =
                            in_pool_members.iter().copied().collect();
                        let matched = binding_target_atoms.iter()
                            .filter(|a| in_pool_member_set.contains(a))
                            .count() as f32;
                        let coverage_of_binding = matched / binding_target_atoms.len() as f32;
                        if coverage_of_binding >= 0.99 {
                            8.0
                        } else if coverage_of_binding >= 0.75 {
                            3.0
                        } else if coverage_of_binding >= 0.5 {
                            1.5
                        } else {
                            1.0
                        }
                    } else { 1.0 };

                    let score = avg_member_act * boost * length_factor * info_factor * binding_boost;
                    let _ = locked_targeted.contains(&nid);  // signal retained for Phase B v2 below
                    // Hold on to raw activation for grounding metric.
                    let _ = act;
                    let _ = target.expanded_size(&n.members);
                    if score > best_score {
                        best_score = score;
                        strongest = Some((r, act));
                    }
                }
            }
        }

        // Outside-grounding detection: nothing in the target pool was
        // sufficiently activated.  Spec §2.2.
        let (strongest_match, strongest_activation) = match strongest {
            Some(x) => (Some(x.0), x.1),
            None    => {
                return AnswerWithGrounding {
                    answer: None,
                    grounding: GroundingReport {
                        input_atom_coverage,
                        strongest_match: None,
                        strongest_match_jaccard: 0.0,
                        composition_used: all_active_concepts,
                        fabric_confidence: 0.0,
                        eem_confidence: None,
                        annealer_confidence: None,
                        integrated_confidence: 0.0,
                        outside_grounding: true,
                        speculation_flag: false,
                        peer_contributions: Vec::new(),
                    },
                    confidence_tier: ConfidenceTier::Ungrounded,
                    next_steps_if_ungrounded: vec![
                        crate::grounding::RequestObservation {
                            domain: "target pool has no activated concept for this input".into(),
                            examples_needed: 1,
                            why: "extend grounding by co-observing query and target streams together".into(),
                            pool: target_pool,
                        },
                    ],
                };
            }
        };

        // Jaccard against the strongest match's member set (for the
        // grounding report only).  Pure measurement, not a gate.
        let strongest_jaccard = if let Some(r) = strongest_match {
            let input_set: std::collections::HashSet<NeuronId> = fired_in_query
                .iter().copied().collect();
            // Member NeuronRefs may live in any pool.  We compare only
            // those whose pool == query_pool — atoms in the target
            // concept that originate from the query stream.
            if let Some(n) = target.get(r.neuron) {
                let concept_in_query: std::collections::HashSet<NeuronId> = n.members.iter()
                    .filter(|m| m.pool == query_pool)
                    .map(|m| m.neuron)
                    .collect();
                if input_set.is_empty() && concept_in_query.is_empty() {
                    0.0
                } else {
                    let inter = input_set.intersection(&concept_in_query).count() as f32;
                    let union = input_set.union(&concept_in_query).count().max(1) as f32;
                    inter / union
                }
            } else { 0.0 }
        } else { 0.0 };

        // Speculation flag: when the strongest match's members span
        // multiple pools (binding concept) OR when the answer arrived
        // via cross-pool propagation rather than direct input retrieval.
        // The latter is detected here as "query_pool != target_pool":
        // the activation reaching target_pool came through axon
        // terminals, which is compositional by definition.
        let speculation_flag = query_pool != target_pool
            || strongest_match.map(|r| {
                target.get(r.neuron)
                    .map(|n| n.is_binding(target_pool))
                    .unwrap_or(false)
            }).unwrap_or(false);

        // Fabric confidence = fraction of propagated activation
        // captured by the strongest match (its share of total
        // concept-level activation in the target pool).
        let total_concept_act: f32 = target_activation.iter()
            .filter(|&(&nid, _)| target.get(nid).map_or(false, |n| !n.is_atom()))
            .map(|(_, &a)| a).sum();
        let fabric_confidence = if total_concept_act > 1e-9 {
            (strongest_activation / total_concept_act).clamp(0.0, 1.0)
        } else { 0.0 };

        // Integrated confidence = fabric_confidence for now (EEM and
        // annealer are not online until phases 5–6; their contribution
        // weights stay implicitly zero).  Spec §4.D will combine all
        // three once they're built.
        let integrated_confidence = fabric_confidence;

        // Decode the strongest match's members back through the target
        // pool's encode/decode contract.  Uses the recursive walker so
        // that a concept-of-concepts gets decoded all the way down to
        // its atom leaves.  Spec §3.4.
        let answer = strongest_match.and_then(|r| {
            target.get(r.neuron).map(|n| {
                target.decode_concept_members(&n.members)
            })
        });

        let outside_grounding = fabric_confidence < 0.01;
        let confidence_tier = ConfidenceTier::from_confidence(
            integrated_confidence, outside_grounding, speculation_flag);

        AnswerWithGrounding {
            answer,
            grounding: GroundingReport {
                input_atom_coverage,
                strongest_match,
                strongest_match_jaccard: strongest_jaccard,
                composition_used: all_active_concepts,
                fabric_confidence,
                eem_confidence: None,
                annealer_confidence: None,
                integrated_confidence,
                outside_grounding,
                speculation_flag,
                peer_contributions: Vec::new(),
            },
            confidence_tier,
            next_steps_if_ungrounded: Vec::new(),
        }
    }

    // ---------------------------------------------------------------
    // Stage 8 — Critical-thinking loop (spec §4.D + §4.B + user's
    // ivy-growth equation-chaining vision).
    //
    // The unbounded-problem-resolution path:
    //   prompt
    //     ↓
    //   fabric retrieval (today's chat) ──→ if confident, return
    //     ↓ low confidence
    //   EEM chain explorer ──→ walk grounded facts across pools by
    //                          composing them through shared concepts;
    //                          look for convergence in the target pool
    //     ↓ no convergence
    //   RequestObservation ──→ honest "I need more knowledge"
    //
    // The annealer also plugs into the chain exploration step: when
    // multiple chains exist with comparable confidence, the annealer's
    // temporal prediction biases toward chains whose convergence
    // members match patterns the substrate has previously seen.
    // ---------------------------------------------------------------

    /// Score the best matching binding for the query's firing atom set.
    /// Returns (precision, recall) of the best binding's query-pool
    /// member set against the actually-firing query atoms.  This is the
    /// substrate's "do I have a trained pattern that matches this
    /// prompt?" signal — used by `integrate_autonomous` to detect
    /// out-of-vocabulary prompts and honestly report outside_grounding
    /// rather than hallucinating an answer from accidentally-overlapping
    /// atoms.
    /// Atom-tier best-binding match — the legacy Stage 9 signature.
    /// Kept for tests and external callers that don't need tier
    /// awareness.  Internally now delegates to
    /// [`Self::best_binding_match_v2`] and discards the tier tag.
    pub fn best_binding_match(&self, query_pool: PoolId) -> (f32, f32) {
        let m = self.best_binding_match_atom_tier(query_pool);
        (m.precision, m.recall)
    }

    /// Stage 11 tier-aware best-binding match.  Tries concept-level
    /// matching first; falls back to atom-level when either side has
    /// fewer than one concept or fails the coverage gate.  Returns
    /// the highest-scoring match across both tiers along with the
    /// tier tag, so callers (notably `integrate_autonomous`) can
    /// reason about whether the match is concept-grade.
    ///
    /// Coverage gate (concept tier only): the fraction of currently-
    /// firing query-pool items that are concepts must be at least
    /// 0.5.  Without this, a single concept lit by noise alongside
    /// many loose atoms would falsely pass the gate at precision=1.0.
    pub fn best_binding_match_v2(&self, query_pool: PoolId) -> BindingMatch {
        let concept = self.best_binding_match_concept_tier(query_pool);
        if concept.tier == MatchTier::Concept {
            return concept;
        }
        self.best_binding_match_atom_tier(query_pool)
    }

    /// Find the best binding for the current query-pool firing state
    /// AND decode its target-pool members verbatim.
    ///
    /// This is the "trained-answer" retrieval path: the binding
    /// stores the exact atoms that co-fired with the query at
    /// training time, so decoding its target-pool members returns
    /// the literal trained response — no scoring formulas, no
    /// avg-activation gymnastics, no concept-emergence side
    /// effects like 'animala' suffix-pollution.
    ///
    /// Match tie-breaking: highest score (precision × recall).
    /// Among ties, prefer the binding whose target-pool member
    /// count is SMALLEST — cleanest binding wins.  Prefer
    /// concept-tier matches over atom-tier when both exist.
    ///
    /// Returns `Some(bytes)` if a binding was found AND its
    /// target-pool members decode to non-empty bytes.  `None`
    /// otherwise (caller falls back to atom-level or chain path).
    pub fn decode_best_trained_binding(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> Option<Vec<u8>> {
        if query_pool == target_pool { return None; }
        let bpid = self.binding_pool_id;
        if bpid == query_pool || bpid == target_pool { return None; }

        // Collect firing query state: atoms and concepts separately.
        let (q_atoms, q_concepts) = {
            let q = self.fabric.pool(query_pool)?;
            let q = q.read();
            let mut atoms = ahash::AHashSet::new();
            let mut concepts = ahash::AHashSet::new();
            for nid in q.currently_firing() {
                match q.get(nid) {
                    Some(n) if n.is_atom() => { atoms.insert(nid); }
                    Some(_)                => { concepts.insert(nid); }
                    None                   => {}
                }
            }
            (atoms, concepts)
        };
        if q_atoms.is_empty() && q_concepts.is_empty() { return None; }

        // Minimum atom-tier score for an OOV-honest match.  A score
        // of `p * r` where p = precision (intersect/bind_atoms.len())
        // and r = recall (intersect/q_atoms.len()).  Empirically:
        //   trained, full-overlap query (e.g. 'ball'->'toy'): 1.0
        //   trained, concept-tier match (e.g. 'hand'->'body'):  >= 1.0
        //   OOV partial atom bleed ('foobarbaz'->food): 0.44
        //     (propagation inflates q_atoms ∩ bind atoms beyond
        //      raw byte overlap, because bidirectional terminals
        //      fire POOL_TEXT atoms co-trained with food)
        //   OOV single-atom bleed ('xyzzy'->'body'):   0.08
        // A 0.50 floor still passes trained queries (full-overlap
        // atom-tier scores 1.0; concept-tier matches always score
        // ≥ 1.0 via the concept-tier bonus when concept_score
        // itself crosses the floor) and rejects OOV bleed.
        // GA-tunable via `BRAIN_MIN_ATOM_SCORE` env var.
        //
        // DYNAMICAL: accepts either a scalar (→ Constant) OR a
        // ControlMode JSON spec.  When DrivenBy, the floor adapts
        // each call to current substrate state.  E.g.
        // `DrivenBy(DecodePrecisionEma, 0.7, 0.2)` means "raise
        // the floor when recent decodes have been confident; lower
        // it when retrieval has been struggling so partial matches
        // can still get through".  Read from query_pool's ControlState.
        let min_atom_score: f32 = {
            let raw = std::env::var("BRAIN_MIN_ATOM_SCORE").ok();
            // Build query-pool state for ControlMode evaluation.
            let state = self.fabric.pool(query_pool)
                .map(|p| p.read().control_state());
            match (raw, state) {
                (Some(s), Some(st)) => {
                    let s = s.trim().to_string();
                    if let Ok(v) = s.parse::<f32>() {
                        v.clamp(0.0, 1.0)
                    } else if let Ok(mode) = serde_json::from_str::<crate::ControlMode>(&s) {
                        mode.evaluate(&st).clamp(0.0, 1.0)
                    } else {
                        0.50
                    }
                }
                _ => self.config.min_atom_score.clamp(0.0, 1.0),
            }
        };

        // Capture the QUERY's last-observed ordered atom sequence
        // from the query pool itself.  Pool::last_observed_sequence
        // is rebuilt every observe_frame() call (no cross-query
        // accumulation, unlike Fabric::current.fired which only
        // resets on advance_tick).  This lets us preempt anagram
        // ambiguity: 'sad' query (last_observed [s,a,d]) picks
        // sad->emotion (ordered text [s,a,d]) over das->animal
        // (ordered text [d,a,s]).
        let query_seq: Vec<NeuronId> = self.fabric.pool(query_pool)
            .map(|p| p.read().last_observed_sequence().to_vec())
            .unwrap_or_default();

        // Walk binding-pool concepts, score each.
        let bp = self.fabric.pool(bpid)?;
        let bp_read = bp.read();
        // (binding_id, score, target_member_count, has_concept_match, seq_match)
        let mut best: Option<(NeuronId, f32, usize, bool, bool)> = None;
        for n in bp_read.iter_neurons() {
            if n.is_atom() { continue; }
            // Partition this binding's members by pool.
            // bind_q_atoms is a Vec (preserves ORDER from the moment
            // fingerprint that promoted this binding).
            let mut bind_q_atoms:    Vec<NeuronId> = Vec::new();
            let mut bind_q_concepts: Vec<NeuronId> = Vec::new();
            let mut bind_target_members: Vec<NeuronId> = Vec::new();
            {
                let q = self.fabric.pool(query_pool)?;
                let q = q.read();
                for m in &n.members {
                    if m.pool == query_pool {
                        match q.get(m.neuron) {
                            Some(qn) if qn.is_atom() => bind_q_atoms.push(m.neuron),
                            Some(_)                  => bind_q_concepts.push(m.neuron),
                            None                     => {}
                        }
                    } else if m.pool == target_pool {
                        bind_target_members.push(m.neuron);
                    }
                }
            }
            if bind_target_members.is_empty() { continue; }
            if bind_q_atoms.is_empty() && bind_q_concepts.is_empty() { continue; }

            // Ordered equality is direct episodic evidence, not a fuzzy
            // overlap score.  Compute it before the grounding floor: broad
            // curricula can make an exact binding's set-based score share a
            // band with OOV partial overlap, but an unseen query cannot equal
            // a stored training sequence.  This is therefore both recall-
            // preserving and OOV-honest.
            let seq_match = !query_seq.is_empty() && bind_q_atoms == query_seq;

            // Compute score: prefer concept-tier when concept members
            // overlap firing concepts; else atom-tier.
            //
            // CRITICAL: intersect must be UNIQUE-set intersect, not
            // multi-set.  bind_q_atoms is a Vec (allows duplicates from
            // multi-source fingerprints — same atom appearing twice in
            // the binding's POOL_TEXT members), while q_atoms is a
            // HashSet (deduplicated firing set).  Without dedup:
            //   bind=[a,a,b,b,c,c,x]  q={a,b,c}
            //   atom_intersect = 6 (counts each duplicate hit)
            //   recall = 6 / 3 = 2.0   ← BROKEN, should cap at 1.0
            // Toddler-collapse bug: this let multi-source bindings
            // overscore single-pair bindings on duplicate atoms.
            let bind_q_atoms_set: ahash::AHashSet<NeuronId> =
                bind_q_atoms.iter().copied().collect();
            let bind_q_concepts_set: ahash::AHashSet<NeuronId> =
                bind_q_concepts.iter().copied().collect();
            let concept_intersect = bind_q_concepts_set.iter()
                .filter(|c| q_concepts.contains(c))
                .count() as f32;
            let concept_score = if !bind_q_concepts_set.is_empty() && !q_concepts.is_empty() {
                let p = concept_intersect / bind_q_concepts_set.len() as f32;
                let r = concept_intersect / q_concepts.len().max(1) as f32;
                p * r
            } else { 0.0 };
            let atom_intersect = bind_q_atoms_set.iter()
                .filter(|a| q_atoms.contains(a))
                .count() as f32;
            let atom_score = if !bind_q_atoms_set.is_empty() && !q_atoms.is_empty() {
                let p = atom_intersect / bind_q_atoms_set.len() as f32;
                let r = atom_intersect / q_atoms.len().max(1) as f32;
                p * r
            } else { 0.0 };
            // OOV-honesty floor applies BOTH to atom-tier and to the
            // raw concept_score.  Without this, a mega-binding with a
            // 797-member POOL_TEXT footprint (runaway concept-of-
            // concept emergence) wins every query with concept_score
            // ≈ 0.005 because the +1.0 bonus pushes its total above
            // any legitimate single-pair binding's atom-tier score.
            // Concept-tier requires BOTH concept_score AND some atom
            // corroboration.  Without the atom check, a single concept
            // matching by partial-substring emergence (e.g. 'fox'
            // concept fires from OOV 'foobarbaz') wins concept-tier
            // preempt and breaks OOV honesty.
            let concept_ok = concept_score >= min_atom_score
                          && atom_score    >= 0.20;
            let atom_ok    = seq_match || atom_score >= min_atom_score;
            if !concept_ok && !atom_ok { continue; }

            // Concept-tier match preempts atom-tier ONLY when the
            // concept_score itself crosses the floor.
            let base_score = if concept_ok {
                concept_score + 1.0   // concept-tier preempt bonus
            } else {
                atom_score
            };
            let has_concept = concept_ok;

            // Hebbian frequency weight: bindings trained more times
            // dominate competing bindings.  Default multiplier =
            // (1 + ln(use_count)).  DYNAMICAL: strength is read each
            // call from BRAIN_FREQ_WEIGHT env var as a ControlMode.
            // Constant(1.0) preserves the canonical formula; higher
            // values amplify the gap between high- and low-uc
            // bindings (helps resolve K-12 ties between bindings
            // with similar use_count); DrivenBy(ConceptCountEma, ...)
            // would let the multiplier grow as the substrate's
            // binding pool grows, sharpening discrimination under
            // dense training.
            let freq_strength: f32 = {
                let raw = std::env::var("BRAIN_FREQ_WEIGHT").ok();
                let st = self.fabric.pool(query_pool)
                    .map(|p| p.read().control_state());
                match (raw, st) {
                    (Some(s), Some(state)) => {
                        let s = s.trim();
                        if let Ok(v) = s.parse::<f32>() { v.clamp(0.0, 8.0) }
                        else if let Ok(mode) = serde_json::from_str::<crate::ControlMode>(s) {
                            mode.evaluate(&state).clamp(0.0, 8.0)
                        } else { 1.0 }
                    }
                    _ => 1.0,
                }
            };
            let uc = n.use_count.max(1) as f32;
            let freq_weight = 1.0 + freq_strength * uc.ln();
            let score = base_score * freq_weight;

            let target_count = bind_target_members.len();

            // Sequence-match preempt (anagram tiebreaker).  When the
            // binding's text-side ordered sequence EXACTLY equals the
            // query's observed firing sequence, this is a stronger
            // signal than just atom-set match.  Distinguishes 'sad'
            // query from 'das' binding even though both have the same
            // multiset.  Empty query_seq disables this check.
            let consider = match &best {
                None => true,
                Some((_, prev_score, prev_target_count, prev_has_concept, prev_seq_match)) => {
                    // Sequence-match preempt is the highest tier.
                    if seq_match && !*prev_seq_match { true }
                    else if !seq_match && *prev_seq_match { false }
                    // Concept-tier beats atom-tier.
                    else if has_concept && !*prev_has_concept { true }
                    else if !has_concept && *prev_has_concept { false }
                    // Same tier — higher score wins.
                    else if score > *prev_score { true }
                    else if score < *prev_score { false }
                    // Tie on score AND on use_count (freq weight) —
                    // tiebreak by target_count.  Direction controlled
                    // by BRAIN_TARGET_TIEBREAK env var as ControlMode
                    // / scalar.  Value > 0.5 = prefer LARGER target
                    // (helps K-12 sad->emotion vs das->animal when
                    // they tie on score and use_count: emotion=7 bytes
                    // > animal=6 bytes).  Default 0.0 = SMALLER target
                    // (legacy behaviour, cleaner toddler decodes).
                    else {
                        let tiebreak_pref: f32 = {
                            let raw = std::env::var("BRAIN_TARGET_TIEBREAK").ok();
                            let st = self.fabric.pool(query_pool)
                                .map(|p| p.read().control_state());
                            match (raw, st) {
                                (Some(s), Some(state)) => {
                                    let s = s.trim();
                                    if let Ok(v) = s.parse::<f32>() { v.clamp(0.0, 1.0) }
                                    else if let Ok(mode) = serde_json::from_str::<crate::ControlMode>(s) {
                                        mode.evaluate(&state).clamp(0.0, 1.0)
                                    } else { 0.0 }
                                }
                                _ => 0.0,
                            }
                        };
                        if tiebreak_pref >= 0.5 {
                            target_count > *prev_target_count
                        } else {
                            target_count < *prev_target_count
                        }
                    }
                }
            };
            if consider {
                best = Some((n.id, score, target_count, has_concept, seq_match));
            }
        }

        let (bnid, winning_score, _, _, _) = best?;
        let bnode = bp_read.get(bnid)?;
        // Decode the binding's target-pool members.
        let target_handle = self.fabric.pool(target_pool)?;
        let t = target_handle.read();
        // A promoted binding can contain both the ordered target atoms and
        // concepts that collapsed from those same atoms in the training
        // moment.  Recursively decoding both serializes the leaves twice
        // (for example `animal` + an `a` concept became `animala`).  The
        // moment's ordered atoms are the lossless response representation;
        // use concepts only for older/concept-only bindings that have no
        // target atoms.
        let target_members: Vec<NeuronRef> = bnode.members.iter()
            .filter(|m| m.pool == target_pool)
            .copied()
            .collect();
        if target_members.is_empty() { return None; }
        let target_atoms: Vec<NeuronRef> = target_members.iter()
            .filter(|m| t.get(m.neuron).is_some_and(|n| n.is_atom()))
            .copied()
            .collect();
        let bytes = if target_atoms.is_empty() {
            t.decode_concept_members(&target_members)
        } else {
            t.decode_concept_members(&target_atoms)
        };
        // Stage 17.5: collect ALL members of the winning binding (across
        // every pool, not just target).  These are the neurons that
        // participated in this successful decode and should receive the
        // reward-modulated salience bump.
        let all_winner_members: Vec<NeuronRef> = bnode.members.clone();
        drop(t);
        drop(bp_read);

        // Feedback: record this decode's score into the query pool's
        // decode_precision_ema so DecodePrecisionEma ControlSignal
        // reflects how confidently the substrate is retrieving lately.
        // ControlModes that read it can adapt — e.g. raise the floor
        // when decodes are confident, lower it when struggling.
        // Divide winning_score by the freq_weight to recover the raw
        // precision×recall (which is what we want as a [0,1] signal).
        let raw_score = if winning_score > 1.0 {
            // Has concept-tier bonus (+1.0).  Recover the freq-weighted
            // concept portion, then clamp to [0,1].
            (winning_score - 1.0).min(1.0)
        } else { winning_score.min(1.0) };
        if let Some(qp) = self.fabric.pool(query_pool) {
            qp.write().record_decode_precision(raw_score);
        }

        // Stage 17.5: brain-emitted salience update.  The winning binding
        // AND each of its members get a salience bump proportional to
        // the decode precision.  Frémaux & Gerstner (2016) three-factor
        // plasticity: pre × post × reward → tag the post-synaptic neurons
        // (here: binding + members) for long-term retention.
        // Frankland & Bontempi (2005): salience-tagged engrams resist
        // consolidation-driven loss.  This is the substrate's own answer
        // to "which of my own neurons matter for my future self?".
        //
        // Bump magnitude: `raw_score * 0.10`.  A perfect decode (score=1.0)
        // contributes 0.10 to salience; salience saturates at 1.0 in
        // Neuron::bump_salience.  Empirically chosen — fast enough that
        // hot bindings reach high salience within a few thousand decodes,
        // slow enough that one lucky decode doesn't dominate.
        let salience_delta = raw_score * 0.10;
        if salience_delta > 0.0 {
            // Bump the binding concept itself.
            if let Some(bp) = self.fabric.pool(bpid) {
                if let Some(n) = bp.write().get_mut(bnid) {
                    n.bump_salience(salience_delta);
                }
            }
            // Group members by their pool, then take one write lock per pool.
            let mut by_pool: std::collections::HashMap<PoolId, Vec<NeuronId>> =
                std::collections::HashMap::new();
            for nref in &all_winner_members {
                by_pool.entry(nref.pool).or_default().push(nref.neuron);
            }
            for (pid, nids) in by_pool {
                if let Some(p) = self.fabric.pool(pid) {
                    let mut w = p.write();
                    for nid in nids {
                        if let Some(n) = w.get_mut(nid) {
                            n.bump_salience(salience_delta);
                        }
                    }
                }
            }
        }

        if bytes.is_empty() { None } else { Some(bytes) }
    }

    /// Decode a target from the joint firing state of several evidence pools.
    /// Every requested pool must be represented in the learned binding and
    /// must independently clear its coverage gate. This makes the readout an
    /// integrated multi-stream moment rather than an average dominated by one
    /// text channel.
    pub fn decode_best_trained_binding_multi(
        &self,
        query_pools: &[PoolId],
        target_pool: PoolId,
    ) -> Option<Vec<u8>> {
        let mut query_ids: Vec<PoolId> = query_pools.iter().copied()
            .filter(|pid| *pid != target_pool && *pid != self.binding_pool_id)
            .collect();
        query_ids.sort_unstable();
        query_ids.dedup();
        if query_ids.is_empty() { return None; }
        let min_pool_score = std::env::var("BRAIN_MULTI_MIN_POOL_SCORE").ok()
            .and_then(|value| value.parse::<f32>().ok()).unwrap_or(0.20).clamp(0.0, 1.0);
        let min_joint_score = std::env::var("BRAIN_MULTI_MIN_JOINT_SCORE").ok()
            .and_then(|value| value.parse::<f32>().ok()).unwrap_or(0.0).clamp(0.0, 1.0);

        struct QueryState {
            atoms: ahash::AHashSet<NeuronId>,
            concepts: ahash::AHashSet<NeuronId>,
            sequence: Vec<NeuronId>,
        }
        let mut states = std::collections::HashMap::new();
        for pid in &query_ids {
            let handle = self.fabric.pool(*pid)?;
            let pool = handle.read();
            let mut atoms = ahash::AHashSet::new();
            let mut concepts = ahash::AHashSet::new();
            for nid in pool.currently_firing() {
                match pool.get(nid) {
                    Some(n) if n.is_atom() => { atoms.insert(nid); }
                    Some(_) => { concepts.insert(nid); }
                    None => {}
                }
            }
            if atoms.is_empty() && concepts.is_empty() { return None; }
            states.insert(*pid, QueryState {
                atoms,
                concepts,
                sequence: pool.last_observed_sequence().to_vec(),
            });
        }

        let binding_handle = self.fabric.pool(self.binding_pool_id)?;
        let bindings = binding_handle.read();
        // binding id, exact-sequence pool count, joint score, target size
        let mut best: Option<(NeuronId, usize, f32, usize)> = None;
        for binding in bindings.iter_neurons().filter(|n| !n.is_atom()) {
            let target_members: Vec<NeuronRef> = binding.members.iter()
                .filter(|m| m.pool == target_pool).copied().collect();
            if target_members.is_empty() { continue; }

            let mut joint_score = 1.0f32;
            let mut exact_count = 0usize;
            let mut valid = true;
            for pid in &query_ids {
                let state = &states[pid];
                let pool_handle = self.fabric.pool(*pid)?;
                let pool = pool_handle.read();
                let mut member_atoms = Vec::new();
                let mut member_concepts = ahash::AHashSet::new();
                for member in binding.members.iter().filter(|m| m.pool == *pid) {
                    match pool.get(member.neuron) {
                        Some(n) if n.is_atom() => member_atoms.push(member.neuron),
                        Some(_) => { member_concepts.insert(member.neuron); }
                        None => {}
                    }
                }
                if member_atoms.is_empty() && member_concepts.is_empty() {
                    valid = false;
                    break;
                }
                let exact = !state.sequence.is_empty() && member_atoms == state.sequence;
                if exact { exact_count += 1; }
                let member_atom_set: ahash::AHashSet<NeuronId> =
                    member_atoms.iter().copied().collect();
                let atom_hits = member_atom_set.intersection(&state.atoms).count() as f32;
                let atom_score = if member_atom_set.is_empty() || state.atoms.is_empty() { 0.0 }
                    else { (atom_hits / member_atom_set.len() as f32)
                         * (atom_hits / state.atoms.len() as f32) };
                let concept_hits = member_concepts.intersection(&state.concepts).count() as f32;
                let concept_score = if member_concepts.is_empty() || state.concepts.is_empty() { 0.0 }
                    else { (concept_hits / member_concepts.len() as f32)
                         * (concept_hits / state.concepts.len() as f32) };
                let pool_score = if exact { 1.0 } else { atom_score.max(concept_score) };
                if pool_score < min_pool_score {
                    valid = false;
                    break;
                }
                // Product penalizes a weak evidence channel; the nth root
                // below keeps scores comparable as topology adds pools.
                joint_score *= pool_score.max(f32::EPSILON);
            }
            if !valid { continue; }
            joint_score = joint_score.powf(1.0 / query_ids.len() as f32);
            if joint_score < min_joint_score { continue; }
            joint_score *= 1.0 + (binding.use_count.max(1) as f32).ln();
            let candidate = (binding.id, exact_count, joint_score, target_members.len());
            let replace = match best {
                None => true,
                Some((_, best_exact, best_score, best_size)) => {
                    exact_count > best_exact
                        || (exact_count == best_exact && joint_score > best_score)
                        || (exact_count == best_exact && joint_score == best_score
                            && target_members.len() < best_size)
                }
            };
            if replace { best = Some(candidate); }
        }

        let (binding_id, _, _, _) = best?;
        let binding = bindings.get(binding_id)?;
        let target_handle = self.fabric.pool(target_pool)?;
        let target = target_handle.read();
        let members: Vec<NeuronRef> = binding.members.iter()
            .filter(|m| m.pool == target_pool).copied().collect();
        let atoms: Vec<NeuronRef> = members.iter()
            .filter(|m| target.get(m.neuron).is_some_and(|n| n.is_atom()))
            .copied().collect();
        let bytes = if atoms.is_empty() {
            target.decode_concept_members(&members)
        } else {
            target.decode_concept_members(&atoms)
        };
        if bytes.is_empty() { None } else { Some(bytes) }
    }

    fn best_binding_match_atom_tier(&self, query_pool: PoolId) -> BindingMatch {
        let q_atoms: ahash::AHashSet<NeuronId> = {
            let q = match self.fabric.pool(query_pool) {
                Some(p) => p,
                None    => return BindingMatch::NONE,
            };
            let q = q.read();
            q.currently_firing()
                .filter(|nid| q.get(*nid).map_or(false, |n| n.is_atom()))
                .collect()
        };
        if q_atoms.is_empty() { return BindingMatch::NONE; }

        let bpid = self.binding_pool_id;
        let bp = match self.fabric.pool(bpid) {
            Some(p) => p,
            None    => return BindingMatch::NONE,
        };
        let bp_read = bp.read();
        let mut best = BindingMatch::NONE;
        for n in bp_read.iter_neurons() {
            if n.is_atom() { continue; }
            let bind_query: ahash::AHashSet<NeuronId> = n.members.iter()
                .filter(|m| m.pool == query_pool)
                .map(|m| m.neuron)
                .collect();
            if bind_query.is_empty() { continue; }
            let intersect = bind_query.iter()
                .filter(|a| q_atoms.contains(a))
                .count() as f32;
            let precision = intersect / bind_query.len() as f32;
            let recall = intersect / q_atoms.len() as f32;
            let candidate = BindingMatch { precision, recall, tier: MatchTier::Atom };
            if candidate.score() > best.score() {
                best = candidate;
            }
        }
        best
    }

    fn best_binding_match_concept_tier(&self, query_pool: PoolId) -> BindingMatch {
        // Pre-collect:
        //   - `query_concept_set` — every concept neuron id in query pool
        //   - `firing_concepts`   — concepts that are currently firing
        //   - `firing_atoms`      — atoms that are currently firing
        //   - `concept_member_atoms` — atoms that are members of any
        //                              firing concept (per §4.D.1 of
        //                              ARCHITECTURE.md, these are
        //                              concept-LAYER evidence not
        //                              atom-LAYER noise)
        let (
            query_concept_set,
            firing_concepts,
            firing_atoms,
            concept_member_atoms,
            firing_total,
        ) = {
            let qp = match self.fabric.pool(query_pool) {
                Some(p) => p,
                None    => return BindingMatch::NONE,
            };
            let q = qp.read();
            let concept_set: ahash::AHashSet<NeuronId> = q.iter_neurons()
                .filter(|n| !n.is_atom())
                .map(|n| n.id)
                .collect();
            let firing: Vec<NeuronId> = q.currently_firing().collect();
            let firing_concepts: ahash::AHashSet<NeuronId> = firing.iter()
                .copied()
                .filter(|nid| concept_set.contains(nid))
                .collect();
            let firing_atoms: ahash::AHashSet<NeuronId> = firing.iter()
                .copied()
                .filter(|nid| !concept_set.contains(nid))
                .collect();
            // Atoms that belong to a firing concept's member set.  When a
            // concept collapses from its constituent atoms during
            // `Pool::observe_frame`, both the concept AND its atoms end
            // up in `currently_firing`.  Counting those atoms as
            // atom-layer noise would discard the trained sequence-precedence
            // signal — the Stage 11A bug fixed here.
            let mut concept_member_atoms: ahash::AHashSet<NeuronId> =
                ahash::AHashSet::new();
            for &cid in &firing_concepts {
                if let Some(c) = q.get(cid) {
                    for m in &c.members {
                        if m.pool == query_pool && firing_atoms.contains(&m.neuron) {
                            concept_member_atoms.insert(m.neuron);
                        }
                    }
                }
            }
            (concept_set, firing_concepts, firing_atoms, concept_member_atoms, firing.len())
        };

        if firing_concepts.is_empty() || firing_total == 0 {
            return BindingMatch::NONE;
        }

        // §4.D.1 layer-aware coverage gate.
        //
        // OLD (Stage 11A, broken): concept_mass = firing_concepts /
        // total_firing.  This treated atoms that are CONSTITUENTS of a
        // firing concept as atom-layer noise, which discarded the
        // sequence-precedence signal the substrate explicitly produces
        // via `collapse_tail_to_concept`.
        //
        // NEW (§4.D.1): the concept layer's evidence INCLUDES both the
        // concept neurons themselves and the atoms that belong to any
        // firing concept's member set (i.e. atoms that just got
        // collapsed UP into a concept).  Loose atoms — atoms firing
        // that are NOT members of any firing concept — are the only
        // atom-layer noise.
        //
        // Example for "apple": firing = {a, p, p, l, e atoms +
        // apple-concept} = 6 things, concept_layer_evidence = 1 concept
        // + 5 atoms-of-concept = 6, total = 6, ratio = 1.0.  Passes.
        //
        // Example for "ball is fun": firing = {b, a, l, l atoms +
        // ball-concept + i, s, f, u, n loose atoms} = 11 things,
        // concept_layer = 1 + 4 = 5, total = 11, ratio = 0.45.  Falls
        // back to atom-tier — correct, because the query IS dominated
        // by loose atoms.
        let concept_layer_evidence =
            firing_concepts.len() + concept_member_atoms.len();
        let layer_coverage = concept_layer_evidence as f32 / firing_total as f32;
        if layer_coverage < 0.5 {
            return BindingMatch::NONE;
        }

        let bpid = self.binding_pool_id;
        let bp = match self.fabric.pool(bpid) {
            Some(p) => p,
            None    => return BindingMatch::NONE,
        };
        let bp_read = bp.read();
        let mut best = BindingMatch::NONE;
        for n in bp_read.iter_neurons() {
            if n.is_atom() { continue; }
            let bind_concepts: ahash::AHashSet<NeuronId> = n.members.iter()
                .filter(|m| m.pool == query_pool)
                .map(|m| m.neuron)
                .filter(|nid| query_concept_set.contains(nid))
                .collect();
            if bind_concepts.is_empty() { continue; }
            let intersect = bind_concepts.iter()
                .filter(|a| firing_concepts.contains(a))
                .count() as f32;
            let precision = intersect / bind_concepts.len() as f32;
            let recall = intersect / firing_concepts.len() as f32;
            let candidate = BindingMatch { precision, recall, tier: MatchTier::Concept };
            if candidate.score() > best.score() {
                best = candidate;
            }
        }
        best
    }

    /// Stage 13A — resonant multi-pool extrusion (the "epoxy mould"
    /// path per the architecture spec).  Where [`Self::integrate`]
    /// retrieves one target concept from a single target pool via a
    /// one-pass propagation, `integrate_resonant` runs
    /// [`Fabric::settle`] (iterative fixed-point with sharpening)
    /// then reads the top-N decoded concepts in *every* requested
    /// target pool simultaneously.
    ///
    /// This is the substrate-level support for "fire a constraint
    /// into the substrate, extrude the coherent multi-pool state."
    /// Categorical retrieval (`integrate`) is unchanged and remains
    /// the contract for `/chat`.
    ///
    /// `top_per_pool`: how many concepts to decode per target pool.
    /// `max_iter`/`eps`: settling parameters; see `Fabric::settle`.
    pub fn integrate_resonant(
        &self,
        query_pool:    PoolId,
        target_pools:  &[PoolId],
        top_per_pool:  usize,
        max_iter:      usize,
        eps:           f32,
    ) -> ResonantExtrusion {
        let settle = self.fabric.settle(
            query_pool,
            max_iter,
            /*top_k for sharpening*/ top_per_pool.max(8),
            eps,
        );

        let mut per_pool: Vec<PoolExtrusion> = Vec::with_capacity(target_pools.len());
        for &tp in target_pools {
            let pool_handle = match self.fabric.pool(tp) {
                Some(p) => p,
                None    => {
                    per_pool.push(PoolExtrusion {
                        pool: tp,
                        decoded: Vec::new(),
                    });
                    continue;
                }
            };
            let pool = pool_handle.read();

            // Build a fast lookup of the settled atom activations in
            // this pool (raw output of the fixed-point iteration).
            let empty: ahash::AHashMap<NeuronId, f32> = ahash::AHashMap::new();
            let atom_acts = settle.pool_activations.get(&tp).unwrap_or(&empty);

            // Rank CONCEPTS by their average-member-activation derived
            // from the settled atom state (the same idea integrate()
            // uses to score concept candidates).  Cross-pool axons
            // target atoms, not concepts, so to read a coherent
            // concept-level extrusion we have to compose it from the
            // member atoms' settled activations.
            //
            // For each concept neuron in the target pool: compute
            // avg(member activations).  Atoms get their settled
            // activation directly.  The combined list is sorted
            // descending; we take the top `top_per_pool`.
            let mut ranked: Vec<(NeuronId, f32, bool)> = Vec::new(); // (id, score, is_concept)
            for n in pool.iter_neurons() {
                if n.is_atom() {
                    let act = atom_acts.get(&n.id).copied().unwrap_or(0.0);
                    if act > 0.001 {
                        ranked.push((n.id, act, false));
                    }
                } else {
                    let in_pool_members: Vec<NeuronId> = n.members.iter()
                        .filter(|m| m.pool == tp)
                        .map(|m| m.neuron)
                        .collect();
                    if in_pool_members.is_empty() { continue; }
                    let sum: f32 = in_pool_members.iter()
                        .map(|nid| atom_acts.get(nid).copied().unwrap_or(0.0))
                        .sum();
                    let avg = sum / in_pool_members.len() as f32;
                    if avg > 0.001 {
                        // Concept score includes a small bonus for
                        // longer concepts (sqrt of length) so that
                        // when atom activations are similar, the
                        // larger emergent unit wins over its atoms.
                        // This matches the spirit of integrate()'s
                        // length factor.
                        let length_bonus = (in_pool_members.len() as f32).sqrt();
                        ranked.push((n.id, avg * length_bonus, true));
                    }
                }
            }
            // Concepts first by their composed score, then atoms.
            // The is_concept flag lets us tiebreak: concept > atom
            // at the same score so emergent units are surfaced.
            ranked.sort_by(|a, b| {
                let s = b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal);
                if s == std::cmp::Ordering::Equal {
                    return b.2.cmp(&a.2);
                }
                s
            });
            ranked.truncate(top_per_pool);

            let mut decoded: Vec<DecodedConcept> = Vec::with_capacity(ranked.len());
            for (nid, score, _is_concept) in ranked {
                let neuron = match pool.get(nid) {
                    Some(n) => n,
                    None    => continue,
                };
                let bytes: Vec<u8> = if neuron.is_atom() {
                    pool.encoding_reassemble(&[(neuron.label.as_str(), 1.0)])
                } else {
                    pool.decode_concept_members(&neuron.members)
                };
                decoded.push(DecodedConcept {
                    neuron:     NeuronRef::new(tp, nid),
                    activation: score,
                    label:      neuron.label.clone(),
                    bytes,
                });
            }
            per_pool.push(PoolExtrusion { pool: tp, decoded });
        }

        ResonantExtrusion {
            iterations_run: settle.iterations_run,
            converged:      settle.converged,
            pools:          per_pool,
        }
    }

    /// Critical-thinking integration.  When fabric retrieval has low
    /// confidence, walks the EEM's grounded-fact graph from the
    /// query's firing concepts to find target-pool concepts reachable
    /// via co-firing chains.  Returns an `AnswerWithGrounding` whose
    /// `answer` is the strongest reached target-pool concept's decoded
    /// bytes, or `None` (with `outside_grounding=true`) when no chain
    /// converges or the prompt is out-of-vocabulary.
    ///
    /// `fabric_confidence_threshold`: below this, the EEM chain path
    /// runs.  Above, the fabric result is returned directly.
    /// `chain_max_depth`: how many hops the ivy-growth walk takes.
    /// `chain_max_visit`: cap on total facts traversed (bounds work).
    /// `binding_match_threshold`: precision floor for trusting the
    /// fabric answer.  When the best binding's query-pool member set
    /// has < this precision against the firing query atoms, the
    /// prompt is treated as out-of-vocabulary regardless of what
    /// fabric returns (prevents "Hello → color" hallucinations).
    /// Default-threshold version of [`Self::integrate_autonomous_tuned`].
    /// Uses 0.70 as the binding-match precision floor — sufficient to
    /// reject out-of-vocabulary prompts (like "Hello" against a
    /// toddler-category brain) while still accepting partial-match
    /// prompts ("doggy" → animal, "blue!" → color).
    pub fn integrate_autonomous(
        &self,
        query_pool:                  PoolId,
        target_pool:                 PoolId,
        fabric_confidence_threshold: f32,
        chain_max_depth:             usize,
        chain_max_visit:             usize,
    ) -> AnswerWithGrounding {
        self.integrate_autonomous_tuned(
            query_pool, target_pool,
            fabric_confidence_threshold,
            chain_max_depth, chain_max_visit,
            0.70,
        )
    }

    pub fn integrate_autonomous_tuned(
        &self,
        query_pool:                  PoolId,
        target_pool:                 PoolId,
        fabric_confidence_threshold: f32,
        chain_max_depth:             usize,
        chain_max_visit:             usize,
        binding_match_threshold:     f32,
    ) -> AnswerWithGrounding {
        // 0. Out-of-vocab gate (Stage 9 + Stage 11 concept tier).
        // SINGLE source of truth for "is this prompt grounded".  Tries
        // concept-tier matching first (sparse, high-precision) and
        // falls back to atom-tier.  The audit-4 rule: this is the only
        // gate site; downstream code does not re-decide.
        let bm = self.best_binding_match_v2(query_pool);
        let binding_score = bm.score();
        if bm.precision < binding_match_threshold {
            return AnswerWithGrounding {
                answer: None,
                grounding: GroundingReport {
                    input_atom_coverage: 0.0,
                    strongest_match: None,
                    strongest_match_jaccard: 0.0,
                    composition_used: Vec::new(),
                    fabric_confidence: binding_score,
                    eem_confidence: None,
                    annealer_confidence: None,
                    integrated_confidence: binding_score,
                    outside_grounding: true,
                    speculation_flag: false,
                    peer_contributions: Vec::new(),
                },
                confidence_tier: ConfidenceTier::Ungrounded,
                next_steps_if_ungrounded: vec![
                    crate::grounding::RequestObservation {
                        domain: format!(
                            "no trained binding matches this prompt at tier={:?} \
                             with sufficient precision",
                            bm.tier).into(),
                        examples_needed: 1,
                        why: format!(
                            "best binding precision={:.2} (tier={:?}) < threshold={:.2}",
                            bm.precision, bm.tier, binding_match_threshold).into(),
                        pool: query_pool,
                    },
                ],
            };
        }

        // 1. Try the fabric path.
        //
        // We already passed the step-0 binding-precision gate, so this
        // prompt is in-vocabulary as far as Stage 9 honesty is
        // concerned.  The inner `integrate` may still flag its own
        // `outside_grounding=true` based on a stricter fabric-
        // confidence threshold — we deliberately *don't* propagate
        // that flag here, because step 0 is now the single source of
        // truth on "is this prompt grounded".  If `integrate` returned
        // a non-empty answer at or above `fabric_confidence_threshold`,
        // accept it instead of letting chain_explore re-decide.
        let fabric_ans = self.integrate(query_pool, target_pool);
        let fabric_has_answer = fabric_ans.answer.as_ref()
            .map(|b| !b.is_empty()).unwrap_or(false);
        if fabric_has_answer
            && fabric_ans.grounding.fabric_confidence >= fabric_confidence_threshold
        {
            // Clear the inner outside_grounding flag — step 0 already
            // owns that decision and said this query IS grounded.
            let mut g = fabric_ans.grounding.clone();
            g.outside_grounding = false;
            return AnswerWithGrounding {
                answer: fabric_ans.answer,
                grounding: g,
                confidence_tier: fabric_ans.confidence_tier,
                next_steps_if_ungrounded: fabric_ans.next_steps_if_ungrounded,
            };
        }

        // 2. Build the seed set: every firing query-pool atom + every
        // firing query-pool concept's atomic leaves.  These are the
        // refs the chain explorer walks from.
        let query_handle = match self.fabric.pool(query_pool) {
            Some(p) => p,
            None    => return fabric_ans,
        };
        let q = query_handle.read();
        let mut seed: Vec<(PoolId, NeuronId)> = Vec::new();
        for nid in q.currently_firing() {
            seed.push((query_pool, nid));
        }
        drop(q);
        if seed.is_empty() { return fabric_ans; }

        // 3. Walk the EEM's grounded-fact graph from those seeds.
        let chain = self.eem.chain_explore(&seed, chain_max_depth, chain_max_visit);

        // 4. Filter reached members to the target pool and rank by
        // chain confidence (product of fact confidences along the
        // shortest path).
        let mut target_candidates: Vec<((PoolId, NeuronId), f32)> = chain
            .reached_members.iter()
            .filter(|((p, _), _)| *p == target_pool)
            .map(|(k, v)| (*k, *v))
            .collect();
        if target_candidates.is_empty() {
            // Chain found nothing in target pool → return the fabric
            // attempt (which may have low confidence; honest signal).
            return fabric_ans;
        }
        target_candidates.sort_by(|a, b|
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 5. Annealer-guided ranking + Stage 11C multi-fact assembly.
        //
        // Ranking: combine chain confidence with the annealer's
        // predicted activation in the target pool, just like the
        // Stage 8 single-pick path did.
        let prediction = self.annealer.predict_next(target_pool);
        let pred_frame = prediction.as_ref().map(|p| &p.predicted_frame);

        let mut ranked: Vec<((PoolId, NeuronId), f32, f32)> = target_candidates.iter()
            .map(|(nref, conf)| {
                let pred = pred_frame.and_then(|f| f.get(&nref.1).copied()).unwrap_or(0.0);
                let score = conf + 0.5 * pred;
                (*nref, *conf, score)
            })
            .collect();
        ranked.sort_by(|a, b|
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Multi-fact gate.  Audit-2 hard cap: at most 4 facts assembled.
        // Audit-10 confidence-delta gate: facts 2..N are only included
        // if their chain confidence is at least 0.6 × the top fact's.
        // Audit-8 single-fact preservation: when only one fact passes,
        // the output is byte-identical to the Stage 8 single-decode
        // path — no separator, no period, no behavior change for the
        // toddler 32-pair regression baseline.
        const MULTI_FACT_CAP: usize = 4;
        const MULTI_FACT_MIN_RATIO: f32 = 0.6;
        const MULTI_FACT_SEP: &[u8] = b". ";

        let top_conf = ranked.first().map(|(_, c, _)| *c).unwrap_or(0.0);
        let selected: Vec<((PoolId, NeuronId), f32)> = ranked.iter()
            .take(MULTI_FACT_CAP)
            .enumerate()
            .filter(|(i, (_, c, _))|
                *i == 0 || *c >= top_conf * MULTI_FACT_MIN_RATIO)
            .map(|(_, (nref, c, _))| (*nref, *c))
            .collect();

        let (best_ref, best_conf) = match selected.first().copied() {
            Some(x) => x,
            None    => return fabric_ans,
        };

        // Multi-fact trigger correction (audit-8 single-fact pin).
        // The reached_members list may contain many distinct atoms
        // even when chain explored a *single* EEM grounded fact
        // (e.g., the single trained pair `dog → animal` exposes 5
        // pool_b atoms — a, n, i, m, l).  Concatenating all 5 with
        // ". " separators produces nonsense like "a. n. i. m. l".
        //
        // The right granularity is *facts*, not member atoms.  We
        // only fire the multi-fact concat path when the chain
        // actually traversed >= 2 distinct EEM facts.  Single-fact
        // chains decode their single best target as before
        // (byte-identical to the Stage 8 path).
        let fire_multi_fact = chain.visited_facts.len() >= 2 && selected.len() >= 2;

        // 6. Decode each selected target neuron.  Single-fact path
        // produces a single byte sequence; multi-fact path joins with
        // the period+space separator.
        let target_handle = match self.fabric.pool(target_pool) {
            Some(p) => p,
            None    => return fabric_ans,
        };
        let t = target_handle.read();

        let decode_one = |nid: NeuronId| -> Option<Vec<u8>> {
            match t.get(nid) {
                Some(n) if n.is_atom() => {
                    let pairs = [(n.label.as_str(), 1.0_f32)];
                    Some(t.encoding_reassemble(&pairs))
                }
                Some(n) => Some(t.decode_concept_members(&n.members)),
                None    => None,
            }
        };

        let answer_bytes: Option<Vec<u8>> = if !fire_multi_fact {
            // Single-fact preservation (audit-8).  Byte-identical to
            // the Stage 8 path: a single-atom trained pair (K->V)
            // returns "V" exactly, no quality gate, because there's
            // nothing to assemble.  The substrate is honest that it
            // has only that one fact and reports it back.
            decode_one(best_ref.1)
        } else {
            // Multi-fact assembly.  Decode each candidate.
            let mut decoded_parts: Vec<Vec<u8>> = Vec::with_capacity(selected.len());
            for (nref, _conf) in selected.iter() {
                if let Some(bytes) = decode_one(nref.1) {
                    if !bytes.is_empty() {
                        decoded_parts.push(bytes);
                    }
                }
            }
            // Atom-soup defense: when the chain explorer walked a
            // fanout of single-byte atoms (e.g. 'l', 'm', 'n', 'i'
            // from an action pool that only ever saw brief words),
            // joining them as "l. m. n. i" produces high-confidence
            // gibberish.  Refuse the join entirely when EVERY
            // decoded part is a single byte — the substrate then
            // honestly reports outside_grounding and the caller
            // (`/brain/ask`, the hypothesis research loop) gets a
            // truthful "I don't know" instead of confident noise.
            // When at least one part is multi-byte (a real concept
            // decode like "alpha" or "beta"), assembly proceeds
            // normally, preserving back-compat with the multi-fact
            // regression pins.
            let all_single_byte = !decoded_parts.is_empty()
                && decoded_parts.iter().all(|p| p.len() == 1);
            if all_single_byte || decoded_parts.is_empty() {
                None
            } else if decoded_parts.len() == 1 {
                Some(decoded_parts.into_iter().next().unwrap())
            } else {
                let mut out = Vec::new();
                for (i, part) in decoded_parts.iter().enumerate() {
                    if i > 0 { out.extend_from_slice(MULTI_FACT_SEP); }
                    out.extend_from_slice(part);
                }
                Some(out)
            }
        };

        // 7. Assemble the grounding.  Multi-fact path also populates
        // `composition_used` so callers can audit which target
        // neurons contributed to the assembled answer.
        let mut g = fabric_ans.grounding.clone();
        g.eem_confidence = Some(best_conf);
        let f = g.fabric_confidence.max(0.0);
        g.integrated_confidence = ((f * best_conf).sqrt()).max(best_conf * 0.5);
        g.strongest_match = Some(NeuronRef::new(target_pool, best_ref.1));
        g.composition_used = if fire_multi_fact {
            selected.iter()
                .map(|(nref, _)| NeuronRef::new(nref.0, nref.1))
                .collect()
        } else {
            // Single-fact path records exactly the one neuron it
            // decoded — the back-compat regression pin in
            // single_fact_returns_single_decoded_answer requires
            // composition_used.len() == 1 here.
            vec![NeuronRef::new(target_pool, best_ref.1)]
        };
        g.outside_grounding = answer_bytes.as_ref().map(|b| b.is_empty()).unwrap_or(true);
        g.speculation_flag = true; // chain-composed answer is by definition speculation
        let tier = ConfidenceTier::from_confidence(
            g.integrated_confidence, g.outside_grounding, g.speculation_flag);
        AnswerWithGrounding {
            answer: answer_bytes,
            grounding: g,
            confidence_tier: tier,
            next_steps_if_ungrounded: Vec::new(),
        }
    }

    /// Telemetry: count grounded facts the EEM has crystallized from
    /// binding emergence.
    pub fn eem_fact_count(&self) -> usize { self.eem.fact_count() }

    // ---------------------------------------------------------------
    // Phase 5 — EEM-augmented integration (spec §4.D + §4.B).
    //
    // `integrate` returns the fabric's view with `eem_confidence:
    // None` because no equation was consulted.  When the caller has
    // an equation that should apply to the current context, they
    // call `integrate_with_equation(query, target, eq, bindings)`
    // and the EEM's confidence enters the grounding report.  The
    // integrated_confidence becomes the geometric mean of fabric and
    // EEM confidences — both must be grounded for the integrated
    // answer to be strong, per spec §2.7.
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Stage 2 — Sequential generation (continuation of spec §4.E).
    //
    // `integrate` returns the strongest concept's decoded bytes for
    // ONE step.  `generate` chains those steps into a sequence: emit
    // a chunk, feed it back as input via observe + advance_tick, find
    // the next non-emitted strongest concept in target_pool, append.
    // Termination is on confidence floor, max step count, or when no
    // unfired concept clears the floor.
    //
    // This is what turns the brain from a recognizer into a speaker.
    // Generation is recurrent retrieval — each emitted chunk becomes
    // the next step's context.  Open-ended novel composition is NOT
    // here; that requires phrase-level concepts (level-3+) trained on
    // enough natural language to have emerged the relevant
    // compositions.  What IS here: the substrate emits whatever
    // sequence its learned concept structure supports.
    // ---------------------------------------------------------------

    /// Generate a sequence of bytes by iterating
    /// integrate-then-observe-back.  Each step:
    ///   1. Propagate from the most-recent firing state (query pool at
    ///      step 0, target pool thereafter).
    ///   2. Consult the temporal annealer (`Annealer::predict_next`)
    ///      for `target_pool` — its predicted-next-frame activations
    ///      are blended additively into each candidate's score so the
    ///      pick is influenced by "what historically follows the
    ///      current state," not just propagation strength alone.  When
    ///      the annealer has <2 history frames the prediction is
    ///      absent and selection falls back to pure density (Stage 2).
    ///   3. Score active target-pool concepts by
    ///         `(act + annealer_weight × pred_act) / expanded_size`
    ///      skipping concepts already emitted in this call.
    ///   4. Decode the winner via the recursive walker and append to
    ///      the output.
    ///   5. Observe the decoded chunk into target_pool so its atoms
    ///      fire and any matching concepts collapse (mini-column
    ///      collapse runs the same way it does for sensor input);
    ///      then `advance_tick` to capture the state into history.
    ///
    /// `max_steps` is a hard cap; `min_confidence` is the activation
    /// floor (concepts below it don't qualify, ending generation).
    /// The set of already-emitted concepts is reset per call.
    ///
    /// Caller is responsible for observing the prompt into `query_pool`
    /// before calling — `generate` itself does NOT take a prompt
    /// argument because the substrate's "prompt" IS the current pool
    /// state, not a fresh input.  This matches how observation works
    /// elsewhere in the API.
    pub fn generate(
        &mut self,
        query_pool:     PoolId,
        target_pool:    PoolId,
        max_steps:      usize,
        min_confidence: f32,
    ) -> Vec<u8> {
        self.generate_weighted(query_pool, target_pool, max_steps, min_confidence, 0.5)
    }

    /// Like [`Brain::generate`] but with an explicit `annealer_weight`
    /// controlling how much the temporal-prediction signal influences
    /// next-concept selection.  0.0 = pure density (Stage 2);
    /// 0.5 = balanced (default); 1.0 = predicted-activation as
    /// important as propagation-activation.
    pub fn generate_weighted(
        &mut self,
        query_pool:      PoolId,
        target_pool:     PoolId,
        max_steps:       usize,
        min_confidence:  f32,
        annealer_weight: f32,
    ) -> Vec<u8> {
        let mut output: Vec<u8> = Vec::new();
        let mut already_emitted: ahash::AHashSet<NeuronRef> = ahash::AHashSet::new();

        for step in 0..max_steps {
            let source = if step == 0 { query_pool } else { target_pool };
            let propagated = self.fabric.propagate(source);

            // Stage 3: ask the annealer what the next frame should
            // look like given the target pool's recent history.  None
            // when there's <2 history frames; selection then falls
            // back to pure-density (Stage 2).
            let prediction = self.annealer.predict_next(target_pool);
            let pred_frame = prediction.as_ref().map(|p| &p.predicted_frame);

            // Find the best non-emitted concept in target_pool.
            let chunk: Option<(NeuronRef, Vec<u8>)> = {
                let empty: AHashMap<NeuronId, f32> = AHashMap::new();
                let target_act = propagated.get(&target_pool).unwrap_or(&empty);
                let pool_arc = match self.fabric.pool(target_pool) {
                    Some(p) => p,
                    None => break,
                };
                let p = pool_arc.read();

                let mut best: Option<(NeuronRef, f32)> = None;
                let mut best_score = f32::NEG_INFINITY;
                for (&nid, &act) in target_act.iter() {
                    if act < min_confidence { continue; }
                    let n = match p.get(nid) {
                        Some(n) => n,
                        None => continue,
                    };
                    if n.is_atom() { continue; }
                    let r = NeuronRef::new(target_pool, nid);
                    if already_emitted.contains(&r) { continue; }
                    let size = p.expanded_size(&n.members).max(1);
                    let pred_act = pred_frame
                        .and_then(|pf| pf.get(&nid).copied())
                        .unwrap_or(0.0);
                    let blended = act + annealer_weight * pred_act;
                    let score = blended / size as f32;
                    if score > best_score {
                        best_score = score;
                        best = Some((r, act));
                    }
                }
                best.and_then(|(r, _)| p.get(r.neuron).map(|n| (r, p.decode_concept_members(&n.members))))
            };

            let (r, bytes) = match chunk {
                Some(x) => x,
                None => break,
            };
            if bytes.is_empty() { break; }

            output.extend_from_slice(&bytes);
            already_emitted.insert(r);

            // Feed the chunk back into target_pool — its atoms fire,
            // its concepts collapse, the annealer captures the frame.
            self.observe(target_pool, &bytes);
            self.advance_tick();
        }

        output
    }

    /// Like [`Brain::integrate`] but additionally consults the
    /// temporal-prediction annealer for the target pool: the
    /// annealer's convergence energy enters `annealer_confidence` in
    /// the grounding report, and `integrated_confidence` becomes the
    /// geometric mean of fabric and annealer confidences.  When the
    /// annealer has fewer than 2 history frames for the target pool,
    /// `annealer_confidence` stays `None`.
    pub fn integrate_with_prediction(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        if let Some(pred) = self.annealer.predict_next(target_pool) {
            base.grounding.annealer_confidence = Some(pred.confidence);
            let f = base.grounding.fabric_confidence;
            let a = pred.confidence;
            base.grounding.integrated_confidence = (f * a).sqrt();
            base.confidence_tier = ConfidenceTier::from_confidence(
                base.grounding.integrated_confidence,
                base.grounding.outside_grounding,
                base.grounding.speculation_flag,
            );
        }
        base
    }

    /// Like [`Brain::integrate`] but additionally consults the EEM:
    /// the named equation is evaluated under `bindings` and its
    /// confidence enters the grounding report.  If the equation is
    /// not registered or any binding is missing, `eem_confidence`
    /// stays `None` (honestly: the EEM could not contribute).
    pub fn integrate_with_equation(
        &self,
        query_pool:    PoolId,
        target_pool:   PoolId,
        equation_name: &str,
        bindings:      &AHashMap<&str, f64>,
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        let application = self.eem.apply_by_name(equation_name, bindings);
        if let Some(app) = application {
            base.grounding.eem_confidence = Some(app.confidence);
            // Combine fabric and EEM confidences as a geometric mean
            // when both are present.  Lifting one without the other
            // would mask uncertainty.
            let f = base.grounding.fabric_confidence;
            let e = app.confidence;
            base.grounding.integrated_confidence = (f * e).sqrt();
            base.confidence_tier = ConfidenceTier::from_confidence(
                base.grounding.integrated_confidence,
                base.grounding.outside_grounding,
                base.grounding.speculation_flag,
            );
        }
        base
    }

    // ---------------------------------------------------------------
    // Phase 7 — Action layer (spec §4.E).
    //
    // Designating an action pool makes that pool's atoms act as action
    // neurons: when they fire (typically through cross-pool propagation
    // from sensor input), `take_action` collects them into
    // `ActionEvent`s with source attribution.  The caller routes the
    // events externally and later calls `feed_outcome` with a score
    // that reinforces or weakens the source→action terminals.
    // ---------------------------------------------------------------

    /// Designate which pool hosts action atoms.  Subsequent atom births
    /// in that pool (via `register_action`) create action neurons.  At
    /// most one action pool per brain; calling again replaces the
    /// designation.
    pub fn designate_action_pool(&mut self, pool_id: PoolId) {
        self.action_pool_id = Some(pool_id);
    }

    pub fn action_pool_id(&self) -> Option<PoolId> {
        self.action_pool_id
    }

    /// Create (or look up) an action atom in the designated action
    /// pool.  Returns the neuron id of the action atom — callers store
    /// this and pair it with their external effect (webhook URL, MQTT
    /// topic, function call name, etc.).
    ///
    /// Panics if no action pool has been designated, or if the
    /// designated pool id is unknown.  Action registration is a
    /// configuration step; getting it wrong is a programmer error,
    /// not a runtime condition.
    pub fn register_action(&mut self, label: String) -> NeuronRef {
        let pool_id = self.action_pool_id
            .expect("designate_action_pool must be called before register_action");
        let pool = self.fabric.pool(pool_id)
            .expect("action pool id is not a registered pool");
        let mut p = pool.write();
        if let Some(existing) = p.label_to_id(&label) {
            return NeuronRef::new(pool_id, existing);
        }
        let now = self.fabric.current_tick();
        let id = p.neuron_count() as NeuronId;
        let neuron = Neuron::new_atom(id, label.clone(), NeuronKind::Excitatory, now);
        p.append_neuron(neuron, label);
        NeuronRef::new(pool_id, id)
    }

    /// Manually fire an action neuron (e.g. during supervised training
    /// where the trainer wants to drive the action while sources are
    /// also firing).  This deposits activation into the action pool;
    /// `take_action` is what surfaces the corresponding event.
    pub fn fire_action(&mut self, action: NeuronRef, strength: f32) {
        if let Some(p) = self.fabric.pool(action.pool) {
            let tick = self.fabric.current_tick();
            p.write().inject_activation(action.neuron, strength, tick);
        }
        // Register in the moment buffer so cross-pool wiring on tick
        // close grows source→action terminals.
        self.fabric.mark_fired(action.pool, action.neuron);
    }

    /// Collect every action neuron that fired this tick, attribute its
    /// sources (currently-firing neurons in other pools whose terminals
    /// target the action neuron), and emit one `ActionEvent` per
    /// firing.  Caller routes externally and later supplies
    /// `feed_outcome`.
    ///
    /// "Fired" means: present in the action pool's `currently_firing`
    /// set (from `fire_action`), OR reached via cross-pool propagation
    /// from any non-action pool with activation ≥ `min_activation`.
    /// The propagation path is what gives the substrate its agency:
    /// sensor input → cross-pool axons → action firing → external
    /// effect → outcome → reinforcement.
    ///
    /// Re-fires of the same action neuron within a single tick are
    /// deduplicated.  Across ticks, each firing produces its own event.
    pub fn take_action(&mut self, min_activation: f32) -> Vec<ActionEvent> {
        let pool_id = match self.action_pool_id {
            Some(p) => p,
            None    => return Vec::new(),
        };
        let pool = match self.fabric.pool(pool_id) {
            Some(p) => p,
            None    => return Vec::new(),
        };

        // Step 1: collect direct firings (from `fire_action` or
        // observed atoms in the action pool itself).
        let mut firings: AHashMap<NeuronId, (String, f32)> = AHashMap::new();
        {
            let p = pool.read();
            for nid in p.currently_firing() {
                if let Some(n) = p.get(nid) {
                    firings.insert(nid, (n.label.clone(), p.activation(nid)));
                }
            }
        }

        // Step 2: propagate from each non-action pool that has firing
        // neurons; merge activation reaching the action pool.
        for src_pool_id in self.fabric.pool_ids() {
            if src_pool_id == pool_id { continue; }
            let any_firing = match self.fabric.pool(src_pool_id) {
                Some(p) => p.read().currently_firing().next().is_some(),
                None    => false,
            };
            if !any_firing { continue; }
            let propagated = self.fabric.propagate(src_pool_id);
            if let Some(act_map) = propagated.get(&pool_id) {
                let p = pool.read();
                for (&nid, &act) in act_map.iter() {
                    if act < min_activation { continue; }
                    if let Some(n) = p.get(nid) {
                        let entry = firings.entry(nid)
                            .or_insert_with(|| (n.label.clone(), 0.0));
                        entry.1 = entry.1.max(act);
                    }
                }
            }
        }

        if firings.is_empty() { return Vec::new(); }

        let tick = self.fabric.current_tick();
        let mut events = Vec::with_capacity(firings.len());

        // Step 3: for each fired action neuron, find currently-firing
        // sources elsewhere in the fabric whose terminals target it.
        // Credit attribution by inverse-terminal lookup — there's no
        // fabric-level routing table, so we scan owning neurons.
        for (action_nid, (action_label, activation)) in firings {
            let action_ref = NeuronRef::new(pool_id, action_nid);
            if self.emitted_this_tick.contains(&action_ref) { continue; }

            let mut sources: Vec<NeuronRef> = Vec::new();
            for src_pool_id in self.fabric.pool_ids() {
                if src_pool_id == pool_id { continue; }
                let src_pool = match self.fabric.pool(src_pool_id) {
                    Some(p) => p,
                    None    => continue,
                };
                let sp = src_pool.read();
                let firing_here: ahash::AHashSet<NeuronId> = sp.currently_firing().collect();
                for nid in firing_here.iter().copied() {
                    if let Some(n) = sp.get(nid) {
                        if n.terminals.iter().any(|t| t.target == action_ref) {
                            sources.push(NeuronRef::new(src_pool_id, nid));
                        }
                    }
                }
            }

            let id = self.next_action_id;
            self.next_action_id += 1;
            let event = ActionEvent {
                id,
                action_neuron: action_ref,
                action_label,
                sources,
                fired_at_tick: tick,
                activation,
            };
            self.pending_actions.insert(id, event.clone());
            self.emitted_this_tick.insert(action_ref);
            events.push(event);
        }

        events
    }

    /// Apply an outcome score to a previously-emitted action.  Positive
    /// scores reinforce source→action terminals; negative scores
    /// weaken them.  The action_id is removed from `pending_actions`
    /// after application — outcomes are one-shot per firing.
    ///
    /// Returns true if the action_id was found and feedback applied.
    /// Returns false if the id is unknown (already-fed-back or never-
    /// emitted).
    pub fn feed_outcome(&mut self, action_id: ActionId, score: f32) -> bool {
        let event = match self.pending_actions.remove(&action_id) {
            Some(e) => e,
            None    => return false,
        };
        let now = self.fabric.current_tick();
        for src in &event.sources {
            let src_pool = match self.fabric.pool(src.pool) {
                Some(p) => p,
                None    => continue,
            };
            let mut sp = src_pool.write();
            let max_w = sp.config.max_weight;
            let mut delta_terminals: usize = 0;
            if let Some(n) = sp.get_mut(src.neuron) {
                // Positive: strengthen via the same idempotent
                // reinforce path.  Negative: directly subtract from
                // the existing terminal's weight (clamped at 0; prune
                // happens on the next housekeeping pass).
                if score >= 0.0 {
                    if n.reinforce_terminal(event.action_neuron, score, now, max_w) {
                        delta_terminals = 1;
                    }
                } else {
                    if let Some(t) = n.terminals.iter_mut()
                        .find(|t| t.target == event.action_neuron)
                    {
                        t.weight = (t.weight + score).max(0.0);
                        t.last_fired_tick = now;
                    }
                }
            }
            sp.total_terminals += delta_terminals;
        }
        true
    }

    /// Number of action events awaiting outcome feedback.  Surfaced
    /// for caller telemetry — a runaway value indicates outcomes are
    /// not being supplied for fired actions.
    pub fn pending_action_count(&self) -> usize {
        self.pending_actions.len()
    }

    // ---------------------------------------------------------------
    // Phase 8 — Distributed network (spec §5).
    //
    // The brain crate defines protocol + data model; transport lives
    // outside (cluster gossip plugs in here via the drain/ingest
    // methods).  Spec §1.7 puts the network as a property of the
    // brain factory, not a baked-in stack.
    // ---------------------------------------------------------------

    pub fn brain_id(&self) -> &str { &self.network.brain_id }
    pub fn set_brain_id(&mut self, id: impl Into<String>) {
        self.network.brain_id = id.into();
    }

    /// Drain all pending outbound motif records.  Transport calls
    /// this periodically (or after each `advance_tick`) and ships
    /// the result over the cluster.  Returns the drained motifs;
    /// the brain's outbound queue is left empty.
    pub fn drain_motif_gossip(&mut self) -> Vec<GossipMotif> {
        std::mem::take(&mut self.network.pending_motif_out)
    }

    /// Number of motif records currently queued for outbound gossip.
    pub fn pending_motif_count(&self) -> usize {
        self.network.pending_motif_out.len()
    }

    /// Ingest motif records gossiped from peer brains.  Records from
    /// this brain's own id are dropped (self-loop guard) — motif
    /// echoes from broadcast topologies should not show up in the
    /// network index.  Repeat records from the same peer about the
    /// same fingerprint update the existing entry rather than
    /// duplicating.
    pub fn ingest_motif_gossip(&mut self, motifs: Vec<GossipMotif>) {
        for mut m in motifs {
            if m.source_brain == self.network.brain_id { continue; }
            m.fingerprint.sort();
            let key = (m.source_brain.clone(), m.fingerprint.clone());
            if let Some(slot) = self.network.received_motifs.iter_mut()
                .find(|(k, _)| *k == key)
            {
                slot.1 = m;
            } else {
                self.network.received_motifs.push((key, m));
            }
        }
    }

    pub fn received_motif_count(&self) -> usize {
        self.network.received_motifs.len()
    }

    pub fn received_motifs_from(&self, peer: &str) -> Vec<&GossipMotif> {
        self.network.received_motifs.iter()
            .filter(|((b, _), _)| b == peer)
            .map(|(_, m)| m)
            .collect()
    }

    /// Snapshot every registered equation as a `GossipEquation`
    /// suitable for cluster broadcast.  Simple whole-state-sync is
    /// honest and idempotent — receivers merge by confidence.
    pub fn export_equations_for_gossip(&self) -> Vec<GossipEquation> {
        let id = self.network.brain_id.clone();
        self.eem.iter_equations().map(|eq| {
            let variable_names = eq.variables.iter()
                .filter_map(|vid| self.eem.variable(*vid).map(|v| v.name.clone()))
                .collect();
            let discipline_name = eq.discipline
                .and_then(|did| self.eem.discipline(did).map(|d| d.name.clone()));
            GossipEquation {
                source_brain:         id.clone(),
                name:                 eq.name.clone(),
                expression:           eq.expression.clone(),
                variable_names,
                discipline_name,
                confidence:           eq.confidence,
                validation_successes: eq.validation_successes,
                validation_failures:  eq.validation_failures,
            }
        }).collect()
    }

    /// Merge peer equation deltas into the local EEM.  Conflict
    /// resolution per spec §5.2: per-equation confidence wins.  A
    /// peer equation strictly higher in confidence than the local
    /// copy replaces the expression and adopts the peer's validation
    /// counters; otherwise the local copy is kept.  Unknown
    /// equations are added at the peer's confidence.
    pub fn ingest_equation_gossip(&mut self, equations: Vec<GossipEquation>) {
        for ge in equations {
            if ge.source_brain == self.network.brain_id { continue; }
            // Variables: register by name (idempotent).
            let var_ids: Vec<_> = ge.variable_names.iter()
                .map(|n| self.eem.register_variable(n.clone(), None))
                .collect();
            let disc_id = ge.discipline_name.as_ref()
                .map(|n| self.eem.register_discipline(n.clone()));

            match self.eem.equation_by_name(&ge.name) {
                Some(eq_id) => {
                    let local_conf = self.eem.confidence(eq_id).unwrap_or(0.0);
                    if ge.confidence > local_conf {
                        // Replace expression and re-seat confidence /
                        // validation counters from the peer.
                        self.eem.replace_equation_expression(eq_id, ge.expression.clone());
                        if let Some(eq) = self.eem.equation_mut(eq_id) {
                            eq.confidence           = ge.confidence;
                            eq.validation_successes = ge.validation_successes;
                            eq.validation_failures  = ge.validation_failures;
                        }
                    }
                }
                None => {
                    let eq_id = self.eem.register_equation(
                        ge.name.clone(),
                        ge.expression.clone(),
                        var_ids,
                        disc_id,
                    );
                    if let Some(eq) = self.eem.equation_mut(eq_id) {
                        eq.confidence           = ge.confidence;
                        eq.validation_successes = ge.validation_successes;
                        eq.validation_failures  = ge.validation_failures;
                    }
                }
            }
        }
    }

    pub fn peer_accuracy(&self, peer: &str) -> Option<&PeerAccuracy> {
        self.network.peer_accuracy.get(peer)
    }

    /// Record an outcome for a peer's contribution.  Drives the
    /// per-peer accuracy rate used by `integrate_with_peers`.
    pub fn report_peer_outcome(&mut self, peer: impl Into<String>, success: bool) {
        let entry = self.network.peer_accuracy.entry(peer.into()).or_default();
        if success {
            entry.successful_contributions = entry.successful_contributions.saturating_add(1);
        } else {
            entry.failed_contributions = entry.failed_contributions.saturating_add(1);
        }
    }

    pub fn known_peers(&self) -> Vec<String> {
        let mut v: Vec<String> = self.network.peer_accuracy.keys().cloned().collect();
        v.sort();
        v
    }

    /// Like [`Brain::integrate`] but folds caller-supplied peer
    /// contributions into the grounding report.  Each peer's
    /// `fabric_confidence` is weighted by its [`PeerAccuracy::rate`]
    /// (unknown peers get 0.5 — neutral) and surfaces in
    /// `GroundingReport.peer_contributions`.  The
    /// `integrated_confidence` becomes a weighted blend of the local
    /// fabric confidence and the average weighted peer confidence.
    pub fn integrate_with_peers(
        &self,
        query_pool:  PoolId,
        target_pool: PoolId,
        peers:       &[PeerContribution],
    ) -> AnswerWithGrounding {
        let mut base = self.integrate(query_pool, target_pool);
        if peers.is_empty() { return base; }

        let mut weighted: Vec<(BrainId, f32)> = Vec::with_capacity(peers.len());
        let mut sum_weights = 0.0_f32;
        let mut sum_weighted_conf = 0.0_f32;
        for p in peers {
            let rate = self.network.peer_accuracy
                .get(&p.brain_id).map(|a| a.rate()).unwrap_or(0.5);
            let weighted_conf = (p.fabric_confidence * rate).clamp(0.0, 1.0);
            weighted.push((p.brain_id.clone(), weighted_conf));
            sum_weights += rate;
            sum_weighted_conf += weighted_conf * rate;
        }

        let local_fabric = base.grounding.fabric_confidence;
        let peer_blend = if sum_weights > 1e-9 {
            sum_weighted_conf / sum_weights
        } else { 0.0 };

        // Blend local + peer 50/50 by weighted contribution mass.
        // A stronger local signal still dominates; a confident-and-
        // accurate peer chorus lifts the integrated score.
        let blended = (local_fabric + peer_blend) / 2.0;
        base.grounding.peer_contributions = weighted;
        base.grounding.integrated_confidence = blended;
        base.confidence_tier = ConfidenceTier::from_confidence(
            base.grounding.integrated_confidence,
            base.grounding.outside_grounding,
            base.grounding.speculation_flag,
        );
        base
    }

    // ---------------------------------------------------------------
    // Phase 10 — Brain factory (spec §3 + §11).
    //
    // Builds a fully-wired brain from a declarative
    // `BrainIdentitySpec` plus a `PoolPrototypeRegistry` that knows
    // how to instantiate encodings.  The Action pool (if any) is
    // auto-designated; the EEM and annealer come up empty for the
    // caller to seed.
    // ---------------------------------------------------------------

    pub fn from_identity(
        identity: &BrainIdentitySpec,
        registry: &PoolPrototypeRegistry,
    ) -> Result<Self, IdentityBuildError> {
        // Detect collisions and binding-pool-id reservations up front
        // so the brain isn't half-built when an error surfaces.
        let binding_pool_id: PoolId = 0;
        let mut seen_ids = ahash::AHashSet::new();
        for ps in &identity.pools {
            if ps.id == binding_pool_id {
                return Err(IdentityBuildError::BindingPoolIdCollision(ps.id));
            }
            if !seen_ids.insert(ps.id) {
                return Err(IdentityBuildError::DuplicatePoolId(ps.id));
            }
        }

        let mut binding_pool_config = PoolConfig::defaults("binding", binding_pool_id);
        binding_pool_config.concept_emergence_threshold = u32::MAX;
        let config = BrainConfig {
            fabric:                      identity.fabric.clone(),
            binding_emergence_threshold: identity.binding_emergence_threshold,
            tentative_emergence_threshold: default_tentative_emergence_threshold(),
            moment_history_window:       identity.moment_history_window,
            min_atom_score:               identity.min_atom_score,
            pressure_band_low:           default_pressure_band_low(),
            pressure_band_high:          default_pressure_band_high(),
            pressure_threshold_max:      default_pressure_threshold_max(),
            pressure_observation_grace:  default_pressure_observation_grace(),
            pressure_adjust_enabled:     default_pressure_adjust_enabled(),
            binding_pool_config,
            eem:                         identity.eem.clone(),
            annealer:                    identity.annealer.clone(),
        };
        let mut brain = Self::new(config);

        for ps in &identity.pools {
            let encoding = registry
                .build(&ps.prototype, &ps.atom_encoding_prefix)
                .ok_or_else(|| IdentityBuildError::UnknownPrototype(ps.prototype.clone()))?;
            brain.create_pool(ps.to_pool_config(), encoding);
        }

        // Auto-designate the first Action pool, if any.  At most one
        // Action pool is meaningful per spec §4.E; later Action pools
        // in the spec are stored but not designated.
        for ps in &identity.pools {
            if matches!(ps.kind, PoolKind::Action) {
                brain.designate_action_pool(ps.id);
                break;
            }
        }

        Ok(brain)
    }

    // ---------------------------------------------------------------
    // Stage 4 — Sleep cycle (spec §6.3).
    //
    // Identifies weak/stale concept neurons across all pools and
    // soft-prunes them: their outgoing terminals are zeroed and all
    // inbound terminals (within-pool AND cross-pool) are removed.
    // Then runs an extra housekeeping pass on every pool so the
    // zeroed terminals drop below the prune floor.
    //
    // Sleep is what makes "survival of fittest" actually surface
    // strong emergence — without it, every cross-pair boundary
    // pattern that crosses the emergence threshold sticks around and
    // dilutes the substrate's concept space.
    // ---------------------------------------------------------------

    /// Soft-prune all concept neurons across all pools whose
    /// `use_count` is below `min_use_count` and whose `last_fired_tick`
    /// is more than `stale_ticks` in the past.
    ///
    /// Returns the total number of concepts pruned.
    ///
    /// This is the **single-shot** form, retained for tests and tools that
    /// don't need to yield to other HTTP traffic.  Per [`ARCHITECTURE.md`]
    /// §17.4, the production sleep path is the decomposed
    /// [`Brain::sleep_pool_phase1`] / [`Brain::sleep_pool_phase2`] /
    /// [`Brain::sleep_pool_housekeeping`] trio invoked by the
    /// [`crate::store`]-aware HTTP handler so other endpoints can interleave.
    /// Note this is `&self` — all the actual mutation happens through
    /// `pool.write()` on per-pool RwLocks.
    pub fn sleep(&self, min_use_count: u64, stale_ticks: u64) -> usize {
        let mut total_pruned = 0usize;
        let mut pruned_refs: ahash::AHashSet<NeuronRef> = ahash::AHashSet::new();
        let pool_ids = self.fabric.pool_ids();

        // Phase 1: per-pool prune; collect (pool, neuron) refs.
        for pid in &pool_ids {
            let pruned = self.sleep_pool_phase1(*pid, min_use_count, stale_ticks);
            total_pruned += pruned.len();
            pruned_refs.extend(pruned.into_iter());
        }

        // Phase 2: clean cross-pool inbound terminals targeting anything
        // pruned in phase 1.
        if !pruned_refs.is_empty() {
            for pid in &pool_ids {
                self.sleep_pool_phase2(*pid, &pruned_refs);
            }
        }

        // Phase 3: extra housekeeping so any zero-weight residuals
        // drop below the prune floor.
        for pid in &pool_ids {
            self.sleep_pool_housekeeping(*pid);
        }

        total_pruned
    }

    /// Phase 1 of [`ARCHITECTURE.md`] §17.4 sleep decomposition — prune one
    /// pool's weak concepts.  Returns the cross-pool refs of pruned
    /// neurons so callers can drive the cross-pool cleanup pass (phase 2)
    /// later, interleaved with other HTTP work.  Brief per-pool write
    /// lock only — other pools stay readable during this call.
    pub fn sleep_pool_phase1(
        &self,
        pool_id:       PoolId,
        min_use_count: u64,
        stale_ticks:   u64,
    ) -> ahash::AHashSet<NeuronRef> {
        let current_tick = self.fabric.current_tick();
        let Some(pool) = self.fabric.pool(pool_id) else {
            return ahash::AHashSet::new();
        };
        let pruned_ids = pool.write()
            .prune_weak_concepts(min_use_count, stale_ticks, current_tick);
        let mut out = ahash::AHashSet::with_capacity(pruned_ids.len());
        for nid in pruned_ids {
            out.insert(NeuronRef::new(pool_id, nid));
        }
        out
    }

    /// Phase 2 of [`ARCHITECTURE.md`] §17.4 sleep decomposition — clean
    /// inbound cross-pool terminals targeting any of `pruned_refs` from
    /// `pool_id`.  Brief per-pool write lock only.
    pub fn sleep_pool_phase2(
        &self,
        pool_id:       PoolId,
        pruned_refs:   &ahash::AHashSet<NeuronRef>,
    ) {
        if pruned_refs.is_empty() { return; }
        let Some(pool) = self.fabric.pool(pool_id) else { return; };
        pool.write().prune_inbound_to(pruned_refs);
    }

    /// Phase 3 of [`ARCHITECTURE.md`] §17.4 sleep decomposition — final
    /// per-pool housekeeping (decay + sub-floor prune).  Brief per-pool
    /// write lock only.
    pub fn sleep_pool_housekeeping(&self, pool_id: PoolId) {
        let now = self.fabric.current_tick();
        let Some(pool) = self.fabric.pool(pool_id) else { return; };
        pool.write().tick_housekeeping(now);
    }

    // (helper for `integrate` — Jaccard on tilde-split label tokens)
    // Defined as a free function below.

    /// Drain queued deferred-promotion candidates per pool.  Called
    /// during sleep — crystallises every concept that crossed the
    /// emergence threshold during observe while W1Z4RD_DEFER_PROMOTION
    /// was set.  Brief per-pool write lock.  Returns total promotions
    /// applied across all pools.
    pub fn sleep_drain_promotions(&self) -> usize {
        let now = self.fabric.current_tick();
        let mut total = 0;
        for pid in self.fabric.pool_ids() {
            let Some(pool) = self.fabric.pool(pid) else { continue; };
            total += pool.write().drain_pending_promotions(now);
        }
        total
    }

    /// Aggregated pending-promotion queue depth across all pools.
    /// Surfaced via /sleep_pressure so the operator can see when the
    /// brain is overdue for a sleep cycle under deferred mode.
    pub fn pending_promotion_count(&self) -> usize {
        self.fabric.pool_ids().into_iter()
            .filter_map(|pid| self.fabric.pool(pid))
            .map(|p| p.read().pending_promotion_count())
            .sum()
    }

    /// Set the domain stamp for every pool's newly-created atoms +
    /// concepts.  Island architecture: operator calls this BEFORE
    /// training a domain-specific corpus so the resulting neurons
    /// cluster into that island.  0 resets to global / unassigned.
    pub fn set_domain_for_new(&self, domain_id: u32) {
        for pid in self.fabric.pool_ids() {
            if let Some(p) = self.fabric.pool(pid) {
                p.write().set_domain_for_new(domain_id);
            }
        }
    }

    /// Per-pool neuron count broken down by domain_id.  Returns a map
    /// keyed by (pool_id, domain_id) → count.  Lets the operator
    /// confirm islands grew as expected during a training run.
    pub fn domain_histogram(&self) -> std::collections::HashMap<(PoolId, u32), usize> {
        let mut out = std::collections::HashMap::new();
        for pid in self.fabric.pool_ids() {
            if let Some(p) = self.fabric.pool(pid) {
                for (dom, cnt) in p.read().domain_histogram() {
                    out.insert((pid, dom), cnt);
                }
            }
        }
        out
    }

    /// Integration cycle — the "X is like Y" substrate.  Finds
    /// structurally similar concept pairs across DIFFERENT domains and
    /// adds bridge terminals between them.
    ///
    /// Similarity metric: cosine on each concept's co-firing signature
    /// (its outgoing-terminal weight vector, indexed by target
    /// NeuronRef).  Two concepts are structurally analogous if they
    /// project into overlapping downstream targets — co-firing
    /// fingerprint matches, NOT label-string matches.  This is the
    /// architecture's intended similarity primitive.
    ///
    /// Bridges are reinforced via normal `reinforce_terminal` with
    /// weight = similarity × 0.1, then strengthened naturally on
    /// subsequent cross-domain co-firing (the soft domain gate in
    /// fabric.rs lets that happen during ordinary training).
    ///
    /// Returns total bridges added across all pools and domain pairs.
    pub fn integrate_islands(
        &self,
        sample_size: usize,
        similarity_threshold: f32,
    ) -> usize {
        let now = self.fabric.current_tick();
        let mut total_bridges = 0usize;

        for pid in self.fabric.pool_ids() {
            let Some(pool) = self.fabric.pool(pid) else { continue };

            // Bucket concept ids by domain, building a co-firing
            // signature for each as we go.  Skip atoms (no analogical
            // content); skip domain 0 (unassigned).
            type Sig = ahash::AHashMap<NeuronRef, f32>;
            let buckets: std::collections::HashMap<u32, Vec<(NeuronId, Sig)>> = {
                let p = pool.read();
                let mut b: std::collections::HashMap<u32, Vec<(NeuronId, Sig)>>
                    = std::collections::HashMap::new();
                for n in p.iter_neurons() {
                    if n.is_atom() || n.domain_id == 0 { continue; }
                    // Co-firing signature = outgoing terminal weights
                    // by target.  Effective weight folds in
                    // consolidation, so well-trained edges dominate
                    // the analogy metric (intended — matures with
                    // training).
                    let mut sig: Sig = ahash::AHashMap::with_capacity(n.terminals.len());
                    for t in &n.terminals {
                        sig.insert(t.target, t.effective_weight());
                    }
                    b.entry(n.domain_id).or_default().push((n.id, sig));
                }
                b
            };

            let domains: Vec<u32> = buckets.keys().copied().collect();
            for i in 0..domains.len() {
                for j in (i + 1)..domains.len() {
                    let dom_x = domains[i];
                    let dom_y = domains[j];

                    let xs: &[(NeuronId, Sig)] = {
                        let slice = &buckets[&dom_x];
                        &slice[..slice.len().min(sample_size)]
                    };
                    let ys: &[(NeuronId, Sig)] = {
                        let slice = &buckets[&dom_y];
                        &slice[..slice.len().min(sample_size)]
                    };

                    let mut pairs: Vec<(NeuronId, NeuronId, f32)> = Vec::new();
                    for (x, sx) in xs {
                        for (y, sy) in ys {
                            let sim = cofiring_cosine(sx, sy);
                            if sim >= similarity_threshold {
                                pairs.push((*x, *y, sim));
                            }
                        }
                    }

                    if pairs.is_empty() { continue; }

                    let mut pw = pool.write();
                    let max_w = pw.config.max_weight;
                    let mut added = 0usize;
                    for (x, y, sim) in &pairs {
                        let w = sim * 0.1;
                        if let Some(nx) = pw.get_mut(*x) {
                            if nx.reinforce_terminal(NeuronRef::new(pid, *y), w, now, max_w) {
                                added += 1;
                            }
                        }
                        if let Some(ny) = pw.get_mut(*y) {
                            if ny.reinforce_terminal(NeuronRef::new(pid, *x), w, now, max_w) {
                                added += 1;
                            }
                        }
                    }
                    pw.total_terminals += added;
                    total_bridges += added;
                }
            }
        }

        total_bridges
    }

    /// Score a moment fingerprint by the mean salience of its participating
    /// neurons.  Stage 17.7 uses this to weight replay sampling toward
    /// moments whose neurons the brain has tagged as important.  Cheap to
    /// compute: O(participants), no fabric-wide scan.
    ///
    /// Returns 0.0 for empty fingerprints (no participants).
    pub fn moment_salience_score(&self, fp: &MomentFingerprint) -> f32 {
        let mut sum = 0.0f32;
        let mut n = 0u32;
        if !fp.ordered_per_pool.is_empty() {
            for (pid, ids) in &fp.ordered_per_pool {
                if let Some(pool) = self.fabric.pool(*pid) {
                    let p = pool.read();
                    for nid in ids {
                        if let Some(neuron) = p.get(*nid) {
                            sum += neuron.salience_ema;
                            n += 1;
                        }
                    }
                }
            }
        } else {
            for &(pid, nid) in &fp.pairs {
                if let Some(pool) = self.fabric.pool(pid) {
                    let p = pool.read();
                    if let Some(neuron) = p.get(nid) {
                        sum += neuron.salience_ema;
                        n += 1;
                    }
                }
            }
        }
        if n == 0 { 0.0 } else { sum / (n as f32) }
    }

    /// Stage 17.7 full — free-energy weighted REPLAY per
    /// [`ARCHITECTURE.md`] §17.7.  Samples `count` moments from
    /// `moment_history` via softmax (Boltzmann) over their salience
    /// scores at temperature `beta`, then re-fires the sampled set.
    ///
    /// Mathematical form: `P(replay m_i) ∝ exp(beta * salience(m_i))`.
    /// - `beta = 0`     → uniform sampling (every moment equally likely)
    /// - `beta → ∞`     → deterministic top-K (the prior partial form)
    /// - `beta ≈ 1–5`   → soft preference; high-salience moments favoured
    ///   but exploration still happens
    ///
    /// This is the canonical free-energy minimisation form (Friston
    /// 2010 active inference, Hinton et al. 1995 wake-sleep) translated
    /// to the moment-buffer scale: low-energy / high-utility moments
    /// (here proxied by salience EMA) are sampled preferentially, but
    /// the temperature parameter lets the brain spend some replay
    /// budget on exploration of less-tagged moments.
    ///
    /// Sampling is **without replacement** — each moment is chosen at
    /// most once per pass.  Reproducible: same `seed` + same brain
    /// state produces the same sampled set.
    pub fn replay_free_energy_weighted(
        &mut self,
        count:    usize,
        strength: f32,
        beta:     f32,
        seed:     u64,
    ) -> usize {
        if count == 0 || strength <= 0.0 || self.moment_history.is_empty() {
            return 0;
        }
        // 1. Score every moment.
        let scored: Vec<(f32, MomentFingerprint)> = self.moment_history
            .iter()
            .map(|fp| (self.moment_salience_score(fp), fp.clone()))
            .collect();
        // 2. Numerically-stable softmax: subtract max before exp.
        let max_score = scored.iter().map(|(s, _)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let safe_max = if max_score.is_finite() { max_score } else { 0.0 };
        let mut weights: Vec<f32> = scored.iter()
            .map(|(s, _)| ((s - safe_max) * beta).exp().max(1e-30))
            .collect();
        let mut remaining_total: f32 = weights.iter().sum();

        // 3. Sample without replacement via the cumulative-distribution
        //    inverse method.  Local xorshift64 — deterministic given seed.
        let mut rng_state = if seed == 0 { 0xDEADBEEF_CAFEBABE_u64 } else { seed };
        let next_f32_unit = |state: &mut u64| -> f32 {
            let mut x = *state;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            *state = x;
            let mixed = x.wrapping_mul(0x2545F4914F6CDD1D);
            ((mixed >> 40) as f32) / ((1u32 << 24) as f32)
        };
        let n_target = count.min(scored.len());
        let mut chosen: Vec<MomentFingerprint> = Vec::with_capacity(n_target);
        for _ in 0..n_target {
            if remaining_total <= 0.0 { break; }
            let u = next_f32_unit(&mut rng_state) * remaining_total;
            let mut acc = 0.0f32;
            let mut picked = 0usize;
            for (i, w) in weights.iter().enumerate() {
                acc += *w;
                if acc >= u { picked = i; break; }
            }
            chosen.push(scored[picked].1.clone());
            remaining_total -= weights[picked];
            weights[picked] = 0.0;
        }

        // 4. Re-fire each chosen moment.  Same injection + tick pattern
        //    as the uniform / top-K replay paths above.
        let mut replayed = 0usize;
        for fp in &chosen {
            let now = self.fabric.current_tick();
            if !fp.ordered_per_pool.is_empty() {
                for (pid, ns) in &fp.ordered_per_pool {
                    if let Some(pool) = self.fabric.pool(*pid) {
                        let mut pp = pool.write();
                        for &nid in ns {
                            pp.inject_activation(nid, strength, now);
                        }
                    }
                }
            } else {
                for &(pid, nid) in &fp.pairs {
                    if let Some(pool) = self.fabric.pool(pid) {
                        pool.write().inject_activation(nid, strength, now);
                    }
                }
            }
            self.fabric.advance_tick();
            replayed += 1;
        }
        replayed
    }

    /// Stage 17.7 partial — salience-weighted REPLAY per
    /// [`ARCHITECTURE.md`] §17.7.  Samples the top-`count` moments from
    /// `moment_history` by their salience score, then re-fires them
    /// (oldest-of-the-top-K first, to preserve any sequential structure).
    /// Same activation injection + tick path as the uniform replay below
    /// — the only difference is *which* moments are chosen.
    ///
    /// Closes the loop with Stage 17.5: high-salience neurons get
    /// preferential replay, which strengthens their terminals (and bumps
    /// their salience again on decode).  Frémaux & Gerstner 2016 three-
    /// factor plasticity at the moment-buffer scale.
    ///
    /// True free-energy-weighted replay using the annealer (boltzmann
    /// sampling by `exp(-beta * free_energy_delta)`) is the full Stage
    /// 17.7 form — ships in a follow-up once the annealer's energy
    /// surface is exposed at the moment-fingerprint granularity.
    pub fn replay_salience_weighted(&mut self, count: usize, strength: f32) -> usize {
        if count == 0 || strength <= 0.0 || self.moment_history.is_empty() {
            return 0;
        }
        // Score every moment in history.  Avoid scoring more than we need
        // to consider — sort by score, take top-K.  At typical
        // moment_history_window sizes (256-1024) this is fast.
        let mut scored: Vec<(f32, MomentFingerprint)> = self.moment_history
            .iter()
            .map(|fp| (self.moment_salience_score(fp), fp.clone()))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(count.min(scored.len()));
        // Re-order by temporal order (the order in moment_history is
        // already temporal; sort by their position there).  We don't
        // have positions captured, but since we cloned from history in
        // its original order before sorting, just re-collect from history
        // selecting those whose fingerprints match.  Cheap because count
        // is small.  Simpler: just replay in score-descending order, which
        // gives an emphasis on most-salient-first.
        let to_replay: Vec<MomentFingerprint> =
            scored.into_iter().map(|(_, fp)| fp).collect();

        let mut replayed = 0usize;
        for fp in &to_replay {
            let now = self.fabric.current_tick();
            if !fp.ordered_per_pool.is_empty() {
                for (pid, ns) in &fp.ordered_per_pool {
                    if let Some(pool) = self.fabric.pool(*pid) {
                        let mut pp = pool.write();
                        for &nid in ns {
                            pp.inject_activation(nid, strength, now);
                        }
                    }
                }
            } else {
                for &(pid, nid) in &fp.pairs {
                    if let Some(pool) = self.fabric.pool(pid) {
                        pool.write().inject_activation(nid, strength, now);
                    }
                }
            }
            self.fabric.advance_tick();
            replayed += 1;
        }
        replayed
    }

    /// P4 sleep-cycle REPLAY (Wilson & McNaughton 1994; McClelland,
    /// McNaughton, O'Reilly 1995 — Complementary Learning Systems).
    /// Re-fires the last `count` moment fingerprints in their original
    /// firing order at reduced activation `strength`, then ticks the
    /// fabric so cross-pool terminals reinforce the same patterns.
    ///
    /// Biological purpose: during sleep / quiet wakefulness, the
    /// hippocampus replays recent activity in compressed form, driving
    /// repeated reactivation of cortical patterns.  This consolidates
    /// recent episodic memories into long-term cortical statistical
    /// representations and is the mechanism by which CLS resolves the
    /// stability/plasticity dilemma.
    ///
    /// Returns the number of fingerprints replayed.  Intended to be
    /// called AFTER `sleep()` prunes weak concepts so replay only
    /// strengthens patterns that survived pruning.
    pub fn replay_recent_moments(&mut self, count: usize, strength: f32) -> usize {
        if count == 0 || strength <= 0.0 { return 0; }
        // Snapshot the last `count` fingerprints.  Replay in original
        // temporal order (oldest first) so any sequential structure
        // between them is preserved.
        let recent: Vec<MomentFingerprint> = self.moment_history.iter()
            .rev().take(count).cloned().collect::<Vec<_>>()
            .into_iter().rev().collect();
        let mut replayed = 0usize;
        for fp in &recent {
            let now = self.fabric.current_tick();
            // Re-inject each member's activation in its source pool.
            // Walks ordered_per_pool (preserves firing order) when
            // available, else falls back to the sorted `pairs`.
            if !fp.ordered_per_pool.is_empty() {
                for (pid, ns) in &fp.ordered_per_pool {
                    if let Some(pool) = self.fabric.pool(*pid) {
                        let mut pp = pool.write();
                        for &nid in ns {
                            pp.inject_activation(nid, strength, now);
                        }
                    }
                }
            } else {
                for &(pid, nid) in &fp.pairs {
                    if let Some(pool) = self.fabric.pool(pid) {
                        pool.write().inject_activation(nid, strength, now);
                    }
                }
            }
            // Advance: drives cross-pool terminal reinforcement,
            // fingerprint re-registration (which may strengthen
            // consolidated bindings), and the standard decay/
            // sparsity pipeline.  This is what turns replay into
            // actual synaptic change.
            self.fabric.advance_tick();
            replayed += 1;
        }
        replayed
    }

    // ---------------------------------------------------------------
    // Phase 9 — Checkpoint / restore (spec §6 + §11).
    //
    // Persists the entire learned state of the brain to a single
    // bincode-encoded file.  Restore rebuilds it verbatim; the only
    // thing the caller re-supplies is the pool encodings (which are
    // stateless trait objects).
    // ---------------------------------------------------------------

    /// Capture this brain's full persistent state into a snapshot
    /// struct.  Use [`crate::persistence::save_snapshot`] to write it
    /// to disk, or hold it in memory for tests.
    pub fn snapshot(&self) -> crate::persistence::BrainSnapshot {
        let moment_history: std::collections::VecDeque<crate::persistence::SerializableFingerprint> =
            self.moment_history.iter()
                .map(|f| crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() })
                .collect();
        let binding_recurrences: Vec<(crate::persistence::SerializableFingerprint, u32)> =
            self.binding_recurrences.iter()
                .map(|(f, &c)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, c))
                .collect();
        let promoted_fingerprints: Vec<(crate::persistence::SerializableFingerprint, NeuronId)> =
            self.promoted_fingerprints.iter()
                .map(|(f, &n)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, n))
                .collect();
        let tentative_promoted: Vec<(crate::persistence::SerializableFingerprint, NeuronId)> =
            self.tentative_promoted.iter()
                .map(|(f, &n)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, n))
                .collect();
        let lifetime_recurrences: Vec<(crate::persistence::SerializableFingerprint, u32)> =
            self.lifetime_recurrences.iter()
                .map(|(f, &c)| (crate::persistence::SerializableFingerprint { pairs: f.pairs.clone() }, c))
                .collect();
        let pending_actions: Vec<(ActionId, ActionEvent)> = self.pending_actions
            .iter().map(|(k, v)| (*k, v.clone())).collect();
        crate::persistence::BrainSnapshot {
            format_version:              crate::persistence::CURRENT_SNAPSHOT_VERSION,
            binding_pool_id:             self.binding_pool_id,
            binding_emergence_threshold: self.config.binding_emergence_threshold,
            moment_history_window:       self.config.moment_history_window,
            fabric:                      self.fabric.snapshot(),
            eem:                         self.eem.snapshot(),
            annealer:                    self.annealer.snapshot(),
            moment_history,
            binding_recurrences,
            promoted_fingerprints,
            tentative_promoted,
            lifetime_recurrences,
            tentative_emergence_threshold: self.config.tentative_emergence_threshold,
            current_threshold:             self.current_threshold,
            total_observations:            self.total_observations,
            action_pool_id:              self.action_pool_id,
            pending_actions,
            next_action_id:              self.next_action_id,
        }
    }

    /// Write this brain's state to `path`.  Convenience wrapper over
    /// `snapshot()` + `persistence::save_snapshot()`.
    ///
    /// Per [`ARCHITECTURE.md`] §17.9, when a WAL store is attached this
    /// also flushes the WAL and emits a `SnapshotMarker` so crash recovery
    /// can fast-forward past replayed events that the bincode snapshot
    /// already covers.  The bincode path is interim until the content-
    /// addressed terminal store ships in full (Stage 17.4+).
    pub fn checkpoint<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        // 1. Streaming-serialize the snapshot to disk (the §17.1 interim
        //    path with peak-RAM = write-buffer, not brain-size).
        crate::persistence::save_snapshot(&self.snapshot(), path)?;

        // 2. Flush the WAL and emit a snapshot marker.  When a NoopStore
        //    is attached (the default, no W1Z4RDV1510N_DATA_DIR set) both
        //    calls are inert.
        let store = self.fabric.store_clone();
        if let Err(e) = store.flush() {
            tracing::warn!("WAL flush failed during checkpoint: {}", e);
        }
        let wall_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        let marker = crate::store::WalEvent::SnapshotMarker {
            tick: self.fabric.current_tick(),
            wall_time_ms,
        };
        if let Err(e) = store.append(&marker) {
            tracing::warn!("WAL marker append failed during checkpoint: {}", e);
        }
        if let Err(e) = store.flush() {
            tracing::warn!("WAL flush after marker failed: {}", e);
        }
        Ok(())
    }

    /// Attach a persistence backend per [`ARCHITECTURE.md`] §17.9.
    /// Fans out through the fabric to every pool.  Idempotent — safe to
    /// call multiple times with the same store.
    pub fn set_store(&mut self, store: std::sync::Arc<dyn crate::store::Store>) {
        self.fabric.set_store(store);
    }

    /// Clone the persistence handle.  Exposes the `Store` to brain_server
    /// for the `/flush` endpoint and any tooling that needs to inspect
    /// the WAL size / force a barrier without going through `checkpoint`.
    pub fn store_clone(&self) -> std::sync::Arc<dyn crate::store::Store> {
        self.fabric.store_clone()
    }

    /// Stage 18.12 step 7 — scan all pools for currently-firing neurons
    /// whose outgoing terminals point to neurons whose home is a remote
    /// cluster peer.  Returns a per-peer batch of deposits ready to be
    /// shipped via `POST /shard/deposit`.
    ///
    /// Per [`ARCHITECTURE.md`] §18.7 — per-tick all-to-all cross-shard
    /// activation deposit batching.  Each entry is
    /// `(target_pool_id, target_neuron_id, deposit_strength)` where
    /// strength = `terminal.effective_weight() * source_activation`.
    ///
    /// Solo mode: returns empty (no remote peers, nothing to ship).
    /// Cluster mode: returns deposits grouped by destination NodeId.
    /// Caller is responsible for actually sending them (the brain crate
    /// doesn't have an HTTP client; brain_server owns transport).
    pub fn scan_cross_shard_deposits(
        &self,
        local_node: crate::store::NodeId,
    ) -> std::collections::HashMap<
            crate::store::NodeId,
            Vec<(PoolId, NeuronId, f32)>
        >
    {
        use std::collections::HashMap as StdHashMap;
        let mut out: StdHashMap<crate::store::NodeId, Vec<(PoolId, NeuronId, f32)>>
            = StdHashMap::new();

        for pid in self.fabric.pool_ids() {
            let Some(pool_arc) = self.fabric.pool(pid) else { continue; };
            let pool = pool_arc.read();
            // Need access to tiered_store; not exposed directly, so we
            // peek via has_tiered_store + a helper.  Until that's wired,
            // we walk each pool's neurons individually.
            if !pool.has_tiered_store() { continue; }

            // For each firing neuron, walk its terminals.
            for nid in pool.currently_firing() {
                let Some(neuron) = pool.get(nid) else { continue; };
                if neuron.terminals.is_empty() { continue; }
                let src_activation = pool.activation(nid);
                if src_activation <= 0.0 { continue; }

                for term in &neuron.terminals {
                    // Get the home of target.neuron.  We need the
                    // TARGET pool's tiered_store for placement.
                    let target_pool_arc = self.fabric.pool(term.target.pool);
                    let target_home = match target_pool_arc {
                        Some(p) => {
                            let tp = p.read();
                            // home_for through whichever store the
                            // target pool has.  Solo pools return local
                            // (so no deposit queued — correct).
                            tp.tiered_home_for(term.target.neuron)
                                .unwrap_or(local_node)
                        }
                        None => local_node,
                    };
                    if target_home == local_node { continue; }
                    let deposit = term.effective_weight() * src_activation;
                    if deposit <= 0.0 { continue; }
                    out.entry(target_home)
                       .or_insert_with(Vec::new)
                       .push((term.target.pool, term.target.neuron, deposit));
                }
            }
        }
        out
    }

    /// Stage 17.6 — map of every pool's Merkle root, keyed by pool id.
    /// First half of the cluster anti-entropy protocol per
    /// [`ARCHITECTURE.md`] §17.6: peers compare these maps to identify
    /// pools whose state has diverged.  The local-only Merkle work
    /// (`Pool::merkle_root`) is from `8c31d1f`; this method just collects
    /// roots across the brain's pools at the brain's current tick.
    pub fn cluster_pool_roots(&self) -> std::collections::HashMap<PoolId, crate::store::PoolRoot> {
        let tick = self.fabric.current_tick();
        let mut out = std::collections::HashMap::new();
        for pid in self.fabric.pool_ids() {
            if let Some(pool) = self.fabric.pool(pid) {
                let root = pool.read().merkle_root(tick);
                out.insert(pid, root);
            }
        }
        out
    }

    /// Stage 17.6 — merge a batch of neurons received from a peer into
    /// `pool_id`.  Per [`ARCHITECTURE.md`] §17.6 cluster anti-entropy:
    /// when the local Merkle root for `pool_id` disagrees with a peer's,
    /// the peer ships its neurons here and we apply the ones we don't
    /// have yet (idempotent on overlapping ids).
    ///
    /// Conservative semantics: only inserts NEW neurons (id >= local
    /// count), never overwrites existing ones.  When inserting, the
    /// **whole peer neuron** is taken — including its terminals,
    /// salience, use_count, and last_fired_tick.  That's the §17.6
    /// "deeper" form: terminal weights propagate across the cluster,
    /// not just topology.
    ///
    /// Returns the count of newly-inserted neurons.
    pub fn cluster_merge_pool(&self, pool_id: PoolId, incoming: Vec<crate::Neuron>) -> usize {
        let Some(pool_arc) = self.fabric.pool(pool_id) else { return 0; };
        let mut pool = pool_arc.write();
        let mut inserted = 0usize;
        for n in incoming {
            // Sequential-id contract: only accept neurons whose id is
            // exactly our next slot.  Prevents id-space conflicts; the
            // peer was presumably trained on the same input order, so
            // ids line up naturally.  Future: a translation table for
            // peers with divergent id-space.
            let next = pool.neuron_count() as crate::NeuronId;
            if n.id != next { continue; }
            if pool.replay_full_neuron(n) {
                inserted += 1;
            }
        }
        inserted
    }

    /// Stage 17.6 — return a snapshot of a single pool's neurons for
    /// cluster sync.  Per [`ARCHITECTURE.md`] §17.6, this is the
    /// "give me your authoritative state for this pool" RPC payload —
    /// a peer fetches this when its Merkle root for `pool_id` disagrees.
    /// Returns the pool's `Vec<Neuron>` cloned at the call moment under
    /// a brief read lock.  For very large pools, a paginated variant is
    /// the follow-up; this first cut is bounded by pool size in RAM.
    pub fn cluster_pool_neurons(&self, pool_id: PoolId) -> Option<Vec<crate::Neuron>> {
        let pool_arc = self.fabric.pool(pool_id)?;
        let pool = pool_arc.read();
        Some(pool.iter_neurons().cloned().collect())
    }

    /// Stage 17.9 — replay a slice of WAL events into this brain to
    /// bring it forward from a snapshot-restored state to its true last-
    /// known state.  Typically the slice comes from
    /// `crate::store::load_events_after_marker`.
    ///
    /// Applies each event:
    /// - `PoolRegistered`: skipped (pools already registered from
    ///   brain.bin or from explicit setup).
    /// - `AtomCreated` / `ConceptEmerged`: insert into the pool at the
    ///   recorded id slot.  Idempotent — if the slot is already filled
    ///   (snapshot already restored), the call returns false and we
    ///   advance.
    /// - `TerminalReinforced` / `NeuronTerminalsPruned`: deferred —
    ///   weight-level updates aren't currently logged at hot-path rate.
    ///   Brain re-learns weights on the next training pass.
    /// - `NeuronEvicted`: marks the neuron as evicted (cold offset
    ///   comes from the PoolSnapshot per Stage 17.4 step 5).
    /// - `TickAdvanced`: advances the fabric's tick counter.
    /// - `SnapshotMarker`: skipped (just a checkpoint barrier).
    ///
    /// Returns a `RecoveryStats` describing how many of each variant
    /// were applied.
    pub fn apply_wal_events(
        &mut self,
        events: &[crate::store::WalEvent],
    ) -> crate::store::RecoveryStats {
        use crate::store::WalEvent as E;
        let mut stats = crate::store::RecoveryStats::default();
        for ev in events {
            stats.observe(ev);
            match ev {
                E::PoolRegistered { .. } => { /* skip: already registered */ }
                E::AtomCreated { pool_id, id, label, kind, born_tick } => {
                    if let Some(pool) = self.fabric.pool(*pool_id) {
                        pool.write().replay_atom_create(
                            *id, label.clone(), *kind, *born_tick,
                        );
                    }
                }
                E::ConceptEmerged { pool_id, id, label, kind, members, born_tick } => {
                    if let Some(pool) = self.fabric.pool(*pool_id) {
                        pool.write().replay_concept_create(
                            *id, label.clone(), *kind,
                            members.clone(), *born_tick,
                        );
                    }
                }
                E::NeuronEvicted { pool_id, neuron_id } => {
                    if let Some(pool) = self.fabric.pool(*pool_id) {
                        pool.write().replay_neuron_evicted(*neuron_id);
                    }
                }
                E::TickAdvanced { new_tick } => {
                    self.fabric.set_tick(*new_tick);
                }
                E::TerminalReinforced(_) | E::NeuronTerminalsPruned { .. } => {
                    // Weight-level updates aren't logged at hot path rate
                    // yet; weights re-learned on next training pass.
                }
                E::SnapshotMarker { .. } => { /* barrier only */ }
            }
        }
        stats
    }

    /// Stage 17.4 full — attach cold tiers to every pool under
    /// `data_dir/cold/pool_{id}.cold`.  After this call, `run_eviction_pass`
    /// can move low-salience concepts to disk.  Binding pool is included
    /// so the API is uniform, but the policy in `run_eviction_pass` will
    /// skip the binding pool by default.
    ///
    /// Returns the number of pools that successfully attached a cold
    /// tier.  Errors per-pool are logged but don't fail the whole call.
    pub fn attach_cold_tiers<P: AsRef<std::path::Path>>(&mut self, data_dir: P) -> usize {
        let dir = data_dir.as_ref().join("cold");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            tracing::warn!("attach_cold_tiers: mkdir failed: {}", e);
            return 0;
        }
        let pool_ids = self.fabric.pool_ids();
        let mut attached = 0;
        for pid in pool_ids {
            let path = dir.join(format!("pool_{}.cold", pid));
            match crate::store::ColdTier::open(&path) {
                Ok(tier) => {
                    let tier_arc = std::sync::Arc::new(tier);
                    if let Some(pool) = self.fabric.pool(pid) {
                        pool.write().set_cold_tier(tier_arc);
                        attached += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("cold tier open failed for pool {}: {}", pid, e);
                }
            }
        }
        attached
    }

    /// Stage 17.4 full — run one eviction pass per
    /// [`ARCHITECTURE.md`] §17.4.  Walks each pool, identifies concept
    /// neurons with low `salience_ema` AND staleness past `min_stale_ticks`,
    /// and evicts up to `target_per_pool` of them (lowest-salience first).
    ///
    /// Atoms are never evicted (refused by `Pool::evict_neuron`).  The
    /// binding pool is skipped entirely — bindings are the answer keys.
    ///
    /// **Stage 17.2 — Hebbian disk layout:** after the salience-sort
    /// pre-filter, candidates are *reordered* by a co-firing proxy
    /// (shared terminal targets) so neurons that fire together land at
    /// adjacent byte offsets in the cold-tier file.  The OS read-ahead
    /// prefetcher then pulls in their neighbors for free when one is
    /// paged back in.  This is Mountcastle's (1997) cortical-column
    /// principle applied to disk layout: physical proximity matches
    /// feature proximity.
    ///
    /// Returns `EvictionStats { pools_visited, neurons_evicted,
    /// errors, wall_time_ms }`.
    pub fn run_eviction_pass(&self, params: EvictionParams) -> EvictionStats {
        let started = std::time::Instant::now();
        let now = self.fabric.current_tick();
        let pool_ids = self.fabric.pool_ids();
        let mut stats = EvictionStats::default();

        for pid in pool_ids {
            // Skip binding pool — bindings are answer keys, must stay hot.
            if pid == self.binding_pool_id { continue; }
            let Some(pool_arc) = self.fabric.pool(pid) else { continue; };
            stats.pools_visited += 1;

            // Phase 1 (read): identify candidates + collect their terminal
            // target sets for the §17.2 co-firing reorder.  Brief read lock.
            let candidates_ordered: Vec<NeuronId> = {
                let p = pool_arc.read();
                // First filter by salience + staleness.
                let mut filtered: Vec<(NeuronId, f32, u64, Vec<NeuronRef>)> = p.iter_neurons()
                    .filter(|n| !n.is_atom())
                    .filter(|n| !p.is_evicted(n.id))
                    .filter(|n| n.salience_ema < params.max_salience_ema)
                    .filter(|n| now.saturating_sub(n.last_fired_tick) >= params.min_stale_ticks)
                    .map(|n| {
                        let tgts: Vec<NeuronRef> = n.terminals.iter()
                            .map(|t| t.target)
                            .collect();
                        (n.id, n.salience_ema, n.last_fired_tick, tgts)
                    })
                    .collect();
                // Pre-sort: lowest salience first, ties by oldest last_fired.
                filtered.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.2.cmp(&b.2))
                });
                filtered.truncate(params.target_per_pool);
                if filtered.is_empty() { Vec::new() }
                else {
                    // §17.2 reorder: greedy nearest-neighbour by Jaccard
                    // similarity of terminal target sets.  Start from the
                    // lowest-salience candidate; each subsequent step
                    // picks the remaining candidate that shares the most
                    // terminal targets with the previous.  Co-fired
                    // neurons (high target overlap) cluster adjacently
                    // in the output order, which determines their disk
                    // offset on append to cold tier.
                    use std::collections::HashSet;
                    let n = filtered.len();
                    let target_sets: Vec<HashSet<NeuronRef>> = filtered.iter()
                        .map(|(_, _, _, tgts)| tgts.iter().copied().collect())
                        .collect();
                    let mut chosen: Vec<bool> = vec![false; n];
                    let mut order: Vec<NeuronId> = Vec::with_capacity(n);
                    // Seed: lowest-salience (= filtered[0]).
                    order.push(filtered[0].0);
                    chosen[0] = true;
                    for _ in 1..n {
                        // Find the previous chosen candidate's target set.
                        let last_idx = order.last()
                            .and_then(|nid| filtered.iter().position(|(i, _, _, _)| i == nid))
                            .unwrap_or(0);
                        let prev_set = &target_sets[last_idx];
                        let mut best: Option<(usize, f32)> = None;
                        for i in 0..n {
                            if chosen[i] { continue; }
                            let ts = &target_sets[i];
                            // Jaccard: |A∩B| / |A∪B|; constant when both
                            // are empty, but we don't reorder isolated
                            // neurons by anything other than salience tiebreak.
                            let inter = prev_set.intersection(ts).count() as f32;
                            let union = prev_set.union(ts).count() as f32;
                            let j = if union == 0.0 { 0.0 } else { inter / union };
                            match best {
                                None => best = Some((i, j)),
                                Some((_, bj)) if j > bj => best = Some((i, j)),
                                _ => {}
                            }
                        }
                        if let Some((idx, _)) = best {
                            order.push(filtered[idx].0);
                            chosen[idx] = true;
                        } else { break; }
                    }
                    order
                }
            };
            if candidates_ordered.is_empty() { continue; }

            // Phase 2 (write): evict each in the Hebbian-locality order.
            // Cold-tier file appends in this exact order → co-fired
            // neurons share disk pages → OS prefetcher pulls neighbours
            // for free on subsequent page-in.
            {
                let mut p = pool_arc.write();
                for cid in candidates_ordered {
                    match p.evict_neuron(cid) {
                        Ok(true)  => { stats.neurons_evicted += 1; }
                        Ok(false) => { /* already evicted or atom */ }
                        Err(e) => {
                            tracing::warn!(
                                "evict_neuron(pool={}, id={}) failed: {}",
                                pid, cid, e,
                            );
                            stats.errors += 1;
                        }
                    }
                }
            }
        }

        stats.wall_time_ms = started.elapsed().as_millis() as u64;
        stats
    }

    /// Stage 17.8 — observable signals for storage-tier dynamical
    /// control.  Per [`ARCHITECTURE.md`] §17.8, the same control
    /// architecture (ControlMode/ControlSignal/ControlState) that
    /// governs the substrate's plasticity knobs also governs the
    /// storage tier; this method computes the storage-tier
    /// observables that future knobs (eviction, replay rate, bloom
    /// resize) read from.
    ///
    /// Today (Stage 17.8): the salience-distribution-entropy and
    /// bloom-load signals are populated from real state; working-set-
    /// pressure / cache-hit-rate / replay-value remain 0.0 until
    /// Stage 17.4 full (eviction actor) and Stage 17.7 full
    /// (free-energy replay) ship.
    pub fn storage_control_state(&self) -> crate::store::StorageControlState {
        // Histogram the salience distribution across all neurons in all
        // pools.  10 bins from 0.0 to 1.0 → 0.1 wide each.  Shannon
        // entropy in bits over this histogram is the diversity signal.
        const NBINS: usize = 10;
        let mut bins = [0u64; NBINS];
        let mut bloom_inserted: u64 = 0;
        let mut bloom_slots:    u64 = 0;
        let mut total_neurons:  u64 = 0;
        let mut evicted_neurons: u64 = 0;
        for pid in self.fabric.pool_ids() {
            if let Some(p) = self.fabric.pool(pid) {
                let pool = p.read();
                for n in pool.iter_neurons() {
                    let s = n.salience.clamp(0.0, 1.0);
                    let idx = ((s * NBINS as f32) as usize).min(NBINS - 1);
                    bins[idx] += 1;
                }
                bloom_inserted += pool.bloom_inserted_keys() as u64;
                // Bloom slot count is a function of the bloom — fetch
                // via byte_size * 2 (4-bit counters → 2 slots per byte).
                bloom_slots += (pool.bloom_byte_size() as u64) * 2;
                total_neurons    += pool.neuron_count() as u64;
                evicted_neurons  += pool.evicted_count() as u64;
            }
        }
        let entropy = crate::store::StorageControlState::entropy_from_bins(&bins);
        let load = if bloom_slots > 0 {
            (bloom_inserted as f32 / bloom_slots as f32).clamp(0.0, 1.0)
        } else { 0.0 };
        // Stage 17.4 full: working_set_pressure = live / total.  1.0 means
        // every neuron is in RAM (full pressure to evict); 0.0 means all
        // evicted.  Caller's ControlMode interprets — typically
        // "DrivenBy(working_set_pressure, scale, offset)" makes eviction
        // more aggressive as pressure climbs.
        let ws_pressure = if total_neurons > 0 {
            ((total_neurons - evicted_neurons) as f32 / total_neurons as f32)
                .clamp(0.0, 1.0)
        } else { 0.0 };

        crate::store::StorageControlState {
            working_set_pressure:       ws_pressure,
            cache_hit_rate:             0.0,  // §17.4 step 2 deferred
            replay_value_score:         0.0,  // §17.7 full
            salience_distribution_entropy: entropy,
            bloom_load:                 load,
        }
    }

    /// Rebuild a brain from a snapshot, supplying fresh encodings
    /// for every pool id in the snapshot.  The binding pool is
    /// included in `encodings`; pass a `BytePassthroughEncoding`
    /// matching the original prefix (`"bind"` for default brains).
    ///
    /// Returns `(brain, missing_pool_ids)` where `missing_pool_ids`
    /// lists pool ids in the snapshot for which no encoding was
    /// supplied — those pools are silently skipped.  Callers should
    /// treat a non-empty missing list as a misconfiguration.
    pub fn from_snapshot(
        snap:      crate::persistence::BrainSnapshot,
        encodings: std::collections::HashMap<PoolId, Box<dyn AtomEncoding>>,
    ) -> (Self, Vec<PoolId>) {
        let (fabric, missing) = Fabric::from_snapshot(snap.fabric, encodings);
        let config = BrainConfig {
            fabric: fabric.config.clone(),
            binding_emergence_threshold: snap.binding_emergence_threshold,
            tentative_emergence_threshold: snap.tentative_emergence_threshold,
            moment_history_window:       snap.moment_history_window,
            min_atom_score:               default_min_atom_score(),
            pressure_band_low:           default_pressure_band_low(),
            pressure_band_high:          default_pressure_band_high(),
            pressure_threshold_max:      default_pressure_threshold_max(),
            pressure_observation_grace:  default_pressure_observation_grace(),
            pressure_adjust_enabled:     default_pressure_adjust_enabled(),
            binding_pool_config:         PoolConfig::defaults("binding", snap.binding_pool_id),
            eem:                         snap.eem.config.clone(),
            annealer:                    snap.annealer.config.clone(),
        };

        let mut moment_history = VecDeque::with_capacity(snap.moment_history_window);
        for f in snap.moment_history {
            moment_history.push_back(MomentFingerprint { pairs: f.pairs, ordered_per_pool: Vec::new() });
        }
        let mut binding_recurrences = AHashMap::new();
        for (f, c) in snap.binding_recurrences {
            binding_recurrences.insert(MomentFingerprint { pairs: f.pairs, ordered_per_pool: Vec::new() }, c);
        }
        let mut promoted_fingerprints = AHashMap::new();
        for (f, n) in snap.promoted_fingerprints {
            promoted_fingerprints.insert(MomentFingerprint { pairs: f.pairs, ordered_per_pool: Vec::new() }, n);
        }
        let mut tentative_promoted = AHashMap::new();
        for (f, n) in snap.tentative_promoted {
            tentative_promoted.insert(MomentFingerprint { pairs: f.pairs, ordered_per_pool: Vec::new() }, n);
        }
        let mut lifetime_recurrences = AHashMap::new();
        for (f, c) in snap.lifetime_recurrences {
            lifetime_recurrences.insert(MomentFingerprint { pairs: f.pairs, ordered_per_pool: Vec::new() }, c);
        }
        let mut pending_actions = AHashMap::new();
        for (k, v) in snap.pending_actions { pending_actions.insert(k, v); }

        let restored_threshold = if snap.current_threshold == 0 {
            snap.binding_emergence_threshold.max(1)
        } else {
            snap.current_threshold
        };
        let brain = Self {
            fabric,
            config,
            binding_pool_id:       snap.binding_pool_id,
            moment_history,
            binding_recurrences,
            lifetime_recurrences,
            tentative_promoted,
            promoted_fingerprints,
            total_observations:      snap.total_observations,
            current_threshold:       restored_threshold,
            last_pressure_check_obs: snap.total_observations,
            action_pool_id:        snap.action_pool_id,
            pending_actions,
            next_action_id:        snap.next_action_id,
            emitted_this_tick:     ahash::AHashSet::new(),
            eem:                   crate::eem::Eem::from_snapshot(snap.eem),
            annealer:              crate::annealer::Annealer::from_snapshot(snap.annealer),
            network:               NetworkState::new(""),
            qa_db:                 QaDatabase::new(4096),
            recent_frames:         AHashMap::new(),
            tuning:                TuningState::default(),
            feedback_loops:        Vec::new(),
            delayed_feedback:      Vec::new(),
            feedback_events_emitted: 0,
        };
        (brain, missing)
    }

    /// Load a brain from `path`.  Convenience wrapper over
    /// `persistence::load_snapshot()` + `from_snapshot()`.
    pub fn restore<P: AsRef<std::path::Path>>(
        path:      P,
        encodings: std::collections::HashMap<PoolId, Box<dyn AtomEncoding>>,
    ) -> std::io::Result<(Self, Vec<PoolId>)> {
        let snap = crate::persistence::load_snapshot(path)?;
        Ok(Self::from_snapshot(snap, encodings))
    }

    pub fn stats(&self) -> BrainStats {
        let mut stats = BrainStats {
            tick:                self.fabric.current_tick(),
            pool_count:          0,
            total_neurons:       0,
            total_concepts:      0,
            total_binding:       0,
            total_terminals:     0,
            binding_pool_id:     self.binding_pool_id,
            fingerprints_window: self.moment_history.len(),
            tentative_bindings:    self.tentative_promoted.len(),
            consolidated_bindings: self.promoted_fingerprints.len(),
            current_threshold:     self.current_threshold,
            total_observations:    self.total_observations,
            binding_pressure:      self.binding_pressure(),
        };
        for pid in self.fabric.pool_ids() {
            if let Some(p) = self.fabric.pool(pid) {
                let pool = p.read();
                stats.pool_count += 1;
                let nc = pool.neuron_count();
                stats.total_neurons += nc;
                let cc = pool.concept_count();
                stats.total_concepts += cc;
                // O(1) — maintained terminal counter (was O(N) walk
                // before pool.total_terminals was introduced).
                stats.total_terminals += pool.total_terminals();
                if pid == self.binding_pool_id {
                    stats.total_binding += cc;
                }
            }
        }
        stats
    }
}
