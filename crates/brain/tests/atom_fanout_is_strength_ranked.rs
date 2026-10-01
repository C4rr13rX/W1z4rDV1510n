//! The atom fan-out bound keeps the STRONGEST terminals, not the earliest.
//!
//! `PoolConfig::max_atom_fanout` bounds a hub atom's fan-out so a byte that
//! appears in every fact cannot acquire one terminal per fact. Until
//! 2026-10-01 it enforced that bound FIRST-COME: `terminals.len() >= cap`
//! refused every later terminal, so an atom that filled on the opening facts
//! of a corpus acquired nothing at all for the rest of it.
//!
//! Measured 2026-10-01, `python tools/scorecard.py --scales 16,64`, scale 16,
//! with the bound lifted entirely (`default_max_atom_fanout() = 0`) and
//! nothing else changed:
//!
//! ```text
//!   scale   integr%         peak_mb                  hub_fanout      terminals
//!      16     44.3 -> 50.8    24.9 -> 28.0 (+12.4%)    512 ->  2,976   68,943 ->  94,370
//!      64     20.5 -> 32.5    40.1 -> 51.9 (+29.4%)    512 -> 11,904   26,827 -> 189,436
//! ```
//!
//! Recall stayed 100.0 % at both scales. So the terminals the bound refused
//! were carrying answers, and the unbounded hub fan-out is one terminal per
//! fact -- which is the RAM the bound exists to prevent, and +29.4 % against a
//! gate that allows 15 %. These tests pin the third option: the same bound, a
//! different selection rule.

use w1z4rd_brain::neuron::{Neuron, NeuronKind, NeuronRef, Terminal};

const POOL: u32 = 0;

fn atom() -> Neuron {
    Neuron::new_atom(0, "a:hub".into(), NeuronKind::Excitatory, 1)
}

fn target(n: u32) -> NeuronRef {
    NeuronRef::new(POOL, n)
}

/// The bound is still a bound. Whatever the selection rule does, fan-out may
/// never exceed `cap` -- that is the whole RAM argument for having one.
#[test]
fn a_displacement_never_grows_the_fan_out() {
    const CAP: usize = 8;
    let mut n = atom();
    for i in 0..64u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
        assert!(
            n.terminals.len() <= CAP,
            "fan-out {} exceeded cap {CAP} after admitting target {i}",
            n.terminals.len()
        );
    }
    assert_eq!(n.terminals.len(), CAP, "a saturated atom should sit at its cap");
}

/// A newcomer that is strictly stronger than the weakest resident terminal
/// takes its slot. Under the first-come rule this atom's resident set was
/// frozen at targets 0..8 forever.
#[test]
fn a_stronger_newcomer_displaces_the_weakest_resident() {
    const CAP: usize = 4;
    let mut n = atom();
    for i in 0..CAP as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
    }
    // Decay the resident set the way `apply_pending_decay` does, leaving
    // target 2 as the weakest.
    for t in n.terminals.iter_mut() {
        t.weight = 0.40;
    }
    n.terminals[2].weight = 0.05;
    let admitted = n.reinforce_terminal_bounded(target(99), 0.5, 2, 4.0, CAP);
    assert!(
        !admitted,
        "a displacement is net-zero fan-out and must not be reported as an add"
    );
    assert_eq!(n.terminals.len(), CAP);
    assert!(
        n.find_terminal(&target(99)).is_some(),
        "the stronger newcomer was refused: {:?}",
        n.terminals.iter().map(|t| t.target.neuron).collect::<Vec<_>>()
    );
    assert!(
        n.find_terminal(&target(2)).is_none(),
        "the weakest resident terminal survived a stronger newcomer"
    );
    // Every other resident terminal is untouched.
    for i in [0u32, 1, 3] {
        assert!(
            n.find_terminal(&target(i)).is_some(),
            "target {i} was displaced instead of the weakest"
        );
    }
}

/// A newcomer no stronger than the weakest resident is refused. Displacement
/// is a strict improvement or it does not happen -- otherwise a saturated
/// atom would churn its whole resident set on every equal-weight fact.
#[test]
fn an_equal_newcomer_is_refused() {
    const CAP: usize = 4;
    let mut n = atom();
    for i in 0..CAP as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
    }
    let before: Vec<u32> = n.terminals.iter().map(|t| t.target.neuron).collect();
    n.reinforce_terminal_bounded(target(99), 0.5, 2, 4.0, CAP);
    let after: Vec<u32> = n.terminals.iter().map(|t| t.target.neuron).collect();
    assert_eq!(
        before, after,
        "an equal-weight newcomer displaced a resident terminal"
    );
}

/// Recall outranks derivation. A `CONSOLIDATION_LOCK` terminal is the
/// 100 %-recall anchor -- `apply_pending_decay` exempts it from decay and
/// pruning, and displacement must exempt it too, or a single fresh 0.5
/// terminal could evict a fact the brain was taught three times.
#[test]
fn a_consolidation_locked_terminal_is_never_displaced() {
    const CAP: usize = 3;
    let mut n = atom();
    for i in 0..CAP as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
    }
    for t in n.terminals.iter_mut() {
        t.weight = 0.01;
        t.consolidation = Neuron::CONSOLIDATION_LOCK;
    }
    let admitted = n.reinforce_terminal_bounded(target(99), 4.0, 2, 4.0, CAP);
    assert!(!admitted);
    assert!(
        n.find_terminal(&target(99)).is_none(),
        "a locked terminal was displaced by a newcomer"
    );
    for i in 0..CAP as u32 {
        assert!(n.find_terminal(&target(i)).is_some(), "locked target {i} lost its slot");
    }
}

/// With one unlocked slot among locked ones, the unlocked one is the only
/// candidate -- the rule picks the weakest DISPLACEABLE terminal, not the
/// weakest terminal.
#[test]
fn only_the_unlocked_slot_is_eligible() {
    const CAP: usize = 3;
    let mut n = atom();
    for i in 0..CAP as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
    }
    for t in n.terminals.iter_mut() {
        t.weight = 0.01;
        t.consolidation = Neuron::CONSOLIDATION_LOCK;
    }
    // Target 1 is unlocked but STRONGER than the locked ones, so a rule that
    // ranked on weight alone would pick a locked slot and refuse.
    n.terminals[1].weight = 0.30;
    n.terminals[1].consolidation = 0;
    n.reinforce_terminal_bounded(target(99), 0.9, 2, 4.0, CAP);
    assert!(
        n.find_terminal(&target(99)).is_some(),
        "the newcomer was refused though an unlocked slot was available"
    );
    assert!(n.find_terminal(&target(1)).is_none(), "the unlocked slot was not the one taken");
    assert!(n.find_terminal(&target(0)).is_some());
    assert!(n.find_terminal(&target(2)).is_some());
}

/// The terminal index must still address every resident terminal after a
/// displacement. The replacement writes in place, so only the evicted target
/// is removed and the newcomer points at the same slot -- a stale entry here
/// would make `find_terminal` return the wrong index and silently reinforce
/// somebody else's dendrite.
#[test]
fn the_terminal_index_survives_displacement_past_the_index_threshold() {
    let cap: usize = Neuron::TERMINAL_INDEX_THRESHOLD + 4;
    let mut n = atom();
    for i in 0..cap as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, cap);
    }
    assert!(
        n.terminal_index_len() > 0,
        "this test is vacuous unless the map index is live at cap {cap}"
    );
    for t in n.terminals.iter_mut() {
        t.weight = 0.40;
    }
    for (slot, newcomer) in [(5usize, 500u32), (0, 501), (cap - 1, 502)] {
        n.terminals[slot].weight = 0.02;
        n.reinforce_terminal_bounded(target(newcomer), 0.5, 2, 4.0, cap);
        let idx = n
            .find_terminal(&target(newcomer))
            .unwrap_or_else(|| panic!("newcomer {newcomer} is not addressable after displacement"));
        assert_eq!(
            n.terminals[idx].target,
            target(newcomer),
            "the index points at the wrong terminal for {newcomer}"
        );
    }
    // Every resident terminal resolves to its own slot.
    for (i, t) in n.terminals.iter().enumerate() {
        let found = n.find_terminal(&t.target).expect("a resident terminal lost its index entry");
        assert_eq!(found, i, "terminal {} resolves to slot {found}, not {i}", t.target.neuron);
    }
    assert_eq!(n.terminals.len(), cap);
}

/// `cap == 0` is unbounded and must behave exactly like `reinforce_terminal`,
/// including returning `true` on a genuine add.
#[test]
fn cap_zero_is_unbounded() {
    let mut n = atom();
    for i in 0..64u32 {
        assert!(
            n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, 0),
            "an unbounded admit of target {i} reported no add"
        );
    }
    assert_eq!(n.terminals.len(), 64);
}

/// THE DECAY HORIZON, pinned as an executable fact rather than a paragraph.
///
/// A terminal is born at weight 0.5. `apply_pending_decay` multiplies by
/// `(1 - decay_rate)^elapsed` on access and deletes the terminal below
/// `prune_floor`, and the pool defaults are `decay_rate = 0.0005`,
/// `prune_floor = 0.01` (`PoolConfig::defaults`). So a terminal that is never
/// re-accessed survives exactly
///
///     ln(0.01 / 0.5) / ln(1 - 0.0005) = 7,822.1 ticks
///
/// and is deleted after that. The scorecard observes once per fact, so
/// `elapsed` is a proxy for CORPUS SIZE: scale 16 is 2,976 ticks and lands at
/// 0.1129, scale 64 is 11,904 ticks and lands at 0.0013. Measured 2026-10-01,
/// that is the only account of the scorecard's terminals FALLING in absolute
/// terms between those two scales -- 68,943 to 26,827 with `evicted_neurons`
/// 0 and `page_outs` 0, so nothing was paged out and the wiring was deleted.
///
/// `CONSOLIDATION_LOCK` is 3 and `consolidation` increments once per DISTINCT
/// TICK, so a fact taught ONCE reaches 1, never locks, and is never exempt.
/// The terminals decay spares are the ones taught three or more times; the
/// ones it deletes are the few-shot ones docs/TEACHING_BENCHMARK.md is about.
///
/// This test does not assert that behaviour is CORRECT -- it is not, and
/// backlog item 7eea7f21 is the fix. It asserts the horizon, so that anyone
/// moving `decay_rate` or `prune_floor` is told in one line what it does to
/// one-shot retention, and so the fix has a pin to flip.
#[test]
fn a_one_shot_terminal_is_deleted_once_the_corpus_exceeds_the_decay_horizon() {
    const DECAY: f32 = 0.0005; // PoolConfig::defaults
    const FLOOR: f32 = 0.01; //  PoolConfig::defaults
    let horizon = (FLOOR / 0.5f32).ln() / (1.0 - DECAY).ln();
    assert!(
        (horizon - 7822.1).abs() < 0.5,
        "the horizon arithmetic moved: {horizon}"
    );

    // Taught once, then a corpus of 2,976 further facts (scale 16).
    let mut survives = atom();
    survives.reinforce_terminal_bounded(target(0), 0.5, 1, 4.0, 0);
    survives.apply_pending_decay(1, DECAY, FLOOR); // bootstrap the decay clock
    let pruned = survives.apply_pending_decay(1 + 2976, DECAY, FLOOR);
    assert_eq!(pruned, 0, "scale 16 is 0.38 of the horizon; nothing should prune");
    assert_eq!(survives.terminals.len(), 1);
    let w = survives.terminals[0].weight;
    assert!(
        // f64 says 0.11286991652; the tolerance is for f32 `powi` drift over
        // 2,976 multiplications, not for uncertainty about the value.
        (w - 0.112870).abs() < 1e-3,
        "weight after 2,976 ticks is {w}, expected 0.112870"
    );

    // The same terminal, a corpus of 11,904 facts (scale 64). 1.52x the
    // horizon, so it is gone.
    let mut deleted = atom();
    deleted.reinforce_terminal_bounded(target(0), 0.5, 1, 4.0, 0);
    deleted.apply_pending_decay(1, DECAY, FLOOR);
    let pruned = deleted.apply_pending_decay(1 + 11904, DECAY, FLOOR);
    assert_eq!(pruned, 1, "scale 64 is 1.52x the horizon; the terminal should be gone");
    assert!(
        deleted.terminals.is_empty(),
        "a fact taught once survived 11,904 ticks: {:?}",
        deleted.terminals
    );

    // Taught three times, so consolidation-locked: exempt at any corpus size.
    // This is the 100 %-recall anchor and the reason recall stays 100.0 at
    // scale 64 while integration falls -- recall's terminals are locked.
    let mut locked = atom();
    locked.reinforce_terminal_bounded(target(0), 0.5, 1, 4.0, 0);
    for tick in 2..=4u64 {
        locked.reinforce_terminal_bounded(target(0), 0.5, tick, 4.0, 0);
    }
    assert!(
        locked.terminals[0].consolidation >= Neuron::CONSOLIDATION_LOCK,
        "consolidation reached {} over 4 distinct ticks, lock is {}",
        locked.terminals[0].consolidation,
        Neuron::CONSOLIDATION_LOCK
    );
    locked.apply_pending_decay(4, DECAY, FLOOR);
    let pruned = locked.apply_pending_decay(4 + 11904, DECAY, FLOOR);
    assert_eq!(pruned, 0, "a consolidation-locked terminal was pruned");
    assert_eq!(locked.terminals.len(), 1);
}

/// An existing target is reinforced in place whether or not the atom is
/// saturated -- a saturated atom must still be able to STRENGTHEN what it
/// already knows, which is how a terminal reaches `CONSOLIDATION_LOCK`.
#[test]
fn a_saturated_atom_can_still_strengthen_an_existing_terminal() {
    const CAP: usize = 4;
    let mut n = atom();
    for i in 0..CAP as u32 {
        n.reinforce_terminal_bounded(target(i), 0.5, 1, 4.0, CAP);
    }
    for tick in 2..6u64 {
        n.reinforce_terminal_bounded(target(1), 0.5, tick, 4.0, CAP);
    }
    let idx = n.find_terminal(&target(1)).expect("target 1 vanished");
    let t: &Terminal = &n.terminals[idx];
    assert_eq!(n.terminals.len(), CAP);
    assert!(t.weight > 0.5, "a saturated atom could not strengthen an existing terminal");
    assert!(
        t.consolidation >= Neuron::CONSOLIDATION_LOCK,
        "consolidation reached {} over 4 distinct ticks, lock is {}",
        t.consolidation,
        Neuron::CONSOLIDATION_LOCK
    );
}
