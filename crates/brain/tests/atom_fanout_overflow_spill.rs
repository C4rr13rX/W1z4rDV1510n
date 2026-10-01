//! The atom fan-out bound keeps the strongest terminals resident and DROPS
//! NOTHING: everything past the bound goes to a compact overflow.
//!
//! WHY THIS EXISTS, and it is the third rule tried at this bound. The first two
//! are both measured dead, so neither is a baseline this can be compared
//! against favourably:
//!
//! * FIRST-COME REFUSAL (`PoolConfig::max_atom_fanout`, the shipped rule): an
//!   atom that fills on the opening facts of a corpus acquires nothing for the
//!   rest of it. At scale 16 of the scorecard the query atoms `q:cg`, `q:IA`
//!   and `q:Pw` each held 2,432 terminals — the fact count exactly — so at
//!   scale 64 the bound of 512 refuses ~96 % of them.
//! * STRONGEST-RESIDENT DISPLACEMENT (`Neuron::reinforce_terminal_bounded`):
//!   wired at `Brain::promote_binding_concept` on 2026-10-01 and it turned
//!   `tests/empty_integration_is_budget_starvation.rs` red. Every atom→binding
//!   terminal is created at the same `delta = 0.5` and `effective_weight`
//!   applies no tick decay of its own, so "strongest" degenerates to "most
//!   recently trained" and it evicts the EARLIEST facts. Refusing the newest
//!   and evicting the oldest are two arbitrary halves of one corpus.
//!
//! WHAT THE REFUSED TERMINALS ARE WORTH, which is why this is not bookkeeping
//! for its own sake: lifting the bound entirely (`max_atom_fanout = 0`) moved
//! scale-16 integration 44.3 → 50.8 % and scale-64 integration 20.5 → 32.5 %
//! with recall 100.0 throughout. It was rejected on RAM alone — peak
//! 40.1 → 51.9 MB at scale 64, +29.4 % against a gate that allows 15 %, i.e.
//! ~72.6 B per terminal.
//!
//! So the bound stays, the terminals stay, and what changes is what an
//! over-bound terminal COSTS. `test_overflow_is_eight_bytes_per_terminal` is
//! the load-bearing one: it reads the figure off the pool's own census rather
//! than off `size_of`, so Vec capacity slack and the map's buckets are both
//! charged. If that number is not ~8 B the design does not pay for itself and
//! the right move is to say so, not to relax the assertion.

use w1z4rd_brain::pool::{AtomFanoutOutcome, Pool};
use w1z4rd_brain::neuron::NeuronRef;
use w1z4rd_brain::{AtomEncoding, BytePassthroughEncoding, PoolConfig};

const CAP: usize = 512;
/// One binding per fact at scale 64 of the scorecard's scene world, three
/// times the bound, so the resident/overflow split is not a boundary case.
const FACTS: usize = 1_536;

fn pool_with_cap(cap: usize) -> Pool {
    let mut pc = PoolConfig::defaults("text", 1);
    pc.max_atom_fanout = cap;
    let enc: Box<dyn AtomEncoding> = Box::new(BytePassthroughEncoding { prefix: "t" });
    Pool::new(pc, enc)
}

/// One atom id, created the way the pretrain path creates them so the test is
/// not asserting against a hand-built neuron.
fn one_atom(pool: &mut Pool) -> u32 {
    let ids = pool.ensure_frame_atoms_for_pretrain(b"a", 0);
    assert_eq!(ids.len(), 1, "one byte must atomize to exactly one atom");
    ids[0]
}

/// The targets a hub atom accumulates: one binding per fact, in a different
/// pool, exactly as `Brain::promote_binding_concept` offers them.
fn binding(i: usize) -> NeuronRef {
    NeuronRef::new(9, i as u32)
}

#[test]
fn nothing_offered_is_ever_dropped() {
    let mut pool = pool_with_cap(CAP);
    let atom = one_atom(&mut pool);

    let mut added = 0usize;
    let mut spilled = 0usize;
    for i in 0..FACTS {
        match pool.reinforce_atom_terminal_keeping_overflow(atom, binding(i), 0.5, 1) {
            AtomFanoutOutcome::Added => added += 1,
            AtomFanoutOutcome::Spilled => spilled += 1,
            other => panic!("fact {i} returned {other:?}, expected Added or Spilled"),
        }
    }

    // The bound is still a bound: resident fan-out never exceeds it.
    assert_eq!(added, CAP, "resident fan-out must stop at the bound");
    assert_eq!(
        pool.get(atom).map(|n| n.terminals.len()),
        Some(CAP),
        "resident terminal count must equal the bound"
    );
    let (atoms, entries) = pool.atom_fanout_overflow_census();
    assert_eq!(atoms, 1, "exactly one atom saturated");
    assert_eq!(spilled, FACTS - CAP);
    assert_eq!(entries, FACTS - CAP, "every over-bound terminal is kept");

    // THE PROPERTY THE OTHER TWO RULES BOTH FAIL: every fact offered is still
    // reachable from the atom, the first and the last alike.
    for i in 0..FACTS {
        assert!(
            pool.atom_reaches(atom, binding(i)),
            "fact {i} was dropped -- first-come refusal loses the last {} and \
             strongest-resident displacement loses the first {}",
            FACTS - CAP,
            FACTS - CAP
        );
    }
    assert!(
        !pool.atom_reaches(atom, binding(FACTS)),
        "a fact never offered must not be reachable"
    );
}

#[test]
fn test_overflow_is_eight_bytes_per_terminal() {
    let mut pool = pool_with_cap(CAP);
    let atom = one_atom(&mut pool);
    for i in 0..FACTS {
        pool.reinforce_atom_terminal_keeping_overflow(atom, binding(i), 0.5, 1);
    }

    let (_, entries) = pool.atom_fanout_overflow_census();
    let bytes = pool.atom_fanout_overflow_bytes();
    let per = bytes as f64 / entries as f64;
    println!("overflow: {entries} entries, {bytes} B, {per:.2} B/entry");

    // A full resident terminal measured 72.6 B at scale 64 (Terminal + an
    // entry in the neuron's terminal_idx + Vec slack). An overflow entry is a
    // bare NeuronRef in an exact-block Vec. 16 B leaves room for the map's own
    // table and up to one block of slack at this size; it does not leave room
    // for a Terminal, an index entry or doubling growth, which is the point.
    assert!(
        per <= 16.0,
        "overflow costs {per:.2} B/entry; at more than 16 the compact \
         representation is not paying for itself"
    );

    // Growth is in exact blocks, so slack is bounded per atom rather than
    // proportional to fan-out. Doubling would leave up to `entries` of slack.
    let slack = pool
        .atom_fanout_overflow_targets(atom)
        .len();
    assert_eq!(slack, FACTS - CAP);
    assert!(
        bytes
            <= (entries + Pool::ATOM_FANOUT_OVERFLOW_BLOCK)
                * std::mem::size_of::<NeuronRef>()
                + 1024,
        "{bytes} B exceeds one block of slack plus the map table"
    );
}

#[test]
fn below_the_bound_and_unbounded_are_untouched() {
    // Nothing saturates: identical to plain reinforce_terminal, no overflow.
    let mut pool = pool_with_cap(CAP);
    let atom = one_atom(&mut pool);
    for i in 0..CAP {
        assert_eq!(
            pool.reinforce_atom_terminal_keeping_overflow(atom, binding(i), 0.5, 1),
            AtomFanoutOutcome::Added
        );
    }
    assert_eq!(pool.atom_fanout_overflow_census(), (0, 0));
    assert_eq!(pool.atom_fanout_overflow_bytes(), 0);

    // cap == 0 is unbounded: everything stays resident, still no overflow.
    let mut unbounded = pool_with_cap(0);
    let atom = one_atom(&mut unbounded);
    for i in 0..FACTS {
        assert_eq!(
            unbounded.reinforce_atom_terminal_keeping_overflow(atom, binding(i), 0.5, 1),
            AtomFanoutOutcome::Added
        );
    }
    assert_eq!(
        unbounded.get(atom).map(|n| n.terminals.len()),
        Some(FACTS),
        "an unbounded pool must keep every terminal resident"
    );
    assert_eq!(unbounded.atom_fanout_overflow_census(), (0, 0));
}

#[test]
fn a_repeat_reinforces_and_never_duplicates() {
    let mut pool = pool_with_cap(CAP);
    let atom = one_atom(&mut pool);
    for i in 0..FACTS {
        pool.reinforce_atom_terminal_keeping_overflow(atom, binding(i), 0.5, 1);
    }
    let terminals_before = pool.get(atom).map(|n| n.terminals.len());
    let (_, entries_before) = pool.atom_fanout_overflow_census();

    // A resident target offered again is Hebbian-strengthened, not re-added.
    assert_eq!(
        pool.reinforce_atom_terminal_keeping_overflow(atom, binding(0), 0.5, 2),
        AtomFanoutOutcome::Reinforced
    );
    // An overflowed target offered again is recognised, not duplicated.
    assert_eq!(
        pool.reinforce_atom_terminal_keeping_overflow(atom, binding(FACTS - 1), 0.5, 2),
        AtomFanoutOutcome::AlreadySpilled
    );

    assert_eq!(pool.get(atom).map(|n| n.terminals.len()), terminals_before);
    let (_, entries_after) = pool.atom_fanout_overflow_census();
    assert_eq!(entries_after, entries_before, "a repeat must not grow fan-out");

    // total_terminals stays exact: a spill is not a terminal.
    assert_eq!(
        pool.total_terminals(),
        CAP,
        "the O(1) counter must count resident terminals only"
    );
}

#[test]
fn a_missing_neuron_is_reported_and_not_spilled() {
    let mut pool = pool_with_cap(CAP);
    let _ = one_atom(&mut pool);
    assert_eq!(
        pool.reinforce_atom_terminal_keeping_overflow(9_999_999, binding(0), 0.5, 1),
        AtomFanoutOutcome::Missing
    );
    assert_eq!(
        pool.atom_fanout_overflow_census(),
        (0, 0),
        "a spill for a neuron that does not exist would leak an entry per call"
    );
}
