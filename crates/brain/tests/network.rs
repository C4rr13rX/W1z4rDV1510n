//! Distributed-network tests per [`ARCHITECTURE.md`] §5 + §11 Phase 8.
//!
//! These prove the protocol surface: motif discoveries auto-emit
//! gossip records, ingestion from peers populates the network index
//! without self-loops, equation deltas merge by confidence, peer
//! contributions land in the grounding report weighted by accuracy.

use w1z4rd_brain::{
    Brain, BrainConfig, BytePassthroughEncoding, GossipEquation, GossipMotif,
    PeerContribution, PoolConfig,
};

fn two_pool_brain(brain_id: &str) -> (Brain, u32, u32) {
    let mut brain = Brain::new(BrainConfig::default());
    brain.set_brain_id(brain_id);
    let pool_a = brain.create_pool(
        PoolConfig::defaults("text", 1),
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );
    let pool_b = brain.create_pool(
        PoolConfig::defaults("audio", 2),
        Box::new(BytePassthroughEncoding { prefix: "a" }),
    );
    (brain, pool_a, pool_b)
}

#[test]
fn binding_promotion_auto_emits_a_motif_for_gossip() {
    // Spec §5.1: motif fingerprints are gossiped on discovery.  When
    // the brain promotes a binding concept (3 reps of the same multi-
    // pool firing), it must queue a GossipMotif for the transport.

    let (mut brain, pool_a, pool_b) = two_pool_brain("brain-alpha");
    assert_eq!(brain.pending_motif_count(), 0,
        "no motifs queued before any observation");

    for _ in 0..3 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }

    assert!(brain.pending_motif_count() >= 1,
        "binding promotion must queue at least one motif");

    let drained = brain.drain_motif_gossip();
    assert!(!drained.is_empty(),
        "drain_motif_gossip must return the queued motif");
    let m = &drained[0];
    assert_eq!(m.source_brain, "brain-alpha",
        "motif must carry source brain id");
    assert!(m.fingerprint.iter().any(|(p, _)| *p == pool_a),
        "motif fingerprint must include pool_a member");
    assert!(m.fingerprint.iter().any(|(p, _)| *p == pool_b),
        "motif fingerprint must include pool_b member");
    assert_eq!(brain.pending_motif_count(), 0,
        "drain must leave outbound queue empty");
}

#[test]
fn ingest_motif_gossip_skips_self_loops_and_dedupes() {
    // Self-loop guard: receiving a motif with our own brain_id must
    // NOT pollute the network index.  Spec §5.1: motif gossip is
    // peer→peer, not self→self.  Repeat ingest from the same peer
    // about the same fingerprint updates the existing entry.

    let (mut brain, _pool_a, _pool_b) = two_pool_brain("brain-alpha");

    let self_motif = GossipMotif {
        source_brain:      "brain-alpha".into(),
        fingerprint:       vec![(1, 10), (2, 20)],
        observation_count: 5,
        local_confidence:  0.9,
        observed_at_tick:  100,
    };
    let peer_motif_v1 = GossipMotif {
        source_brain:      "brain-beta".into(),
        fingerprint:       vec![(1, 10), (2, 20)],
        observation_count: 4,
        local_confidence:  0.7,
        observed_at_tick:  90,
    };
    let peer_motif_v2 = GossipMotif {
        source_brain:      "brain-beta".into(),
        fingerprint:       vec![(2, 20), (1, 10)],  // same after sort
        observation_count: 6,
        local_confidence:  0.85,
        observed_at_tick:  110,
    };

    brain.ingest_motif_gossip(vec![self_motif, peer_motif_v1, peer_motif_v2]);

    assert_eq!(brain.received_motif_count(), 1,
        "self-loop dropped + repeat from beta deduped → exactly 1 entry");
    let from_beta = brain.received_motifs_from("brain-beta");
    assert_eq!(from_beta.len(), 1);
    assert_eq!(from_beta[0].observation_count, 6,
        "repeat ingest must update record to the latest values");
}

#[test]
fn equation_gossip_export_carries_local_eem_state() {
    let mut brain = Brain::new(BrainConfig::default());
    brain.set_brain_id("brain-alpha");
    let v = brain.eem_mut().register_variable("a", None);
    let eq = brain.eem_mut().register_equation("ident", "a", vec![v], None);
    for _ in 0..4 { brain.eem_mut().report_validation(eq, true); }
    let conf = brain.eem().confidence(eq).unwrap();

    let exported = brain.export_equations_for_gossip();
    assert_eq!(exported.len(), 1);
    let ge = &exported[0];
    assert_eq!(ge.source_brain, "brain-alpha");
    assert_eq!(ge.name, "ident");
    assert_eq!(ge.expression, "a");
    assert_eq!(ge.variable_names, vec!["a".to_string()]);
    assert!((ge.confidence - conf).abs() < 1e-6);
    assert_eq!(ge.validation_successes, 4);
}

#[test]
fn equation_ingest_merges_by_confidence_higher_wins() {
    // Spec §5.2: per-equation confidence is the conflict-resolution
    // mechanism.  A peer equation with higher confidence replaces
    // the local one; lower confidence is ignored.

    let mut brain = Brain::new(BrainConfig::default());
    brain.set_brain_id("brain-alpha");
    let v = brain.eem_mut().register_variable("x", None);
    let eq = brain.eem_mut().register_equation("law", "x", vec![v], None);
    // local confidence starts at 0.5; bump it to ~0.6 with two
    // successes (default success_gain = 0.05).
    brain.eem_mut().report_validation(eq, true);
    brain.eem_mut().report_validation(eq, true);
    let local_before = brain.eem().confidence(eq).unwrap();
    assert!(local_before > 0.5);

    // Peer with HIGHER confidence + different expression → wins.
    let stronger = GossipEquation {
        source_brain:         "brain-beta".into(),
        name:                 "law".into(),
        expression:           "x * 2".into(),
        variable_names:       vec!["x".into()],
        discipline_name:      None,
        confidence:           0.95,
        validation_successes: 20,
        validation_failures:  1,
    };
    brain.ingest_equation_gossip(vec![stronger]);
    let after_strong = brain.eem().equation_by_name("law").unwrap();
    assert_eq!(brain.eem().equation(after_strong).unwrap().expression, "x * 2",
        "higher-confidence peer must replace the expression");
    assert!((brain.eem().confidence(after_strong).unwrap() - 0.95).abs() < 1e-6);

    // Peer with LOWER confidence → ignored.
    let weaker = GossipEquation {
        source_brain:         "brain-gamma".into(),
        name:                 "law".into(),
        expression:           "x + 99".into(),
        variable_names:       vec!["x".into()],
        discipline_name:      None,
        confidence:           0.3,
        validation_successes: 0,
        validation_failures:  10,
    };
    brain.ingest_equation_gossip(vec![weaker]);
    assert_eq!(brain.eem().equation(after_strong).unwrap().expression, "x * 2",
        "lower-confidence peer must NOT replace expression");
}

#[test]
fn unknown_peer_equation_is_added_at_peer_confidence() {
    // A peer-only equation must enter the local EEM at the peer's
    // confidence.  This is how knowledge propagates.

    let mut brain = Brain::new(BrainConfig::default());
    brain.set_brain_id("brain-alpha");
    assert_eq!(brain.eem().equation_count(), 0);

    let new_law = GossipEquation {
        source_brain:         "brain-beta".into(),
        name:                 "new_law".into(),
        expression:           "a + b".into(),
        variable_names:       vec!["a".into(), "b".into()],
        discipline_name:      Some("physics".into()),
        confidence:           0.8,
        validation_successes: 10,
        validation_failures:  2,
    };
    brain.ingest_equation_gossip(vec![new_law]);
    let eq_id = brain.eem().equation_by_name("new_law")
        .expect("ingested equation must be registered");
    assert!((brain.eem().confidence(eq_id).unwrap() - 0.8).abs() < 1e-6);
    assert_eq!(brain.eem().equation(eq_id).unwrap().validation_successes, 10);
}

#[test]
fn peer_accuracy_default_is_neutral_then_moves_with_outcomes() {
    let mut brain = Brain::new(BrainConfig::default());
    assert!(brain.peer_accuracy("unknown").is_none());

    brain.report_peer_outcome("peer-1", true);
    brain.report_peer_outcome("peer-1", true);
    brain.report_peer_outcome("peer-1", false);
    let acc = brain.peer_accuracy("peer-1").unwrap();
    assert_eq!(acc.successful_contributions, 2);
    assert_eq!(acc.failed_contributions, 1);
    assert!((acc.rate() - 2.0 / 3.0).abs() < 1e-6);
}

#[test]
fn integrate_with_peers_lists_weighted_contributions_in_grounding() {
    // Spec §5.3: peer contributions enter integration weighted by
    // each peer's accuracy track record.  The grounding report must
    // expose the per-peer weighted confidence and the new integrated
    // score.

    let (mut brain, pool_a, pool_b) = two_pool_brain("brain-alpha");
    for _ in 0..6 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");

    // Peer-1 has a great track record; Peer-2 is unknown (neutral).
    for _ in 0..8 { brain.report_peer_outcome("peer-1", true); }

    let peers = vec![
        PeerContribution {
            brain_id:              "peer-1".into(),
            fabric_confidence:     0.9,
            eem_confidence:        None,
            annealer_confidence:   None,
            strongest_match_label: None,
        },
        PeerContribution {
            brain_id:              "peer-2".into(),
            fabric_confidence:     0.4,
            eem_confidence:        None,
            annealer_confidence:   None,
            strongest_match_label: None,
        },
    ];

    let answer = brain.integrate_with_peers(pool_a, pool_b, &peers);
    assert_eq!(answer.grounding.peer_contributions.len(), 2,
        "both peers must appear in the grounding report");

    // Peer-1's weight should be the higher of the two — accuracy ≈ 1.0
    // and fabric_confidence 0.9 → ~0.9.  Peer-2 unknown → ~0.5*0.4 = 0.2.
    let p1_weight = answer.grounding.peer_contributions.iter()
        .find(|(b, _)| b == "peer-1").map(|(_, w)| *w).unwrap();
    let p2_weight = answer.grounding.peer_contributions.iter()
        .find(|(b, _)| b == "peer-2").map(|(_, w)| *w).unwrap();
    assert!(p1_weight > p2_weight,
        "accurate peer must outweigh unknown peer: p1={} p2={}",
        p1_weight, p2_weight);
}

#[test]
fn integrate_with_peers_no_peers_is_identical_to_integrate() {
    // Honest behavior under no peers: no degradation of the answer.

    let (mut brain, pool_a, pool_b) = two_pool_brain("brain-alpha");
    for _ in 0..6 {
        brain.observe(pool_a, b"X");
        brain.observe(pool_b, b"Y");
        brain.advance_tick();
    }
    brain.observe(pool_a, b"X");

    let base = brain.integrate(pool_a, pool_b);
    let with_no_peers = brain.integrate_with_peers(pool_a, pool_b, &[]);
    assert_eq!(base.grounding.integrated_confidence,
               with_no_peers.grounding.integrated_confidence);
    assert!(with_no_peers.grounding.peer_contributions.is_empty());
}
