use w1z4rd_brain::{BytePassthroughEncoding, Pool, PoolConfig};

#[test]
fn prediction_replays_learned_concept_collapse_without_learning() {
    let mut config = PoolConfig::defaults("text", 1);
    config.recent_atoms_window = 32;
    config.max_concept_member_count = 2;
    config.concept_emergence_threshold = 2;
    let mut pool = Pool::new(
        config,
        Box::new(BytePassthroughEncoding { prefix: "t" }),
    );

    pool.observe_frame(b"ab", 1, None);
    pool.observe_frame(b"ab", 2, None);
    assert!(pool.concept_count() > 0);
    let neurons_before = pool.neuron_count();

    let atoms = pool.activate_known_frame_for_prediction(b"ab");
    assert_eq!(atoms.len(), 2);
    assert!(pool.currently_firing().any(|id| {
        pool.get(id).is_some_and(|neuron| !neuron.is_atom())
    }));
    assert_eq!(pool.neuron_count(), neurons_before);
}
