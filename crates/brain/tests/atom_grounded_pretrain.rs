use w1z4rd_brain::{BytePassthroughEncoding, Pool, PoolConfig};

#[test]
fn pretrain_promotes_recurring_raw_sequence_with_atom_members() {
    let mut config = PoolConfig::defaults("text", 1);
    config.max_concept_member_count = 8;
    let mut pool = Pool::new(config, Box::new(BytePassthroughEncoding { prefix: "t" }));
    let frames = vec![b"cat cat".to_vec(), b"catfish cat".to_vec()];

    let report = pool.pretrain_recurring_patterns(&frames, 1, 3, 128);

    assert!(report.concepts_promoted > 0);
    let cat = pool
        .iter_neurons()
        .find(|neuron| !neuron.is_atom() && pool.decode_concept_members(&neuron.members) == b"cat")
        .expect("the recurring c-a-t sequence should be promoted");
    assert!(cat.members.iter().all(|member| {
        pool.get(member.neuron)
            .is_some_and(|neuron| neuron.is_atom())
    }));
}
