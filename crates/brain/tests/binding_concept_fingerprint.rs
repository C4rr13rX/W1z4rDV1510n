use w1z4rd_brain::{Brain, BrainConfig, BytePassthroughEncoding, PoolConfig};

#[test]
fn promoted_binding_retains_firing_prompt_concepts() {
    let mut config = BrainConfig::default();
    config.binding_emergence_threshold = 3;
    config.tentative_emergence_threshold = 3;
    let mut brain = Brain::new(config);

    let mut prompt_config = PoolConfig::defaults("prompt", 1);
    prompt_config.recent_atoms_window = 64;
    prompt_config.max_concept_member_count = 4;
    prompt_config.concept_emergence_threshold = 2;
    let mut response_config = PoolConfig::defaults("response", 2);
    response_config.max_concept_member_count = 1;
    let prompt = brain.create_pool(
        prompt_config,
        Box::new(BytePassthroughEncoding { prefix: "p" }),
    );
    let response = brain.create_pool(
        response_config,
        Box::new(BytePassthroughEncoding { prefix: "r" }),
    );

    for _ in 0..5 {
        brain.observe(prompt, b"abab");
        brain.observe(response, b"yes");
        brain.advance_tick();
    }

    let prompt_pool = brain.fabric().pool(prompt).unwrap();
    let prompt_pool = prompt_pool.read();
    let binding_pool = brain.fabric().pool(brain.binding_pool_id()).unwrap();
    let binding_pool = binding_pool.read();
    assert!(binding_pool.iter_neurons().filter(|n| !n.is_atom()).any(|binding| {
        binding.members.iter().any(|member| {
            member.pool == prompt
                && prompt_pool.get(member.neuron).is_some_and(|n| !n.is_atom())
        })
    }));
}
