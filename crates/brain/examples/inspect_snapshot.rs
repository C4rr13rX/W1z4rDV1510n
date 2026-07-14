use std::collections::HashMap;
use std::path::PathBuf;
use w1z4rd_brain::{AtomEncoding, Brain, BytePassthroughEncoding, PoolId};

fn main() -> std::io::Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "usage: inspect_snapshot <brain.bin>",
            )
        })?;
    let restore = std::env::args().any(|arg| arg == "--restore");
    let snapshot = w1z4rd_brain::persistence::load_snapshot(&path)?;
    println!(
        "tick={} pools={}",
        snapshot.fabric.tick,
        snapshot.fabric.pools.len()
    );
    let mut all_neurons = 0usize;
    let mut all_concepts = 0usize;
    let mut all_terminals = 0usize;
    for pool_id in &snapshot.fabric.pool_order {
        let Some(pool) = snapshot.fabric.pools.get(pool_id) else {
            continue;
        };
        let neurons = pool.neurons.len();
        let concepts = pool
            .neurons
            .iter()
            .filter(|neuron| !neuron.is_atom())
            .count();
        let terminals: usize = pool
            .neurons
            .iter()
            .map(|neuron| neuron.terminals.len())
            .sum();
        let max_fanout = pool
            .neurons
            .iter()
            .map(|neuron| neuron.terminals.len())
            .max()
            .unwrap_or(0);
        println!(
            "pool={} neurons={} concepts={} terminals={} max_fanout={}",
            pool_id, neurons, concepts, terminals, max_fanout,
        );
        all_neurons += neurons;
        all_concepts += concepts;
        all_terminals += terminals;
    }
    println!(
        "total_neurons={} total_concepts={} total_terminals={}",
        all_neurons, all_concepts, all_terminals,
    );
    if restore {
        let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
        for pool_id in &snapshot.fabric.pool_order {
            encodings.insert(
                *pool_id,
                Box::new(BytePassthroughEncoding { prefix: "inspect" }),
            );
        }
        let started = std::time::Instant::now();
        let (brain, missing) = Brain::from_snapshot(snapshot, encodings);
        println!(
            "restored_in_seconds={:.3} missing_pools={} stats={:?}",
            started.elapsed().as_secs_f64(),
            missing.len(),
            brain.stats(),
        );
    }
    Ok(())
}
