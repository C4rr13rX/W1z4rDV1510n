use std::collections::HashMap;
use std::path::Path;
use w1z4rd_brain::{AtomEncoding, Brain, BytePassthroughEncoding, PoolId};

#[test]
fn live_87gb_wbrain_if_present_restores_cleanly() {
    let p = Path::new(r"D:\Projects\W1z4rDV1510n\brain-data\brain.wbrain");
    if !p.exists() {
        return;
    }
    let mut encodings: HashMap<PoolId, Box<dyn AtomEncoding>> = HashMap::new();
    for pid in 0u32..32u32 {
        let prefix: &'static str = Box::leak(format!("p{pid}").into_boxed_str());
        encodings.insert(pid, Box::new(BytePassthroughEncoding { prefix }));
    }
    let (_brain, missing) = Brain::restore_wbrain(p, encodings)
        .expect("87.55 GB live brain.wbrain must restore cleanly via LegacyWbrainBrainMetadata + LegacyWbrainPoolMetadata");
    println!("Restored 87.55 GB live brain.wbrain cleanly! missing={missing:?}");
}
