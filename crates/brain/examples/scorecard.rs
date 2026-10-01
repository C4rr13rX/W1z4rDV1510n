//! Scorecard: does the brain stay small while it stays right?
//!
//! One scene world -- rooms full of objects, each object with properties --
//! trained at a given scale, then probed three ways:
//!
//! - recall:      trained facts ("r03 lamp color?" -> "red"). Must stay 100%.
//! - integration: facts never trained but true in the world, derivable by
//!                chaining two trained ones ("r03 lamp on" -> "desk", "r03
//!                desk material" -> "oak", so "r03 lamp on material?" ->
//!                "oak"). Must never drop.
//! - footprint:   neurons, terminals, the largest fan-out on any one neuron
//!                (the hub that cannot leave RAM), and an estimate of
//!                resident bytes. Must stay flat as the world grows.
//!
//! Prints ONE JSON line. Peak process memory is measured from outside by
//! tools/capped.py, which also enforces the cap -- see tools/scorecard.py.
//!
//! Run: `cargo run --release --example scorecard -p w1z4rd-brain -- --scale 4`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use w1z4rd_brain::brain::hash_table_bytes;
use w1z4rd_brain::{AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, Neuron, NeuronRef,
    PoolConfig, Terminal};

/// Live heap bytes and their high-water mark, counted by the allocator itself.
///
/// `est_resident_mb` is a census of structures the brain knows it owns; the
/// peak is measured from OUTSIDE by tools/capped.py. So the gap between them
/// had no owner and could not be attributed: it was either heap the census
/// fails to count, transient allocation the census cannot see because it is
/// already freed, or process overhead the brain never asked for. Those need
/// different fixes, and nothing distinguished them. Every byte the program
/// requests passes through `GlobalAlloc`, so counting here splits the three
/// without a platform memory API:
///
///   peak_rss - HEAP_PEAK  = process overhead (runtime, binary, allocator arenas)
///   HEAP_PEAK - HEAP_LIVE = transient churn, freed before the mark was taken
///   HEAP_LIVE - accounted = live structures the census does not count
static HEAP_LIVE: AtomicUsize = AtomicUsize::new(0);
static HEAP_PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

/// Adds `delta` to the live total and raises the high-water mark.
fn grew(delta: usize) {
    let live = HEAP_LIVE.fetch_add(delta, Ordering::Relaxed) + delta;
    HEAP_PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(l) };
        if !p.is_null() {
            grew(l.size());
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        HEAP_LIVE.fetch_sub(l.size(), Ordering::Relaxed);
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        // A failed realloc leaves the old block live, so only a non-null
        // result may move the counter.
        let q = unsafe { System.realloc(p, l, new) };
        if !q.is_null() {
            if new >= l.size() {
                grew(new - l.size());
            } else {
                HEAP_LIVE.fetch_sub(l.size() - new, Ordering::Relaxed);
            }
        }
        q
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// (live, peak) heap megabytes at the moment of the call.
fn heap_mb() -> (f64, f64) {
    (
        HEAP_LIVE.load(Ordering::Relaxed) as f64 / 1_048_576.0,
        HEAP_PEAK.load(Ordering::Relaxed) as f64 / 1_048_576.0,
    )
}

const QUERY_POOL: u32 = 1;
const ANSWER_POOL: u32 = 2;
const ROOMS_PER_SCALE: usize = 8;
const EPOCHS: usize = 2;

const OBJECTS: &[&str] = &["bed", "chair", "mirror", "desk", "lamp", "door", "window", "paper"];
const COLORS: &[&str] = &["red", "blue", "green", "white", "black", "grey"];
const MATERIALS: &[&str] = &["oak", "steel", "glass", "cloth", "pine", "brass"];
/// Objects that rest on another object, and what they rest on.
const RESTS_ON: &[(&str, &str)] = &[("lamp", "desk"), ("paper", "desk"), ("mirror", "door")];
/// One room in this many has the `beside?` phrasing trained, so the paraphrase
/// family is a held-out PHRASING rather than a word the brain has never seen.
const BESIDE_TRAINED_EVERY: usize = 4;
/// Objects per room carrying the `next {obj} color?` family. Held down because
/// every probe is an inference, and inference is the slow phase.
const NEXT_COLOR_OBJECTS: usize = 2;

/// A question and the answer the world says is true.
struct Probe {
    query: String,
    answer: String,
}

/// One held-out question shape, named and counted on its own.
///
/// The score used to be a single number over a single shape ("{room} {obj} on
/// material?", three per room), so a mechanism that chains rests-on and nothing
/// else scored 100% and M3's "integration >= 90%" could be met without the
/// brain deriving anything general. A family carries its own count and
/// percentage so one family's win cannot read as integration.
struct Family {
    name: &'static str,
    /// Trained edges the answer must be composed from. Reported so a reader can
    /// see at a glance that the set is not all 2-hop.
    hops: usize,
    probes: Vec<Probe>,
}

/// The scene world at one scale: trained facts and untrained-but-true probes.
struct SceneWorld {
    facts: Vec<Probe>,
    families: Vec<Family>,
}

impl SceneWorld {
    fn new(scale: usize) -> Self {
        let rooms = ROOMS_PER_SCALE * scale;
        let name = |r: usize| format!("r{r:03}");
        // The multiplier on `salt` must be COPRIME with the list length or the
        // world is far smaller than it looks. It was 3 against 6-long lists, and
        // 3*salt mod 6 only ever takes {0, 3}: every room had exactly TWO
        // colours and TWO materials across all eight objects. Worse for what
        // this file measures, 3*(7+1) and 3*(3+1) are both 0 mod 6, so for the
        // (paper, desk) pair the trained "{room} paper material?" answer was
        // IDENTICAL to the integration answer for "{room} paper on material?" --
        // one of the three probes per room was solvable by the prefix shortcut
        // the world exists to defeat. 17 is coprime with both 6 and 8.
        let pick = |list: &[&'static str], r: usize, salt: usize| {
            list[(r * 31 + salt * 17) % list.len()].to_string()
        };
        let color = |r: usize, i: usize| pick(COLORS, r, i);
        let material = |r: usize, i: usize| pick(MATERIALS, r, i + 1);
        let idx = |o: &str| OBJECTS.iter().position(|x| *x == o).unwrap();
        // The distractor: what `{obj}` sits NEXT TO, which is never what it
        // rests ON, and is chosen so its material differs from the base's. A
        // chain that follows `near?` instead of `on?` therefore returns a
        // trained string that is the wrong answer, rather than returning
        // nothing -- the only way a wrong chain can be told from no chain.
        let decoy = |r: usize, obj: &str, base: &str| -> String {
            let want = material(r, idx(base));
            OBJECTS
                .iter()
                .find(|d| **d != obj && **d != base && material(r, idx(d)) != want)
                .expect("6 materials over 8 objects always leave a differing decoy")
                .to_string()
        };

        let mut facts = Vec::new();
        for r in 0..rooms {
            let room = name(r);
            let next = name((r + 1) % rooms);
            for (i, obj) in OBJECTS.iter().enumerate() {
                facts.push(Probe { query: format!("{room} {obj} color?"), answer: color(r, i) });
                facts.push(Probe { query: format!("{room} {obj} material?"), answer: material(r, i) });
            }
            for (obj, base) in RESTS_ON {
                facts.push(Probe { query: format!("{room} {obj} on?"), answer: (*base).into() });
                facts.push(Probe { query: format!("{room} {obj} near?"), answer: decoy(r, obj, base) });
            }
            // The second relation: room adjacency. Nothing about it is
            // rests-on, so a mechanism that only chains rests-on scores zero on
            // three of the four families below.
            facts.push(Probe { query: format!("{room} next?"), answer: next.clone() });
            if r % BESIDE_TRAINED_EVERY == 0 {
                facts.push(Probe { query: format!("{room} beside?"), answer: next });
            }
        }

        let mut on_material = Vec::new();
        let mut next_color = Vec::new();
        let mut next_on_material = Vec::new();
        let mut beside_next = Vec::new();
        for r in 0..rooms {
            let room = name(r);
            let nr = (r + 1) % rooms;
            for (obj, base) in RESTS_ON {
                on_material.push(Probe {
                    query: format!("{room} {obj} on material?"),
                    answer: material(r, idx(base)),
                });
            }
            for k in 0..NEXT_COLOR_OBJECTS {
                let i = (r + 3 * k) % OBJECTS.len();
                let obj = OBJECTS[i];
                next_color.push(Probe {
                    query: format!("{room} next {obj} color?"),
                    answer: color(nr, i),
                });
            }
            let (obj, base) = RESTS_ON[r % RESTS_ON.len()];
            next_on_material.push(Probe {
                query: format!("{room} next {obj} on material?"),
                answer: material(nr, idx(base)),
            });
            if r % BESIDE_TRAINED_EVERY != 0 {
                beside_next.push(Probe { query: format!("{room} beside?"), answer: name(nr) });
            }
        }
        let families = vec![
            Family { name: "on_material", hops: 2, probes: on_material },
            Family { name: "next_color", hops: 2, probes: next_color },
            Family { name: "next_on_material", hops: 3, probes: next_on_material },
            Family { name: "beside_next", hops: 1, probes: beside_next },
        ];
        Self { facts, families }
    }

    fn integration_count(&self) -> usize {
        self.families.iter().map(|f| f.probes.len()).sum()
    }
}

/// The brain under test, built the way examples/lab.rs builds it.
struct Subject {
    brain: Brain,
}

impl Subject {
    fn new() -> Self {
        let mut cfg = BrainConfig::default();
        cfg.binding_emergence_threshold = 3;
        cfg.moment_history_window = 256;
        let mut brain = Brain::new(cfg);
        for (name, id, prefix) in [("query", QUERY_POOL, "q"), ("answer", ANSWER_POOL, "a")] {
            let mut pc = PoolConfig::defaults(name, id);
            pc.recent_atoms_window = 2048;
            pc.concept_emergence_threshold = 2;
            pc.max_concept_member_count = 64;
            pc.decay_rate = 0.0001;
            pc.prune_floor = 0.005;
            brain.create_pool(pc, Box::new(BytePassthroughEncoding { prefix }) as Box<dyn AtomEncoding>);
        }
        Self { brain }
    }

    /// Trains the way the node's /brain/pretrain route does: one binding
    /// episode per fact, shuffled each epoch.
    fn train(&mut self, facts: &[Probe]) {
        for epoch in 0..EPOCHS {
            for i in shuffled(facts.len(), epoch) {
                self.brain.pretrain_binding_episode(&[
                    (QUERY_POOL, facts[i].query.as_bytes().to_vec()),
                    (ANSWER_POOL, facts[i].answer.as_bytes().to_vec()),
                ]);
            }
        }
    }

    /// Answers the way the node's authoritative QA path does
    /// (crates/node/src/brain_api.rs): the trained binding first, integrate()
    /// as the fallback, then release whatever the query paged in.
    fn recall(&mut self, query: &str) -> Vec<u8> {
        self.brain.observe_read_only(QUERY_POOL, query.as_bytes());
        let legacy = self.brain.integrate(QUERY_POOL, ANSWER_POOL);
        let answer = self.brain.decode_best_trained_binding(QUERY_POOL, ANSWER_POOL).or(legacy.answer);
        let _ = self.brain.finish_read_only_inference();
        answer.unwrap_or_default()
    }

    /// The inference path. `recall` releases what the query paged in and this
    /// did not, so every integration probe leaked its working set -- and
    /// integration is the phase whose probe count this change multiplied.
    fn infer(&mut self, query: &str) -> Vec<u8> {
        self.brain.observe_read_only(QUERY_POOL, query.as_bytes());
        let answer = self
            .brain
            .integrate_autonomous(QUERY_POOL, ANSWER_POOL, 100.0, 3, 200)
            .answer;
        let _ = self.brain.finish_read_only_inference();
        answer.unwrap_or_default()
    }

    /// Where the bytes actually are, at the moment of the call: the
    /// Brain-level maps, the per-pool side structures that survive eviction,
    /// and the neuron bodies. `est_resident_mb` only ever counted the last of
    /// those, which is why it reads 24 MB inside a 2.47 GB process.
    fn census(&self) -> serde_json::Value {
        let mut side = std::collections::BTreeMap::<&str, usize>::new();
        for pid in self.brain.fabric().pool_ids() {
            let Some(pool) = self.brain.fabric().pool(pid) else { continue };
            let s = pool.read().side_structure_bytes();
            for (k, v) in [
                ("concept_sequence_index", s.concept_sequence_index),
                ("concept_multiset_index", s.concept_multiset_index),
                ("label_index", s.label_index),
                ("sequence_ledger", s.sequence_ledger),
                ("evicted_set", s.evicted_set),
                ("cold_offsets", s.cold_offsets),
                ("neuron_slot_table", s.neuron_slot_table),
                ("transient_firing", s.transient_firing),
            ] {
                *side.entry(k).or_default() += v;
            }
        }
        // Fan-out is the last scorecard number still linear in the corpus
        // (152 at scale 1, 9,728 at scale 64 -- exactly one per fact). Which
        // neuron carries it decides whether bounding it is safe, so name it
        // rather than reporting a bare maximum.
        let mut top: Vec<serde_json::Value> = Vec::new();
        for pid in self.brain.fabric().pool_ids() {
            let Some(pool) = self.brain.fabric().pool(pid) else { continue };
            let pool = pool.read();
            let mut ranked: Vec<(usize, u32, String, bool, usize)> = pool
                .iter_neurons()
                .map(|n| (n.terminals.len(), n.id, n.label.clone(), n.is_atom(), n.members.len()))
                .collect();
            ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for (fanout, id, label, is_atom, members) in ranked.into_iter().take(3) {
                top.push(serde_json::json!({
                    "pool": pid, "id": id, "fanout": fanout,
                    "is_atom": is_atom, "members": members,
                    "label": label.chars().take(48).collect::<String>(),
                }));
            }
        }
        // Neuron bodies are the largest per-fact pot left (13.9 MB over 9,776
        // neurons at scale 64 -- ~1,490 bytes each for members that are ~22
        // NeuronRef, so 176 B). Name the components rather than reporting the
        // total: `footprint()` counts Vec CAPACITY but never the `terminal_idx`
        // AHashMap each neuron carries, and a hash map allocates for its
        // capacity, not its length.
        let mut body = std::collections::BTreeMap::<&str, usize>::new();
        let mut fanout_hist = std::collections::BTreeMap::<&str, usize>::new();
        let mut fanout_terms = std::collections::BTreeMap::<&str, usize>::new();
        let mut neurons = 0usize;
        for pid in self.brain.fabric().pool_ids() {
            let Some(pool) = self.brain.fabric().pool(pid) else { continue };
            let pool = pool.read();
            for n in pool.iter_neurons() {
                neurons += 1;
                *body.entry("struct").or_default() += std::mem::size_of::<Neuron>();
                *body.entry("label").or_default() += n.label.capacity();
                *body.entry("members").or_default() +=
                    n.members.capacity() * std::mem::size_of::<NeuronRef>();
                *body.entry("terminals").or_default() +=
                    n.terminals.capacity() * std::mem::size_of::<Terminal>();
                // One spelling of the table's real size, shared with the
                // Brain-level census: capacity is NOT the bucket count.
                *body.entry("terminal_idx").or_default() +=
                    hash_table_bytes::<NeuronRef, usize>(n.terminal_index_capacity());
                // Where a per-neuron index would still be worth its bytes: a
                // linear scan over `terminals` answers the same question, so
                // the map only earns its keep on high fan-out neurons. Bucket
                // the fan-out so a threshold is chosen from the distribution
                // rather than guessed.
                let f = n.terminals.len();
                let bucket = if f == 0 { "f0" } else if f < 8 { "f1_7" }
                    else if f < 32 { "f8_31" } else if f < 64 { "f32_63" }
                    else if f < 128 { "f64_127" } else { "f128_up" };
                *fanout_hist.entry(bucket).or_default() += 1;
                *fanout_terms.entry(bucket).or_default() += f;
            }
        }
        serde_json::json!({
            "global": self.brain.global_index_sizes(),
            "pool_side_bytes": side,
            "pool_side_total_bytes": side.values().sum::<usize>(),
            "neuron_body_bytes": body,
            "neuron_body_total_bytes": body.values().sum::<usize>(),
            "neuron_body_count": neurons,
            "fanout_neurons": fanout_hist,
            "fanout_terminals": fanout_terms,
            "top_fanout": top,
        })
    }

    /// Every byte the census can name: Brain-level maps, per-pool side
    /// structures, neuron bodies. Compared against `HEAP_LIVE` this is the one
    /// number that says whether a residual is an uncounted structure or not a
    /// structure at all.
    fn accounted_bytes(&self) -> usize {
        let c = self.census();
        let at = |a: &str, b: &str| {
            c.get(a)
                .and_then(|v| if b.is_empty() { Some(v) } else { v.get(b) })
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize
        };
        at("global", "total_bytes") + at("pool_side_total_bytes", "")
            + at("neuron_body_total_bytes", "")
    }

    /// (hub fan-out, estimated resident bytes) over every pool.
    fn footprint(&self) -> (usize, usize) {
        let terminal = std::mem::size_of::<w1z4rd_brain::Terminal>();
        // A terminal used to cost a terminal_idx entry too. It no longer does:
        // the index is kept only above Neuron::TERMINAL_INDEX_THRESHOLD, so the
        // map is charged from its OWN capacity below rather than per terminal.
        // Charging it per terminal read 4.1 MB at scale 16 both before and
        // after the change that removed 4.3 MB of it.
        let per_terminal = terminal;
        let per_member = std::mem::size_of::<w1z4rd_brain::NeuronRef>();
        let per_neuron = std::mem::size_of::<w1z4rd_brain::Neuron>();
        let (mut hub, mut bytes) = (0usize, 0usize);
        for pid in self.brain.fabric().pool_ids() {
            let Some(pool) = self.brain.fabric().pool(pid) else { continue };
            for n in pool.read().iter_neurons() {
                hub = hub.max(n.terminals.len());
                // CAPACITY, not len. A Vec grown by push holds up to 2x what
                // it uses, and a `len`-only estimate is how this number came
                // to read 5.9 MB against a 55.8 MB training peak at scale 64.
                bytes += per_neuron
                    + n.label.capacity()
                    + n.members.capacity() * per_member
                    + n.terminals.capacity() * per_terminal
                    + hash_table_bytes::<w1z4rd_brain::NeuronRef, usize>(
                        n.terminal_index_capacity(),
                    );
            }
        }
        (hub, bytes)
    }
}

fn shuffled(n: usize, epoch: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s: u64 = 0xC0FF_EECA_FEBA_BE ^ (epoch as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for i in (1..n).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s as usize) % (i + 1));
    }
    idx
}

fn score(probes: &[Probe], mut ask: impl FnMut(&str) -> Vec<u8>) -> (f64, f64) {
    let t0 = Instant::now();
    let hits = probes.iter().filter(|p| ask(&p.query) == p.answer.as_bytes()).count();
    let ms_per = t0.elapsed().as_secs_f64() * 1000.0 / probes.len().max(1) as f64;
    (100.0 * hits as f64 / probes.len().max(1) as f64, ms_per)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let scale = args
        .iter()
        .position(|a| a == "--scale")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1usize);

    // `--phase train|recall|infer|all` stops after that phase. Peak process
    // memory is measured from OUTSIDE (tools/capped.py), so running the same
    // scale once per phase attributes the peak to a phase without needing any
    // platform memory API. Default `all` is what the gate runs.
    let phase = args
        .iter()
        .position(|a| a == "--phase")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("all")
        .to_string();
    let census_wanted = args.iter().any(|a| a == "--census");

    let world = SceneWorld::new(scale);
    // The probe set is built before the brain, so this mark separates heap the
    // BRAIN allocates from heap the harness allocates for its own questions.
    let (live_world, _) = heap_mb();
    let mut subject = Subject::new();
    // A brain with two empty pools. Whatever this holds is fixed construction
    // cost -- the EEM's equation tables, the annealer, the fabric -- and is NOT
    // part of any per-fact residual, however large it looks at small scale.
    let (live_new, _) = heap_mb();
    let t0 = Instant::now();
    subject.train(&world.facts);
    let train_s = t0.elapsed().as_secs_f64();

    let census_after_train = census_wanted.then(|| subject.census());
    // Marks taken at each phase boundary. Training is the phase that allocates
    // what is left (measured at scale 64: train 55.8 / recall 58.3 / infer 58.7
    // MB of peak RSS), so the mark after training is the one that matters.
    let (live_train, peak_train) = heap_mb();
    let accounted_train = subject.accounted_bytes() as f64 / 1_048_576.0;
    let (mut recall_pct, mut recall_ms) = (f64::NAN, f64::NAN);
    let (mut integration_pct, mut infer_ms) = (f64::NAN, f64::NAN);
    let mut family_rows: Vec<serde_json::Value> = Vec::new();
    if phase != "train" {
        (recall_pct, recall_ms) = score(&world.facts, |q| subject.recall(q));
    }
    if phase != "train" && phase != "recall" {
        // Per family, then the overall figure as a hit-weighted total. A single
        // family at 100% against three at 0% must not be able to print as 90%,
        // so the breakdown is reported beside the total and never instead of it.
        let (mut hits, mut probes, mut secs) = (0usize, 0usize, 0.0f64);
        for fam in &world.families {
            let t0 = Instant::now();
            let h = fam.probes.iter().filter(|p| subject.infer(&p.query) == p.answer.as_bytes()).count();
            secs += t0.elapsed().as_secs_f64();
            hits += h;
            probes += fam.probes.len();
            family_rows.push(serde_json::json!({
                "name": fam.name,
                "hops": fam.hops,
                "probes": fam.probes.len(),
                "hits": h,
                "pct": 100.0 * h as f64 / fam.probes.len().max(1) as f64,
            }));
        }
        integration_pct = 100.0 * hits as f64 / probes.max(1) as f64;
        infer_ms = secs * 1000.0 / probes.max(1) as f64;
    }
    let (live_end, peak_end) = heap_mb();
    let accounted_end = subject.accounted_bytes() as f64 / 1_048_576.0;
    let (hub, bytes) = subject.footprint();
    let s = subject.brain.stats();
    if let Some(after_train) = census_after_train {
        eprintln!(
            "census after train: {}",
            serde_json::to_string_pretty(&after_train).unwrap_or_default()
        );
        eprintln!(
            "census after {phase}: {}",
            serde_json::to_string_pretty(&subject.census()).unwrap_or_default()
        );
    }

    println!(
        "{}",
        serde_json::json!({
            "scale": scale,
            "phase": phase,
            "facts": world.facts.len(),
            "integration_probes": world.integration_count(),
            "recall_pct": recall_pct,
            "integration_pct": integration_pct,
            "integration_families": family_rows,
            "train_s": train_s,
            "recall_ms": recall_ms,
            "infer_ms": infer_ms,
            "neurons": s.total_neurons,
            "concepts": s.total_concepts,
            "terminals": s.total_terminals,
            "hub_fanout": hub,
            "est_resident_mb": bytes as f64 / 1_048_576.0,
            // The allocator's own books. `heap_peak_mb` is every byte this
            // process ever held at once; `peak_mb` (tools/capped.py, from
            // outside) minus it is process overhead the brain never allocated.
            "heap_live_world_mb": live_world,
            "heap_live_new_mb": live_new,
            "heap_live_train_mb": live_train,
            "heap_peak_train_mb": peak_train,
            "accounted_train_mb": accounted_train,
            "heap_live_mb": live_end,
            "heap_peak_mb": peak_end,
            "accounted_mb": accounted_end,
            "unaccounted_live_mb": live_end - accounted_end,
        })
    );
}

/// What makes an integration probe honest, asserted over the GENERATED world at
/// every scale the scorecard runs -- not by inspection, and not by any claim
/// inside crates/brain, which never sees a probe.
///
/// Three properties, each of which has silently failed here before:
///
/// 1. DERIVABLE. Every probe answer is the composition of trained facts, looked
///    up through the fact table rather than recomputed from the generator, so a
///    generator bug cannot agree with itself.
/// 2. NOT SHORTCUTTABLE. For each probe, the most similar TRAINED question
///    answers something else. "Most similar" is taken under two metrics a
///    shortcut would plausibly use -- longest common prefix and character-bigram
///    Dice -- and over the whole ARGMAX SET, because a tie broken arbitrarily is
///    not evidence either way.
/// 3. NOT TRAINED. No probe query appears in the trained set.
#[cfg(test)]
mod honesty {
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// Scales the double-metric sweep runs at. It is O(probes x facts), so the
    /// largest scorecard scale is checked for structure only (`scale_64_*`).
    const METRIC_SCALES: &[usize] = &[1, 4, 16];

    fn facts_map(w: &SceneWorld) -> HashMap<&str, &str> {
        w.facts.iter().map(|p| (p.query.as_str(), p.answer.as_str())).collect()
    }

    fn family<'a>(w: &'a SceneWorld, name: &str) -> &'a Family {
        w.families.iter().find(|f| f.name == name).expect("family present")
    }

    fn answers<'a>(f: &'a Family) -> HashMap<&'a str, &'a str> {
        f.probes.iter().map(|p| (p.query.as_str(), p.answer.as_str())).collect()
    }

    fn lcp(a: &str, b: &str) -> usize {
        a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
    }

    /// Sorted, deduplicated character bigrams. Built once per query: the sweep
    /// is quadratic and rebuilding these inside it is what makes it slow.
    fn bigrams(s: &str) -> Vec<u16> {
        let mut v: Vec<u16> = s
            .as_bytes()
            .windows(2)
            .map(|w| u16::from(w[0]) << 8 | u16::from(w[1]))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    fn shared(a: &[u16], b: &[u16]) -> usize {
        let (mut i, mut j, mut n) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    n += 1;
                    i += 1;
                    j += 1;
                }
            }
        }
        n
    }

    /// Every trained index maximising `key`. Returning the whole set rather than
    /// one winner is the point: with a single winner, a tie decided by vector
    /// order would let the assertion pass or fail on nothing.
    fn argmax_set<T: Copy + PartialOrd>(n: usize, mut key: impl FnMut(usize) -> T) -> Vec<usize> {
        let mut best = key(0);
        let mut set = vec![0usize];
        for i in 1..n {
            let k = key(i);
            if k > best {
                best = k;
                set.clear();
                set.push(i);
            } else if !(k < best) {
                set.push(i);
            }
        }
        set
    }

    /// Property 2, both metrics, every family, at one scale.
    fn no_probe_is_shortcuttable(scale: usize) {
        let w = SceneWorld::new(scale);
        let tq: Vec<&str> = w.facts.iter().map(|p| p.query.as_str()).collect();
        let ta: Vec<&str> = w.facts.iter().map(|p| p.answer.as_str()).collect();
        let tb: Vec<Vec<u16>> = tq.iter().map(|q| bigrams(q)).collect();
        let trained: HashSet<&str> = tq.iter().copied().collect();
        for fam in &w.families {
            for p in &fam.probes {
                assert!(
                    !trained.contains(p.query.as_str()),
                    "scale {scale} family {}: probe {:?} is TRAINED, so it measures recall",
                    fam.name,
                    p.query
                );
                for i in argmax_set(tq.len(), |i| lcp(&p.query, tq[i])) {
                    assert_ne!(
                        ta[i], p.answer,
                        "scale {scale} family {}: prefix-nearest trained {:?} -> {:?} already \
                         answers {:?}, so a prefix shortcut scores this probe",
                        fam.name, tq[i], ta[i], p.query
                    );
                }
                // 2*shared/(na+nb), compared by cross-multiplication so no two
                // scores are equal by floating-point accident.
                let pb = bigrams(&p.query);
                for i in argmax_set(tq.len(), |i| {
                    Dice(2 * shared(&pb, &tb[i]), pb.len() + tb[i].len())
                }) {
                    assert_ne!(
                        ta[i], p.answer,
                        "scale {scale} family {}: bigram-nearest trained {:?} -> {:?} already \
                         answers {:?}",
                        fam.name, tq[i], ta[i], p.query
                    );
                }
            }
        }
    }

    /// An exact rational, ordered without floats.
    #[derive(Copy, Clone, PartialEq)]
    struct Dice(usize, usize);

    impl PartialOrd for Dice {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            (self.0 * o.1).partial_cmp(&(o.0 * self.1))
        }
    }

    #[test]
    fn no_probe_is_shortcuttable_at_every_metric_scale() {
        for &s in METRIC_SCALES {
            no_probe_is_shortcuttable(s);
        }
    }

    /// Property 1: each family's answer is the composition of trained facts, and
    /// the chain is exactly as long as the family claims.
    #[test]
    fn every_family_answer_is_derivable_from_trained_facts() {
        for &scale in METRIC_SCALES {
            let w = SceneWorld::new(scale);
            let f = facts_map(&w);
            let rooms = ROOMS_PER_SCALE * scale;
            let (om, nc, nom, bn) = (
                answers(family(&w, "on_material")),
                answers(family(&w, "next_color")),
                answers(family(&w, "next_on_material")),
                answers(family(&w, "beside_next")),
            );
            for r in 0..rooms {
                let room = format!("r{r:03}");
                let next = f[format!("{room} next?").as_str()];
                for (obj, _) in RESTS_ON {
                    let base = f[format!("{room} {obj} on?").as_str()];
                    assert_eq!(
                        om[format!("{room} {obj} on material?").as_str()],
                        f[format!("{room} {base} material?").as_str()],
                        "on_material must be on? then material?"
                    );
                }
                for k in 0..NEXT_COLOR_OBJECTS {
                    let obj = OBJECTS[(r + 3 * k) % OBJECTS.len()];
                    assert_eq!(
                        nc[format!("{room} next {obj} color?").as_str()],
                        f[format!("{next} {obj} color?").as_str()],
                        "next_color must be next? then color?"
                    );
                }
                let (obj, _) = RESTS_ON[r % RESTS_ON.len()];
                let base = f[format!("{next} {obj} on?").as_str()];
                assert_eq!(
                    nom[format!("{room} next {obj} on material?").as_str()],
                    f[format!("{next} {base} material?").as_str()],
                    "next_on_material must be next? then on? then material?"
                );
                if r % BESIDE_TRAINED_EVERY == 0 {
                    assert_eq!(f[format!("{room} beside?").as_str()], next, "beside? == next?");
                } else {
                    assert_eq!(
                        bn[format!("{room} beside?").as_str()], next,
                        "beside_next is next? under a phrasing held out for this room"
                    );
                }
            }
        }
    }

    /// Four families, not all the same depth, not all the same relation, and the
    /// paraphrase family's wording is one the trained set actually uses.
    #[test]
    fn four_families_one_three_hop_one_paraphrase_one_non_rests_on() {
        let w = SceneWorld::new(4);
        assert!(w.families.len() >= 4, "families: {}", w.families.len());
        for fam in &w.families {
            assert!(!fam.probes.is_empty(), "family {} is empty", fam.name);
        }
        assert!(w.families.iter().any(|f| f.hops == 3), "no 3-hop family");
        // Non-rests-on: next_color is composed from `next?` and `color?`, and
        // RESTS_ON appears nowhere in it. Asserted by deleting rests-on from the
        // world's reach: no next_color answer is an OBJECT, which is the only
        // thing rests-on returns.
        for p in &family(&w, "next_color").probes {
            assert!(!OBJECTS.contains(&p.answer.as_str()), "next_color leaned on rests-on");
        }
        // The paraphrase is held out per ROOM, not invented: the same wording is
        // trained elsewhere, so the family tests generalising a phrasing rather
        // than guessing an unseen word.
        let trained: HashSet<&str> = w.facts.iter().map(|p| p.query.as_str()).collect();
        assert!(
            trained.iter().any(|q| q.ends_with(" beside?")),
            "beside? is never trained, so beside_next is unlearnable rather than held out"
        );
    }

    /// The distractor. `near?` is trained, is never what the object rests on,
    /// and its material differs from the base's -- so a chain that follows the
    /// wrong relation returns a trained string that is wrong, which is the only
    /// way a wrong chain can be distinguished from no chain at all.
    #[test]
    fn a_wrong_chain_returns_a_trained_distractor() {
        for &scale in METRIC_SCALES {
            let w = SceneWorld::new(scale);
            let f = facts_map(&w);
            let om = answers(family(&w, "on_material"));
            for r in 0..ROOMS_PER_SCALE * scale {
                let room = format!("r{r:03}");
                for (obj, _) in RESTS_ON {
                    let base = f[format!("{room} {obj} on?").as_str()];
                    let near = f[format!("{room} {obj} near?").as_str()];
                    assert_ne!(near, base, "{room} {obj}: decoy equals the base");
                    let right = om[format!("{room} {obj} on material?").as_str()];
                    assert_eq!(right, f[format!("{room} {base} material?").as_str()]);
                    assert_ne!(
                        f[format!("{room} {near} material?").as_str()], right,
                        "{room} {obj}: near-chain returns the RIGHT material, so the \
                         distractor cannot catch a wrong chain"
                    );
                }
            }
        }
    }

    /// The coprime fix. `salt * 3` against a 6-long list took only {0, 3} mod 6,
    /// so every room had two colours and two materials, and the (paper, desk)
    /// probe's answer equalled the trained `paper material?`.
    #[test]
    fn every_room_uses_every_colour_and_material() {
        let w = SceneWorld::new(1);
        let f = facts_map(&w);
        for r in 0..ROOMS_PER_SCALE {
            let room = format!("r{r:03}");
            for (prop, list) in [("color", COLORS), ("material", MATERIALS)] {
                let seen: HashSet<&str> = OBJECTS
                    .iter()
                    .map(|o| f[format!("{room} {o} {prop}?").as_str()])
                    .collect();
                assert_eq!(seen.len(), list.len(), "{room} {prop}: {seen:?}");
            }
        }
    }

    /// Scale 64 is the stress scale and the quadratic sweep is too slow there,
    /// so it is checked for the structural properties that do not need it.
    #[test]
    fn scale_64_probes_are_untrained_and_every_family_is_populated() {
        let w = SceneWorld::new(64);
        let trained: HashSet<&str> = w.facts.iter().map(|p| p.query.as_str()).collect();
        for fam in &w.families {
            assert!(!fam.probes.is_empty(), "{} empty at scale 64", fam.name);
            for p in &fam.probes {
                assert!(!trained.contains(p.query.as_str()), "{:?} is trained", p.query);
            }
        }
        assert_eq!(w.integration_count(), w.families.iter().map(|f| f.probes.len()).sum::<usize>());
    }
}
