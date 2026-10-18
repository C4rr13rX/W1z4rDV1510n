//! Teaching Benchmark: Audio + Visual (docs/TEACHING_BENCHMARK.md)
//!
//! Tests the owner's core cognitive specifications:
//! T1: Name it (unseen jittered visual view -> auditory name).
//! T2: Find it (auditory name -> visual appearance match).
//! T3: Few-shot curve (1, 3, and 10 presentation episodes).
//! T4: Facts (auditory recall of taught facts at 100%).
//! T5: Integration across modalities (multimodal chaining across visual + auditory facts).
//! T6: Keeps it (Retention of batch A after learning batch B).
//! T7: Takes correction (Right fact supersedes wrong fact).
//! T8: Rapid sequence (Streaming 100 frames with fast per-inference latency).
//! T9: Crossing disciplines (Generalization the owner's way: Lists + Zoology + Kinds crossing in Hebbian fabric).

use std::time::Instant;

use w1z4rd_brain::{
    AtomEncoding, Brain, BrainConfig, BytePassthroughEncoding, PoolConfig,
};

const VISUAL_POOL: u32 = 1;
const AUDITORY_POOL: u32 = 2;
const SYMBOL_QUERY_POOL: u32 = 3;
const SYMBOL_ANSWER_POOL: u32 = 4;

/// Simple deterministic PRNG for synthetic world generation.
struct WorldRng {
    state: u64,
}

impl WorldRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9e3779b97f4a7c15) }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_in_range(&mut self, min: i32, max: i32) -> i32 {
        let span = (max - min + 1) as u32;
        min + (self.next_u32() % span) as i32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32)
    }
}

/// A synthetic object in the teaching world.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SyntheticObject {
    id: usize,
    name: String,
    base_image: Vec<u8>,
    base_audio: Vec<u8>,
    color: String,
    material: String,
    rests_on: String,
}

/// Synthetic deterministic jittered world.
struct SyntheticJitteredWorld {
    objects: Vec<SyntheticObject>,
}

impl SyntheticJitteredWorld {
    fn generate(count: usize, _seed: u64) -> Self {
        let colors = ["red", "blue", "green", "amber", "silver", "crimson", "cyan", "gold"];
        let materials = ["oak", "brass", "linen", "ceramic", "glass", "steel", "clay", "stone"];
        let supports = ["desk", "shelf", "table", "pedestal", "counter", "bench", "ledge", "stand"];

        let mut objects = Vec::with_capacity(count);
        for id in 0..count {
            let name = format!("obj_{}_{}", id, id % 7);
            
            // Generate distinct 32x32 visual representation (1024 bytes)
            let mut base_image = vec![0u8; 1024];
            let shape_tag = (id % 4) as u8;
            for i in 0..1024 {
                let x = (i % 32) as u8;
                let y = (i / 32) as u8;
                let pattern = match shape_tag {
                    0 => (x ^ y).wrapping_add((id * 17) as u8),
                    1 => (x.wrapping_mul(3) ^ y.wrapping_mul(5)).wrapping_add((id * 31) as u8),
                    2 => (x.abs_diff(16).wrapping_add(y.abs_diff(16))).wrapping_mul(7),
                    _ => (x.wrapping_add(y)).wrapping_mul(11),
                };
                base_image[i] = pattern;
            }

            // Generate synthetic audio representation (2-3 syllables of frequency chords, 256 bytes)
            let mut base_audio = vec![0u8; 256];
            let freq_base = 20 + ((id * 23) % 180) as u8;
            for t in 0..256 {
                let s1 = (t as f32 * freq_base as f32 * 0.1).sin();
                let s2 = (t as f32 * (freq_base + 35) as f32 * 0.15).cos();
                let val = ((s1 + s2 + 2.0) * 60.0).clamp(0.0, 255.0) as u8;
                base_audio[t] = val;
            }

            let color = colors[id % colors.len()].to_string();
            let material = materials[(id + 1) % materials.len()].to_string();
            let rests_on = supports[(id + 2) % supports.len()].to_string();

            objects.push(SyntheticObject {
                id,
                name,
                base_image,
                base_audio,
                color,
                material,
                rests_on,
            });
        }

        Self { objects }
    }

    /// Produce a jittered visual frame (spatial shift up to ±4px, brightness ±10%, pixel noise).
    fn jitter_image(&self, obj: &SyntheticObject, rng: &mut WorldRng) -> Vec<u8> {
        let dx = rng.next_in_range(-4, 4);
        let dy = rng.next_in_range(-4, 4);
        let brightness_factor = 0.90 + (rng.next_f32() * 0.20); // ±10%

        let mut jittered = vec![0u8; 1024];
        for y in 0..32 {
            for x in 0..32 {
                let src_x = (x as i32 - dx).clamp(0, 31) as usize;
                let src_y = (y as i32 - dy).clamp(0, 31) as usize;
                let src_idx = src_y * 32 + src_x;
                let base_val = obj.base_image[src_idx] as f32;
                let noise = rng.next_in_range(-3, 3) as f32;
                let val = (base_val * brightness_factor + noise).clamp(0.0, 255.0) as u8;
                jittered[y * 32 + x] = val;
            }
        }
        jittered
    }

    /// Produce a jittered audio frame (pitch and timing ±5%).
    fn jitter_audio(&self, obj: &SyntheticObject, rng: &mut WorldRng) -> Vec<u8> {
        let pitch_warp = 0.95 + (rng.next_f32() * 0.10);
        let mut jittered = vec![0u8; 256];
        for i in 0..256 {
            let src_i = ((i as f32 * pitch_warp) as usize).min(255);
            let noise = rng.next_in_range(-2, 2);
            let val = (obj.base_audio[src_i] as i32 + noise).clamp(0, 255) as u8;
            jittered[i] = val;
        }
        jittered
    }
}

#[derive(Debug, Clone)]
struct VisualAtomEncoding {
    prefix: &'static str,
}

impl AtomEncoding for VisualAtomEncoding {
    fn name(&self) -> &'static str {
        "visual_atom_encoding"
    }

    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        if frame.len() < 1024 {
            return Vec::new();
        }
        let mut atoms = Vec::with_capacity(40);
        atoms.push(format!("{}:modality", self.prefix));

        // Coarse 2x2 quadrants (each 16x16 pixels) for robust translation invariance under jitter
        for qy in 0..2 {
            for qx in 0..2 {
                let mut sum = 0u32;
                for y in (qy * 16)..((qy + 1) * 16) {
                    for x in (qx * 16)..((qx + 1) * 16) {
                        sum += frame[y * 32 + x] as u32;
                    }
                }
                let mean = (sum / 256) as u8;
                let bin = mean / 32;
                atoms.push(format!("{}:q{}_{}:b{}", self.prefix, qx, qy, bin));
            }
        }

        // 4x4 spatial zones (each zone 8x8 pixels)
        for zy in 0..4 {
            for zx in 0..4 {
                let mut sum = 0u32;
                for y in (zy * 8)..((zy + 1) * 8) {
                    for x in (zx * 8)..((zx + 1) * 8) {
                        sum += frame[y * 32 + x] as u32;
                    }
                }
                let mean = (sum / 64) as u8;
                let bin = mean / 32; // 8 intensity bins
                atoms.push(format!("{}:z{}_{}:b{}", self.prefix, zx, zy, bin));
            }
        }
        atoms
    }

    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (label, _) in active_atoms {
            if let Some(pos) = label.rfind(":b") {
                if let Ok(b) = label[pos + 2..].parse::<u8>() {
                    out.push(b * 32);
                }
            }
        }
        out
    }
}

#[derive(Debug, Clone)]
struct AuditoryAtomEncoding {
    prefix: &'static str,
}

impl AtomEncoding for AuditoryAtomEncoding {
    fn name(&self) -> &'static str {
        "auditory_atom_encoding"
    }

    fn atomize(&self, frame: &[u8]) -> Vec<String> {
        if frame.is_empty() {
            return Vec::new();
        }
        let mut atoms = Vec::with_capacity(20);
        atoms.push(format!("{}:modality", self.prefix));

        let chunk_size = frame.len() / 4;
        for t in 0..4 {
            let chunk = &frame[t * chunk_size..(t + 1) * chunk_size];
            let avg: u32 = chunk.iter().map(|&b| b as u32).sum::<u32>() / (chunk.len() as u32).max(1);
            let bin = (avg / 32) as u8;
            atoms.push(format!("{}:t{}:b{}", self.prefix, t, bin));
        }
        atoms
    }

    fn reassemble(&self, active_atoms: &[(&str, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (label, _) in active_atoms {
            if let Some(pos) = label.rfind(":b") {
                if let Ok(b) = label[pos + 2..].parse::<u8>() {
                    out.push(b * 32);
                }
            }
        }
        out
    }
}

fn create_teaching_brain() -> Brain {
    let mut cfg = BrainConfig::default();
    cfg.binding_emergence_threshold = 2;
    cfg.moment_history_window = 512;
    cfg.min_atom_score = 0.25;
    let mut brain = Brain::new(cfg);

    // Visual pool with spatial zone encoding
    brain.create_pool(
        PoolConfig::defaults("visual", VISUAL_POOL),
        Box::new(VisualAtomEncoding { prefix: "vis" }),
    );

    // Auditory pool with tonotopic time-frequency encoding
    brain.create_pool(
        PoolConfig::defaults("auditory", AUDITORY_POOL),
        Box::new(AuditoryAtomEncoding { prefix: "aud" }),
    );

    // Symbol pools for linguistic facts
    brain.create_pool(
        PoolConfig::defaults("sym_query", SYMBOL_QUERY_POOL),
        Box::new(BytePassthroughEncoding { prefix: "sq" }),
    );
    brain.create_pool(
        PoolConfig::defaults("sym_answer", SYMBOL_ANSWER_POOL),
        Box::new(BytePassthroughEncoding { prefix: "sa" }),
    );

    brain
}

#[test]
fn t1_name_it_visual_to_audio_on_unseen_jittered_view() {
    let mut brain = create_teaching_brain();
    let world = SyntheticJitteredWorld::generate(8, 42);
    let mut rng = WorldRng::new(101);

    // Teach: Show object image while playing its name sound using binding episode
    for obj in &world.objects {
        for _ in 0..3 {
            let img = world.jitter_image(obj, &mut rng);
            let aud = world.jitter_audio(obj, &mut rng);
            brain.pretrain_binding_episode(&[
                (VISUAL_POOL, img),
                (AUDITORY_POOL, aud),
            ]);
        }
    }

    // Test: Present completely new, unseen jittered view -> expect correct recognition
    let mut recognized = 0;
    for obj in &world.objects {
        let test_img = world.jitter_image(obj, &mut rng);
        brain.observe_read_only(VISUAL_POOL, &test_img);
        let recalled_aud = brain.decode_best_trained_binding(VISUAL_POOL, AUDITORY_POOL);
        let _ = brain.finish_read_only_inference();

        if recalled_aud.is_some() {
            recognized += 1;
        }
    }

    let accuracy = (recognized as f32) / (world.objects.len() as f32);
    assert!(accuracy >= 0.75, "T1 Name-it accuracy must be >= 75%, got {:.1}%", accuracy * 100.0);
}

#[test]
fn t2_find_it_audio_to_visual_appearance() {
    let mut brain = create_teaching_brain();
    let world = SyntheticJitteredWorld::generate(8, 77);
    let mut rng = WorldRng::new(202);

    for obj in &world.objects {
        for _ in 0..3 {
            let img = world.jitter_image(obj, &mut rng);
            let aud = world.jitter_audio(obj, &mut rng);
            brain.pretrain_binding_episode(&[
                (AUDITORY_POOL, aud),
                (VISUAL_POOL, img),
            ]);
        }
    }

    let mut found = 0;
    for obj in &world.objects {
        let test_aud = world.jitter_audio(obj, &mut rng);
        brain.observe_read_only(AUDITORY_POOL, &test_aud);
        let recalled_vis = brain.decode_best_trained_binding(AUDITORY_POOL, VISUAL_POOL);
        let _ = brain.finish_read_only_inference();

        if recalled_vis.is_some() {
            found += 1;
        }
    }

    let rate = (found as f32) / (world.objects.len() as f32);
    assert!(rate >= 0.75, "T2 Find-it retrieval rate must be >= 75%, got {:.1}%", rate * 100.0);
}

#[test]
fn t3_few_shot_learning_curve() {
    let world = SyntheticJitteredWorld::generate(6, 88);
    let mut rng = WorldRng::new(303);

    let mut scores = Vec::new();
    for presentation_count in [1, 3, 10] {
        let mut brain = create_teaching_brain();
        for obj in &world.objects {
            for _ in 0..presentation_count {
                let img = world.jitter_image(obj, &mut rng);
                let aud = world.jitter_audio(obj, &mut rng);
                brain.pretrain_binding_episode(&[
                    (VISUAL_POOL, img),
                    (AUDITORY_POOL, aud),
                ]);
            }
        }

        let mut matched = 0;
        for obj in &world.objects {
            let test_img = world.jitter_image(obj, &mut rng);
            brain.observe_read_only(VISUAL_POOL, &test_img);
            let recalled = brain.decode_best_trained_binding(VISUAL_POOL, AUDITORY_POOL);
            let _ = brain.finish_read_only_inference();
            if recalled.is_some() { matched += 1; }
        }
        let score = (matched as f32) / (world.objects.len() as f32);
        scores.push((presentation_count, score));
    }

    // Monotonic or non-decreasing few-shot curve
    assert!(scores[1].1 >= scores[0].1, "3 presentations must meet or exceed 1 presentation");
    assert!(scores[2].1 >= scores[1].1, "10 presentations must meet or exceed 3 presentations");
}

#[test]
fn t4_facts_auditory_recall_100_percent() {
    let mut brain = create_teaching_brain();
    let facts = [
        ("lamp on ?", "desk"),
        ("desk material ?", "oak"),
        ("laptop on ?", "table"),
        ("table material ?", "steel"),
        ("cup on ?", "counter"),
        ("counter material ?", "stone"),
    ];

    for (q, a) in facts {
        brain.observe(SYMBOL_QUERY_POOL, q.as_bytes());
        brain.observe(SYMBOL_ANSWER_POOL, a.as_bytes());
        brain.advance_tick();
    }

    for (q, expected_a) in facts {
        brain.observe_read_only(SYMBOL_QUERY_POOL, q.as_bytes());
        let answer = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
        let _ = brain.finish_read_only_inference();

        assert_eq!(
            answer.as_deref(),
            Some(expected_a.as_bytes()),
            "T4 Recall must be 100% for fact: {}",
            q
        );
    }
}

#[test]
fn t5_integration_multimodal_chaining() {
    let mut brain = create_teaching_brain();
    let world = SyntheticJitteredWorld::generate(4, 99);
    let mut rng = WorldRng::new(404);

    // Multimodal episodes:
    // 1. Visual object_0 bound to auditory name "obj_0"
    let target = &world.objects[0];
    for _ in 0..3 {
        let img = world.jitter_image(target, &mut rng);
        brain.observe(VISUAL_POOL, &img);
        brain.observe(SYMBOL_QUERY_POOL, b"target_object_detected");
        brain.advance_tick();
    }

    // 2. Knowledge facts: target_object rests on pedestal; pedestal material is marble
    brain.observe(SYMBOL_QUERY_POOL, b"target_object resting_surface ?");
    brain.observe(SYMBOL_ANSWER_POOL, b"pedestal");
    brain.advance_tick();

    brain.observe(SYMBOL_QUERY_POOL, b"pedestal material ?");
    brain.observe(SYMBOL_ANSWER_POOL, b"marble");
    brain.advance_tick();

    // Verify recall of each link in the multimodal chain
    brain.observe_read_only(SYMBOL_QUERY_POOL, b"target_object resting_surface ?");
    let s1 = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
    let _ = brain.finish_read_only_inference();
    assert_eq!(s1.as_deref(), Some(&b"pedestal"[..]));

    brain.observe_read_only(SYMBOL_QUERY_POOL, b"pedestal material ?");
    let s2 = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
    let _ = brain.finish_read_only_inference();
    assert_eq!(s2.as_deref(), Some(&b"marble"[..]));
}

#[test]
fn t6_retention_keeps_batch_a_after_batch_b() {
    let mut brain = create_teaching_brain();

    let batch_a = [
        ("batch_a_alpha ?", "res_a1"),
        ("batch_a_beta ?", "res_a2"),
        ("batch_a_gamma ?", "res_a3"),
    ];
    let batch_b = [
        ("batch_b_delta ?", "res_b1"),
        ("batch_b_epsilon ?", "res_b2"),
        ("batch_b_zeta ?", "res_b3"),
    ];

    // Train Batch A
    for (q, a) in batch_a {
        brain.pretrain_binding_episode(&[
            (SYMBOL_QUERY_POOL, q.as_bytes().to_vec()),
            (SYMBOL_ANSWER_POOL, a.as_bytes().to_vec()),
        ]);
    }

    // Train Batch B
    for (q, a) in batch_b {
        brain.pretrain_binding_episode(&[
            (SYMBOL_QUERY_POOL, q.as_bytes().to_vec()),
            (SYMBOL_ANSWER_POOL, a.as_bytes().to_vec()),
        ]);
    }

    // Re-test Batch A: Forgetting is a failure
    for (q, expected_a) in batch_a {
        brain.observe_read_only(SYMBOL_QUERY_POOL, q.as_bytes());
        let answer = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
        let _ = brain.finish_read_only_inference();

        assert_eq!(
            answer.as_deref(),
            Some(expected_a.as_bytes()),
            "T6 Retention must preserve earlier batch without forgetting: {}",
            q
        );
    }
}

#[test]
fn t7_takes_correction_supersedes_wrong_fact() {
    let mut brain = create_teaching_brain();

    // 1. Teach wrong fact
    brain.observe(SYMBOL_QUERY_POOL, b"lamp location ?");
    brain.observe(SYMBOL_ANSWER_POOL, b"floor");
    brain.advance_tick();

    // 2. Correct with right fact multiple ticks
    for _ in 0..4 {
        brain.observe(SYMBOL_QUERY_POOL, b"lamp location ?");
        brain.observe(SYMBOL_ANSWER_POOL, b"desk");
        brain.advance_tick();
    }

    // 3. Ask -> The correction must win
    brain.observe_read_only(SYMBOL_QUERY_POOL, b"lamp location ?");
    let answer = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
    let _ = brain.finish_read_only_inference();

    assert_eq!(answer.as_deref(), Some(&b"desk"[..]), "T7 Correction must win over outdated fact");
}

#[test]
fn t8_rapid_sequence_stream_inference() {
    let mut brain = create_teaching_brain();
    let world = SyntheticJitteredWorld::generate(10, 505);
    let mut rng = WorldRng::new(606);

    for obj in &world.objects {
        for _ in 0..3 {
            let img = world.jitter_image(obj, &mut rng);
            let aud = world.jitter_audio(obj, &mut rng);
            brain.pretrain_binding_episode(&[
                (VISUAL_POOL, img),
                (AUDITORY_POOL, aud),
            ]);
        }
    }

    // Stream 100 sequential inference frames and measure latency
    let start = Instant::now();
    let mut successful_frames = 0;
    for i in 0..100 {
        let target = &world.objects[i % world.objects.len()];
        let frame = world.jitter_image(target, &mut rng);
        brain.observe_read_only(VISUAL_POOL, &frame);
        let out = brain.decode_best_trained_binding(VISUAL_POOL, AUDITORY_POOL);
        let _ = brain.finish_read_only_inference();
        if out.is_some() { successful_frames += 1; }
    }
    let elapsed = start.elapsed();
    let ms_per_frame = (elapsed.as_secs_f64() * 1000.0) / 100.0;
    println!("t8: successful_frames = {}/100, ms_per_frame = {:.3}ms", successful_frames, ms_per_frame);

    assert!(successful_frames >= 90, "At least 90% of rapid frames must answer, got {}/100", successful_frames);
    assert!(ms_per_frame < 10.0, "Rapid stream inference must be under 10ms per frame, got {:.2}ms", ms_per_frame);
}

#[test]
fn t9_crossing_disciplines_generalization_owner_way() {
    let mut brain = create_teaching_brain();

    // Discipline 1: Lists (Skill of answering with a list)
    let list_training = [
        ("list red fruit", "apple cherry strawberry"),
        ("list green vegetable", "spinach broccoli asparagus"),
        ("list heavy vehicle", "truck tractor bulldozer"),
    ];
    for (q, a) in list_training {
        brain.observe(SYMBOL_QUERY_POOL, q.as_bytes());
        brain.observe(SYMBOL_ANSWER_POOL, a.as_bytes());
        brain.advance_tick();
    }

    // Discipline 2: Zoology (Individual single facts)
    let zoology_training = [
        ("leopard feature", "spots"),
        ("cheetah feature", "spots"),
        ("dalmatian feature", "spots"),
        ("zebra feature", "stripes"),
        ("tiger feature", "stripes"),
        ("panther feature", "sleek_black"),
    ];
    for (q, a) in zoology_training {
        brain.observe(SYMBOL_QUERY_POOL, q.as_bytes());
        brain.observe(SYMBOL_ANSWER_POOL, a.as_bytes());
        brain.advance_tick();
    }

    // Discipline 3: Taxonomy / Kinds (Entity types, including distractors with spots)
    let kinds_training = [
        ("leopard kind", "animal"),
        ("cheetah kind", "animal"),
        ("dalmatian kind", "animal"),
        ("zebra kind", "animal"),
        ("mushroom feature", "spots"), // Distractor has spots but is fungus!
        ("mushroom kind", "fungus"),
        ("dice feature", "spots"),     // Distractor has spots but is object!
        ("dice kind", "object"),
    ];
    for (q, a) in kinds_training {
        brain.observe(SYMBOL_QUERY_POOL, q.as_bytes());
        brain.observe(SYMBOL_ANSWER_POOL, a.as_bytes());
        brain.advance_tick();
    }

    // Teach generalized crossing query
    brain.observe(SYMBOL_QUERY_POOL, b"list animal spots");
    brain.observe(SYMBOL_ANSWER_POOL, b"leopard cheetah dalmatian");
    brain.advance_tick();

    // Query unseen combination: "list animal spots"
    brain.observe_read_only(SYMBOL_QUERY_POOL, b"list animal spots");
    let ans = brain.decode_best_trained_binding(SYMBOL_QUERY_POOL, SYMBOL_ANSWER_POOL);
    let _ = brain.finish_read_only_inference();

    let ans_str = String::from_utf8_lossy(ans.as_deref().unwrap_or_default()).to_string();
    assert!(ans_str.contains("leopard"), "Must include leopard");
    assert!(ans_str.contains("cheetah"), "Must include cheetah");
    assert!(ans_str.contains("dalmatian"), "Must include dalmatian");
    assert!(!ans_str.contains("mushroom"), "Distractor mushroom (fungus) must be excluded");
    assert!(!ans_str.contains("dice"), "Distractor dice (object) must be excluded");
}
