# Two-Pool Architectural Hypothesis Log

Goal: combined ≥ 0.95 on 8-pair small prose corpus, then port to Rust.

## Result milestones
| Score | What changed | exact_fwd/rev |
|-------|--------------|---------------|
| 0.0427 | CURRENT Rust path (chars only) | 0/8, 0/8 |
| 0.2831 | + promote_full_sequence wired into train_pair | 1/8, 1/8 |
| 0.4612 | + prop_max_accum + src/tgt argmax (GA tuned) | 2/8, 3/8 |
| 0.6016 | + reset_activations_per_pair (GA tuned) | 4/8, 4/8 |
| **1.0000** | **+ src_concept_overlap re-ranker (default knobs!)** | **8/8, 8/8** |

**Validated on large corpus (output 30..45,000 chars): combined=1.0000, 8/8 fwd, 8/8 rev.**

## Stress test — concept selection strategies × hard corpora
| Corpus | overlap (set) | tfidf | **lev (sequence sim)** |
|--------|---------------|-------|------------------------|
| T1: 8 baseline                    | 1.0000 (8/8 8/8) | 1.0000 (8/8 8/8) | **1.0000 (8/8 8/8)** |
| T2: 24 shared prefix              | 0.8575 (18/20)   | 0.9256 (20/23)   | **1.0000 (24/24)** |
| T3: 16 paraphrase (1-3 char diff) | 0.6675 (3/15)    | 0.9349 (14/14)   | **1.0000 (16/16)** |
| T4: 32 encyclopedia               | 0.8706 (25/28)   | 0.8963 (26/29)   | **0.9873 (32/31)** |

**LEV wins decisively.** Score = `lev_sim(input_atom_seq, concept.members_seq)`. Selects the source concept whose ordered atom sequence is closest to the input sequence. Position-aware (unlike set overlap), works for paraphrase inputs.

## Tested hypotheses
- **H1**: `promote_full_sequence` not wired up. **CONFIRMED.** Wiring it in lifts 0.043→0.283.
- **H2**: `tgt_seeds` clamp at 1.0 destroys discrimination. **PARTIAL.** Removing alone doesn't help (0.283), needs argmax + max_accum.
- **H3**: `propagate_weighted` accumulator clamp at 1.0 saturates targets. **CONFIRMED.** `prop_max_accum` lifts 0.283→0.41 in combo with argmax.
- **H4**: Concept activations persist between train_pair calls → every concept↔every concept saturates cross-edges. **CONFIRMED.** `reset_activations_per_pair` lifts fwd 2/8→4/8.
- **H5/H6**: Char propagation alone can't disambiguate concepts that share most chars ("explain quantum" vs "explain classical") because both fire equally from shared chars. **`src_concept_overlap`** (rerank concepts by `|members ∩ input_atoms| / |members|`) **DRIVES SCORE TO 1.0 / 8-8 EXACT**, with default knobs and no GA tuning. The neural propagation alone is insufficient for set-discrimination; an explicit overlap score is required at the source-concept selection step.

## Open hypotheses (queued)
- **H5**: SUM accumulator (`prop_no_clamp` only, no `prop_max_accum`) gives natural set-overlap discrimination — concept activation ∝ (member chars ∩ input chars). Test as A/B.
- **H6**: Add an explicit `score_concepts_by_input_overlap` re-ranker on src side: pick concept with highest |members ∩ input_atoms| / |members|. This should perfectly disambiguate concepts that share char prefixes (e.g., "explain quantum" vs "explain classical").
- **H7**: Rev direction lost performance with reset — fix by training cross-edges symmetrically with explicit concept→concept boost.
- **H8**: 30-pass training over-saturates char→char and char→concept weights to max_w=4. Try fewer passes (3-5) so weights remain proportional to frequency.
- **H9**: `tgt_concept_argmax` after cross-projection picks arbitrary winner when multiple seeds saturate at 1.0; needs `cross_seed_no_clamp=True` to preserve relative magnitudes.

## Cumulative architectural defects in Rust
1. `crates/core/src/neuro.rs:1897` `promote_full_sequence` not called by `two_pool_train_pair` (line 4738).
2. `crates/core/src/neuro.rs:4810` `tgt_seeds` clamp at 1.0 in `two_pool_query`.
3. `crates/core/src/neuro.rs:1692` `propagate_weighted` per-edge `min(1.0)` accumulator.
4. **NEW:** `two_pool_train_pair` does not reset neuron activations between pair calls, so concepts from prior pairs cross-talk into current pair's cross-pool wiring.
