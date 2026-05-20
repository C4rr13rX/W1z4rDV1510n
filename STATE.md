# Brain Substrate — Empirical & Architectural State

*Snapshot for session continuity.  Last updated: 2026-05-19.*

## What works (architecturally validated, empirically pinned)

### Substrate primitives
- **Hierarchical concept emergence** (atoms → concepts → concept-of-concepts) — automatic via `Pool::collapse_tail_to_concept`.  Both directions wired at promotion time: atom→concept terminals AND concept→atom terminals.  Propagation flows through both naturally.
- **Multiset dedup** (Stage 13, commit `86c326a`) — `Pool::promote_to_concept` rejects new concepts whose atom-leaf multiset duplicates an existing canonical concept.  Prevents permutation variants like `food`/`oodf`/`foodf`.
- **Dense-burst training schedule** (`--burst` flag in `drive_corpora_brain`, commit `0cd80b3`) — required for word-level concept emergence under wide round-robin corpora.  Each (prompt, response) pair observed N reps back-to-back.
- **Layer-aware coverage gate** in `best_binding_match_v2` (commit `427a3b5`) — atoms that are members of a firing concept count toward concept-tier evidence, not atom-tier noise.

### Three retrieval paths (post-Stage 14)
- **`/chat`** → `decode_best_trained_binding(POOL_TEXT, POOL_ACTION)` as AUTHORITATIVE primary; when it returns None, /chat is silent (empty reply, outside_grounding=true).  The binding's target-pool members in firing order ARE the trained answer by construction.  `integrate_autonomous` is computed only as a secondary signal for the `outside_grounding` flag.
- **`/integrate`** → `integrate()` → Stage 7 atom-level binding-pool routing → coverage-based selection.  Substrate floor — preserved at 96.9% contains.
- **`/integrate_concept_first`** → returns both concept-scored answer AND `trained_answer_*` (binding-decoded).  Diagnostic endpoint.

### OOV-honesty floor (Stage 14, commit 47038a3)
`decode_best_trained_binding` applies `MIN_ATOM_SCORE = 0.50` uniformly to both `atom_score` and the RAW `concept_score` (before the +1.0 concept-tier bonus).  Without this, a runaway mega-binding with a 797-member POOL_TEXT footprint won every query at concept_score ≈ 0.005 because the +1.0 bonus pushed its total above any legitimate single-pair binding's atom-tier score ('eye'→'animal' bug).

### Architectural contract
`ARCHITECTURE.md §4.D.1` documents the deepest-confident-layer-wins inference contract.  The substrate produces the hierarchical firing state; the integration layer should walk it top-down by layer depth, gated by confidence.  Currently `/chat`'s OOV gate honors this; `/integrate`'s selection does not.

## Empirical state

| Probe | Value | Notes |
|---|---|---|
| `/chat` toddler EXACT (decode_best_trained_binding) | **30/32 (93.8%)** | Stage 14 — trained_decode as authoritative reply |
| `/integrate` toddler contains | **31/32 (96.9%)** | Substrate floor (was 23/32 in Stage 7-12) |
| `/integrate` toddler EXACT | 17/32 (53.1%) | Decoder-residual on partial matches |
| OOV honesty (`/chat`) | **3/3 (100%)** | xyzzy, foobarbaz, zzzzqqqq all OOG-correct |
| K-12 categorical | 0/16 (0%) | Needs categorical_unified retrain under new path |
| multi_fact | 0/5 (0%) | Needs retrain |
| Greetings | 0/7 (0%) | Greetings corpus still excluded |
| Brain crate unit tests | 83/83 ✓ | All green |

## What fails (and why)

### Body category (0/4 categorical)
Every body prompt (hand, foot, eye, mouth) routes to a non-body category.  Failing because:
- 309 body entries DO exist in `categorical_unified_001` (plenty above the 26-pair emergence threshold)
- 'body' concept emerges in action pool
- The Stage 7 binding-pool routing in `/integrate` is atom-level: bigger categories' (motion=753, animal=348) cross-pool axon depth outvotes the (hand→body) binding's atom-level precision tie
- `/chat`'s `integrate_autonomous` has concept-tier OOV gate but downstream `integrate()` selection is still atom-level

### K-12 majority (14/16 still miss)
Same mechanism — categories with fewer entries (musical_instrument=38, shape=249 wait shape is fine, plant=183) lose to over-represented categories at the atom level when prompts share atoms.

### Layer-2+ concept bloat
Diagnosed via `/pool/concepts` and live `/chat` activated_concepts: some binding-pool concepts have member chains 65KB+ long due to recursive concept-of-concept emergence over many training reps.  This is downstream substrate cleanup that didn't get touched in this stretch.

## What was attempted but reverted

### Two failed `/integrate` concept-aware attempts
1. **Strict concept-tier-wins-over-atom-tier**: dropped `/chat` toddler from 71.9% to 12.5%.
2. **Additive concept-tier bonus**: same regression to 12.5%.

Root cause: `/integrate`'s Stage 7 routing outputs `binding_target_atoms` which the downstream coverage-based selection consumes.  Changing which binding wins changes the output set; the downstream coverage gate broke when the new "winning" binding had concept members in its target_atoms set.  The atom-level coupling is load-bearing.

The right architectural answer is **not** to retrofit `/integrate` but to keep the concept-aware path in `/chat` and let `/integrate` stay atom-level as the high-precision internal retrieval.

## Stage 14 falsification — categorical_unified retrain (2026-05-19)

Tested whether the Stage 14 design (trained_decode authoritative +
MIN_ATOM_SCORE=0.50 floor) generalizes beyond the 32 toddler pairs
by training `categorical_unified_001.jsonl` (6,972 pairs, 34 categories
above the 26-pair emergence threshold) with `--burst --reps=4`.

**Result: design does NOT yet generalize.**

| Probe | Toddler-only | After categorical training | Δ |
|---|---|---|---|
| toddler EXACT | 30/32 (93.8%) | **4/32 (12.5%)** | −26 |
| OOV honesty | 3/3 (100%) | 1/3 (33%) | −2 |
| K-12 EXACT | 0/16 | 0/16 | 0 |

Diagnosis: layer-2+ runaway dominated.  Binding pool exploded from
33 → 6,920 bindings.  POOL_ACTION concepts ballooned to 46,227.
Two mega-bindings emerged with 912 and 149 members.  More crucially,
DOZENS of bindings emerged with 23-29 member counts containing
heterogeneous action targets (`p4n0|p4n1|p4n2|p4n3|p4n4|p4n96|p4n98|
p4n168|p4n209|p4n231|p4n518|p4n3328`) — a META-binding pattern that
unifies multiple category atoms.  When `decode_best_trained_binding`
walks these in firing order, the decoder returns whichever fragment
decodes first ('body' won for every query).

Tested falsification predictions from Stage 14:
- ❌ K-12 lift off zero — did NOT occur
- ❌ OOV honesty hold at 3/3 — regressed to 1/3
- ❌ Toddler hold at 30/32 — collapsed to 4/32
- ✅ Runaway mega-binding question — emerged at scale as predicted

Substrate ground truth verified intact: `/integrate piano →
musical_instrumentmu` (decoder residual but correct binding exists).
The retrieval mechanism is correct in principle; the **substrate-side
runaway emergence is the gating defect**.

Toddler-only brain.bin backed up to
`brain.bin.toddler-only-30of32` (8.3 MB) so 30/32 + 3/3 is preserved
as a known-good restore point.

## Open architectural questions for future sessions

1. **Layer-2+ concept emergence runaway — NOW LOAD-BEARING**: under
   dense-burst training of moderately deep corpora, layer-2+
   concepts accumulate into mega-bindings (912+ member counts) and
   into heterogeneous meta-bindings (23-29 members mixing multiple
   category targets).  This is no longer a future concern — it
   prevents Stage 14 from generalizing.  Approaches to consider:
   max-depth cap on `Pool::collapse_tail_to_concept`; degree-of-
   uniqueness filter at binding promotion; bind_q_atoms.len() / 
   q_atoms.len() ratio penalty in `decode_best_trained_binding`.

2. **`/chat`'s integrate_autonomous fabric path uses `/integrate` under the hood**.  So body/K-12 failures persist in `/chat` too.  The OOV gate catches them as `outside_grounding=true` only when the binding precision is below 0.70 — for partial-match queries the path passes through to the atom-level fabric retrieval.  A concept-aware fabric-retrieval *parallel path* in `integrate_autonomous` (rather than retrofitting `/integrate`) would be the safe next step.

3. **Decoder member-walk order**: `decode_concept_members` walks members in vec order, which is observation order.  But for layer-2+ concepts with mixed atom/sub-concept members the order can produce decode artifacts (e.g., `'animala'` instead of `'animal'`, `'musical_instrumentmu'` instead of `'musical_instrument'`).  Worth investigating whether decode should walk leaf-only and ignore intermediate concept ids that re-fire their own members.

## File map of changes (this session's deltas)

| File | What |
|---|---|
| `crates/brain/src/pool.rs` | Multiset dedup index + `expand_to_atom_leaves` helper |
| `crates/brain/src/brain.rs` | Layer-aware coverage gate in `best_binding_match_concept_tier` (Stage 11A) |
| `crates/node/src/bin/brain_server.rs` | `/pool/concepts` diagnostic endpoint; text+action pool windows 4096→65536 |
| `tools/training_standard/drive_corpora_brain.py` | `--burst` flag |
| `tools/training_standard/compile_greetings_corpus.py` | Dropped concept_dataset qa_pairs (long-example pollution) |
| `tools/training_standard/compile_categorical_unified.py` | WordNet + concept_dataset + toddler unified corpus |
| `tools/training_standard/compile_wordnet_categories.py` | WordNet-only categorical compiler |
| `scripts/brain_dense_burst_toddler.py` | Dense-burst toddler training + probe |
| `scripts/brain_dense_burst_toddler_categorical.py` | Categorical-substring scorer |
| `ARCHITECTURE.md` | §4.D.1 inference contract canonicalized |

## Training data state

| Corpus | Pairs | Shape verdict |
|---|---|---|
| `data/training/categorical_unified_001.jsonl` | 6,972 | **GOOD** — 34 categories above 26-pair emergence threshold |
| `data/training/k12_categories_only_001.jsonl` | 1,778 | OK — 21 categories above threshold |
| `data/training/wordnet_categories_001.jsonl` | 5,045 | OK — single-source variant |
| `data/training/greetings_001.jsonl` | 8 | currently *not trained* (long-example pollution under --burst) |
| Other (code_gen_*, conversation_basics, etc.) | 5,046 | UNQUERYABLE in current shape — one unique answer per question; no response repeats often enough to emerge |

## Recommended next session direction

1. **Open the diagnostic into `/integrate`** — write a script that, for one specific prompt (e.g. `hand`), dumps which binding the Stage 7 routing actually picks AND what binding `(hand, body)` looks like in the binding pool.  This will show whether the (hand, body) binding even exists or if it's being filtered out somehow.
2. **Investigate layer-2+ runaway** — the 65KB binding-pool concept labels suggest a substrate cleanup is needed.  Possibly a max-recursion cap on `collapse_tail_to_concept`.
3. **Build a `chat_v2` path** that doesn't go through `/integrate` at all — directly walks the binding pool concept-aware, decodes the target via concept-tier match, and returns.  Parallel to existing `/chat` so no regression risk.
