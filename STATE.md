# Brain Substrate — Empirical & Architectural State

*Snapshot for session continuity.  Last updated: 2026-05-18.*

## What works (architecturally validated, empirically pinned)

### Substrate primitives
- **Hierarchical concept emergence** (atoms → concepts → concept-of-concepts) — automatic via `Pool::collapse_tail_to_concept`.  Both directions wired at promotion time: atom→concept terminals AND concept→atom terminals.  Propagation flows through both naturally.
- **Multiset dedup** (Stage 13, commit `86c326a`) — `Pool::promote_to_concept` rejects new concepts whose atom-leaf multiset duplicates an existing canonical concept.  Prevents permutation variants like `food`/`oodf`/`foodf`.
- **Dense-burst training schedule** (`--burst` flag in `drive_corpora_brain`, commit `0cd80b3`) — required for word-level concept emergence under wide round-robin corpora.  Each (prompt, response) pair observed N reps back-to-back.
- **Layer-aware coverage gate** in `best_binding_match_v2` (commit `427a3b5`) — atoms that are members of a firing concept count toward concept-tier evidence, not atom-tier noise.

### Two retrieval paths
- **`/chat`** → `integrate_autonomous` → `best_binding_match_v2` (concept-tier aware) → fabric retrieval → chain explorer fallback → multi-fact assembler (Stage 11C).  This is the user-facing path.
- **`/integrate`** → `integrate()` → Stage 7 atom-level binding-pool routing → coverage-based selection.  This is the internal retrieval path; **stays atom-level by load-bearing design.**

### Architectural contract
`ARCHITECTURE.md §4.D.1` documents the deepest-confident-layer-wins inference contract.  The substrate produces the hierarchical firing state; the integration layer should walk it top-down by layer depth, gated by confidence.  Currently `/chat`'s OOV gate honors this; `/integrate`'s selection does not.

## Empirical state

| Probe | Value | Notes |
|---|---|---|
| `/chat` toddler (the no-regression floor) | **23/32 (71.9%)** | Held across all Stage 7-13 work |
| `/integrate` toddler strict-substring | **25/32 (78.1%)** | +2 above floor |
| `/integrate` toddler categorical (3+ byte) | 26/32 (81.2%) | Architectural-correct metric |
| K-12 categorical | 2/16 (12.5%) | First non-zero hits via dense-burst + dedup |
| multi_fact | 1/5 (20%) | First non-zero hit |
| OOV honesty | 2/3 (66.7%) | xyzzy, zzzzqqqq honestly OOG |
| Greetings | 1/7 (14.3%) | Down from 2/7 (greetings corpus dropped to prevent pollution) |
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

## Open architectural questions for future sessions

1. **Layer-2+ concept emergence runaway**: under dense-burst training of moderately deep corpora, layer-2+ concepts accumulate with very long member chains.  Should there be a max-depth cap on concept-of-concept emergence?  Or a degree-of-uniqueness filter?

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
