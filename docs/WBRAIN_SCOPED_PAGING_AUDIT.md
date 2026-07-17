# `.wbrain` scoped-paging audit

This is the implementation map for the invariant that an idle brain keeps
neuron bodies serialized and inference wakes only knowledge reachable from the
input. It is architecture work, not brain-configuration advice.

## Required end state

- A sleeping neuron occupies a compact address entry, not a full `Neuron`.
- Startup does not hydrate neuron bodies, concept membership, recurrence
  history, or complete binding-routing maps.
- Statistics use persisted counters and compact histograms.
- Maintenance may stream neuron IDs sequentially under an explicit budget.
- Inference must obtain candidate IDs from input-keyed disk indexes; it may
  never fall back to iterating every sleeping neuron.
- A page-in metric must report IDs awakened, bytes read, index nodes visited,
  resident bytes, and return-to-sleep count for each request.

## Current whole-pool iteration map

| Site | Current purpose | Required replacement |
|---|---|---|
| `brain.rs:890` | concept count for tuning | Persisted pool concept counter |
| `brain.rs:971` | locked-terminal count | Persisted counter updated with terminal mutations |
| `brain.rs:1746` | rebuild binding indexes | Bounded sequential maintenance pass only |
| `brain.rs:1773` | bounded binding-index rebuild | Bounded sequential maintenance pass; never startup inference |
| `brain.rs:2195` | best binding during inference | Input-keyed binding candidate index |
| `brain.rs:2775` | fallback binding candidates | Remove full-scan fallback; union exact, feature, and motif indexes |
| `brain.rs:3188` | ordered binding match | Input-keyed sequence/feature candidate index |
| `brain.rs:4009` | binding match | Input-keyed feature candidate index |
| `brain.rs:4054` | identify query concepts | Current firing set plus compact per-ID kind bit |
| `brain.rs:4135` | binding concept match | Input-keyed concept/member index |
| `brain.rs:4228` | rank active neurons | Current activation IDs only; page members for active concepts |
| `brain.rs:5585` | analogy integration | Explicit bounded maintenance stream |
| `brain.rs:6303` | cluster export | Explicit diagnostic stream with limit/cursor |
| `brain.rs:6586` | eviction selection | Resident-neuron map only |
| `brain.rs:6715` | salience histogram | Persisted compact histogram |
| `fabric.rs:428` | tier orchestrator scan | Resident-neuron map only |
| `brain_api.rs:2694` | concept listing | Cursor-based disk diagnostic |
| `brain_api.rs:2751` | binding diagnostics | Input-keyed candidate index |
| `brain_server.rs:1032` | concept export | Cursor-based disk diagnostic |

Line numbers describe commit `cef923e` and should be refreshed after related
edits.

### First routed path

`decode_best_trained_binding_with_context` no longer uses its whole-binding-pool
fallback. It unions exact sequence postings, flattened feature-atom postings,
and the eight narrowest character-motif posting lists, then pages only those
binding bodies before scoring. The lazy-restore fixture now exercises this
single-pool path as well as the existing multi-pool path. GitHub Actions run
`29616946268` passed the streaming migration, sleeping payload, scoped lazy
recall, and node migration-stage recovery gates on Windows.

`decode_best_trained_binding_multi` now preserves the candidate IDs produced by
its hydration phase and scores only those IDs. The hydration phase combines
active-neuron terminals, exact sequence postings, feature-atom postings, and
the eight narrowest motif lists, pages those bindings and their named member
trees, and returns the routed set instead of discarding it. This removes the
second inference-time whole-binding-pool scan. It is covered by the same
successful scoped lazy-recall gate in run `29616946268`.

## Resident structures still violating the invariant

1. `Pool.neurons: Vec<Neuron>` creates one full Rust object for every sleeping
   ID. Terminal and member heaps can now be released, but the fixed object
   remains.
2. `PoolContainerManifest.neuron_offsets: Vec<Option<u64>>` hydrates the entire
   address table.
3. `WbrainPoolMetadata.neuron_kinds`, `concept_slots`, and `born_ticks` hydrate
   parallel full-pool vectors.
4. `Brain.binding_sequence_index`, `binding_feature_atom_index`, and
   `binding_motif_index` are rebuilt and restored as complete resident maps.
5. Several inference paths still use whole-binding-pool scans when an exact
   index key is absent.

## Implementation order

1. Replace inference scan fallbacks with the union of exact sequence, feature
   atom, character motif, and active-concept candidate IDs. Honest OOV is
   preferable to waking an unrelated entire brain.
2. Move the three binding indexes into `.wbrain` disk hash records with compact
   root directories and request-local result vectors.
3. Replace `Vec<Neuron>` with stable compact slots plus a resident-neuron map.
   `ensure_loaded(id)` inserts one body; idle serialization removes it.
4. Move offset and kind/concept/born metadata to paged fixed-width tables.
5. Convert diagnostic APIs to bounded cursor streams.
6. Run cold/warm latency, bytes-read, awakened-neuron, peak-RAM, and
   return-to-sleep gates before resuming corpus training.

## Admission rule

Do not claim neuron-level paging from a zero-resident-terminal count alone.
Admission requires proving that fixed metadata and routing memory do not scale
as full `Neuron` objects or full binding maps, and that an unseen prompt cannot
trigger a whole-pool fallback scan.
