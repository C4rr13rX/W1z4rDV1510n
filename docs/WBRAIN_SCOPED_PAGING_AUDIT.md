# `.wbrain` scoped-paging audit

This is the implementation map for the invariant that an idle brain keeps
neuron bodies serialized and inference wakes only knowledge reachable from the
input. It is architecture work, not brain-configuration advice.

## Required end state

- A sleeping neuron occupies a compact address entry, not a full `Neuron`.
- Startup does not hydrate neuron bodies, concept membership, recurrence
  history, historical label maps, or complete binding-routing maps.
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
| `brain.rs:2195` | best binding during inference | Routed through sequence, feature, and active-terminal candidates |
| `brain.rs:2775` | fallback binding candidates | Remove full-scan fallback; union exact, feature, and motif indexes |
| `brain.rs:3188` | ordered binding match | Input-keyed sequence/feature candidate index |
| `brain.rs:4009` | binding match | Input-keyed feature candidate index |
| `brain.rs:4054` | identify query concepts | Uses only the current firing set and compact sleeping-slot identity |
| `brain.rs:4135` | binding concept match | Routed through feature and active-terminal candidates |
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

The grounding decoder and both `best_binding_match` confidence tiers now share
the same input-routed candidate resolver. It unions exact ordered-sequence
postings, feature-atom postings, and binding terminals reachable from the
currently firing query neurons, pages only those binding bodies, and returns
honest OOV when the union is empty. Concept-tier classification now separates
atoms from concepts by inspecting only the current firing IDs; it no longer
constructs a whole-pool concept set. The lazy restore fixture sleeps the entire
brain independently before confidence matching and grounded integration to
prove both paths can restore the trained binding without a binding-pool scan.

### Neuron body slots

`Pool.neurons` now has two representations. A dense legacy brain stores one
boxed `Neuron` per occupied ID. An idle `.wbrain` pool retains only scalar slot
and concept counts plus a hash map of the bodies actually resident for the
current request. `ensure_loaded(id)` inserts one body; returning to idle
persists and removes it. There is no full-pool pointer vector or per-ID evicted
set.

### Paged address and identity table

The first full migration attempt with compact bodies was safely aborted at
8.26 GB private memory. The source remained intact and the monitor still
reported 12.9 GiB physically available. Measurements showed capacity doubling
in `Vec<Option<u64>>` offsets plus parallel kind/concept/birth vectors; neuron
payload streaming itself remained bounded.

Container version 2 replaced those resident vectors during large migration
with one fixed-width 24-byte on-disk slot record containing the current neuron
record offset, birth tick, kind, and flags. Initial writes are published in
65,536-neuron contiguous batches. Reopen reads one slot record by stable ID;
concept maintenance streams slot flags and request-time bodies independently.
The manifest keeps no resident offset vector for a paged pool.

The first v2 full-scale rerun advanced to a 7.99 GB generated container while
private memory plateaued at 2.02 GB instead of climbing to the 8 GB abort
limit. That run was intentionally stopped after the plateau was traced to the
remaining historical binding-label `HashMap`; completing it would have
produced a container that reopened with the same multi-gigabyte index.

Container version 3 moves historical atom and binding labels into immutable
on-disk hash generations. Exact lookup reads one bucket chain and verifies the
stable neuron slot is still present. Labels created by live training form a
small resident overlay, are flushed as a disk delta at idle/manifest commit,
and are then removed from both the store and pool maps. Prediction and
neurogenesis now use the same overlay-plus-disk lookup path.

Version 3 also separates logical neuron count from allocated slot capacity.
Migration reserves bounded growth space, and exhausting it relocates the
fixed-width slot table with a bounded copy rather than hydrating it. The
regression suite creates neurons beyond the initial reserve, reopens the
container, and proves exact body and label lookup for the new ID.

Container version 4 moves the three historical binding-routing maps into an
immutable on-disk posting index. Finalization streams each binding once and
emits exact ordered-sequence, feature-atom, and character-motif postings
without retaining a corpus-sized map. Inference hashes only the keys derived
from the active input, reads the corresponding bucket chains, and pages the
returned binding IDs. Bindings learned after migration remain in a small live
overlay until the next idle transition; idle writes that overlay as a new disk
generation and clears it from RAM. The regression suite verifies zero
resident posting entries after rebuild, after live learning and sleep, and
after reopen, while preserving exact deduplication and trained recall.

## Resident structures still violating the invariant

1. Live concept-sequence overlays are still serialized in pool metadata; they
   need the same immutable-delta treatment before indefinitely long training.
2. A feature shared by an extreme number of bindings can still produce a large
   request-local posting vector. Preserve accuracy while replacing this with a
   cursor/intersection strategy before mobile admission.
3. Immutable posting generations need bounded compaction after many separate
   train/sleep cycles so lookup cost does not grow with generation count.
4. Diagnostic/export and analogy paths listed above still need explicit
   cursor/budget enforcement; request-time binding decoders no longer have a
   whole-pool fallback.

## Implementation order

1. Replace inference scan fallbacks with the union of exact sequence, feature
   atom, character motif, and active-concept candidate IDs. Honest OOV is
   preferable to waking an unrelated entire brain.
2. Move the three binding indexes into `.wbrain` disk hash records with compact
   root directories and request-local result vectors. **Implemented in
   container version 4; full-scale rerun pending.**
3. Replace `Vec<Neuron>` with stable compact slots plus request-scoped resident
   bodies. `ensure_loaded(id)` inserts one body; idle serialization removes it.
   **Implemented and covered by persistence/inference gates.**
4. Move offset and kind/concept/born metadata to paged fixed-width tables.
   **Implemented in container version 2.**
5. Move historical labels to disk and preserve post-migration neurogenesis.
   **Implemented in container version 3.**
6. Convert diagnostic APIs to bounded cursor streams.
7. Run cold/warm latency, bytes-read, awakened-neuron, peak-RAM, and
   return-to-sleep gates before resuming corpus training.

## Admission rule

Do not claim neuron-level paging from a zero-resident-terminal count alone.
Admission requires proving that fixed metadata and routing memory do not scale
as full `Neuron` objects or full binding maps, and that an unseen prompt cannot
trigger a whole-pool fallback scan.
