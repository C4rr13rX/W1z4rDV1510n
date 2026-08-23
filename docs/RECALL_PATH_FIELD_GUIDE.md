# Recall Path Field Guide

## Purpose and scope

How a request reaches an answer in `/brain/chat`, what to verify before
changing any part of it, and which changes have already been measured and
rejected.

This exists because the recall path punishes assumption. Its failures are
silent: a route can score a candidate perfectly and still return nothing,
and a similarity metric can look better on paper while destroying the exact
matches it was meant to help. Every claim below was measured on the
persisted programming brain, and each names the artefact that proves it.

Read this before editing `decode_best_binding_by_char_motifs_*`,
`recall_derived`, or the answer-branch chain in `crates/node/src/brain_api.rs`.

## Verify these four facts first

They are cheap to check, they are all documented elsewhere in this repo, and
each one has cost real time when assumed instead.

| Fact | How to verify | Why it matters |
|---|---|---|
| **Atoms are bytes, not words** | `curl /stats` — compare `total_neurons` with `total_concepts`. Measured 2026-08-23: 2,548,947 vs 2,548,068, so 879 atoms exist across the whole fabric. Also `ARCHITECTURE.md` lines 27, 190, 281. | Any design that treats an atom as a subject term will select hubs. Every document shares the same ~256 byte atoms per prefix. |
| **A section is keyed by its heading** | Read a corpus row: `prompt` is the key, `response` is the prose. | A heading carries a median 2.2% of its section's vocabulary and 18% of headings share no content word with their prose, so a question about the content cannot reach it by key similarity alone. |
| **Which branch actually answers** | Send the request and read `intent_diagnostics.answer_branch`. | The answer is chosen by an `if/else` chain. An earlier arm that matches and returns `None` ends the chain — a later arm never runs, however well it scored. |
| **What a declined route scored** | Read `recall_derived_best_score` and `recall_derived_best_margin`. | `recall_derived_score` is populated only on success. Without the `best_*` fields every abstention looks identical and there is nothing to tune against. |

## The path

1. **Exact match** (`raw_is_exact`) — the request is a trained key verbatim.
2. **Labelled composition** — `composition_features` from the instruction
   intent table; drives manifest and component routes.
3. **`recall_derived`** — runs only when no label applied and the request is
   not an exact hit. Scores candidates by char-motif Dice, gated on
   `UNLABELED_RECALL_MIN_SCORE` and a margin.
4. **`no_answer`** — honest abstention.

`recall_derived` is the arm that serves any request the intent table has no
word for, which is every non-programming question.

## Scoring contract

- **Selection** uses motif rarity: posting lists are sorted by length and the
  rarest are kept. This is where discrimination belongs.
- **Scoring** is unweighted Dice over shared motifs. Do not also weight the
  score by rarity — see the rejected entry below.
- **Margin** scales with certainty: `min_margin * (1 - score)`. A weak match
  faces the full requirement; a near-certain one does not.
- **A duplicate is not a rival.** A textbook prints the same section in
  several places, so `runner_score` skips candidates whose decoded bytes
  equal the winner's. Agreement between duplicates is evidence FOR an answer.

## Measured and rejected

Do not re-attempt these without new evidence.

| Change | Result |
|---|---|
| Route on learned atoms via `binding_feature_atom_index` | Selects hubs. `clusterStatus` (a Python function) and a statistics exercise page won every physics question at 0.49–0.89 under `max`, product, coverage, and `sqrt(coverage x precision)`. Atoms are bytes. |
| Weight motifs by `1 / ln(postings)` in the score | Margins sharpened ~37x but exact matches collapsed: "the first law of thermodynamics" fell from 1.0 to 0.084. Weighting the denominator penalises short keys that match perfectly. |
| Plain containment (`overlap / min(len)`) | Lifts wrong answers along with right ones. |
| Length-blended Dice | "pH a measure of acidity" scored 0.473 against a correct 0.469 for an entropy question. |
| IDF-weighted containment | "The SI unit for pressure is the pascal" scored 0.570 for a question about force. |
| Query reduced to content words only | 3 of 6 questions cleared the floor, all on wrong sections. |
| Lower `UNLABELED_RECALL_MIN_SCORE` | Answerable and unanswerable ranges overlap (0.048–0.113 against 0.076–0.113). Admitting the former admits "summarise the internal memo that was never written". |

## Char motifs are not the weak link

Verified 2026-08-23: 3-grams already bridge morphology — `wooden`/`wood` and
`colder`/`cold` score 0.67, `electrons`/`electron` 0.92. When a question and
the section answering it share few motifs, they genuinely use different words
for the same idea. That gap is semantic and will not close by changing the
string metric.

## Measurement discipline

- **Sample more than once.** A live curriculum trains underneath the brain,
  so the same question can answer correctly and then abstain minutes later.
- **A single passing probe is not verification.** One post-deploy sample
  passed and a ten-sample repeat scored 0/10.
- **Check that the binary you tested is the one you built.** Resolve
  `/proc/<pid>/exe`; `pgrep -f w1z4rd_brain_server` also matches the
  supervisor, whose command line contains the binary path.
