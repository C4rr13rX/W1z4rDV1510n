# Island Architecture — Empirical Validation (Phase 4)

Production brain (no islands) lost canonical specificity during a 15K-row
code ingest: "dog is a kind of" started returning "sport" instead of
"animal".  Dense fabric homogenized old training across the same global
weight space as the new training.

## Test setup (commit 3680791)

Fresh fabric on :8097, W1Z4RD_DOMAIN_MODE=1 active.

| Step | Action |
|---|---|
| 1 | POST /set_domain { domain_id: 1 }                |
| 2 | Train greetings_001, 8 reps, 64 cross-pool pairs |
| 3 | POST /sleep, baseline benchmark greetings        |
| 4 | POST /set_domain { domain_id: 2 }                |
| 5 | Train categorical_unified_001, 8 reps, 55,776 cross-pool pairs |
| 6 | POST /sleep, RE-benchmark greetings              |
| 7 | Benchmark categorical                            |

## Result

Domain histogram after both training passes:
  pool 1 (text):    1,634 @ dom=1  +  547,709 @ dom=2
  pool 4 (action):  2,279 @ dom=1  +   28,343 @ dom=2
  pool 0 (binding): 6,927 @ dom=0

Greetings benchmark before categorical:  FAIL PASS FAIL FAIL  (1/4)
Greetings benchmark after  categorical:  FAIL PASS FAIL FAIL  (1/4)

IDENTICAL.  Categorical training had zero observable effect on
greetings recall.  The same single prompt passes; the same three fail.

Categorical benchmark: 3/6 (piano, triangle, speak pass) — matches
prior fresh-fabric trajectory, so the domain gate did not damage
in-domain learning.

## Implication

When neurons carry domain tags and the cross-pool wiring path respects
them, training one domain does not perturb another domain's weights.
Continuous online training with multiple domains becomes safe — old
domains stay intact as new ones are added.
