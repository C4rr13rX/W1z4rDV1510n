# Market Brain Metric Fusion Audit

## Rule

Historical peak metrics are observations, not interchangeable components.  A
candidate's accuracy, balanced accuracy, MCC, coverage, expectancy, profit
factor, calibration error, and drawdown remain attached to its candidate ID,
evaluation signature, fold depth, learner, and complete companion summary.
They must never be copied into a synthetic "best of" score.

Run the repeatable audit with:

```powershell
python scripts/market_metric_fusion_audit.py `
  --state-dir runtime/market-evolution `
  --output runtime/benchmarks/market-metric-fusion-audit-20260813.json
```

The report separates all historical observations, candidates that completed
all requested protected folds, the current dataset signature, and current
full-retention candidates.  ECE and drawdown are minimized; the other listed
metrics are maximized.  Records with no evaluated fold are excluded.

## 2026-08-13 finding

The apparent top line did not belong to one retained model.  The strongest
historical one-fold observations included 67.49% accuracy, 66.83% balanced
accuracy, MCC 0.339, expectancy 0.003649, and profit factor 1.377, but those
metrics came from different candidates (or different metric winners) on the
old `2cbca6bd9e1ddef8` signature and did not complete three folds.  Across the
entire ledger, the full-retention maxima were only 56.94% accuracy, 55.96%
balanced accuracy, MCC 0.129, expectancy 0.000135, and profit factor 1.012;
these also did not all belong to one candidate or signature.

The discrepancy was therefore provenance plus selection depth, not evidence
that one architecture had retained every advertised maximum.  Current live
numbers must come from the current-signature full-retention section of the
generated report and the current state/champion report.

## Causal fusion

The previous `extra_trees_hybrid` claimed to combine the classifier champion
with a return-ranking specialist, but it copied only the specialist's cutoff.
Both fitted submodels still used the champion's tree coordinates and recency.
That was an incomplete architectural fusion.

Evolution schema 14 gives the return-selection submodel independent tree
count, leaf capacity, minimum leaf support, and recency half-life.  A hybrid is
launched only when its profitable selective point and coverage anchor share
the same signed return-tree phenotype and the champion's exact feature view.
It preserves the classifier coordinates, copies the return coordinates and
coverage cutoff, records every distinct source candidate as a parent, and
starts unscored.  It receives no inherited metric: it must pass the normal
fresh prescreen, all protected folds, unseen-asset, profitability,
calibration, neural, ghost, and live gates.

This is the only defensible way to "fuse the highs": combine a causal model
mechanism, then measure the resulting model from scratch.
