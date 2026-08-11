# Market Brain Ghost-Trading Admission Contract

## Objective

The market brain predicts a future return state for a named instrument and
horizon from causal OHLCV, volume/flow, volatility, regime, cross-market, and
news streams.  The caller expands the symbolic return state around the current
price to obtain a price forecast and interval.  No held-out observation may be
admitted to learning until its horizon has elapsed and its outcome is known.

Ghost trading is an evaluation stage, not evidence that real-money trading is
safe or profitable.  Real-money admission requires a separate operational and
risk review.

## Leakage controls

- Split by event time, never randomly.
- Purge at least one complete prediction horizon between train and evaluation.
- Fit bucket boundaries, calibration, confidence thresholds, and every other
  statistic on training data only.
- News is eligible only when its publication timestamp is at or before the
  prediction timestamp.  Ingestion/fetch time is not a substitute.
- Overlapping copies of the same pair are deduplicated before splitting.
- A final untouched period is evaluated once after configuration selection.
- Read-only prediction must leave the brain tick and stable topology unchanged.

## Starting floor

All values are measured after estimated spread, slippage, and fees.

| Gate | Minimum for ghost trading |
|---|---:|
| Walk-forward folds | 3 |
| Covered directional observations per fold | 200 |
| Directional coverage | 70% |
| Directional balanced accuracy | 55% |
| Directional accuracy | 58% |
| Improvement over best causal baseline | 5 percentage points |
| Matthews correlation coefficient | 0.15 |
| Expected calibration error | 0.10 maximum |
| Net expectancy per acted prediction | Positive in every fold |
| Profit factor | 1.20 |
| Maximum simulated drawdown | 15% maximum |
| p95 warm inference | 250 ms on the training host |
| p95 cold neuron-paged inference | 750 ms on the training host |
| Stable topology/tick during evaluation | Required |

Directional coverage is actionable up/down predictions divided by all
directional outcomes; parsed abstentions or `sideways` predictions do not
count as coverage.  Drawdown uses timestamp-ordered, equal-weight portfolio
returns so simultaneous correlated markets do not each receive an implicit
100% capital allocation.

The working target is deliberately above the floor: at least 62% directional
accuracy, 58% balanced accuracy, 0.25 MCC, profit factor 1.35, and positive net
expectancy across every fold and the untouched final period.  Accuracy may not
be raised by predicting only a dominant class; coverage, balanced accuracy,
MCC, and the baseline margin are co-equal gates.

## Experience admission

After a prediction horizon closes, record the prediction, causal feature
streams, realized return, fees/slippage, and resulting error.  Successful and
failed experiences are both eligible for training.  Admit them transactionally
only after the outcome is externally confirmed, then re-run protected
walk-forward and retention gates before replacing the accepted brain.

## Evolution evidence discipline

- Reserve each generation for at least one controlled, single-coordinate
  descendant of the best candidate that completed every requested fold. A
  high-scoring one-fold frontier must not consume the entire search budget.
- Evaluate that controlled descendant before expensive speculative models on
  constrained worker pools so an observe-and-adjust cycle yields actionable
  evidence within its time budget.
- When a confidence change improves accuracy but worsens profit factor or net
  expectancy, mirror the change across the full-fold champion before bisecting
  the boundary. Never march farther into a known economics regression merely
  because raw accuracy increased.
- A profitable partial-fold frontier may contribute one exclusive causal
  feature program at a time to the full-fold champion. Wholesale crossover is
  not evidence that the transferred feature caused an improvement. The
  resulting candidate must repeat all coverage, calibration, unseen-asset,
  expectancy, and profit-factor gates before it can replace the champion.
- Persist every evaluated phenotype, including failures, under the evaluation
  signature. Restarts and later generations must reuse that signed evidence
  rather than silently repeating a disproven experiment.
- When a return-magnitude model finds a profitable selective tail but loses
  direction at the coverage floor, preserve the validated direction model and
  use an independently fitted magnitude pool only for abstention ranking. Such
  a hybrid may launch only after signed evidence identifies both the selective
  profit region and a coverage-clearing threshold, and it still must pass all
  folds before either component is admitted.
- Each materially different profitable near-coverage learner class must retain
  an independent repair lane. A later multiscale or tree frontier may not
  silently replace the strongest ordinary regressor's experiment; descendants
  bracket the smallest coverage-clearing quantile and preserve protected-fold
  reversals as signed evidence. If a quantile descendant produces an identical
  acted set and coverage, the next bounded step must escape that score plateau
  rather than treating a numerically different genome as new behavior. An
  identical descendant must not replace the upper frontier endpoint merely
  because its quantile is lower; doing so destroys the ancestry that proves
  the plateau and resets automatic correction.

## Stop conditions

Do not start ghost trading if results depend on one asset, one market regime,
one fold, news published after the decision, unreported abstentions, or a
baseline that was computed with weaker information than the brain received.
