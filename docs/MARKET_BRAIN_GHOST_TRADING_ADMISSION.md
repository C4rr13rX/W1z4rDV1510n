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

## Stop conditions

Do not start ghost trading if results depend on one asset, one market regime,
one fold, news published after the decision, unreported abstentions, or a
baseline that was computed with weaker information than the brain received.
