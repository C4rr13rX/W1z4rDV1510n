# Perpetual Market-Brain Evolution

`scripts/market_evolution_service.py` is the canonical restart-safe evolution
loop. It loads the causal corpus once, runs a bounded parallel population,
retains every result and failure, keeps elites, and creates mutations and
crossovers until `runtime/market-evolution/STOP` appears.

The genome evolves feature and derived-relationship selection, confidence
scope, screening parameters, Wizard binding/concept thresholds, and supervised
presentation density. Screening parameters are only a cheap information test:
no classifier state is copied into the Wizard brain. Numeric relationships
enter neural validation as deterministic named character/atom magnitude frames
through `brains/market_predictor_evolution.identity.toml`.

Evolution schema 2 also evolves bounded causal feature programs and a recency
half-life. Programs combine only same-moment or prior-derived inputs with
canonical operations (addition, subtraction, product, stabilized ratio,
absolute gap, signed square-root product, regime gate, and bounded nonlinear
mix). Asset-relative rolling normalization uses only observations preceding
the decision row; cross-sectional ranks use only markets observable at the
same timestamp. The screening fit balances asset and outcome exposure and
decays old observations according to the evolved half-life. These genes are
rendered into named atom-grounded frames for the Wizard gate rather than
copying the screening model.
The screening learner family may mutate among directional gradient boosting,
return regression, and extremely randomized trees. This is an information and
relationship-discovery gene only: learner parameters and fitted weights are
never installed in the Wizard brain.

## Protected stages

```text
causal data
  -> genetic feature/configuration screening
  -> fresh isolated Wizard micro-brain smoke gate
  -> three-fold Wizard gate
  -> one untouched final period
  -> promotion candidate (never automatic live-money authorization)
```

Every genome fits before a purged calibration interval. Its attention threshold
is selected on calibration data only, then it is scored on future familiar
assets and completely withheld assets. Fitness uses the weakest section, never
an average or best fold. The final interval is excluded from genetic fitness.
The evaluation end is the newest timestamp supported by at least 75 percent of
eligible instruments, after which the active instruments are deterministically
split into trained and entirely held-out assets. This prevents one stale market
from forcing every fold far into the past while preserving a true instrument
holdout.

The complete floor is defined in
`docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md`: observation count, actionable
coverage, accuracy, balanced accuracy, MCC, baseline margin, calibration,
after-cost expectancy, profit factor, and equal-weight portfolio drawdown are
co-equal gates.

## Current causal data

- canonical hourly OHLCV and buy/sell flow;
- checksum-verified Binance public spot and perpetual archives;
- quote volume, trade count, taker flow, futures/spot basis, premium, funding;
- same-time market breadth built before any future-label filtering;
- 11,909 deduplicated historical news articles bounded by publication time.

## Neural validation

`scripts/market_evolution_brain_gate.py` trains each fold in a new Wizard brain.
Horizon, instrument, evolved causal relationships, and outcome have distinct
pools. A smoke result cannot promote a brain; at least three neural folds are
required before the untouched-final test.

The process-isolated experimental alternative,
`scripts/market_evolution_supervisor.py`, uses the separate
`runtime/market-evolution-process-isolated/` directory. Do not point it at the
canonical service state.

## Operation

The workstation-friendly launch currently in use is:

```powershell
python scripts/market_evolution_service.py `
  --population 8 --workers 2 --brain-gate-every 1
```

Generated state is ignored below `runtime/market-evolution/`:

- `state.json` is the atomic restart coordinate;
- `events.jsonl` is the append-only decision ledger;
- `candidates/` retains genomes and results;
- `champion.json` is the current surrogate champion;
- `brain-gate-reports/` and `brain-gates/` hold neural evidence/state.

The causal feature matrix is fingerprinted and cached at
`runtime/cache/market-evolution-dataset-v2.joblib`; primary OHLCV files,
supplemental derivatives, news, horizon, stride, seed, and evolution schema all
participate in cache invalidation. Use `scripts/market_evolution_validate.py`
to rerun a micro-sweep winner on the complete protected three-fold contract
before spending time on its Wizard gate.

Create `runtime/market-evolution/STOP` for a cooperative stop after the active
generation. Remove it before restarting. The service resumes from `state.json`.
Ghost trading remains evaluation, and the evolution loop never authorizes
real-money trading.
