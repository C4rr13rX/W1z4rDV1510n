# Perpetual Market-Brain Evolution

## Purpose

The market evolution loop continuously searches for a better causal feature
configuration and Wizard-brain configuration without allowing a lucky fold,
data leakage, or a surrogate classifier to replace the accepted brain.
Generated corpora, candidate brains, logs, and ledgers live below `runtime/`
or the CoolCryptoUtilities data directory and remain ignored. The scripts,
identities, contracts, and experiments are tracked.

Ghost trading remains an evaluation stage. Neither a surrogate pass nor a
micro-brain pass authorizes real-money trading.

## Live pipeline

```text
causal datasets
  -> feature/expression genomes
  -> purged walk-forward + complete unseen-asset screening
  -> isolated Wizard micro-brain smoke gate
  -> three-fold Wizard gate
  -> one untouched final period
  -> promotion candidate (never automatic live trading)
```

`scripts/market_evolution_service.py` is the authoritative perpetual service.
It loads the data once, runs a bounded population with two workers by default,
persists every candidate and failure, retains elites, and produces new genomes
with crossover and mutation. It reserves a final interval that fitness never
sees. The evolved genes cover:

- causal feature selection and explicit derived relationships;
- price, flow, futures/spot basis, premium, funding, market breadth, and news;
- confidence selectivity, constrained so abstention cannot evade coverage;
- classifier screening parameters used only as a cheap information test;
- Wizard binding/concept thresholds and supervised presentations used by the
  isolated neural gate.

`scripts/market_evolution_brain_gate.py` constructs fresh Wizard micro-brains
from `brains/market_predictor_evolution.identity.toml`. Selected numeric
features remain character/atom grounded as deterministic named magnitude
buckets in pool 15. Horizon and instrument context fire in separate pools,
and the future outcome remains a separate action pool. No classifier state is
copied into the Wizard brain.

The alternate `scripts/market_evolution_supervisor.py` is a process-isolated
experimental harness for the immutable `MarketGenome` schema. It deliberately
uses `runtime/market-evolution-process-isolated/`; do not point it at the live
service state directory.

## Data currently admitted to evolution

- canonical deduplicated hourly OHLCV representatives;
- buy/sell volume flow from the existing corpus;
- checksum-verified Binance public spot and perpetual hourly archives;
- quote volume, trade count, taker-buy flow, futures/spot basis, premium, and
  funding, aligned without using values published after the decision;
- same-time cross-sectional breadth computed from all assets before filtering
  future direction labels;
- 11,909 deduplicated historical crypto-news articles, bounded by publication
  timestamp and filtered away from unrelated generic advisories.

## Admission rules

Fitness uses the weakest known-asset or unseen-asset section across every
walk-forward fold. A candidate cannot pass unless each section satisfies the
full contract in `docs/MARKET_BRAIN_GHOST_TRADING_ADMISSION.md`, including 200
acted observations, 70% actionable coverage, baseline margin, balanced
accuracy, MCC, calibration, after-cost expectancy, profit factor, and
equal-weight portfolio drawdown. The untouched final period is never part of
genetic fitness.

The service writes:

- `runtime/market-evolution/state.json` — atomic restart coordinate;
- `runtime/market-evolution/events.jsonl` — append-only decisions;
- `runtime/market-evolution/candidates/` — every genome and result;
- `runtime/market-evolution/champion.json` — current surrogate champion;
- `runtime/market-evolution/brain-gate-reports/` — neural smoke/full evidence.

Failed and inaccurate candidates remain evidence; they are not layered onto
the accepted brain.

## Operations

The current launch is intentionally modest so other workstation activity can
continue:

```powershell
python scripts/market_evolution_service.py --population 8 --workers 2
```

Create `runtime/market-evolution/STOP` for a cooperative stop after the active
generation. Removing that file and launching the same command resumes from the
atomic state. Status is read from `state.json`, `champion.json`, and the tail of
`events.jsonl`.

The service automatically starts one isolated neural smoke gate at bounded
intervals. A full three-fold gate can be run with the same gate script by
omitting the smoke overrides. Only a complete three-fold pass is eligible for
the untouched-final evaluation.
