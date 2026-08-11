# Perpetual Market-Brain Evolution

`scripts/market_evolution_service.py` is the canonical restart-safe evolution
loop. It loads the causal corpus once, runs a bounded parallel population,
retains every result and failure, keeps elites, and creates mutations and
crossovers until `runtime/market-evolution/STOP` appears.

While running, the controller fingerprints OHLCV, derivatives, and news at
generation boundaries (every 30 minutes by default). When new evidence
arrives, it reloads the corpus, retains the heritable genomes, clears every
corpus-dependent fitness and neural score, and revalidates all survivors.
Neural-gate results that finish against an older corpus fingerprint are logged
as stale and cannot enter selection or the untouched-final queue. Use
`--dataset-refresh-seconds 0` only for an intentionally frozen experiment.

The genome evolves feature and derived-relationship selection, confidence
scope, screening parameters, Wizard binding/concept thresholds, and supervised
presentation density. Screening parameters are only a cheap information test:
no classifier state is copied into the Wizard brain. Numeric relationships
enter neural validation as deterministic named character/atom magnitude frames
through `brains/market_predictor_evolution.identity.toml`.

Evolution schema 6 evolves bounded causal feature programs and a recency
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
return regression, broad-market-plus-instrument-residual regression, and
extremely randomized trees. The decomposed family learns two supervised heads
and genetically weights their recombination; future market aggregates are
targets only and never enter a decision feature. This is an information and
relationship-discovery gene only: learner parameters and fitted weights are
never installed in the Wizard brain.

The same genome now describes a bounded developmental graph: an observable
regime feature and one-to-three specialist regions, a 1/6/12/24-hour temporal
curriculum, broad-market versus instrument-residual weighting, and explicit
regime, arbitration, and realized-experience pools. The Wizard brain still
learns the associations and motifs; the genome supplies scopes and wiring, not
answers. Small isolated neural gates contribute a bounded tie-breaking score
inside a candidate's protected fold stage, so neural evidence can steer brain
genes but can never promote a candidate or outrank earning another complete
protected fold.

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

Before evaluation, the gate performs non-pruning neuron-wise sleep and a
checkpoint, then requires zero resident terminals. That maintenance request
has a bounded 15-minute client timeout because million-terminal experimental
brains can take several minutes to serialize; ordinary inference retains its
short timeout. A timeout never converts into a score or bypasses lifecycle
evidence.

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
- `accuracy_improvements.jsonl` contains only strict new accuracy highs for
  complete protected folds or isolated Wizard-brain gates (separated by exact
  corpus signature and evidence source); `accuracy_best.json` stores the
  restart-safe baselines used to decide whether a high is new;
- `candidates/` retains genomes and results;
- `champion.json` is the current surrogate champion;
- `brain-gate-reports/` and `brain-gates/` hold neural evidence/state.

`scripts/market_evolution_watchdog.py` is the lightweight operational owner.
It waits without loading the dataset until the configured free-RAM floor is
met, starts the evolution worker, publishes `supervisor_status.json`, and
restarts unexpected worker exits. `launch_revenir.ps1` starts this watchdog
instead of starting the memory-intensive worker directly.

The causal feature matrix is fingerprinted and cached at
`runtime/cache/market-evolution-dataset-v4.joblib`; primary OHLCV files,
supplemental derivatives, news, horizon, stride, seed, and evolution schema all
participate in cache invalidation. Use `scripts/market_evolution_validate.py`
to rerun a micro-sweep winner on the complete protected three-fold contract
before spending time on its Wizard gate.

Create `runtime/market-evolution/STOP` for a cooperative stop after the active
generation. Remove it before restarting. The service resumes from `state.json`.
Ghost trading remains evaluation, and the evolution loop never authorizes
real-money trading.

Schema 3 filters generic software advisories out of the global crypto-news
stream and exposes causal event-regime counts (regulation/macro, security,
institutional, liquidation, exchange, stablecoin, network, and whale) plus
asset-specific sentiment acceleration. These are neutral observations: their
directional meaning must be learned and survive every protected fold.

Schema 6 retains bounded future returns at multiple horizons as supervised
outcomes only. They are never decision features. Presentation density is
shared across horizons, preventing the temporal curriculum from multiplying
neural growth. Confirmed ghost outcomes can be encoded as after-cost success
or failure in the experience pool and are evaluated by the same weakest-regime
contract.

Schema 7 evolves a bounded conservative multiplier on regression temperature.
It cannot change a score's sign or confidence rank, so it cannot manufacture
accuracy or choose easier observations; it only prevents a directionally
useful lineage from expressing confidence unsupported by protected future data.

Schema 8 composes regime routing with market/residual decomposition as a
separate preserved learner species. Each observable-state expert independently
learns broad-market motion and instrument-specific residual motion; the evolved
market weight recombines them. Future returns remain supervised targets only,
and the router sees only decision-time context.

Selection retains the champion plus every learner species. A positive-expectancy
directional frontier above the multimetric prescreen floor but outside the calibration gate
also produces monotonic 4x and 8x temperature descendants. Those descendants
preserve its features and sign model exactly, allowing confidence honesty to be
tested without waiting for unrelated mutations or losing the lineage.

When a lineage reaches at least two protected folds but fails a later regime,
the controller also reserves two repair descendants in the next generation.
They preserve the causal feature lineage while testing sharply shorter memory
and two forms of observable-state routing (regime regression and
regime-routed market/residual decomposition). This turns the failed protected
fold into evolutionary pressure without exposing the untouched final period or
weakening any gate. Random mutation, crossover, learner-species preservation,
and reflexivity variants continue alongside these targeted descendants.

`status.json` is an atomic live heartbeat. It changes to `evaluating` at the
start of a generation, updates after every completed candidate with the latest
protected summary, and changes to `generation_complete` after the durable
restart coordinate is written. A cooperative stop is honored even while the
service is waiting for the configured free-memory reserve.
Transient Windows sharing violations are retried and heartbeat publication is
best-effort: a dashboard, indexer, or antivirus reader cannot terminate the
durable evolution loop. After a controller restart, an already-running neural
gate is detected and allowed to finish before another gate is launched.
An isolated neural hypothesis gets at most two durable automatic launches
without producing a report. Repeated report-less timeouts assign only the
ordinary negative neural advisory score and quarantine that hypothesis from
automatic relaunch; they cannot reject its statistical phenotype, alter a
protected metric, or affect live admission. A later explicit report clears the
retry ceiling, so repaired neural infrastructure can supply fresh evidence.

Schema 9 adds competitive reflexivity as an independent causal feature pool.
It observes bounded participant direction, consensus, disagreement, leverage
intensity, price alignment, and pressure acceleration from decision-time flow,
basis, funding, and price state. These are neutral relationships, not a rule to
follow or fade the crowd. The brain learns continuation, squeeze, reversal, and
abstention motifs from subsequent after-cost outcomes; the GA decides when this
pool and its cross-pool bindings participate.

Schema 10 makes specialist-pool topology heritable. A genome may isolate up to
eight selected raw or evolved causal features into reproducibly named sensory
pools, mutate their membership and concept-emergence thresholds, remove them,
or inherit them through crossover. Generated identities assign collision-free
pool IDs and fire those specialists in the same observation as price, flow,
news, regime, arbitration, and outcome classes. The fabric's existing
co-firing rule then grows and repeatedly strengthens atom- and concept-level
cross-pool terminals. Because statistical surrogates cannot measure neural
topology, the strongest new-pool hypothesis is explicitly queued for an
isolated Wizard-brain gate. Old schema states load with an empty topology gene
and immediately receive one bounded topology experiment.

The reliability scheduler also treats repeated identical protected outcomes as
evidence. When
six or more orientation-aware reliability trials from one feature pool collapse
to the same accuracy, balance, MCC, coverage, expectancy, and profit-factor
signature, that pool/threshold dimension is declared locally uninformative.
The scheduler stops infinitesimal quantile bisection and tests compact
flow/derivatives, news/regime, and relative-trend correctness specialists. This
keeps the perpetual search causal and novel without relaxing any promotion,
coverage, calibration, unseen-asset, profitability, or ghost-only gate.

Frontier repair and champion exploitation have separate budgets. High-accuracy
one-fold frontiers may continue bounded temporal-reversal research, while one
ordinary population slot performs an evidence-backed coordinate search around
the fully evaluated three-fold champion. Each descendant changes exactly one
cutoff, tree-complexity, recency, or calibration coordinate, and signed prior
phenotypes are never repeated. A partial-fold frontier cannot enter this path.

An ordinary histogram regressor can produce a discontinuous coverage frontier:
nearby confidence quantiles select the same tied tree leaves until one step
admits an entire leaf. Once signed descendants prove both a plateau above the
coverage floor and a lower-quantile profitability reversal, the controller
reserves one `continuous_rank_regressor` descendant. It preserves the fitted
return model, directional probability, prediction, features, and threshold,
and adds only a deterministic `1e-10` decision-time feature projection to the
abstention score. That perturbation cannot reorder materially different model
confidences; it makes observations inside an exactly tied leaf individually
selectable. If the same-threshold ablation preserves the frontier, subsequent
coverage bisection remains inside this ranked species; reverting those children
to the ordinary regressor would recreate the proven leaf plateau. Each child
records both its immediate ranked parent and the durable coverage frontier so
later reserved search lanes cannot overwrite its protected population slot. The
descendant must still pass every ordinary coverage,
profitability, calibration, known/unseen-asset, protected-fold, neural, and
ghost-only admission gate. Historical descendants suppress repeated probes.
