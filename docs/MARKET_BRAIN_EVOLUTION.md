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

Do not lower the workstation's `3.5 GiB` free-RAM floor to push through a
stall. Both the watchdog's launch gate and the service's generation-boundary
gate count consecutive low-memory samples. After four samples they may ask
Windows to evict reclaimable resident pages from a process only when all of
these conditions hold: its executable name exactly matches
`w1z4rd_node.exe`, its command contains `api --addr 127.0.0.1:8090`, it is the
only match, and its loopback `/health` endpoint passes before the operation.
Health is checked again afterward and attempts are rate-limited for 15
minutes. The operation never terminates the node, changes a genome, relaxes an
evaluation threshold, or enables trading. It also does not reduce private
commitment, so repeated pressure remains an architecture signal requiring
bounded neural retrieval or neuron-scoped sleep/checkpoint rather than more
aggressive trimming. Every attempted action is appended as
`memory_reclamation_attempt` evidence.

The local long-lived market node still uses the legacy monolithic
`brain-data/brain.bin` plus separate cold-tier shards. Use the staged migration
preflight before scheduling its conversion to the same neuron-addressable
`.wbrain` architecture used by the programming brain:

```powershell
& scripts/prepare_market_brain_wbrain_migration.ps1
```

The default invocation is read-only and prints exact RAM, disk, source-size,
and process-ownership evidence. `-Execute` is accepted only when one exact
healthy `127.0.0.1:8090` node owns the legacy state and every resource gate
passes. It builds the current resumable migrator, requires a successful neural
checkpoint, gracefully stops that exact node, hard-links the immutable settled
source into an isolated runtime directory, and monitors the conversion under
the private-memory circuit breaker. Its `finally` path always restarts and
health-checks the untouched legacy production directory. A completed staged
container is not promoted automatically; cold-open, behavior, memory, Django,
production-manager, and ghost-only gates must pass before a separate atomic
switch-over. Partial staging remains resumable and can never shadow the live
`brain.bin`.

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

The learned genome-outcome pool also has an explicit plateau response. Its
chronological validation must beat the historical-mean baseline before it can
advise reproduction. Normally it alternates balanced-safe and profit-discovery
acquisition. Once the full-fold champion is unchanged for 24 generations, it
adds an accuracy-discovery turn every third generation, rewarding predicted
directional accuracy, balanced accuracy, and MCC while strongly penalizing
predicted coverage below 60 percent or normalized profit factor below 1.0. The
pool still owns exactly one advisory slot and cannot supply fitness or admission
evidence; the selected child remains unevaluated until every protected fold and
unchanged safety gate signs its result.

Tree capacity is refined as a finite, evidence-backed integer coordinate. When
three otherwise identical tree phenotypes on the same evaluation signature
bracket a strong interior result, the controller compares their combined
direction, MCC, economics, and coverage rather than continuing the original
coarse four-leaf walk past a known weaker endpoint. It tests the midpoint toward
the stronger neighbouring endpoint first, then the other midpoint, and stops
when the adjacent integer values are already signed. This refinement owns at
most one ordinary reproduction slot, cannot mix results across dataset
signatures, and supplies no fitness or admission evidence by itself.

The first live refinement under signature `11a90...` narrowed the coarse
8/12/16-leaf bracket through leaf 10 and then leaf 11. Leaf 11 became the
strongest signed local point at 61.59% directional accuracy, 62.46% balanced
accuracy, MCC 0.251, positive expectancy, and profit factor 1.151. Because
10/11/12 were then adjacent, the adviser terminated instead of spending a slot
on leaf 14. That result remains research evidence only: one protected fold and
roughly 51.43% minimum coverage do not satisfy admission.

Return-tree threshold repair must preserve that exact signed phenotype. The
coverage controller groups confidence results by every predictive coordinate
except the confidence quantile, so a weak lower cutoff from a different leaf
capacity cannot become the opposite endpoint of a bracket. When a profitable,
positive-expectancy return lineage is selective, its next cutoff child inherits
its leaf capacity, calibration, features, programs, and topology and names that
signed candidate as its parent. Only after this same-phenotype curve is
exhausted may the scheduler fall through to a hybrid or a new topology. This
lets useful coordinates compose without granting partial-fold evidence any
fitness or admission authority. While such a profitable positive-expectancy
curve remains actionable, it owns the one shared specialist slot ahead of new
feature-program transfer; once its finite proposals are signed, program search
automatically resumes.

The controller also retains signed failures on that exact return-tree curve.
They are negative boundary evidence, not reusable fitness: an unsafe cutoff is
included in the structure-local bracket and in canonical phenotype
deduplication, so it cannot be proposed again under a fresh lineage ID. The
first live composed cutoff, candidate `053b48cdec3e2b48` at quantile
`0.1454066`, raised coverage to 63.50 percent but fell to 43.49 percent
directional accuracy, profit factor 0.641, and negative expectancy. It was
correctly rejected. The next finite experiment must therefore bisect between
that signed unsafe cutoff and the profitable selective endpoint rather than
repeat either endpoint. Return-tree telemetry counts descendants of retained
independent frontiers as return-tree candidates even when the current
classifier is not their immediate parent.

After deployment, the controller found a tighter signed 12-leaf curve and
tested its midpoint as candidate `6e207432aecc93f0` at quantile `0.1697279`.
It improved to 58.33 percent accuracy, 59.55 percent balanced accuracy, profit
factor 1.029, positive expectancy, and 56.86 percent coverage. It still missed
the protected accuracy, MCC, acted-observation, and coverage floors and was
therefore rejected. This is useful boundary evidence, not a promotion.

A profitable return cutoff that comes within one percentage point of required
coverage but has already fallen below 60 percent directional accuracy is a
signed Pareto boundary. The controller stops lowering that scalar: doing so
would predictably exchange the accuracy objective for marginal coverage. Its
next bounded trial keeps the exact cutoff, leaf capacity, calibration,
features, and programs, and increases only the return tree's minimum leaf
support. Every topology coordinate is derived from the signed return parent,
not reconstructed from the classifier champion. Generation 1171 established
this trigger on the 11-leaf lineage at 58.64 percent accuracy, 59.90 percent
balanced accuracy, MCC 0.203, PF 1.0068, positive expectancy, and 59.73 percent
coverage. The smoothing child remains ordinary research evidence and must pass
all unchanged protected folds and admission gates.

This near-boundary topology repair has priority across the whole return-tree
family, not merely within one leaf-capacity curve. Otherwise an unused cutoff
from a more distant sibling can consume the shared specialist slot forever
while the closest Pareto boundary is never repaired. Candidates are ordered by
coverage proximity, then accuracy and economics; only after their finite
topology coordinates are signed does scalar search on sibling curves resume.

Topology coordinates are themselves causal branches. If the first increase in
minimum leaf support causes a severe reversal (accuracy below 50 percent or PF
below 0.8), larger minimum-leaf settings are not attempted. The controller
records that signed failure and advances to leaf capacity while restoring the
parent's minimum support and cutoff. Candidate `6f2bcbcaf7cea5e0` supplied this
evidence: support `8 -> 12` crossed coverage at 61.09 percent but collapsed to
42.93 percent accuracy, PF 0.626, and negative expectancy. It was rejected; the
next live-ledger proposal holds q `0.1611117`, restores support 8, and changes
only leaf capacity `11 -> 9`.

Signed coordinate descendants are recovered through lineage closure, not only
same-structure cutoff keys. This prevents a restart from forgetting that a
topology branch failed. The ledger already contained the leaf-9 result at the
same cutoff (`42%` accuracy and PF `0.66`), so smaller capacity is now stopped
as well. The next bounded proposal restores leaves 11 and support 8, holds q
`0.1611117`, and changes only recency half-life to 75 percent of the parent.

That recency ablation produced candidate `5ad1805e85eb837c`: 59.87 percent
accuracy, 60.51 percent balanced accuracy, MCC 0.210, PF 1.048, positive
expectancy, and 57.92 percent coverage. It remained rejected because coverage
was 2.08 percentage points below prescreen and ECE was 19.71 percent. A useful
topology interaction must not be lost merely because the family-wide Pareto
guard blocks broad cutoff descent. When a new topology retains at least 59.5
percent accuracy, 59 percent balanced accuracy, positive expectancy, PF 1.0,
55 percent coverage, and prescreen-valid ECE, the controller may test one
bounded 0.5--1.5 percentage-point cutoff recovery derived from that exact
topology. A result below those quality bounds stops the interaction; it does
not relax any fold, unseen-asset, calibration, profitability, ghost, or live
gate.

Live candidate `f853c5684321bb34` exercised that interaction on the stronger
12-leaf parent. Moving q from `0.1820959` to `0.1670959` raised coverage from
55.05 to 58.22 percent and retained PF 1.042 plus positive expectancy, but
accuracy fell from 60.00 to 58.60 percent. The child therefore failed
prescreen and, because it crossed below the 59.5-percent continuation bound,
cannot trigger another cutoff descent in this lane.

Plateau retirement applies to local refinement as well as the coarse pool
schedule. Once a reliability feature pool is classified inert or
outcome-plateaued, it cannot win the later best-score selection merely because
one of its historical points outranks the remaining active pool. This closes a
live loop in which version-8 `combined` reliability produced the same protected
outcome 99 times at numerically distinct quantiles. Local refinement now ranks
only active pools; all statistical and neural gates remain unchanged.
Generation 1196 supplied the live proof: the reserved child switched to
`flow_news` candidate `32d723e59b0e1d14` instead of another `combined` alias.
It raised coverage to 74.51 percent but scored only 52.69 percent accuracy, PF
0.936, and negative expectancy, so prescreen rejected it. That signed failure
is useful scheduler evidence and has no promotion effect.

Minimum-leaf response is not monotonic. On the same 12-leaf return phenotype,
support 12 was worse than support 8, but support 20 recovered to 60.26 percent
accuracy, 61.20 percent balanced accuracy, PF 1.019, and positive expectancy.
Its bounded cutoff child `73ac58fe4f342b67` then reached 59.75 percent
accuracy, 60.15 percent balanced accuracy, PF 1.033, positive expectancy, and
59.28 percent coverage. A scheduler must not prune a topology direction from
one merely dominated intermediate. Once a recovered high-support endpoint is
within one coverage point, the controller now signs its immediate support
neighbors at fixed cutoff before changing coordinates.

Generation 1200 independently found a smaller 10-leaf topology,
`a7261f157d6dd22d`, with 60.26 percent accuracy, 61.49 percent balanced
accuracy, PF 1.109, positive expectancy, and 54.75 percent coverage. Because it
already clears the working direction and PF thresholds and misses the general
55-percent recovery bound by only 0.25 percentage points, it receives exactly
one bounded cutoff-recovery trial. This narrow economics override requires at
least 60 percent accuracy and balanced accuracy, PF 1.10, positive expectancy,
54 percent coverage, and valid ECE. It does not alter prescreen, multi-fold,
unseen-asset, calibration, ghost, or live admission requirements.

Coverage repair treats repeated identical outcomes on either side of the
coverage floor as evidence of a discrete tree-score plateau. In particular,
two lower-threshold descendants that admit the same harmful leaf are enough to
stop asymptotic scalar bisection. The lane preserves the model, features,
programs, and frontier threshold while switching only to deterministic
within-tie confidence ranking; that descendant still has to earn the ordinary
coverage, accuracy, profitability, calibration, unseen-asset, and ghost gates.
If the same-threshold ranked ablation itself crosses coverage but reproduces
the protected-fold accuracy/profit reversal, the discontinuity is not a tie
artifact. The lane then stops changing the scalar threshold and tries one
previously unseen causal margin interaction at the frontier threshold. It
preserves the original feature core and fit hyperparameters, records both the
durable frontier and failed ranked probe as parents, and remains subject to all
unchanged gates. Historical signed phenotypes prevent cycling through an
already disproven interaction. The interaction catalogue is finite. When every
catalogued interaction has signed evidence, the scheduler holds the frontier's
observations, programs, fit coordinates, and threshold fixed and tests, in
order, decomposed-return, extra-trees-return, and multiscale-return learners.
These are isolated architecture ablations, not promotion evidence. Once they
are also exhausted, the reserved lane yields to general evolution; it never
falls back to the already disproven scalar bisection.
