import json
import math
import random
from pathlib import Path

import pytest
import numpy as np

import scripts.market_evolution_service as evolution

from scripts.market_evolution_service import (
    Genome, add_derived_features, attach_causal_normalization, brain_feedback_score,
    attach_news_features, crossover, dataset_signature, evaluation_scope,
    evaluation_signature, decompose_returns, fit_regime_decomposed_regressor,
    introduce_calibration_variants, introduce_directional_frontier_variants,
    introduce_missing_learner_species, introduce_reflexivity_variant,
    introduce_regime_repair_variants,
    introduce_emergent_pool_variant,
    load_dataset_cached, mutate, passes_floor,
    passes_prescreen, program_name, program_value, recover_pending_gate,
    brain_accuracy_summary, record_accuracy_improvement,
    regression_probability_scale, seed_genomes, select_diverse_elites,
    random_emergent_pool,
    preserve_emergent_pool_elite,
    write_live_status,
)


def test_genome_identity_is_deterministic_and_mutation_stays_bounded():
    parent = seed_genomes(5, random.Random(1))[0]
    clone = Genome(**{**parent.__dict__, "features": list(reversed(parent.features))}).finalize()
    assert clone.genome_id == parent.genome_id
    child = mutate(parent, 2, random.Random(3))
    assert 0 <= child.confidence_quantile <= .30
    assert 1 <= child.calibration_safety <= 12
    assert 2 <= child.binding_threshold <= 9
    assert len(child.features) >= 8


def test_regime_learner_always_retains_its_routing_observation():
    candidate = seed_genomes(6, random.Random(21))[0]
    candidate.learner_kind = "regime_regressor"
    candidate.regime_feature = "funding_rate"
    candidate.regime_bins = 1
    candidate.features = [name for name in candidate.features if name != "funding_rate"]
    candidate.finalize()
    assert "funding_rate" in candidate.features


def test_crossover_preserves_a_viable_feature_genome():
    parents = seed_genomes(5, random.Random(2))
    child = crossover(parents[1], parents[2], 4, random.Random(4))
    assert len(child.features) >= 8
    assert child.generation == 4
    assert len(child.parents) >= 1


def test_seed_population_honors_requested_size():
    assert len(seed_genomes(4, random.Random(2))) == 4


def test_floor_requires_every_relationship_not_accuracy_alone():
    passing = {
        "acted_observations": 200, "coverage": .7, "directional_accuracy": .58,
        "directional_balanced_accuracy": .55, "baseline_margin": .05, "mcc": .15,
        "ece": .1, "net_expectancy": .0001, "profit_factor": 1.2,
        "max_portfolio_drawdown": .15,
    }
    assert passes_floor(passing)
    assert not passes_floor({**passing, "net_expectancy": 0})
    assert not passes_floor({**passing, "coverage": .69})


def test_prescreen_is_lower_than_admission_but_still_multimetric():
    viable = {
        "acted_observations": 150, "coverage": .6, "directional_accuracy": .54,
        "directional_balanced_accuracy": .52, "mcc": .04, "ece": .2,
        "profit_factor": .85,
    }
    assert passes_prescreen(viable)
    assert not passes_floor(viable)
    assert not passes_prescreen({**viable, "mcc": .039})


def test_derived_features_use_only_present_causal_values():
    features = {
        "spot_taker_imbalance": .2, "futures_taker_imbalance": -.1,
        "flow_divergence": -.3, "futures_spot_basis": .01, "funding_rate": .0001,
        "r6": .02, "r24": .04, "rv24": .01, "rv168": .02,
        "market_median_r6": .01, "trend_vote": 3, "market_breadth_r6": .6,
        "futures_quote_ratio24": 1.4, "spot_quote_ratio24": 1.1,
        "basis_z24": .8, "funding_z168": -.2,
        "imbalance_acceleration": .05, "basis_delta6": .001,
        "funding_delta24": .00001,
    }
    rows = [{"features": features}]
    add_derived_features(rows)
    assert rows[0]["features"]["flow_consensus"] == .05
    assert abs(rows[0]["features"]["breadth_gap_r6"] - .01) < 1e-12
    assert -1 <= rows[0]["features"]["participant_direction"] <= 1
    assert 0 <= rows[0]["features"]["participant_consensus"] <= 1
    assert rows[0]["features"]["participant_disagreement"] >= 0
    assert 0 <= rows[0]["features"]["crowding_intensity"] <= 1


def test_news_features_do_not_read_future_publications(tmp_path):
    path = tmp_path / "news.json"
    path.write_text('{"articles":['
                    '{"timestamp":100,"headline":"BTC rises","tokens":["BTC"],"sentiment":"positive"},'
                    '{"timestamp":300,"headline":"BTC falls","tokens":["BTC"],"sentiment":"negative"}'
                    ']}')
    rows = [{"timestamp": 200, "asset": "WBTC", "features": {}}]
    attach_news_features(rows, path)
    assert rows[0]["features"]["asset_news_count_24h"] > 0
    assert rows[0]["features"]["asset_news_sentiment_24h"] == 1.0


def test_news_features_filter_generic_advisories_and_expose_event_regimes(tmp_path):
    path = tmp_path / "news.json"
    path.write_text(json.dumps({"articles": [
        {"timestamp": 100, "headline": "generic package vulnerability",
         "article": "software advisory", "tokens": ["SECURITY"],
         "sentiment": "negative", "source": "GitHub Security Advisories"},
        {"timestamp": 110, "headline": "Bitcoin ETF receives regulatory approval",
         "article": "institutional fund launch", "tokens": ["BTC"],
         "sentiment": "positive", "source": "CoinDesk"},
    ]}))
    rows = [{"timestamp": 120, "asset": "WBTC", "features": {}}]
    attach_news_features(rows, path)
    features = rows[0]["features"]
    assert features["news_count_24h"] == pytest.approx(math.log1p(1))
    assert features["news_institutional_24h"] > 0
    assert features["asset_news_sentiment_acceleration"] == 0


def test_unfinished_champion_gate_is_recovered_after_restart(tmp_path):
    champion = seed_genomes(4, random.Random(7))[0]
    assert recover_pending_gate(tmp_path, champion, None) == champion.genome_id
    report = tmp_path / "brain-gate-reports" / f"{champion.genome_id}.smoke.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    assert recover_pending_gate(tmp_path, champion, None) is None
    assert recover_pending_gate(tmp_path, champion, "newer") == "newer"


def test_dataset_signature_covers_primary_and_news_inputs(tmp_path):
    primary = tmp_path / "primary.json"
    primary.write_text("[]")
    news = tmp_path / "news.json"
    news.write_text("[]")
    features = tmp_path / "supplemental" / "features"
    features.mkdir(parents=True)
    (features / "BTC.json").write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"selected":[{"path":"' +
                        str(primary).replace("\\", "\\\\") + '"}]}')
    before = dataset_signature(manifest, features.parent, news)
    primary.write_text("[1]")
    after_primary = dataset_signature(manifest, features.parent, news)
    news.write_text("[2]")
    after_news = dataset_signature(manifest, features.parent, news)
    assert before != after_primary
    assert after_primary != after_news


def test_evaluation_signature_changes_without_invalidating_feature_cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"selected":[]}')
    supplemental = tmp_path / "supplemental"
    (supplemental / "features").mkdir(parents=True)
    data = dataset_signature(manifest, supplemental)
    first = evaluation_signature(data, folds=3, test_days=42, calibration_days=30,
                                 final_days=21, horizon=12, cost_bps=20)
    second = evaluation_signature(data, folds=3, test_days=56, calibration_days=30,
                                  final_days=21, horizon=12, cost_bps=20)
    assert first != second
    assert data == dataset_signature(manifest, supplemental)


def test_new_corpus_evidence_invalidates_scores_but_preserves_lineages():
    population = seed_genomes(6, random.Random(14))
    identities = {genome.genome_id for genome in population}
    for genome in population:
        genome.fitness = 123.0
        genome.result = {"status": "screened"}
    refreshed = evolution.invalidate_population_for_new_evidence(
        population, 19, random.Random(15)
    )
    assert identities & {genome.genome_id for genome in refreshed}
    assert all(genome.fitness is None and genome.result is None
               for genome in refreshed)
    assert {genome.learner_kind for genome in refreshed} == set(evolution.LEARNER_KINDS)


def test_regression_temperature_preserves_zero_boundary():
    scores = np.asarray([-10, -8, -6, -4, -2, 2, 4, 6, 8, 10] * 4, dtype=float)
    labels = np.asarray([-1, -1, -1, 1, -1, 1, 1, -1, 1, 1] * 4, dtype=int)
    scale = regression_probability_scale(scores, labels)
    assert scale > 0
    assert (scores / scale >= 0).tolist() == (scores >= 0).tolist()


def test_return_decomposition_reconstructs_each_observation_without_features():
    rows = [
        {"timestamp": 1, "return": .03}, {"timestamp": 1, "return": .01},
        {"timestamp": 2, "return": -.02}, {"timestamp": 2, "return": .04},
    ]
    market, residual = decompose_returns(rows)
    realized = np.asarray([row["return"] for row in rows])
    assert np.allclose(market + residual, realized)
    assert market.tolist() == pytest.approx([.02, .02, .01, .01])


def test_evolved_program_is_deterministic_bounded_and_same_moment():
    program = {"op": "regime_gate", "left": "r6", "right": "funding_rate", "scale": 2}
    assert program_name(program) == program_name(dict(reversed(list(program.items()))))
    assert program_value({"r6": .03, "funding_rate": -.001}, program) == -.06
    assert abs(program_value({"r6": 1e20, "funding_rate": 1},
                             {**program, "op": "mul"})) <= 1e6


def test_causal_normalization_does_not_read_a_future_row():
    def row(timestamp, value, asset="A"):
        features = {name: 0.0 for name in (
            "r1", "r6", "r12", "r24", "r72", "r168", "rv24",
            "volatility_ratio", "volume_ratio24", "flow_imbalance",
            "market_median_r6", "market_breadth_r6", "relative_market_r6",
            "futures_spot_basis", "funding_rate", "flow_divergence",
        )}
        features["r6"] = value
        return {"timestamp": timestamp, "asset": asset, "features": features}
    prefix = [row(index, float(index)) for index in range(20)]
    attach_causal_normalization(prefix)
    before = prefix[-1]["features"]["causal_z14_r6"]
    extended = [row(index, float(index)) for index in range(20)] + [row(20, -999.0)]
    attach_causal_normalization(extended)
    assert extended[-2]["features"]["causal_z14_r6"] == before


def test_evaluation_scope_discards_only_stale_quarter_and_resplits():
    dataset = {"asset_ends": {f"A{i}": float(i) for i in range(8)}}
    training, holdout, end = evaluation_scope(dataset, set(dataset["asset_ends"]), "seed")
    assert end == 1.0
    assert training.isdisjoint(holdout)
    assert len(training | holdout) == 7


def test_dataset_cache_rejects_invalid_payload_and_round_trips(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"selected":[]}')
    supplemental = tmp_path / "supplemental"
    (supplemental / "features").mkdir(parents=True)
    cache = tmp_path / "dataset.joblib"
    evolution.joblib.dump({"signature": "wrong", "dataset": None}, cache)
    expected = {"rows": [{"timestamp": 1}], "assets": ["A"]}
    monkeypatch.setattr(evolution, "load_dataset", lambda *args: expected)
    first = load_dataset_cached(manifest, supplemental, 12, 12, "seed", None, cache)
    monkeypatch.setattr(
        evolution, "load_dataset",
        lambda *args: (_ for _ in ()).throw(AssertionError("cache was not used")),
    )
    second = load_dataset_cached(manifest, supplemental, 12, 12, "seed", None, cache)
    assert first == expected
    assert second == expected


def test_diverse_elites_preserve_learner_species_without_overriding_fitness():
    population = seed_genomes(6, random.Random(9))
    for index, genome in enumerate(population):
        genome.fitness = 100.0 - index
    elites = select_diverse_elites(population, 6)
    assert elites[0].genome_id == population[0].genome_id
    assert {genome.learner_kind for genome in elites} == {
        "classifier", "regressor", "extra_trees", "decomposed_regressor",
        "regime_regressor", "regime_decomposed_regressor",
    }


def test_resumed_population_immediately_receives_new_learner_species():
    population = seed_genomes(4, random.Random(2))
    population = [genome for genome in population if genome.learner_kind != "decomposed_regressor"]
    population.append(genome_from := Genome(**population[-1].__dict__))
    updated = introduce_missing_learner_species(population, 9, random.Random(3))
    decomposed = next(genome for genome in updated
                      if genome.learner_kind == "decomposed_regressor")
    assert decomposed.generation == 9
    assert decomposed.parents == [population[0].genome_id]


def test_neural_feedback_is_bounded_and_uses_weakest_section():
    report = {"folds": [{"sections": {
        "known": {"metrics": {"directional_accuracy": .8,
                                "directional_balanced_accuracy": .8,
                                "mcc": .6, "profit_factor": 2}},
        "unseen": {"metrics": {"directional_accuracy": .4,
                                 "directional_balanced_accuracy": .4,
                                 "mcc": -.2, "profit_factor": .5}},
    }}]}
    assert -25 <= brain_feedback_score(report) < 0
    assert brain_feedback_score(None) == -25


def test_new_calibration_gene_is_seeded_at_wide_monotonic_scales():
    population = seed_genomes(5, random.Random(4))
    for genome in population:
        genome.calibration_safety = 1.0
        genome.finalize()
    updated = introduce_calibration_variants(population, 11)
    assert {genome.calibration_safety for genome in updated} >= {1.0, 2.0, 4.0, 8.0}


def test_regime_decomposed_species_routes_market_and_residual_specialists():
    genome = seed_genomes(6, random.Random(8))[5]
    genome.features = [genome.features[0], genome.features[1]]
    genome.regime_feature = genome.features[0]
    genome.regime_bins = 2
    genome.min_samples_leaf = 8
    genome.max_iter = 80
    values = np.asarray([[float(index % 2), float(index)]
                         for index in range(240)], dtype=np.float32)
    market = np.asarray([.01 if row[0] else -.01 for row in values])
    residual = np.asarray([.001 * math.sin(row[1]) for row in values])
    model = fit_regime_decomposed_regressor(
        genome, values, market, residual, np.ones(len(values)), 17,
    )
    prediction = model.predict(values[:8])
    assert prediction.shape == (8,)
    assert np.isfinite(prediction).all()


def test_directional_frontier_seeds_calibration_without_changing_sign_model():
    population = seed_genomes(12, random.Random(12))
    base = population[1]
    base.calibration_safety = 1.0
    base.result = {"summary": {
        "min_accuracy": .59, "min_balanced_accuracy": .56,
        "min_mcc": .13, "min_expectancy": .002,
        "max_ece": .30,
    }}
    updated = introduce_directional_frontier_variants(population, [base], 4)
    variants = updated[-2:]
    assert [genome.calibration_safety for genome in variants] == [4.0, 8.0]
    assert all(genome.learner_kind == base.learner_kind for genome in variants)
    assert all(genome.features == base.features for genome in variants)
    assert all(genome.parents == [base.genome_id] for genome in variants)


def test_prescreen_directional_frontier_can_repair_calibration_before_next_fold():
    population = seed_genomes(12, random.Random(19))
    base = population[1]
    base.calibration_safety = 1.0
    base.result = {"summary": {
        "min_accuracy": .56, "min_balanced_accuracy": .54, "min_mcc": .08,
        "min_expectancy": .001, "max_ece": .27,
    }}
    updated = introduce_directional_frontier_variants(population, [base], 5)
    variants = updated[-2:]
    assert [genome.calibration_safety for genome in variants] == [4.0, 8.0]
    assert all(genome.parents == [base.genome_id] for genome in variants)


def test_deep_cross_regime_failure_seeds_targeted_repair_descendants():
    population = seed_genomes(12, random.Random(15))
    base = population[3]
    base.recency_half_life_days = 600
    base.fitness = 2200
    base.result = {
        "status": "screened", "evaluated_folds": 3,
        "summary": {"min_accuracy": .47},
    }
    updated = introduce_regime_repair_variants(
        population, population, 8, random.Random(16),
    )
    variants = updated[-2:]
    assert {genome.learner_kind for genome in variants} == {
        "regime_regressor", "regime_decomposed_regressor",
    }
    assert {genome.regime_bins for genome in variants} == {2, 3}
    assert max(genome.recency_half_life_days for genome in variants) <= 360
    assert all(genome.parents == [base.genome_id] for genome in variants)
    assert all(genome.fitness is None and genome.result is None for genome in variants)


def test_shallow_failures_do_not_displace_random_exploration():
    population = seed_genomes(12, random.Random(17))
    for genome in population:
        genome.fitness = 300
        genome.result = {
            "status": "prescreen_reject", "evaluated_folds": 1,
            "summary": {"min_accuracy": .57},
        }
    original = [genome.genome_id for genome in population]
    updated = introduce_regime_repair_variants(
        population, population, 9, random.Random(18),
    )
    assert [genome.genome_id for genome in updated] == original


def test_live_status_failure_never_stops_evolution(monkeypatch, tmp_path):
    def denied(path: Path, payload):
        raise PermissionError("dashboard reader briefly owns status")

    monkeypatch.setattr(evolution, "atomic_json", denied)
    write_live_status(tmp_path, "evaluating", 12, completed=3)


def test_competence_condition_is_frozen_and_causal():
    rows = [
        {"features": {"rv24": value}}
        for value in (-2.0, -0.5, 0.5, 2.0)
    ]
    condition = [{
        "kind": "feature", "feature": "rv24", "side": "high",
        "threshold": 0.5, "label": "high rv24",
    }]
    mask = evolution.condition_mask(rows, condition)
    assert mask.tolist() == [False, False, True, True]
    # Targets are deliberately absent: routing cannot inspect future labels.
    assert all("target" not in row for row in rows)


def test_competence_condition_supports_auditable_conjunctions():
    rows = [
        {"features": {"rv24": rv, "market_breadth_r6": breadth}}
        for rv, breadth in ((.1, .8), (.1, .2), (.9, .8), (.9, .2))
    ]
    setup = [{
        "kind": "all", "label": "low volatility + high breadth",
        "clauses": [
            {"kind": "feature", "feature": "rv24", "side": "low",
             "threshold": .2},
            {"kind": "feature", "feature": "market_breadth_r6", "side": "high",
             "threshold": .7},
        ],
    }]
    assert evolution.condition_mask(rows, setup).tolist() == [True, False, False, False]


def test_conditional_ghost_floor_remains_separate_from_live_floor():
    local_edge = {
        "acted_observations": 40, "directional_accuracy": .56,
        "directional_balanced_accuracy": .53, "mcc": .06,
        "net_expectancy": .001, "profit_factor": 1.08,
        "coverage": .20, "ece": .15, "baseline_margin": .01,
        "max_portfolio_drawdown": .25,
    }
    assert evolution.competence_passes(local_edge)
    assert not passes_floor(local_edge)


@pytest.mark.parametrize("size,expected_elites", [(4, 2), (8, 3), (12, 5)])
def test_elite_budget_always_reserves_real_exploration(size, expected_elites):
    assert evolution.elite_budget(size) == expected_elites
    assert size - expected_elites >= evolution.minimum_novel_candidates(size)


def test_breeding_cannot_silently_retain_the_entire_population():
    evaluated = seed_genomes(8, random.Random(40))
    for index, genome in enumerate(evaluated):
        genome.fitness = float(index)
        genome.result = {"status": "prescreen_reject"}
    following, health = evolution.breed_population(
        evaluated, 12, random.Random(41), {}
    )
    assert len(following) == 8
    assert len({genome.genome_id for genome in following}) == 8
    assert health["elite_budget"] == 3
    assert health["offspring_created"] == 5
    assert sum(genome.fitness is None for genome in following) == 5


def test_novelty_guard_repairs_a_fully_stale_population():
    population = seed_genomes(8, random.Random(42))
    for genome in population:
        genome.fitness = 1.0
    protected_leader = population[0].genome_id
    repaired, injected = evolution.ensure_novelty(
        population, 13, random.Random(43)
    )
    assert repaired[0].genome_id == protected_leader
    assert injected >= 2
    assert sum(genome.fitness is None for genome in repaired) >= 2


def test_news_context_cannot_disappear_from_evolution():
    evaluated = seed_genomes(8, random.Random(44))
    for index, genome in enumerate(evaluated):
        genome.features = sorted(
            set(genome.features) - set(evolution.NEWS_SPECIALIST_FEATURES)
        )
        genome.fitness = float(index)
        genome.finalize()
    following, _ = evolution.breed_population(
        evaluated, 14, random.Random(45), {}
    )
    following = evolution.introduce_news_context_variant(
        following, evaluated, 14, random.Random(46)
    )
    specialists = [
        genome for genome in following
        if set(genome.features) & set(evolution.NEWS_SPECIALIST_FEATURES)
    ]
    assert specialists
    assert any(genome.fitness is None for genome in specialists)
    assert any(
        "news" in json.dumps(program) for genome in specialists
        for program in genome.feature_programs
    )


def test_curriculum_does_not_buy_accuracy_with_collapsed_coverage():
    selective = evolution.curriculum_fitness(
        fold_count=1, min_accuracy=.59, min_balanced=.56, min_mcc=.21,
        min_margin=-.008, min_coverage=.47, min_observations=127,
        min_expectancy=-.0028, min_profit=.79, max_ece=.096,
        max_drawdown=.71, conditional_ghost_pass=False,
        conditional_ghost_accuracy=0,
    )
    admissible = evolution.curriculum_fitness(
        fold_count=1, min_accuracy=.56, min_balanced=.55, min_mcc=.13,
        min_margin=.02, min_coverage=.72, min_observations=190,
        min_expectancy=.0002, min_profit=1.03, max_ece=.07,
        max_drawdown=.30, conditional_ghost_pass=False,
        conditional_ghost_accuracy=0,
    )
    assert admissible > selective


def test_accuracy_still_improves_fitness_when_other_evidence_is_equal():
    evidence = dict(
        fold_count=1, min_balanced=.55, min_mcc=.13, min_margin=.02,
        min_coverage=.72, min_observations=190, min_expectancy=.0002,
        min_profit=1.03, max_ece=.07, max_drawdown=.30,
        conditional_ghost_pass=False, conditional_ghost_accuracy=0,
    )
    assert evolution.curriculum_fitness(min_accuracy=.58, **evidence) > (
        evolution.curriculum_fitness(min_accuracy=.56, **evidence)
    )


def test_high_accuracy_low_coverage_lineage_gets_repair_descendants():
    evaluated = seed_genomes(8, random.Random(47))
    base = evaluated[-1]
    base.confidence_quantile = .30
    base.calibration_safety = 4.0
    base.fitness = 400
    base.result = {"summary": {
        "min_accuracy": .59, "min_balanced_accuracy": .56,
        "min_mcc": .20, "min_coverage": .594,
        "min_expectancy": .0001, "min_profit_factor": 1.01,
    }}
    following, _ = evolution.breed_population(
        evaluated, 15, random.Random(48), {}
    )
    for genome in following:
        genome.emergent_pools = [{
            "features": [genome.features[0]], "concept_threshold": 5,
        }]
        genome.finalize()
    repaired = evolution.introduce_coverage_repair_variants(
        following, evaluated, 15
    )
    variants = [
        variant for variant in repaired
        if variant.parents == [base.genome_id] and variant.fitness is None
        and variant.confidence_quantile < .30
    ]
    assert [round(value, 3) for value in sorted(
        variant.confidence_quantile for variant in variants
    )] == [.291, .295]
    assert all(variant.calibration_safety == 4.0 for variant in variants)
    assert all(variant.parents == [base.genome_id] for variant in variants)
    assert all(variant.fitness is None for variant in variants)
    assert any(
        genome.emergent_pools and genome.parents != [base.genome_id]
        for genome in repaired
    )


def test_coverage_repair_bisects_verified_reversal_boundary():
    evaluated = seed_genomes(8, random.Random(62))
    upper = evaluated[-1]
    upper.confidence_quantile = .27
    upper.result = {"summary": {
        "min_accuracy": .63, "min_balanced_accuracy": .62,
        "min_mcc": .25, "min_coverage": .594,
        "min_expectancy": .0004, "min_profit_factor": 1.04,
    }}
    upper.finalize()
    lower = Genome(**{
        **upper.__dict__, "confidence_quantile": .26,
        "generation": 99, "parents": [upper.genome_id],
        "genome_id": "", "fitness": 1200,
        "result": {"folds": [{}, {}], "summary": {
            "min_accuracy": .47, "min_coverage": .601,
        }},
    }).finalize()
    assert evolution.genome_structure_key(upper) == evolution.genome_structure_key(lower)
    following, _ = evolution.breed_population(
        evaluated, 31, random.Random(63), {}
    )
    repaired = evolution.introduce_coverage_repair_variants(
        following, [upper], 31, lower
    )
    variants = [
        genome for genome in repaired if genome.parents == [upper.genome_id]
    ]
    assert sorted(round(genome.confidence_quantile, 4) for genome in variants) == [
        .265, .2675,
    ]


def test_narrow_coverage_reversal_launches_margin_reordering_children():
    evaluated = seed_genomes(8, random.Random(71))
    upper = evaluated[-1]
    upper.confidence_quantile = .264
    upper.result = {"summary": {
        "min_accuracy": .632, "min_balanced_accuracy": .621,
        "min_mcc": .25, "min_coverage": .597,
        "min_expectancy": .0004, "min_profit_factor": 1.04,
    }}
    upper.finalize()
    lower = Genome(**{
        **upper.__dict__, "confidence_quantile": .263,
        "generation": 99, "parents": [upper.genome_id],
        "genome_id": "", "fitness": 1200,
        "result": {
            "folds": [
                {
                    "known_asset_future": {"directional_accuracy": .65},
                    "unseen_asset_future": {"directional_accuracy": .63},
                },
                {
                    "known_asset_future": {"directional_accuracy": .47},
                    "unseen_asset_future": {"directional_accuracy": .53},
                },
            ],
            "summary": {"min_accuracy": .47, "min_mcc": -.03},
        },
    }).finalize()
    following, _ = evolution.breed_population(
        evaluated, 32, random.Random(72), {}
    )
    repaired = evolution.introduce_coverage_repair_variants(
        following, [upper], 32, lower
    )
    variants = [
        genome for genome in repaired if genome.parents == [upper.genome_id]
    ]
    assert len(variants) == 2
    assert all(genome.confidence_quantile == .263 for genome in variants)
    assert all(
        evolution.genome_structure_key(genome)
        != evolution.genome_structure_key(upper)
        for genome in variants
    )
    new_programs = {
        evolution.program_name(program)
        for genome in variants for program in genome.feature_programs
    } - {
        evolution.program_name(program) for program in upper.feature_programs
    }
    assert len(new_programs) == 2


def test_failed_margin_children_escalate_to_multiscale_memory():
    evaluated = seed_genomes(8, random.Random(73))
    upper = evaluated[-1]
    upper.confidence_quantile = .264
    upper.recency_half_life_days = 300
    upper.result = {"summary": {
        "min_accuracy": .632, "min_balanced_accuracy": .621,
        "min_mcc": .25, "min_coverage": .597,
        "min_expectancy": .0004, "min_profit_factor": 1.04,
    }}
    upper.finalize()
    lower = Genome(**{
        **upper.__dict__, "confidence_quantile": .263,
        "generation": 99, "parents": [upper.genome_id], "genome_id": "",
        "result": {
            "folds": [
                {"known_asset_future": {"directional_accuracy": .64},
                 "unseen_asset_future": {"directional_accuracy": .63}},
                {"known_asset_future": {"directional_accuracy": .46},
                 "unseen_asset_future": {"directional_accuracy": .53}},
            ],
            "summary": {"min_accuracy": .46, "min_mcc": -.04},
        },
    }).finalize()
    failed = []
    for index, op in enumerate(("mul", "abs_gap")):
        child = Genome(**{
            **upper.__dict__,
            "feature_programs": [
                *upper.feature_programs,
                {"op": op, "left": "r1", "right": "rv24", "scale": 1},
            ],
            "generation": 100 + index, "parents": [upper.genome_id],
            "genome_id": "", "fitness": 1200,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .46, "min_coverage": .60,
            }},
        }).finalize()
        failed.append(child)
    following, _ = evolution.breed_population(
        evaluated, 33, random.Random(74), {}
    )
    repaired = evolution.introduce_coverage_repair_variants(
        following, [upper, *failed], 33, lower
    )
    variants = [
        genome for genome in repaired if genome.parents == [upper.genome_id]
    ]
    assert len(variants) == 2
    assert all(genome.learner_kind == "multiscale_regressor" for genome in variants)
    assert sorted(genome.recency_half_life_days for genome in variants) == [300, 600]
    assert all(genome.confidence_quantile == .263 for genome in variants)


def test_failed_multiscale_children_probe_new_memory_scales():
    evaluated = seed_genomes(8, random.Random(173))
    upper = evaluated[-1]
    upper.learner_kind = "multiscale_regressor"
    upper.confidence_quantile = .264
    upper.recency_half_life_days = 300
    upper.result = {"summary": {
        "min_accuracy": .632, "min_balanced_accuracy": .621,
        "min_mcc": .25, "min_coverage": .597,
        "min_expectancy": .0004, "min_profit_factor": 1.04,
    }}
    upper.finalize()
    lower = Genome(**{
        **upper.__dict__, "confidence_quantile": .263,
        "generation": 199, "parents": [upper.genome_id], "genome_id": "",
        "result": {"summary": {"min_accuracy": .46, "min_mcc": -.04}},
    }).finalize()
    failed = []
    for index, op in enumerate(("mul", "abs_gap")):
        failed.append(Genome(**{
            **upper.__dict__,
            "feature_programs": [
                *upper.feature_programs,
                {"op": op, "left": "r1", "right": "rv24", "scale": 1},
            ],
            "generation": 200 + index, "parents": [upper.genome_id],
            "genome_id": "", "fitness": 1200,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .46, "min_coverage": .60,
            }},
        }).finalize())
    following, _ = evolution.breed_population(
        evaluated, 34, random.Random(174), {}
    )
    repaired = evolution.introduce_coverage_repair_variants(
        following, [upper, *failed], 34, lower
    )
    variants = [
        genome for genome in repaired if genome.parents == [upper.genome_id]
    ]
    assert len(variants) == 2
    assert sorted(genome.recency_half_life_days for genome in variants) == [150, 600]
    assert all(genome.confidence_quantile == .263 for genome in variants)

    prior_memory_failures = []
    for index, memory in enumerate((150, 600)):
        prior_memory_failures.append(Genome(**{
            **upper.__dict__, "recency_half_life_days": memory,
            "confidence_quantile": .263,
            "generation": 210 + index, "parents": [upper.genome_id],
            "genome_id": "", "fitness": 1300,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .48, "min_coverage": .61,
            }},
        }).finalize())
    repaired_again = evolution.introduce_coverage_repair_variants(
        following, [upper, *failed, *prior_memory_failures], 35, lower
    )
    widened = [
        genome for genome in repaired_again
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert sorted(genome.recency_half_life_days for genome in widened) == [75, 1200]

    widened_failures = []
    for index, memory in enumerate((75, 1200)):
        widened_failures.append(Genome(**{
            **upper.__dict__, "recency_half_life_days": memory,
            "confidence_quantile": .263,
            "generation": 220 + index, "parents": [upper.genome_id],
            "genome_id": "", "fitness": 1400,
            "result": {"evaluated_folds": 3, "summary": {
                "min_accuracy": .47, "min_coverage": .66,
            }},
        }).finalize())
    oriented_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures],
        36, lower,
    )
    oriented = [
        genome for genome in oriented_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(oriented) == 2
    assert all(genome.calibration_orientation for genome in oriented)
    assert sorted(genome.recency_half_life_days for genome in oriented) == [150, 300]

    no_flip_failures = []
    for index, genome in enumerate(oriented):
        no_flip_failures.append(Genome(**{
            **genome.__dict__, "generation": 230 + index,
            "genome_id": "", "fitness": 1500,
            "result": {
                "evaluated_folds": 2,
                "folds": [
                    {"multiscale_calibration": {"direction": 1.0}},
                    {"multiscale_calibration": {"direction": 1.0}},
                ],
                "summary": {"min_accuracy": .48, "min_coverage": .63},
            },
        }).finalize())
    reliability_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures],
        37, lower,
    )
    reliability = [
        genome for genome in reliability_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(reliability) == 2
    assert all(genome.calibration_reliability for genome in reliability)
    assert all(genome.calibration_reliability_version == 1 for genome in reliability)
    assert not any(genome.calibration_orientation for genome in reliability)
    assert sorted(genome.recency_half_life_days for genome in reliability) == [150, 300]

    linear_reliability_failures = [
        Genome(**{
            **genome.__dict__, "generation": 240 + index,
            "genome_id": "", "fitness": 1600,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .46, "min_coverage": .68,
            }},
        }).finalize()
        for index, genome in enumerate(reliability)
    ]
    nonlinear_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures],
        38, lower,
    )
    nonlinear = [
        genome for genome in nonlinear_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(nonlinear) == 2
    assert all(genome.calibration_reliability_version == 2 for genome in nonlinear)
    assert sorted(genome.recency_half_life_days for genome in nonlinear) == [150, 300]

    nonlinear_failures = [
        Genome(**{
            **genome.__dict__, "generation": 250 + index,
            "genome_id": "", "fitness": 1700,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .46, "min_coverage": .67,
            }},
        }).finalize()
        for index, genome in enumerate(nonlinear)
    ]
    isolated_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures,
         *nonlinear_failures],
        39, lower,
    )
    isolated = [
        genome for genome in isolated_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(isolated) == 2
    assert all(genome.calibration_reliability_version == 3 for genome in isolated)
    assert {genome.calibration_reliability_pool for genome in isolated} == {
        "trend_regime", "flow_news",
    }

    isolated_failures = [
        Genome(**{
            **genome.__dict__, "generation": 260 + index,
            "genome_id": "", "fitness": 1800,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .47, "min_coverage": .67,
            }},
        }).finalize()
        for index, genome in enumerate(isolated)
    ]
    integrated_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures,
         *nonlinear_failures, *isolated_failures],
        40, lower,
    )
    integrated = [
        genome for genome in integrated_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(integrated) == 2
    assert all(genome.calibration_reliability_version == 4 for genome in integrated)
    assert all(genome.calibration_reliability_pool == "combined"
               for genome in integrated)
    assert set(evolution.RELIABILITY_FEATURE_POOLS["flow_news"]) <= set(
        evolution.RELIABILITY_FEATURE_POOLS["combined"]
    )

    integrated_failures = [
        Genome(**{
            **genome.__dict__, "generation": 270 + index,
            "genome_id": "", "fitness": 1900,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .46, "min_coverage": .63,
            }},
        }).finalize()
        for index, genome in enumerate(integrated)
    ]
    recent_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures,
         *nonlinear_failures, *isolated_failures, *integrated_failures],
        41, lower,
    )
    recent = [
        genome for genome in recent_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(recent) == 2
    assert all(genome.calibration_reliability_version == 5 for genome in recent)
    assert all(genome.calibration_reliability_pool == "flow_news"
               for genome in recent)

    recent_failures = [
        Genome(**{
            **genome.__dict__, "generation": 280 + index,
            "genome_id": "", "fitness": 2000,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": .47, "min_coverage": .70,
            }},
        }).finalize()
        for index, genome in enumerate(recent)
    ]
    decay_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures,
         *nonlinear_failures, *isolated_failures, *integrated_failures,
         *recent_failures],
        42, lower,
    )
    decay_variants = [
        genome for genome in decay_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(decay_variants) == 2
    assert all(genome.calibration_reliability_version == 6
               for genome in decay_variants)
    assert all(genome.calibration_reliability_pool == "flow_news"
               for genome in decay_variants)
    assert {genome.calibration_reliability_decay for genome in decay_variants} == {
        .75, 3.5,
    }
    assert len({genome.recency_half_life_days for genome in decay_variants}) == 1
    assert decay_variants[0].recency_half_life_days == 75

    mature_decay_failures = []
    for index, (decay, accuracy, balanced, mcc, expectancy, profit) in enumerate((
        (.75, .4843, .4923, -.0167, -.00222, .802),
        (3.5, .4943, .4993, -.0016, -.00148, .866),
        (.25, .4839, .4933, -.0144, -.00243, .784),
        (5.0, .4765, .4765, -.0483, -.00231, .809),
        (2.125, .4952, .4949, -.0107, -.00165, .850),
        (4.25, .4916, .4938, -.0128, -.00165, .851),
        (3.875, .4943, .4999, -.0002, -.00143, .870),
        (2.8125, .4972, .4979, -.0043, -.00136, .876),
    )):
        mature_decay_failures.append(Genome(**{
            **decay_variants[0].__dict__,
            "calibration_reliability_decay": decay,
            "generation": 310 + index, "genome_id": "", "fitness": 2100,
            "result": {"evaluated_folds": 2, "summary": {
                "min_accuracy": accuracy, "min_balanced_accuracy": balanced,
                "min_mcc": mcc, "min_expectancy": expectancy,
                "min_profit_factor": profit, "min_coverage": .72,
            }},
        }).finalize())
    threshold_population = evolution.introduce_coverage_repair_variants(
        following,
        [upper, *failed, *prior_memory_failures, *widened_failures,
         *no_flip_failures, *linear_reliability_failures,
         *nonlinear_failures, *isolated_failures, *integrated_failures,
         *recent_failures, *mature_decay_failures],
        43, lower,
    )
    threshold_variants = [
        genome for genome in threshold_population
        if genome.parents == [upper.genome_id] and genome.fitness is None
    ]
    assert len(threshold_variants) == 2
    assert all(genome.calibration_reliability_version == 7
               for genome in threshold_variants)
    assert all(genome.calibration_reliability_decay == 2.8125
               for genome in threshold_variants)
    assert all(genome.recency_half_life_days == 75
               for genome in threshold_variants)
    assert sorted(genome.confidence_quantile for genome in threshold_variants) == [
        .283, .30,
    ]


def test_multiscale_regressor_tunes_blend_on_calibration_only():
    class FixedModel:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def predict(self, _values):
            return self.values

    model = evolution.MultiscaleRegressor(
        FixedModel([2, 2, -2, -2]), FixedModel([-3, -3, 3, 3])
    )
    values = np.zeros((4, 1))
    labels = np.asarray([1, 1, -1, -1], dtype=np.int8)
    model.tune(values, labels)
    assert model.short_weight == .75
    assert np.array_equal(np.sign(model.predict(values)), labels)


def test_multiscale_orientation_requires_clear_calibration_advantage():
    class FixedModel:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def predict(self, _values):
            return self.values

    labels = np.asarray(([1] * 20) + ([-1] * 20), dtype=np.int8)
    inverse = -labels.astype(np.float64)
    model = evolution.MultiscaleRegressor(
        FixedModel(inverse), FixedModel(inverse), allow_orientation=True
    )
    model.tune(np.zeros((40, 1)), labels)
    assert model.direction == -1.0
    assert np.array_equal(np.sign(model.predict(np.zeros((40, 1)))), labels)


def test_calibration_reliability_ranks_abstention_without_flipping_sign():
    class ScoreModel:
        def predict(self, values):
            return values[:, 0]

    scores = np.tile(np.asarray([1.0, -1.0]), 60)
    regime = np.repeat(np.asarray([1.0, -1.0]), 60)
    values = np.column_stack((scores, regime))
    base_sign = np.where(scores >= 0, 1, -1).astype(np.int8)
    labels = np.where(regime > 0, base_sign, -base_sign).astype(np.int8)
    surrogate = evolution.Surrogate(ScoreModel(), "regressor", score_scale=1.0)
    assert surrogate.fit_reliability(values, labels, [1])
    probability = surrogate.probability(values)
    prediction = np.where(probability >= .5, 1, -1).astype(np.int8)
    confidence = surrogate.selection_confidence(values)
    assert np.array_equal(prediction, base_sign)
    assert confidence[regime > 0].mean() > confidence[regime < 0].mean()


def test_continuous_rank_regressor_only_breaks_confidence_ties():
    class TwoLevelModel:
        def predict(self, values):
            return np.where(values[:, 0] < 0, .01, .02)

    values = np.asarray([
        [-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
        [1.0, -2.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
    ], dtype=np.float64)
    ordinary = evolution.Surrogate(TwoLevelModel(), "regressor")
    ranked = evolution.Surrogate(TwoLevelModel(), "continuous_rank_regressor")

    base = ordinary.selection_confidence(values)
    selection = ranked.selection_confidence(values)

    assert len(np.unique(base[:4])) == 1
    assert len(np.unique(selection[:4])) == 4
    assert max(selection[:4]) < min(selection[4:])
    assert np.array_equal(ranked.predict(values), ordinary.predict(values))
    assert np.array_equal(selection, ranked.selection_confidence(values))


def test_nonlinear_calibration_reliability_discovers_feature_interactions():
    class ScoreModel:
        def predict(self, values):
            return values[:, 0]

    combinations = np.asarray([
        [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0],
    ])
    regimes = np.repeat(combinations, 60, axis=0)
    scores = np.tile(np.asarray([1.0, -1.0]), 120)
    values = np.column_stack((scores, regimes))
    base_sign = np.where(scores >= 0, 1, -1).astype(np.int8)
    correct = regimes[:, 0] * regimes[:, 1] > 0
    labels = np.where(correct, base_sign, -base_sign).astype(np.int8)
    surrogate = evolution.Surrogate(ScoreModel(), "regressor", score_scale=1.0)

    assert surrogate.fit_reliability(values, labels, [1, 2], version=2)
    prediction = np.where(surrogate.probability(values) >= .5, 1, -1)
    confidence = surrogate.selection_confidence(values)

    assert np.array_equal(prediction, base_sign)
    assert confidence[correct].mean() > confidence[~correct].mean() + .25


def test_reliability_orientation_inverts_only_abstention_rank_on_later_calibration():
    class ScoreModel:
        def predict(self, values):
            return values[:, 0]

    scores = np.tile(np.asarray([1.0, -1.0]), 80)
    regime = np.repeat(np.asarray([1.0, -1.0]), 80)
    values = np.column_stack((scores, regime))
    base_sign = np.where(scores >= 0, 1, -1).astype(np.int8)
    fit_labels = np.where(regime > 0, base_sign, -base_sign).astype(np.int8)
    later_labels = np.where(regime > 0, -base_sign, base_sign).astype(np.int8)
    surrogate = evolution.Surrogate(ScoreModel(), "regressor", score_scale=1.0)

    assert surrogate.fit_reliability(values, fit_labels, [1], version=8)
    assert surrogate.tune_reliability_orientation(values, later_labels) == -1.0
    assert np.array_equal(surrogate.predict(values), base_sign)
    confidence = surrogate.selection_confidence(values)
    assert confidence[regime < 0].mean() > confidence[regime > 0].mean()


def test_orientation_aware_scheduler_explores_both_feature_pools_without_repeats():
    base = seed_genomes(1, random.Random(176))[0]
    base.confidence_quantile = .237396
    first = evolution.next_oriented_reliability_variants([], base.confidence_quantile)
    assert first == (("flow_news", .237396), ("combined", .237396))
    evidence = [
        Genome(**{
            **base.__dict__, "calibration_reliability": True,
            "calibration_reliability_version": 8,
            "calibration_reliability_pool": pool,
            "confidence_quantile": quantile,
            "genome_id": "", "fitness": 1200,
            "result": {"summary": {"min_accuracy": .49}},
        }).finalize()
        for pool, quantile in first
    ]
    second = evolution.next_oriented_reliability_variants(
        evidence, base.confidence_quantile
    )
    assert second == (("flow_news", .257396), ("combined", .257396))

    evidence[0].result["folds"] = [
        {"multiscale_calibration": {"reliability_direction": 0.0}},
        {"multiscale_calibration": {"reliability_direction": 0.0}},
    ]
    after_inert_pool = evolution.next_oriented_reliability_variants(
        evidence, base.confidence_quantile
    )
    assert after_inert_pool == (
        ("combined", .257396), ("combined", .217396)
    )


def test_orientation_scheduler_escapes_outcome_plateau_into_compact_pools():
    base = seed_genomes(1, random.Random(177))[0]
    base.confidence_quantile = .237396
    evidence = []
    for index, quantile in enumerate((
        .197396, .217396, .237396, .257396, .277396, .297396,
    )):
        evidence.append(Genome(**{
            **base.__dict__, "calibration_reliability": True,
            "calibration_reliability_version": 8,
            "calibration_reliability_pool": "combined",
            "confidence_quantile": quantile,
            "generation": 400 + index, "genome_id": "", "fitness": 1200,
            "result": {"summary": {
                "min_accuracy": .5027, "min_balanced_accuracy": .5063,
                "min_mcc": .0131, "min_coverage": .7928,
                "min_expectancy": -.0007, "min_profit_factor": .9319,
            }},
        }).finalize())
    for index, quantile in enumerate((.237396, .257396)):
        evidence.append(Genome(**{
            **base.__dict__, "calibration_reliability": True,
            "calibration_reliability_version": 8,
            "calibration_reliability_pool": "flow_news",
            "confidence_quantile": quantile,
            "generation": 500 + index, "genome_id": "", "fitness": 1100,
            "result": {
                "folds": [
                    {"multiscale_calibration": {"reliability_direction": 0.0}},
                    {"multiscale_calibration": {"reliability_direction": 0.0}},
                ],
                "summary": {"min_accuracy": .49},
            },
        }).finalize())

    proposals = evolution.next_oriented_reliability_variants(
        evidence, base.confidence_quantile
    )

    assert proposals == (
        ("flow_derivatives", .237396), ("news_regime", .237396)
    )
    assert all(pool not in {"combined", "flow_news"} for pool, _ in proposals)


def test_full_fold_champion_coordinate_search_is_single_axis_and_remembers_evidence():
    population = seed_genomes(8, random.Random(178))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {"min_accuracy": .566, "min_profit_factor": .958},
    }
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None
        genome.emergent_pools = []
    first = evolution.introduce_champion_coordinate_variant(
        population, champion, [], 900
    )
    child = next(
        genome for genome in first
        if champion.genome_id in genome.parents and genome.fitness is None
    )
    assert child.confidence_quantile == max(
        0.0, champion.confidence_quantile - .01
    )
    unchanged = (
        set(champion.__dict__) - {
            "confidence_quantile", "generation", "parents", "fitness",
            "result", "genome_id",
        }
    )
    assert all(getattr(child, name) == getattr(champion, name) for name in unchanged)

    child.fitness = 1200
    child.result = {
        "evaluation_signature": "scope-a", "evaluated_folds": 3,
        "requested_folds": 3, "summary": {"min_accuracy": .55},
    }
    next_population = seed_genomes(8, random.Random(179))
    for genome in next_population:
        genome.fitness = None
        genome.result = None
        genome.emergent_pools = []
    second = evolution.introduce_champion_coordinate_variant(
        next_population, champion, [child], 901
    )
    next_child = next(
        genome for genome in second
        if champion.genome_id in genome.parents and genome.fitness is None
    )
    assert next_child.confidence_quantile == min(
        .30, champion.confidence_quantile + .01
    )


def test_partial_fold_frontier_cannot_drive_champion_coordinate_search():
    population = seed_genomes(8, random.Random(180))
    frontier = population[0]
    frontier.fitness = 2500
    frontier.result = {"evaluated_folds": 1, "requested_folds": 3}
    before = [genome.genome_id for genome in population]
    after = evolution.introduce_champion_coordinate_variant(
        population, frontier, [], 902
    )
    assert [genome.genome_id for genome in after] == before


def test_full_fold_champion_coordinate_search_survives_emergent_saturation():
    population = seed_genomes(8, random.Random(181))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None
        genome.emergent_pools = [{
            "name": f"emergent_{genome.genome_id}",
            "features": ["r1"], "concept_threshold": 2,
        }]

    after = evolution.introduce_champion_coordinate_variant(
        population, champion, [], 903
    )

    descendants = [
        genome for genome in after
        if champion.genome_id in genome.parents and genome.fitness is None
    ]
    assert len(descendants) == 1
    assert descendants[0].confidence_quantile == max(
        0.0, champion.confidence_quantile - .01
    )


def test_champion_coordinate_search_mirrors_accuracy_economics_tradeoff():
    population = seed_genomes(8, random.Random(182))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5664, "min_profit_factor": .9576,
            "min_expectancy": -.00048,
        },
    }
    observed = Genome(**{
        **champion.__dict__,
        "confidence_quantile": champion.confidence_quantile - .01,
        "generation": 903, "parents": [champion.genome_id],
        "genome_id": "", "fitness": 2390,
        "result": {
            "evaluated_folds": 3, "requested_folds": 3,
            "summary": {
                "min_accuracy": .5678, "min_profit_factor": .9409,
                "min_expectancy": -.00069,
            },
        },
    }).finalize()
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None
        genome.emergent_pools = []

    after = evolution.introduce_champion_coordinate_variant(
        population, champion, [observed], 904
    )

    child = next(
        genome for genome in after
        if champion.genome_id in genome.parents and genome.fitness is None
    )
    assert child.confidence_quantile == pytest.approx(
        champion.confidence_quantile + .01
    )


def test_profitable_frontier_program_isolated_on_full_fold_champion():
    population = seed_genomes(8, random.Random(183))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    frontier = population[1]
    frontier.fitness = 2300
    frontier.feature_programs = [{
        "op": "regime_gate", "left": "funding_z168",
        "right": "crowd_price_alignment", "scale": 2.7,
    }]
    frontier.result = {"summary": {
        "min_profit_factor": 1.05, "min_expectancy": .0005,
    }}
    protected_parent = "multiscale-frontier"
    for genome in population[2:]:
        genome.fitness = None
        genome.result = None
        genome.parents = [protected_parent]
        genome.finalize()
    coordinate = population[2]
    coordinate.parents = [champion.genome_id]
    coordinate.finalize()

    after = evolution.introduce_champion_profit_program_variant(
        population, champion, frontier, [], 905, {protected_parent}
    )

    transfers = [
        genome for genome in after
        if champion.genome_id in genome.parents
        and len(genome.feature_programs) > len(champion.feature_programs)
    ]
    assert len(transfers) == 1
    assert transfers[0].confidence_quantile == champion.confidence_quantile
    program_name = evolution.program_name(frontier.feature_programs[0])
    assert program_name in {
        evolution.program_name(program) for program in transfers[0].feature_programs
    }
    assert sum(
        protected_parent in genome.parents for genome in after
    ) >= 1


def test_failed_nearby_program_transfer_advances_to_next_hypothesis():
    population = seed_genomes(8, random.Random(184))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {"min_accuracy": .566, "min_profit_factor": .958},
    }
    first = {
        "op": "signed_sqrt_product", "left": "cross_rank_r1",
        "right": "causal_z14_r24", "scale": 3.2,
    }
    second = {
        "op": "regime_gate", "left": "funding_z168",
        "right": "crowd_price_alignment", "scale": 2.7,
    }
    frontier = population[1]
    frontier.fitness = 2300
    frontier.feature_programs = [first, second]
    frontier.result = {"summary": {
        "min_profit_factor": 1.05, "min_expectancy": .0005,
    }}
    failed = Genome(**{
        **champion.__dict__,
        "feature_programs": [*champion.feature_programs, first],
        "confidence_quantile": champion.confidence_quantile - .005,
        "generation": 904, "parents": ["prior-champion"],
        "genome_id": "", "fitness": 2300,
        "result": {
            "evaluation_signature": "scope-a",
            "evaluated_folds": 3, "requested_folds": 3,
            "summary": {"min_accuracy": .563, "min_profit_factor": .907},
        },
    }).finalize()
    for genome in population[2:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_profit_program_variant(
        population, champion, frontier, [failed], 906
    )

    transfer = next(
        genome for genome in after
        if champion.genome_id in genome.parents
        and len(genome.feature_programs) > len(champion.feature_programs)
    )
    names = {evolution.program_name(program) for program in transfer.feature_programs}
    assert evolution.program_name(first) not in names
    assert evolution.program_name(second) in names


def test_exhausted_profit_frontier_advances_without_spending_two_slots():
    population = seed_genomes(8, random.Random(185))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {"min_accuracy": .566, "min_profit_factor": .958},
    }
    exhausted_program = {
        "op": "signed_sqrt_product", "left": "cross_rank_r1",
        "right": "causal_z14_r24", "scale": 3.2,
    }
    next_program = {
        "op": "regime_gate", "left": "hour_sin",
        "right": "causal_z60_market_breadth_r6", "scale": 3.8,
    }
    exhausted, following = population[1:3]
    for frontier, program, profit in (
        (exhausted, exhausted_program, 1.05),
        (following, next_program, 1.21),
    ):
        frontier.fitness = 2300
        frontier.feature_programs = [program]
        frontier.result = {"summary": {
            "min_profit_factor": profit, "min_expectancy": .0005,
        }}
    failed = Genome(**{
        **champion.__dict__,
        "feature_programs": [*champion.feature_programs, exhausted_program],
        "generation": 904, "parents": ["prior-champion"],
        "genome_id": "", "fitness": 2300,
        "result": {
            "evaluation_signature": "scope-a",
            "evaluated_folds": 3, "requested_folds": 3,
            "summary": {"min_accuracy": .55, "min_profit_factor": .90},
        },
    }).finalize()
    for genome in population[3:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_profit_program_from_frontiers(
        population, champion, [exhausted, following], [failed], 907
    )

    transfers = [
        genome for genome in after
        if champion.genome_id in genome.parents
        and len(genome.feature_programs) > len(champion.feature_programs)
    ]
    assert len(transfers) == 1
    names = {evolution.program_name(program) for program in transfers[0].feature_programs}
    assert evolution.program_name(next_program) in names


def test_accurate_program_transfer_gets_selective_profit_followup():
    population = seed_genomes(8, random.Random(186))
    champion = population[0]
    champion.fitness = 2400
    champion.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {"min_accuracy": .5664, "min_profit_factor": .958},
    }
    program = {
        "op": "regime_gate", "left": "hour_sin",
        "right": "causal_z60_market_breadth_r6", "scale": 3.8,
    }
    frontier = population[1]
    frontier.fitness = 2300
    frontier.feature_programs = [program]
    frontier.result = {"summary": {
        "min_profit_factor": 1.21, "min_expectancy": .002,
    }}
    observed = Genome(**{
        **champion.__dict__,
        "feature_programs": [*champion.feature_programs, program],
        "generation": 905, "parents": [champion.genome_id],
        "genome_id": "", "fitness": 2380,
        "result": {
            "evaluation_signature": "scope-a",
            "evaluated_folds": 3, "requested_folds": 3,
            "summary": {"min_accuracy": .5668, "min_profit_factor": .932},
        },
    }).finalize()
    for genome in population[2:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_profit_program_variant(
        population, champion, frontier, [observed], 908
    )

    followup = next(
        genome for genome in after
        if champion.genome_id in genome.parents
        and len(genome.feature_programs) > len(champion.feature_programs)
    )
    assert followup.confidence_quantile == pytest.approx(
        champion.confidence_quantile + .01
    )
    assert evolution.program_name(program) in {
        evolution.program_name(item) for item in followup.feature_programs
    }


def test_full_fold_tree_champion_can_launch_return_magnitude_specialist():
    population = seed_genomes(8, random.Random(187))
    champion = population[0]
    champion.learner_kind = "extra_trees"
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    protected_parent = "active-frontier"
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None
        genome.parents = [protected_parent]
        genome.finalize()

    after = evolution.introduce_champion_return_tree_variant(
        population, champion, [], 909, {protected_parent}
    )

    specialists = [
        genome for genome in after
        if genome.learner_kind == "extra_trees_regressor"
        and champion.genome_id in genome.parents
    ]
    assert len(specialists) == 1
    assert specialists[0].features == champion.features
    assert specialists[0].feature_programs == champion.feature_programs
    assert sum(protected_parent in genome.parents for genome in after) >= 1


def test_profitable_return_tree_repairs_coverage_before_more_topology():
    population = seed_genomes(8, random.Random(192))
    champion = population[0]
    champion.learner_kind = "extra_trees"
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    observed = Genome(**{
        **champion.__dict__, "learner_kind": "extra_trees_regressor",
        "generation": 910, "parents": [champion.genome_id],
        "genome_id": "", "fitness": 405,
        "result": {
            "evaluated_folds": 1, "requested_folds": 3,
            "summary": {
                "min_accuracy": .6115, "min_balanced_accuracy": .6127,
                "min_mcc": .225, "min_profit_factor": 1.002,
                "min_coverage": .483,
            },
        },
    }).finalize()
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_return_tree_variant(
        population, champion, [observed], 911
    )

    repair = next(
        genome for genome in after
        if genome.learner_kind == "extra_trees_regressor"
    )
    expected_step = max(.02, min(.08, (.60 - .483) * .5))
    assert repair.confidence_quantile == pytest.approx(
        observed.confidence_quantile - expected_step
    )


def test_return_tree_bisects_signed_quality_coverage_boundary():
    population = seed_genomes(8, random.Random(194))
    champion = population[0]
    champion.learner_kind = "extra_trees"
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    evidence = []
    for quantile, accuracy, balanced, mcc, profit, coverage in (
        (.18, .611, .612, .225, .999, .483),
        (.12, .462, .462, -.077, .761, .619),
    ):
        evidence.append(Genome(**{
            **champion.__dict__, "learner_kind": "extra_trees_regressor",
            "confidence_quantile": quantile,
            "generation": 914, "parents": [champion.genome_id],
            "genome_id": "", "fitness": 400,
            "result": {
                "evaluated_folds": 1, "requested_folds": 3,
                "summary": {
                    "min_accuracy": accuracy,
                    "min_balanced_accuracy": balanced, "min_mcc": mcc,
                    "min_profit_factor": profit, "min_coverage": coverage,
                },
            },
        }).finalize())
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_return_tree_variant(
        population, champion, evidence, 915
    )

    repair = next(
        genome for genome in after
        if genome.learner_kind == "extra_trees_regressor"
    )
    assert repair.confidence_quantile == pytest.approx(.15)


def test_return_tree_evidence_launches_direction_magnitude_hybrid():
    population = seed_genomes(8, random.Random(196))
    champion = population[0]
    champion.learner_kind = "extra_trees"
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    evidence = []
    for quantile, accuracy, profit, coverage in (
        (.18, .611, 1.002, .483),
        (.12, .462, .761, .619),
    ):
        evidence.append(Genome(**{
            **champion.__dict__, "learner_kind": "extra_trees_regressor",
            "confidence_quantile": quantile,
            "generation": 918, "parents": [champion.genome_id],
            "genome_id": "", "fitness": 400,
            "result": {
                "evaluated_folds": 1, "requested_folds": 3,
                "summary": {
                    "min_accuracy": accuracy,
                    "min_balanced_accuracy": accuracy,
                    "min_mcc": .20 if accuracy > .60 else -.07,
                    "min_profit_factor": profit, "min_coverage": coverage,
                },
            },
        }).finalize())
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_return_tree_variant(
        population, champion, evidence, 919
    )

    hybrid = next(
        genome for genome in after
        if genome.learner_kind == "extra_trees_hybrid"
    )
    assert hybrid.confidence_quantile == pytest.approx(.12)
    assert hybrid.features == champion.features


def test_hybrid_uses_classifier_direction_and_return_magnitude_for_selection():
    values = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    labels = np.asarray([-1, -1, 1, 1], dtype=np.int8)
    returns = np.asarray([-.01, -.002, .003, .02], dtype=np.float64)
    direction = evolution.ExtraTreesClassifier(
        n_estimators=20, random_state=3,
    ).fit(values, labels)
    magnitude = evolution.ExtraTreesRegressor(
        n_estimators=20, random_state=4,
    ).fit(values, returns)
    model = evolution.Surrogate(
        evolution.ExtraTreesHybridModel(direction, magnitude),
        "extra_trees_hybrid",
    )

    probability = model.base_probability(values)
    selection = model.selection_confidence(values)

    assert probability[0] < .5 < probability[-1]
    assert selection[-1] > selection[1]
    assert np.all(selection >= 0)


def test_return_tree_smooths_ranking_before_admitting_weaker_signals():
    population = seed_genomes(8, random.Random(195))
    champion = population[0]
    champion.learner_kind = "extra_trees"
    champion.min_samples_leaf = 8
    champion.fitness = 2400
    champion.result = {"evaluated_folds": 3, "requested_folds": 3}
    observed = Genome(**{
        **champion.__dict__, "learner_kind": "extra_trees_regressor",
        "confidence_quantile": .15,
        "generation": 916, "parents": [champion.genome_id],
        "genome_id": "", "fitness": 415,
        "result": {
            "evaluated_folds": 1, "requested_folds": 3,
            "summary": {
                "min_accuracy": .60, "min_balanced_accuracy": .603,
                "min_mcc": .207, "min_profit_factor": .986,
                "min_coverage": .539,
            },
        },
    }).finalize()
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    after = evolution.introduce_champion_return_tree_variant(
        population, champion, [observed], 917
    )

    repair = next(
        genome for genome in after
        if genome.learner_kind == "extra_trees_regressor"
    )
    assert repair.confidence_quantile == pytest.approx(.15)
    assert repair.min_samples_leaf == 12
    assert repair.max_leaf_nodes == champion.max_leaf_nodes


def test_nearby_return_tree_evidence_survives_champion_quantile_handoff(tmp_path):
    old = seed_genomes(1, random.Random(193))[0]
    old.learner_kind = "extra_trees"
    old.confidence_quantile = .1796
    old.finalize()
    champion = Genome(**{
        **old.__dict__, "confidence_quantile": .1821,
        "generation": 913, "parents": [old.genome_id], "genome_id": "",
        "fitness": 2400, "result": {
            "evaluated_folds": 3, "requested_folds": 3, "summary": {},
        },
    }).finalize()
    champion.finalize()
    return_tree = Genome(**{
        **old.__dict__, "learner_kind": "extra_trees_regressor",
        "generation": 912, "parents": [old.genome_id],
        "genome_id": "", "fitness": 405,
        "result": {
            "evaluation_signature": "scope-a",
            "evaluated_folds": 1, "requested_folds": 3,
            "summary": {"min_accuracy": .611, "min_profit_factor": 1.002},
        },
    }).finalize()
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates" / f"{return_tree.genome_id}.json").write_text(
        json.dumps(return_tree.__dict__), encoding="utf-8"
    )

    evidence = evolution.load_nearby_return_tree_evidence(
        tmp_path, champion, "scope-a"
    )

    assert [genome.genome_id for genome in evidence] == [return_tree.genome_id]


def test_extra_trees_return_regressor_ranks_signed_magnitude():
    genome = seed_genomes(1, random.Random(188))[0]
    values = np.linspace(-2.0, 2.0, 80, dtype=np.float32).reshape(-1, 1)
    target = .02 * values[:, 0]
    model = evolution.new_extra_trees_return_regressor(genome, 19).fit(
        values, target
    )
    prediction = model.predict(values)
    assert prediction[0] < 0 < prediction[-1]
    assert prediction[-1] > prediction[len(prediction) // 2]


def test_champion_replacement_rejects_material_profitability_regression():
    incumbent, candidate = seed_genomes(2, random.Random(189))
    incumbent.fitness = 2394.11
    incumbent.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5664, "min_balanced_accuracy": .5589,
            "min_mcc": .1234, "min_baseline_margin": .0117,
            "min_coverage": .6489, "min_expectancy": -.000481,
            "min_profit_factor": .9581, "max_ece": .1536,
            "max_drawdown": .736,
        },
    }
    candidate.fitness = 2394.27
    candidate.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5673, "min_balanced_accuracy": .5593,
            "min_mcc": .1246, "min_baseline_margin": .0117,
            "min_coverage": .6519, "min_expectancy": -.000532,
            "min_profit_factor": .9537, "max_ece": .1505,
            "max_drawdown": .765,
        },
    }
    assert not evolution.champion_replacement_allowed(candidate, incumbent)


def test_champion_replacement_accepts_bounded_pareto_gain():
    incumbent, candidate = seed_genomes(2, random.Random(190))
    incumbent.fitness = 2394.07
    incumbent.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5664, "min_balanced_accuracy": .5589,
            "min_mcc": .1234, "min_baseline_margin": .0117,
            "min_coverage": .6474, "min_expectancy": -.000488,
            "min_profit_factor": .9576, "max_ece": .1536,
            "max_drawdown": .735,
        },
    }
    candidate.fitness = 2394.11
    candidate.result = {
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5664, "min_balanced_accuracy": .5589,
            "min_mcc": .1234, "min_baseline_margin": .0117,
            "min_coverage": .6489, "min_expectancy": -.000481,
            "min_profit_factor": .9581, "max_ece": .1536,
            "max_drawdown": .736,
        },
    }
    assert evolution.champion_replacement_allowed(candidate, incumbent)


def test_unsafe_champion_rolls_back_to_signed_parent(tmp_path):
    incumbent, candidate = seed_genomes(2, random.Random(191))
    incumbent.fitness = 2394.11
    incumbent.result = {
        "evaluation_signature": "scope-a",
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5664, "min_balanced_accuracy": .5589,
            "min_mcc": .1234, "min_baseline_margin": .0117,
            "min_coverage": .6489, "min_expectancy": -.000481,
            "min_profit_factor": .9581, "max_ece": .1536,
            "max_drawdown": .736,
        },
    }
    candidate.fitness = 2394.27
    candidate.parents = [incumbent.genome_id]
    candidate.result = {
        "evaluation_signature": "scope-a",
        "evaluated_folds": 3, "requested_folds": 3,
        "summary": {
            "min_accuracy": .5673, "min_balanced_accuracy": .5593,
            "min_mcc": .1246, "min_baseline_margin": .0117,
            "min_coverage": .6519, "min_expectancy": -.000532,
            "min_profit_factor": .9537, "max_ece": .1505,
            "max_drawdown": .765,
        },
    }
    (tmp_path / "candidates").mkdir()
    (tmp_path / "candidates" / f"{incumbent.genome_id}.json").write_text(
        json.dumps(incumbent.__dict__), encoding="utf-8"
    )

    restored, rejected = evolution.rollback_unsafe_champion(
        tmp_path, candidate, "scope-a"
    )

    assert restored.genome_id == incumbent.genome_id
    assert rejected == [candidate.genome_id]


def test_reliability_decay_search_brackets_best_protected_evidence():
    base = seed_genomes(1, random.Random(175))[0]
    evidence = []
    for index, (decay, accuracy, balanced, mcc, expectancy, profit) in enumerate((
        (.75, .4843, .4923, -.0167, -.00222, .802),
        (3.5, .4943, .4993, -.0016, -.00148, .866),
        (.25, .4839, .4933, -.0144, -.00243, .784),
        (5.0, .4765, .4765, -.0483, -.00231, .809),
        (2.125, .4952, .4949, -.0107, -.00165, .850),
        (4.25, .4916, .4938, -.0128, -.00165, .851),
        (3.875, .4943, .4999, -.0002, -.00143, .870),
        (2.8125, .4972, .4979, -.0043, -.00136, .876),
    )):
        evidence.append(Genome(**{
            **base.__dict__, "calibration_reliability": True,
            "calibration_reliability_version": 6,
            "calibration_reliability_pool": "flow_news",
            "calibration_reliability_decay": decay,
            "generation": 300 + index, "genome_id": "", "fitness": 1200,
            "result": {"summary": {
                "min_balanced_accuracy": balanced, "min_mcc": mcc,
                "min_accuracy": accuracy, "min_expectancy": expectancy,
                "min_profit_factor": profit,
            }},
        }).finalize())

    assert evolution.next_reliability_decays(evidence) == (2.46875, 3.15625)


def test_multiscale_frontier_persists_and_gets_independent_coverage_repair(tmp_path):
    evaluated = seed_genomes(8, random.Random(75))
    primary = evaluated[-1]
    primary.confidence_quantile = .264
    primary.result = {"summary": {
        "min_accuracy": .632, "min_balanced_accuracy": .621,
        "min_mcc": .25, "min_coverage": .597,
        "min_expectancy": .0004, "min_profit_factor": 1.04,
    }}
    primary.finalize()
    multiscale = Genome(**{
        **primary.__dict__, "learner_kind": "multiscale_regressor",
        "confidence_quantile": .263, "generation": 101,
        "parents": [primary.genome_id], "genome_id": "", "fitness": 470,
        "result": {"summary": {
            "min_accuracy": .645, "min_balanced_accuracy": .640,
            "min_mcc": .29, "min_coverage": .572,
            "min_expectancy": .0004, "min_profit_factor": 1.04,
        }},
    }).finalize()
    saved = evolution.update_multiscale_frontier(tmp_path, None, multiscale)
    assert saved.genome_id == multiscale.genome_id
    assert evolution.load_multiscale_frontier(tmp_path).genome_id == multiscale.genome_id

    following, _ = evolution.breed_population(
        evaluated, 34, random.Random(76), {}
    )
    repaired = evolution.introduce_coverage_repair_variants(
        following, [primary, multiscale], 34,
        multiscale_frontier=multiscale,
    )
    variants = [
        genome for genome in repaired if genome.parents == [multiscale.genome_id]
    ]
    assert len(variants) == 2
    assert all(genome.learner_kind == "multiscale_regressor" for genome in variants)
    assert all(genome.confidence_quantile < multiscale.confidence_quantile
               for genome in variants)

    reversal = Genome(**{
        **multiscale.__dict__, "confidence_quantile": .18,
        "generation": 102, "parents": [multiscale.genome_id],
        "genome_id": "", "fitness": 1200,
        "result": {
            "folds": [
                {"known_asset_future": {"directional_accuracy": .64},
                 "unseen_asset_future": {"directional_accuracy": .63}},
                {"known_asset_future": {"directional_accuracy": .46},
                 "unseen_asset_future": {"directional_accuracy": .53}},
            ],
            "summary": {"min_accuracy": .46, "min_mcc": -.05},
        },
    }).finalize()
    following, _ = evolution.breed_population(
        evaluated, 35, random.Random(77), {}
    )
    bisected = evolution.introduce_coverage_repair_variants(
        following, [primary, multiscale], 35,
        multiscale_frontier=multiscale,
        multiscale_reversal_frontier=reversal,
    )
    bisection_children = [
        genome for genome in bisected if genome.parents == [multiscale.genome_id]
    ]
    assert sorted(round(genome.confidence_quantile, 5)
                  for genome in bisection_children) == [.2215, .24225]

    boundary = Genome(**{
        **multiscale.__dict__, "confidence_quantile": .242,
        "generation": 103, "parents": [multiscale.genome_id],
        "genome_id": "", "fitness": 460,
        "result": {"summary": {
            "min_accuracy": .635, "min_balanced_accuracy": .632,
            "min_mcc": .27, "min_coverage": .592,
            "min_expectancy": -.0004, "min_profit_factor": .97,
        }},
    }).finalize()
    saved_boundary = evolution.update_multiscale_boundary_frontier(
        tmp_path, None, boundary
    )
    assert saved_boundary.genome_id == boundary.genome_id
    assert (evolution.load_multiscale_boundary_frontier(tmp_path).genome_id
            == boundary.genome_id)
    tight_reversal = Genome(**{
        **boundary.__dict__, "confidence_quantile": .221,
        "generation": 104, "parents": [boundary.genome_id],
        "genome_id": "", "fitness": 1200,
        "result": {
            "folds": [
                {"known_asset_future": {"directional_accuracy": .64},
                 "unseen_asset_future": {"directional_accuracy": .63}},
                {"known_asset_future": {"directional_accuracy": .45},
                 "unseen_asset_future": {"directional_accuracy": .53}},
            ],
            "summary": {"min_accuracy": .45, "min_mcc": -.08},
        },
    }).finalize()
    following, _ = evolution.breed_population(
        evaluated, 36, random.Random(79), {}
    )
    tightened = evolution.introduce_coverage_repair_variants(
        following, [primary, multiscale, boundary], 36,
        multiscale_frontier=multiscale,
        multiscale_reversal_frontier=tight_reversal,
        multiscale_boundary_frontier=boundary,
    )
    tight_children = [
        genome for genome in tightened if genome.parents == [boundary.genome_id]
    ]
    assert sorted(round(genome.confidence_quantile, 5)
                  for genome in tight_children) == [.2315, .23675]


def test_extra_trees_frontier_persists_and_expands_coverage(tmp_path):
    evaluated = seed_genomes(8, random.Random(81))
    frontier = evaluated[2]
    frontier.learner_kind = "extra_trees"
    frontier.confidence_quantile = .281
    frontier.fitness = 415
    frontier.result = {"summary": {
        "min_accuracy": .655, "min_balanced_accuracy": .650,
        "min_mcc": .304, "min_coverage": .424,
        "min_expectancy": .0014, "min_profit_factor": 1.13,
    }}
    frontier.finalize()
    saved = evolution.update_extra_trees_frontier(tmp_path, None, frontier)
    assert saved.genome_id == frontier.genome_id
    assert evolution.load_extra_trees_frontier(tmp_path).genome_id == frontier.genome_id

    following, _ = evolution.breed_population(
        evaluated, 40, random.Random(82), {}
    )
    repaired = evolution.introduce_extra_trees_coverage_variant(
        following, frontier, 40
    )
    children = [
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    ]
    assert len(children) == 1
    assert children[0].learner_kind == "extra_trees"
    assert children[0].confidence_quantile < frontier.confidence_quantile

    reversal = Genome(**{
        **frontier.__dict__, "confidence_quantile": .22,
        "generation": 41, "parents": [frontier.genome_id],
        "genome_id": "", "fitness": 1200,
        "result": {
            "folds": [
                {"known_asset_future": {"directional_accuracy": .65},
                 "unseen_asset_future": {"directional_accuracy": .64}},
                {"known_asset_future": {"directional_accuracy": .46},
                 "unseen_asset_future": {"directional_accuracy": .48}},
            ],
            "summary": {"min_accuracy": .46, "min_mcc": -.08},
        },
    }).finalize()
    following, _ = evolution.breed_population(
        evaluated, 41, random.Random(83), {}
    )
    bisected = evolution.introduce_extra_trees_coverage_variant(
        following, frontier, 41, reversal
    )
    child = next(
        genome for genome in bisected if genome.parents == [frontier.genome_id]
    )
    assert child.confidence_quantile == pytest.approx((.281 + .22) / 2)


def test_extra_trees_soft_accuracy_boundary_triggers_bisection():
    frontier = seed_genomes(4, random.Random(181))[-1]
    frontier.learner_kind = "extra_trees"
    frontier.confidence_quantile = .251
    frontier.result = {"summary": {
        "min_accuracy": .635, "min_balanced_accuracy": .635,
        "min_mcc": .27, "min_coverage": .532,
        "min_expectancy": .0011, "min_profit_factor": 1.10,
    }}
    frontier.finalize()
    boundary = Genome(**{
        **frontier.__dict__, "confidence_quantile": .230,
        "generation": 201, "parents": [frontier.genome_id], "genome_id": "",
        "result": {"evaluated_folds": 1, "summary": {
            "min_accuracy": .618, "min_balanced_accuracy": .618,
            "min_mcc": .236, "min_coverage": .563,
            "min_expectancy": .00069, "min_profit_factor": 1.062,
        }},
    }).finalize()
    assert evolution.compatible_reversal_rank(boundary, frontier) is not None
    population = seed_genomes(8, random.Random(182))
    repaired = evolution.introduce_extra_trees_coverage_variant(
        population, frontier, 42, boundary
    )
    child = next(
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    )
    assert child.confidence_quantile == pytest.approx((.251 + .230) / 2)


def test_profitable_primary_regressor_gets_independent_coverage_lane():
    frontier = seed_genomes(1, random.Random(197))[0]
    frontier.learner_kind = "regressor"
    frontier.confidence_quantile = .21823
    frontier.fitness = 410
    frontier.result = {"evaluated_folds": 1, "summary": {
        "min_accuracy": .6467, "min_balanced_accuracy": .6452,
        "min_mcc": .2923, "min_coverage": .5976,
        "min_acted_observations": 150, "min_expectancy": .00221,
        "min_profit_factor": 1.2156,
    }}
    frontier.finalize()
    population = seed_genomes(8, random.Random(198))
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    repaired = evolution.introduce_primary_coverage_variant(
        population, frontier, [], 920
    )

    child = next(
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    )
    assert child.learner_kind == "regressor"
    assert .21 < child.confidence_quantile < frontier.confidence_quantile
    assert child.feature_programs == frontier.feature_programs


def test_primary_coverage_lane_bisects_a_protected_fold_reversal():
    frontier = seed_genomes(1, random.Random(199))[0]
    frontier.learner_kind = "regressor"
    frontier.confidence_quantile = .218
    frontier.fitness = 410
    frontier.result = {"evaluated_folds": 1, "summary": {
        "min_accuracy": .646, "min_balanced_accuracy": .645,
        "min_mcc": .29, "min_coverage": .597,
        "min_expectancy": .002, "min_profit_factor": 1.21,
    }}
    frontier.finalize()
    reversal = Genome(**{
        **frontier.__dict__, "confidence_quantile": .214,
        "generation": 921, "parents": [frontier.genome_id],
        "genome_id": "", "fitness": 1300,
        "result": {"evaluated_folds": 2, "summary": {
            "min_accuracy": .49, "min_balanced_accuracy": .48,
            "min_mcc": -.03, "min_coverage": .62,
            "min_expectancy": -.001, "min_profit_factor": .89,
        }},
    }).finalize()
    population = seed_genomes(8, random.Random(200))
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    repaired = evolution.introduce_primary_coverage_variant(
        population, frontier, [reversal], 922
    )

    child = next(
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    )
    assert child.confidence_quantile == pytest.approx(.216)


def test_primary_coverage_lane_escapes_an_identical_score_plateau():
    frontier = seed_genomes(1, random.Random(201))[0]
    frontier.learner_kind = "regressor"
    frontier.confidence_quantile = .218
    frontier.fitness = 410
    frontier.result = {"evaluated_folds": 1, "summary": {
        "min_accuracy": .646, "min_balanced_accuracy": .645,
        "min_mcc": .29, "min_coverage": .5976,
        "min_expectancy": .002, "min_profit_factor": 1.21,
    }}
    frontier.finalize()
    plateau = Genome(**{
        **frontier.__dict__, "confidence_quantile": .2155,
        "generation": 923, "parents": [frontier.genome_id],
        "genome_id": "", "fitness": 410,
        "result": {"evaluated_folds": 1, "summary": {
            "min_accuracy": .646, "min_balanced_accuracy": .645,
            "min_mcc": .29, "min_coverage": .5976,
            "min_expectancy": .002, "min_profit_factor": 1.21,
        }},
    }).finalize()
    population = seed_genomes(8, random.Random(202))
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    repaired = evolution.introduce_primary_coverage_variant(
        population, frontier, [plateau], 924
    )

    child = next(
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    )
    assert child.confidence_quantile == pytest.approx(.2055)


def test_primary_coverage_lane_brackets_transitive_plateau_evidence():
    frontier = seed_genomes(1, random.Random(203))[0]
    frontier.learner_kind = "regressor"
    frontier.confidence_quantile = .218
    frontier.fitness = 410
    good = {
        "min_accuracy": .646, "min_balanced_accuracy": .645,
        "min_mcc": .29, "min_coverage": .5976,
        "min_acted_observations": 150, "min_expectancy": .002,
        "min_profit_factor": 1.21,
    }
    frontier.result = {"evaluated_folds": 1, "summary": good}
    frontier.finalize()
    plateau_one = Genome(**{
        **frontier.__dict__, "confidence_quantile": .215,
        "generation": 925, "parents": [frontier.genome_id],
        "genome_id": "", "fitness": 410,
        "result": {"evaluated_folds": 1, "summary": good},
    }).finalize()
    plateau_two = Genome(**{
        **frontier.__dict__, "confidence_quantile": .212,
        "generation": 926, "parents": [plateau_one.genome_id],
        "genome_id": "", "fitness": 410,
        "result": {"evaluated_folds": 1, "summary": good},
    }).finalize()
    reversal = Genome(**{
        **frontier.__dict__, "confidence_quantile": .205,
        "generation": 927, "parents": [frontier.genome_id],
        "genome_id": "", "fitness": 1260,
        "result": {"evaluated_folds": 2, "summary": {
            "min_accuracy": .449, "min_balanced_accuracy": .446,
            "min_mcc": -.11, "min_coverage": .78,
            "min_acted_observations": 184, "min_expectancy": -.0025,
            "min_profit_factor": .79,
        }},
    }).finalize()
    population = seed_genomes(8, random.Random(204))
    for genome in population[1:]:
        genome.fitness = None
        genome.result = None

    repaired = evolution.introduce_primary_coverage_variant(
        population, frontier, [plateau_one, plateau_two, reversal], 928
    )

    child = next(
        genome for genome in repaired if genome.parents == [frontier.genome_id]
    )
    assert child.learner_kind == "continuous_rank_regressor"
    assert child.confidence_quantile == pytest.approx(frontier.confidence_quantile)
    assert child.features == frontier.features

    child.fitness = frontier.fitness
    child.result = {"evaluated_folds": 1, "summary": good}
    second_population = seed_genomes(8, random.Random(206))
    for genome in second_population[1:]:
        genome.fitness = None
        genome.result = None
    continued = evolution.introduce_primary_coverage_variant(
        second_population, frontier,
        [plateau_one, plateau_two, reversal, child], 929,
    )
    threshold_child = next(
        genome for genome in continued if genome.parents == [child.genome_id]
    )
    assert threshold_child.learner_kind == "continuous_rank_regressor"
    assert threshold_child.confidence_quantile == pytest.approx((.218 + .205) / 2)
    assert threshold_child.features == frontier.features


def test_structure_evidence_recovers_indirect_descendants(tmp_path):
    frontier = seed_genomes(1, random.Random(205))[0]
    frontier.result = {"evaluation_signature": "scope-a", "summary": {}}
    frontier.fitness = 400
    frontier.finalize()
    direct = Genome(**{
        **frontier.__dict__, "confidence_quantile": .19,
        "parents": [frontier.genome_id], "genome_id": "",
    }).finalize()
    indirect = Genome(**{
        **frontier.__dict__, "confidence_quantile": .18,
        "parents": [direct.genome_id], "genome_id": "",
    }).finalize()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for candidate in (direct, indirect):
        (candidates / f"{candidate.genome_id}.json").write_text(
            json.dumps(candidate.__dict__), encoding="utf-8"
        )

    recovered = evolution.load_structure_evidence(
        tmp_path, frontier, "scope-a"
    )

    assert {candidate.genome_id for candidate in recovered} == {
        direct.genome_id, indirect.genome_id,
    }


def test_compatible_reversal_frontier_is_structure_scoped():
    upper = seed_genomes(4, random.Random(69))[-1]
    upper.confidence_quantile = .27
    upper.finalize()
    lower = Genome(**{
        **upper.__dict__, "confidence_quantile": .26,
        "generation": 70, "parents": [upper.genome_id], "genome_id": "",
        "result": {
            "folds": [
                {
                    "known_asset_future": {"directional_accuracy": .64},
                    "unseen_asset_future": {"directional_accuracy": .62},
                },
                {
                    "known_asset_future": {"directional_accuracy": .47},
                    "unseen_asset_future": {"directional_accuracy": .46},
                },
            ],
            "summary": {"min_mcc": -.05},
        },
    }).finalize()
    rank = evolution.compatible_reversal_rank(lower, upper)
    assert rank is not None and rank[0] == .26

    unrelated = seed_genomes(5, random.Random(70))[-1]
    unrelated.confidence_quantile = .26
    unrelated.result = lower.result
    unrelated.finalize()
    assert evolution.compatible_reversal_rank(unrelated, upper) is None


def test_near_coverage_frontier_survives_population_turnover(tmp_path):
    candidate = seed_genomes(8, random.Random(49))[-1]
    candidate.fitness = 390
    candidate.result = {"summary": {
        "min_accuracy": .593, "min_balanced_accuracy": .563,
        "min_mcc": .15, "min_coverage": .594,
        "min_acted_observations": 149, "min_profit_factor": 1.01,
        "min_expectancy": .0001,
    }}
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / f"{candidate.genome_id}.json").write_text(
        json.dumps(candidate.__dict__), encoding="utf-8"
    )
    recovered = evolution.load_coverage_frontier(tmp_path)
    assert recovered is not None
    assert recovered.genome_id == candidate.genome_id
    assert (tmp_path / "coverage_frontier.json").is_file()

    worse = seed_genomes(8, random.Random(50))[-1]
    worse.result = {"summary": {
        "min_accuracy": .60, "min_balanced_accuracy": .57,
        "min_mcc": .16, "min_coverage": .50,
        "min_acted_observations": 125, "min_profit_factor": .90,
        "min_expectancy": -.001,
    }}
    assert evolution.update_coverage_frontier(
        tmp_path, recovered, worse
    ).genome_id == candidate.genome_id
    same_mask_lower_threshold = Genome(**{
        **candidate.__dict__, "confidence_quantile": .15,
        "genome_id": "", "fitness": candidate.fitness,
    }).finalize()
    assert evolution.update_coverage_frontier(
        tmp_path, recovered, same_mask_lower_threshold
    ).genome_id == candidate.genome_id
    (candidates / f"{same_mask_lower_threshold.genome_id}.json").write_text(
        json.dumps(same_mask_lower_threshold.__dict__), encoding="utf-8"
    )
    (tmp_path / "coverage_frontier.json").write_text(json.dumps({
        "genome": same_mask_lower_threshold.__dict__,
    }), encoding="utf-8")
    assert evolution.load_coverage_frontier(tmp_path).genome_id == candidate.genome_id


def test_regime_shift_frontier_recovers_verified_temporal_reversal(tmp_path):
    candidate = seed_genomes(8, random.Random(51))[-1]
    candidate.result = {
        "folds": [
            {
                "known_asset_future": {"directional_accuracy": .61},
                "unseen_asset_future": {"directional_accuracy": .59},
            },
            {
                "known_asset_future": {"directional_accuracy": .47},
                "unseen_asset_future": {"directional_accuracy": .45},
            },
        ],
        "summary": {"min_mcc": -.08},
    }
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / f"{candidate.genome_id}.json").write_text(
        json.dumps(candidate.__dict__), encoding="utf-8"
    )

    assert evolution.regime_shift_rank(candidate) == (.59, -.45, -.08)
    recovered = evolution.load_regime_shift_frontier(tmp_path)
    assert recovered is not None
    assert recovered.genome_id == candidate.genome_id
    assert (tmp_path / "regime_shift_frontier.json").is_file()

    candidate.result["folds"][1]["unseen_asset_future"][
        "directional_accuracy"
    ] = .51
    candidate.result["folds"][1]["known_asset_future"][
        "directional_accuracy"
    ] = .52
    assert evolution.regime_shift_rank(candidate) is None


def test_regime_shift_frontier_launches_rotating_stability_children():
    frontier = seed_genomes(8, random.Random(52))[-1]
    frontier.recency_half_life_days = 400
    frontier.l2_regularization = 2
    frontier.result = {
        "folds": [
            {
                "known_asset_future": {"directional_accuracy": .60},
                "unseen_asset_future": {"directional_accuracy": .59},
            },
            {
                "known_asset_future": {"directional_accuracy": .46},
                "unseen_asset_future": {"directional_accuracy": .45},
            },
        ],
        "summary": {"min_mcc": -.1},
    }
    frontier.finalize()
    population = seed_genomes(8, random.Random(53))
    for genome in population:
        genome.fitness = None
        genome.emergent_pools = [{
            "features": [genome.features[0]], "concept_threshold": 5,
        }]
        genome.finalize()
    protected_parent = "coverage-parent"
    population[-1].parents = [protected_parent]
    population[-1].finalize()

    evolved = evolution.introduce_regime_shift_variants(
        population, frontier, 20, {protected_parent}
    )
    children = [
        genome for genome in evolved if genome.parents == [frontier.genome_id]
    ]
    assert len(children) == 1
    assert children[0].learner_kind == "regressor"
    assert children[0].recency_half_life_days == 600
    assert all(genome.l2_regularization == 8 for genome in children)
    routed_population = evolution.introduce_regime_shift_variants(
        seed_genomes(8, random.Random(65)), frontier, 21
    )
    routed = next(
        genome for genome in routed_population
        if genome.parents == [frontier.genome_id]
    )
    assert routed.learner_kind == "regime_decomposed_regressor"
    assert routed.recency_half_life_days == 1600
    assert routed.l2_regularization == 8
    assert routed.regime_feature == "market_median_r6"
    assert routed.regime_bins == 2
    assert any(genome.parents == [protected_parent] for genome in evolved)
    assert any(
        genome.emergent_pools and genome.parents != [frontier.genome_id]
        for genome in evolved
    )


def test_pending_queue_prioritizes_active_accuracy_repairs():
    (coverage, shift, multiscale, champion, ordinary, slow_shift,
     coverage_child) = seed_genomes(
        7, random.Random(54)
    )
    multiscale.learner_kind = "multiscale_regressor"
    multiscale.finalize()
    multiscale_child = evolution.mutate(multiscale, 30, random.Random(55))
    multiscale_child.parents = [multiscale.genome_id]
    multiscale_child.finalize()
    slow_shift.parents = [shift.genome_id]
    coverage_child.parents = [coverage.genome_id]
    slow_shift.finalize()
    coverage_child.finalize()
    champion_child = evolution.mutate(champion, 30, random.Random(56))
    champion_child.parents = [champion.genome_id]
    champion_child.finalize()

    ordered = evolution.prioritize_pending_genomes(
        [slow_shift, ordinary, multiscale_child, coverage_child, champion_child],
        coverage, shift, multiscale, None, champion,
    )
    assert ordered == [
        champion_child, coverage_child, multiscale_child, ordinary, slow_shift,
    ]


def test_resume_seeders_cannot_erase_active_frontier_children():
    population = seed_genomes(8, random.Random(59))
    protected_parent = "active-accuracy-frontier"
    for genome in population:
        genome.learner_kind = "classifier"
        genome.features = sorted(set(genome.features) - evolution.REFLEXIVITY_FEATURES)
        genome.emergent_pools = []
        genome.finalize()
    protected_child = population[-1]
    protected_child.parents = [protected_parent]
    protected_child.finalize()
    protected_id = protected_child.genome_id

    population = evolution.introduce_missing_learner_species(
        population, 30, random.Random(60), {protected_parent}
    )
    population = evolution.introduce_reflexivity_variant(
        population, 30, {protected_parent}
    )
    population = evolution.introduce_emergent_pool_variant(
        population, 30, random.Random(61), {protected_parent}
    )
    assert any(genome.genome_id == protected_id for genome in population)


def test_surplus_expensive_regime_candidates_are_converted():
    population = seed_genomes(8, random.Random(67))
    shift_frontier = seed_genomes(2, random.Random(68))[1]
    for genome in population:
        genome.fitness = None
    for genome in population[-3:]:
        genome.learner_kind = "regime_decomposed_regressor"
        genome.regime_bins = 3
        genome.finalize()
    population[-2].parents = [shift_frontier.genome_id]
    population[-2].finalize()

    updated, converted = evolution.cap_expensive_regime_candidates(
        population, 40, shift_frontier
    )
    remaining = [
        genome for genome in updated
        if genome.fitness is None
        and genome.learner_kind == "regime_decomposed_regressor"
    ]
    assert converted == 2
    assert len(remaining) == 1
    assert remaining[0].parents == [shift_frontier.genome_id]

    lone_random = seed_genomes(4, random.Random(78))
    for genome in lone_random:
        genome.fitness = None
    lone_random[-1].learner_kind = "regime_decomposed_regressor"
    lone_random[-1].parents = ["unverified-random-parent"]
    lone_random[-1].finalize()
    updated, converted = evolution.cap_expensive_regime_candidates(
        lone_random, 41, shift_frontier
    )
    assert converted == 1
    assert all(genome.learner_kind != "regime_decomposed_regressor"
               for genome in updated)


def test_untargeted_multiscale_candidates_are_converted():
    population = seed_genomes(7, random.Random(80))
    for genome in population:
        genome.fitness = None
    protected_parent = "active-multiscale-boundary"
    population[-2].learner_kind = "multiscale_regressor"
    population[-2].parents = [protected_parent]
    population[-2].finalize()
    duplicate = Genome(**{
        **population[-2].__dict__,
        "confidence_quantile": population[-2].confidence_quantile + .000004,
        "genome_id": "", "fitness": None, "result": None,
    }).finalize()
    population[-1] = duplicate
    population[-3].learner_kind = "multiscale_regressor"
    population[-3].parents = ["unrelated-mutation"]
    population[-3].finalize()

    updated, converted = evolution.cap_expensive_multiscale_candidates(
        population, 42, {protected_parent}
    )
    remaining = [
        genome for genome in updated
        if genome.fitness is None and genome.learner_kind == "multiscale_regressor"
    ]
    assert converted == 2
    assert len(remaining) == 1
    assert remaining[0].parents == [protected_parent]


def test_resumed_multiscale_duplicate_of_evaluated_probe_is_converted():
    population = seed_genomes(6, random.Random(206))
    protected_parent = "active-multiscale-boundary"
    evaluated = population[-2]
    evaluated.learner_kind = "multiscale_regressor"
    evaluated.parents = [protected_parent]
    evaluated.fitness = 1300
    evaluated.result = {"status": "prescreen_reject", "summary": {}}
    evaluated.finalize()
    pending = Genome(**{
        **evaluated.__dict__,
        "confidence_quantile": evaluated.confidence_quantile + .000004,
        "genome_id": "", "fitness": None, "result": None,
    }).finalize()
    population[-1] = pending

    updated, converted = evolution.cap_expensive_multiscale_candidates(
        population, 43, {protected_parent}
    )

    assert converted == 1
    assert updated[-2].learner_kind == "multiscale_regressor"
    assert updated[-1].learner_kind == "regressor"


def test_genome_outcome_pool_learns_live_metric_rankings():
    evidence = []
    for index in range(120):
        genome = seed_genomes(1, random.Random(300 + index))[0]
        coordinate = (index % 20) / 20.0
        genome.confidence_quantile = .03 + .24 * coordinate
        genome.generation = index
        genome.fitness = 300 + index
        genome.result = {
            "status": "prescreen_reject",
            "evaluated_folds": 1 + index % 3,
            "requested_folds": 3,
            "summary": {
                "min_accuracy": .48 + .16 * coordinate,
                "min_balanced_accuracy": .47 + .15 * coordinate,
                "min_mcc": -.04 + .30 * coordinate,
                "min_coverage": .82 - .30 * coordinate,
                "min_profit_factor": .72 + 1.1 * coordinate,
                "min_expectancy": -.002 + .004 * coordinate,
                "max_ece": .18 - .08 * coordinate,
                "max_drawdown": 1.2 - .5 * coordinate,
            },
        }
        genome.finalize()
        evidence.append(genome)

    pool = evolution.train_genome_outcome_pool(evidence)

    assert pool is not None
    assert pool.examples == 120
    assert len(pool.models) == 3
    prediction, uncertainty = pool.predict(evidence[-2:])
    assert prediction.shape == (2, len(evolution.OUTCOME_POOL_TARGETS))
    assert uncertainty.shape == prediction.shape
    accuracy_index = evolution.OUTCOME_POOL_TARGETS.index("accuracy")
    assert pool.validation_rank_correlation[accuracy_index] > .25


def test_outcome_pool_profit_target_changes_acquisition_priority():
    conservative = np.asarray([.7, .62, .60, .62, .65, .35, .55, .7, .7])
    profitable = conservative.copy()
    profitable[5] = .75
    uncertainty = np.zeros_like(conservative)

    assert evolution.outcome_acquisition(
        profitable, uncertainty
    ) > evolution.outcome_acquisition(conservative, uncertainty)


def test_active_outcome_pool_reserves_only_one_reproduction_slot():
    class FixedOutcomeModel:
        def predict(self, values):
            target = np.asarray([.8, .64, .62, .62, .66, .72, .58, .7, .7])
            return np.tile(target, (len(values), 1))

    population = seed_genomes(8, random.Random(421))
    evaluated = seed_genomes(8, random.Random(422))
    for index, genome in enumerate(evaluated):
        genome.fitness = 2000 - index
        genome.result = {
            "evaluated_folds": 3, "requested_folds": 3,
            "summary": {"min_accuracy": .56, "min_profit_factor": 1.0},
        }
    for genome in population[3:]:
        genome.fitness = None
        genome.result = None
    width = len(evolution.genome_outcome_vector(population[0]))
    targets = len(evolution.OUTCOME_POOL_TARGETS)
    pool = evolution.GenomeOutcomePool(
        np.zeros(width), np.ones(width), np.zeros(targets), np.ones(targets),
        [FixedOutcomeModel(), FixedOutcomeModel()], 120,
        [.05] * targets, [.10] * targets, [.4] * targets, True,
    )

    before = {genome.genome_id for genome in population}
    updated, report = evolution.introduce_outcome_pool_variant(
        population, evaluated, pool, 430, random.Random(423)
    )

    introduced = [genome for genome in updated if genome.genome_id not in before]
    assert report["active"] is True and report["proposed"] is True
    assert len(introduced) == 1
    assert introduced[0].fitness is None
    assert report["genome_id"] == introduced[0].genome_id


def test_reflexivity_variant_is_seeded_without_erasing_learner_diversity():
    population = seed_genomes(12, random.Random(14))
    for genome in population:
        genome.features = sorted(set(genome.features) - evolution.REFLEXIVITY_FEATURES)
        genome.finalize()
    original_species = {genome.learner_kind for genome in population}
    updated = introduce_reflexivity_variant(population, 7)
    reflexive = [genome for genome in updated
                 if set(genome.features) & evolution.REFLEXIVITY_FEATURES]
    assert len(reflexive) == 1
    assert evolution.REFLEXIVITY_FEATURES <= set(reflexive[0].features)
    assert original_species <= {genome.learner_kind for genome in updated}


def test_accuracy_log_contains_only_strict_comparable_improvements(tmp_path):
    assert not record_accuracy_improvement(
        tmp_path, "corpus-a", "protected_surrogate", .55,
        generation=1, genome_id="baseline",
    )
    assert not (tmp_path / "accuracy_improvements.jsonl").exists()
    assert not record_accuracy_improvement(
        tmp_path, "corpus-a", "protected_surrogate", .54,
        generation=2, genome_id="worse",
    )
    assert record_accuracy_improvement(
        tmp_path, "corpus-a", "protected_surrogate", .57,
        generation=3, genome_id="better", metrics={"min_mcc": .2},
    )
    rows = [json.loads(line) for line in
            (tmp_path / "accuracy_improvements.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "accuracy_increased"
    assert rows[0]["previous_accuracy"] == .55
    assert rows[0]["accuracy"] == .57
    # A new corpus/source starts its own silent baseline.
    assert not record_accuracy_improvement(
        tmp_path, "corpus-b", "protected_surrogate", .80,
        generation=4, genome_id="new-corpus",
    )
    assert not record_accuracy_improvement(
        tmp_path, "corpus-a", "isolated_wizard_brain", .60,
        generation=4, genome_id="neural-baseline",
    )


def test_stale_brain_gate_detection_is_age_and_state_scoped(monkeypatch, tmp_path):
    class Entry:
        def __init__(self, pid, command, created):
            self.info = {
                "pid": pid, "cmdline": command, "create_time": created,
            }

    state_text = str(tmp_path.resolve())
    entries = [
        Entry(10, ["python", "market_evolution_brain_gate.py",
                   "--candidate", f"{state_text}\\candidates\\a.json"], 100),
        Entry(11, ["python", "market_evolution_brain_gate.py",
                   "--candidate", f"{state_text}\\candidates\\b.json"], 950),
        Entry(12, ["python", "market_evolution_brain_gate.py",
                   "--candidate", "D:\\unrelated\\candidates\\c.json"], 100),
        Entry(13, ["python", "unrelated.py", state_text], 100),
    ]

    class FakePsutil:
        AccessDenied = RuntimeError
        NoSuchProcess = LookupError

        @staticmethod
        def process_iter(_attributes):
            return entries

    monkeypatch.setattr(evolution, "psutil", FakePsutil)
    assert evolution.stale_external_brain_gate_pids(
        tmp_path, 300, now=1000
    ) == [10]


def test_stale_brain_gate_recovery_attempts_only_discovered_pids(
    monkeypatch, tmp_path,
):
    attempted = []
    monkeypatch.setattr(
        evolution, "stale_external_brain_gate_pids",
        lambda state_dir, timeout: [20, 21],
    )
    monkeypatch.setattr(
        evolution, "terminate_process_tree",
        lambda pid: attempted.append(pid) or pid == 21,
    )
    assert evolution.recover_stale_external_brain_gates(tmp_path, 900) == [21]
    assert attempted == [20, 21]


def test_interrupted_candidates_restore_only_for_exact_evaluation(tmp_path):
    population = seed_genomes(2, random.Random(64))
    for genome in population:
        genome.fitness = None
        genome.result = None
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    completed = Genome(**{
        **population[0].__dict__, "fitness": 123.0,
        "result": {"evaluation_signature": "scope-a", "status": "reject"},
    })
    mismatched = Genome(**{
        **population[1].__dict__, "fitness": 456.0,
        "result": {"evaluation_signature": "scope-b", "status": "reject"},
    })
    (candidates / f"{completed.genome_id}.json").write_text(
        json.dumps(completed.__dict__), encoding="utf-8"
    )
    (candidates / f"{mismatched.genome_id}.json").write_text(
        json.dumps(mismatched.__dict__), encoding="utf-8"
    )

    restored, identities = evolution.restore_completed_candidates(
        population, tmp_path, "scope-a"
    )
    assert identities == [completed.genome_id]
    assert restored[0].fitness == 123.0
    assert restored[1].fitness is None

    legacy_population = seed_genomes(1, random.Random(66))
    legacy = Genome(**{
        **legacy_population[0].__dict__, "fitness": 789.0,
        "result": {"status": "legacy-interrupted"},
    })
    legacy_path = candidates / f"{legacy.genome_id}.json"
    legacy_path.write_text(json.dumps(legacy.__dict__), encoding="utf-8")
    restored, identities = evolution.restore_completed_candidates(
        legacy_population, tmp_path, "scope-a",
        legacy_after=legacy_path.stat().st_mtime - 1,
    )
    assert identities == [legacy.genome_id]
    assert restored[0].result["evaluation_signature"] == "scope-a"


def test_identical_phenotype_reuses_signed_evidence_across_generations(tmp_path):
    original = seed_genomes(1, random.Random(164))[0]
    original.fitness = 432.1
    original.result = {
        "evaluation_signature": "scope-a", "status": "prescreen_reject",
        "summary": {"min_accuracy": .61},
    }
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / f"{original.genome_id}.json").write_text(
        json.dumps(original.__dict__), encoding="utf-8"
    )
    descendant = Genome(**{
        **original.__dict__, "generation": original.generation + 1,
        "parents": [original.genome_id], "genome_id": "",
        "fitness": None, "result": None,
    }).finalize()
    assert descendant.genome_id != original.genome_id
    restored, identities = evolution.restore_completed_candidates(
        [descendant], tmp_path, "scope-a"
    )
    assert identities == [descendant.genome_id]
    assert restored[0].genome_id == descendant.genome_id
    assert restored[0].parents == [original.genome_id]
    assert restored[0].fitness == 432.1
    assert restored[0].result["summary"]["min_accuracy"] == .61


def test_retired_reliability_v0_reuses_only_ordinary_phenotype_evidence(tmp_path):
    ordinary = seed_genomes(1, random.Random(174))[0]
    ordinary.fitness = 432.1
    ordinary.result = {
        "evaluation_signature": "scope-a", "status": "ordinary-evidence",
    }
    retired = Genome(**{
        **ordinary.__dict__, "calibration_reliability": True,
        "calibration_reliability_version": 0, "genome_id": "",
        "fitness": 999.0,
        "result": {
            "evaluation_signature": "scope-a", "status": "retired-evidence",
        },
    }).finalize()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for evidence in (ordinary, retired):
        (candidates / f"{evidence.genome_id}.json").write_text(
            json.dumps(evidence.__dict__), encoding="utf-8"
        )

    target = Genome(**{
        **retired.__dict__, "generation": retired.generation + 1,
        "parents": [retired.genome_id], "genome_id": "",
        "fitness": None, "result": None,
    }).finalize()
    restored, identities = evolution.restore_completed_candidates(
        [target], tmp_path, "scope-a"
    )

    assert evolution.genome_evaluation_key(target) == evolution.genome_evaluation_key(
        ordinary
    )
    assert identities == [target.genome_id]
    assert restored[0].fitness == 432.1
    assert restored[0].result["status"] == "ordinary-evidence"


def test_direct_descendant_evidence_persists_search_memory(tmp_path):
    parent = seed_genomes(1, random.Random(264))[0]
    child = Genome(**{
        **parent.__dict__, "generation": parent.generation + 1,
        "parents": [parent.genome_id], "genome_id": "", "fitness": 1200,
        "result": {"evaluation_signature": "scope-a", "evaluated_folds": 2},
    }).finalize()
    stale = Genome(**{
        **child.__dict__, "generation": child.generation + 1,
        "genome_id": "", "fitness": 1300,
        "result": {"evaluation_signature": "scope-b", "evaluated_folds": 3},
    }).finalize()
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    for candidate in (child, stale):
        (candidates / f"{candidate.genome_id}.json").write_text(
            json.dumps(candidate.__dict__), encoding="utf-8"
        )
    evidence = evolution.load_direct_descendant_evidence(
        tmp_path, {parent.genome_id}, "scope-a"
    )
    assert [candidate.genome_id for candidate in evidence] == [child.genome_id]


def test_conclusive_multifold_anti_signal_skips_brain_gate(tmp_path):
    candidate = seed_genomes(1, random.Random(364))[0]
    candidate.fitness = 2200
    candidate.result = {
        "evaluated_folds": 3,
        "summary": {
            "min_accuracy": .47, "min_mcc": -.10,
            "min_expectancy": -.002, "min_profit_factor": .77,
        },
    }
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / f"{candidate.genome_id}.json").write_text(
        json.dumps(candidate.__dict__), encoding="utf-8"
    )
    assert not evolution.brain_gate_obligation_viable(
        tmp_path, candidate.genome_id
    )
    candidate.result["evaluated_folds"] = 1
    (candidates / f"{candidate.genome_id}.json").write_text(
        json.dumps(candidate.__dict__), encoding="utf-8"
    )
    assert evolution.brain_gate_obligation_viable(tmp_path, candidate.genome_id)


def test_emergent_pool_gene_is_deterministic_bounded_and_feature_scoped():
    candidate = seed_genomes(6, random.Random(31))[0]
    candidate.emergent_pools = [
        {"features": ["r6"], "concept_threshold": 99},
        {"features": ["not_a_feature"], "concept_threshold": 4},
    ]
    candidate.finalize()
    assert len(candidate.emergent_pools) == 1
    assert candidate.emergent_pools[0]["name"].startswith("emergent_")
    assert candidate.emergent_pools[0]["features"] == ["r6"]
    assert candidate.emergent_pools[0]["concept_threshold"] == 12
    first_id = candidate.genome_id
    candidate.finalize()
    assert candidate.genome_id == first_id


def test_resumed_population_immediately_launches_pool_topology_experiment():
    population = seed_genomes(6, random.Random(32))
    assert not any(genome.emergent_pools for genome in population)
    updated = introduce_emergent_pool_variant(population, 7, random.Random(33))
    specialists = [genome for genome in updated if genome.emergent_pools]
    assert len(specialists) == 1
    specialist = specialists[0]
    assert specialist.fitness is None and specialist.result is None
    assert specialist.parents
    assert random_emergent_pool(specialist, random.Random(34)) is not None


def test_emergent_topology_species_survives_flat_surrogate_selection():
    evaluated = seed_genomes(6, random.Random(35))
    specialist = evaluated[-1]
    specialist.emergent_pools = [{"features": [specialist.features[0]],
                                  "concept_threshold": 5}]
    specialist.finalize()
    following = [genome for genome in evaluated[:-1]]
    following.append(seed_genomes(1, random.Random(36))[0])
    assert not any(genome.emergent_pools for genome in following)
    preserved = preserve_emergent_pool_elite(following, list(reversed(evaluated)))
    assert any(genome.genome_id == specialist.genome_id for genome in preserved)


def test_brain_accuracy_summary_uses_weakest_neural_section():
    report = {"folds": [{"sections": {
        "known": {"metrics": {"directional_accuracy": .72,
                                "directional_balanced_accuracy": .68,
                                "mcc": .3, "profit_factor": 1.4, "ece": .08}},
        "unseen": {"metrics": {"directional_accuracy": .59,
                                 "directional_balanced_accuracy": .56,
                                 "mcc": .16, "profit_factor": 1.2, "ece": .10}},
    }}]}
    summary = brain_accuracy_summary(report)
    assert summary["min_accuracy"] == .59
    assert summary["min_balanced_accuracy"] == .56
    assert summary["sections"] == 2
