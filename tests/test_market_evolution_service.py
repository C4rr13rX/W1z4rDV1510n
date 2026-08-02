import json
import math
import random

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
    load_dataset_cached, mutate, passes_floor,
    passes_prescreen, program_name, program_value, recover_pending_gate,
    regression_probability_scale, seed_genomes, select_diverse_elites,
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
