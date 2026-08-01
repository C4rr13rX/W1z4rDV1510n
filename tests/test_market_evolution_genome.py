import random

from scripts.market_evolution_genome import crossover, load_genome, mutate, report_fitness, seed


def test_market_genomes_mutate_and_cross_without_invalid_features():
    rng = random.Random(19)
    left = seed(rng, 0)
    right = seed(rng, 1)
    for _ in range(100):
        left = mutate(left, rng)
        left.validate()
        assert load_genome(left.as_json()) == left
    child = crossover(left, right, rng)
    child.validate()
    assert child.parents == (left.genome_id, right.genome_id)


def test_fitness_uses_weakest_fold_and_requires_every_gate():
    passing = {"directional_accuracy": .63, "directional_balanced_accuracy": .60,
               "mcc": .28, "coverage": .75, "acted_observations": 220,
               "net_expectancy": .002, "profit_factor": 1.4, "ece": .05,
               "baseline_margin": .08}
    report = {"feature_sets": {"genome": [
        {"sections": {"known_asset_future": {"selective": dict(passing)},
                      "unseen_asset_future": {"selective": dict(passing)}}}
        for _ in range(3)
    ]}}
    score, summary = report_fitness(report)
    assert score > 0 and summary["admitted"]
    report["feature_sets"]["genome"][2]["sections"]["unseen_asset_future"]["selective"]["net_expectancy"] = -.001
    _, failed = report_fitness(report)
    assert not failed["admitted"]
    assert failed["minima"]["expectancy"] == -.001
