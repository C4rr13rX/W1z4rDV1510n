import random

from scripts.ga_architecture_genome import crossover, fitness, mutate, seed


def test_structural_mutations_remain_valid_and_materialize_isolated_instances():
    rng = random.Random(7)
    parent = seed("market", rng)
    for i in range(40):
        child = mutate(parent, rng, i)
        child.validate()
        spec = child.materialize(f"runtime/test/{child.name}")
        assert spec["deployment"]["data_dir"].endswith(child.name)
        assert spec["deployment"]["feedback_loops"]


def test_composition_and_prediction_have_more_fitness_weight_than_recall_alone():
    recall_only = fitness({"exact_recall": 1.0})
    generalized = fitness({"exact_recall": 0.8, "composition": 0.8,
                           "directional_edge_over_baseline": 0.2})
    assert generalized > recall_only


def test_crossover_combines_topology_and_records_lineage():
    rng = random.Random(7)
    left = seed("coding", rng)
    right = mutate(left, rng, 1)
    left.inference_mode = "integrate"
    right.inference_mode = "chat"
    child = crossover(left, right, rng, 2)
    child.validate()
    assert child.parents == [left.name, right.name]
    assert {p.name for p in left.pools} <= {p.name for p in child.pools}
    assert child.inference_mode == "hybrid"
