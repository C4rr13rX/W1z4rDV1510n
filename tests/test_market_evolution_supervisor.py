import json
import random

from scripts.market_evolution_genome import seed
from scripts.market_evolution_supervisor import (
    atomic_json, dataset_signature, next_population, unique_population,
)


def test_population_preserves_elite_and_produces_unique_children():
    rng = random.Random(4)
    population = unique_population([seed(rng, i) for i in range(3)], 6, rng)
    ranked = [(genome, {"score": float(100 - index)}) for index, genome in enumerate(population)]
    following = next_population(ranked, 6, 2, rng)
    assert len({genome.genome_id for genome in following}) == 6
    assert population[0] in following and population[1] in following


def test_atomic_state_and_dataset_signature_change_with_data(tmp_path):
    manifest = tmp_path / "manifest.json"
    supplement = tmp_path / "supplement"
    features = supplement / "features"
    features.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    (features / "BTCUSDT.json").write_text("{}", encoding="utf-8")
    first = dataset_signature(manifest, supplement)
    (features / "BTCUSDT.json").write_text('{"rows":[]}', encoding="utf-8")
    assert dataset_signature(manifest, supplement) != first
    state = tmp_path / "state.json"
    atomic_json(state, {"ok": True})
    assert json.loads(state.read_text())["ok"] is True
