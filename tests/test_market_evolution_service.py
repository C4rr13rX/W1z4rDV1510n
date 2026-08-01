import random

from scripts.market_evolution_service import (
    Genome, add_derived_features, attach_news_features, crossover, mutate,
    dataset_signature, passes_floor, recover_pending_gate, seed_genomes,
)


def test_genome_identity_is_deterministic_and_mutation_stays_bounded():
    parent = seed_genomes(5, random.Random(1))[0]
    clone = Genome(**{**parent.__dict__, "features": list(reversed(parent.features))}).finalize()
    assert clone.genome_id == parent.genome_id
    child = mutate(parent, 2, random.Random(3))
    assert 0 <= child.confidence_quantile <= .30
    assert 2 <= child.binding_threshold <= 9
    assert len(child.features) >= 8


def test_crossover_preserves_a_viable_feature_genome():
    parents = seed_genomes(5, random.Random(2))
    child = crossover(parents[1], parents[2], 4, random.Random(4))
    assert len(child.features) >= 8
    assert child.generation == 4
    assert len(child.parents) >= 1


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


def test_derived_features_use_only_present_causal_values():
    features = {
        "spot_taker_imbalance": .2, "futures_taker_imbalance": -.1,
        "flow_divergence": -.3, "futures_spot_basis": .01, "funding_rate": .0001,
        "r6": .02, "r24": .04, "rv24": .01, "rv168": .02,
        "market_median_r6": .01, "trend_vote": 3, "market_breadth_r6": .6,
        "futures_quote_ratio24": 1.4, "spot_quote_ratio24": 1.1,
    }
    rows = [{"features": features}]
    add_derived_features(rows)
    assert rows[0]["features"]["flow_consensus"] == .05
    assert abs(rows[0]["features"]["breadth_gap_r6"] - .01) < 1e-12


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
