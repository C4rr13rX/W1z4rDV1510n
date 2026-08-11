import tomllib

from scripts.market_evolution_brain_gate import (
    add_baselines,
    available_port,
    emergent_pool_layout,
    feature_family,
    feature_frame,
    genome_feature_frame,
    quantize,
    retain_attempt,
    render_identity,
    settle_brain,
    streams,
)
from scripts.market_evolution_service import Genome, program_name


def genome():
    return Genome(features=["r6", "funding_rate"], learning_rate=.1, max_iter=100,
                  max_leaf_nodes=12, min_samples_leaf=20, l2_regularization=1,
                  confidence_quantile=.2, binding_threshold=7, concept_threshold=9,
                  presentations=3).finalize()


def test_feature_frame_is_named_deterministic_and_atom_grounded():
    row = {"features": {"r6": .0123, "funding_rate": -.0001}}
    assert feature_frame(row, genome().features) == "funding_rate=n1e-4 r6=p1.2e-2"
    assert quantize(0) == "zero"


def test_genome_feature_frame_includes_heritable_expression():
    candidate = genome()
    candidate.feature_programs = [
        {"op": "sub", "left": "r6", "right": "funding_rate", "scale": 1}
    ]
    candidate.finalize()
    frame = genome_feature_frame(
        {"features": {"r6": .01, "funding_rate": .001}}, candidate
    )
    assert "evolved_" in frame
    assert "=p9e-3" in frame


def test_feature_families_fire_in_separate_atom_grounded_pools():
    candidate = genome()
    candidate.features = ["r6", "funding_rate"]
    row = {"asset": "ETH", "features": {"r6": .01, "funding_rate": .001}}
    fired = streams(row, candidate, 12)
    pool_ids = [pool_id for pool_id, _ in fired]
    assert len(pool_ids) == len(set(pool_ids))
    assert feature_family("r6") == "price"
    assert feature_family("funding_rate") == "derivatives"
    assert feature_family("participant_consensus") == "reflexivity"
    assert any("r6=" in frame for _, frame in fired)
    assert any("funding_rate=" in frame for _, frame in fired)
    assert any("regime_feature=" in frame for _, frame in fired)
    assert any("specialists=" in frame for _, frame in fired)


def test_isolated_feature_launches_specialist_pool_and_cofires_with_classes():
    candidate = genome()
    candidate.feature_programs = [
        {"op": "sub", "left": "r6", "right": "funding_rate", "scale": 1}
    ]
    candidate.finalize()
    evolved_name = program_name(candidate.feature_programs[0])
    candidate.emergent_pools = [{
        "features": [evolved_name], "concept_threshold": 4,
    }]
    candidate.finalize()
    row = {"asset": "ETH", "features": {"r6": .01, "funding_rate": .001}}
    fired = streams(row, candidate, 12)
    layout = emergent_pool_layout(candidate)
    dynamic_id = layout[0]["id"]
    assert dynamic_id >= 100
    assert any(pool_id == dynamic_id and evolved_name in frame
               for pool_id, frame in fired)
    assert any(pool_id == 15 for pool_id, _ in fired)
    assert any(pool_id == 18 for pool_id, _ in fired)
    assert not any(pool_id == 21 and evolved_name in frame
                   for pool_id, frame in fired)


def test_rendered_identity_applies_brain_genes_without_lowering_outcome_threshold(tmp_path):
    template = tmp_path / "template.toml"
    template.write_text('binding_emergence_threshold = 3\n[[pools]]\nkind = "SensoryInput"\nconcept_emergence_threshold = 5\n[[pools]]\nkind = "Action"\nconcept_emergence_threshold = 3\n')
    output = tmp_path / "identity.toml"
    render_identity(template, output, genome())
    text = output.read_text()
    assert "binding_emergence_threshold = 7" in text
    assert 'kind = "SensoryInput"\nconcept_emergence_threshold = 9' in text
    assert 'kind = "Action"\nconcept_emergence_threshold = 3' in text


def test_rendered_identity_declares_dynamic_pool_with_matching_route(tmp_path):
    candidate = genome()
    candidate.emergent_pools = [{"features": ["r6"], "concept_threshold": 6}]
    candidate.finalize()
    template = tmp_path / "template.toml"
    template.write_text('name = "test"\nbinding_emergence_threshold = 3\n')
    output = tmp_path / "identity.toml"
    render_identity(template, output, candidate)
    pool = emergent_pool_layout(candidate)[0]
    text = output.read_text()
    assert f'name = "{pool["name"]}"' in text
    assert f'id = {pool["id"]}' in text
    assert "concept_emergence_threshold = 6" in text
    parsed = tomllib.loads(text)
    declared = next(item for item in parsed["pools"] if item["id"] == pool["id"])
    assert declared["name"] == pool["name"]
    assert declared["kind"] == "SensoryInput"


def test_dynamic_gate_port_can_be_reserved():
    port = available_port()
    assert 1024 < port <= 65535


def test_neural_gate_settles_without_pruning_before_evaluation():
    class Client:
        def __init__(self):
            self.posts = []
            self.stats = iter([
                {"resident_terminals": 2048, "tick": 12},
                {"resident_terminals": 0, "tick": 12},
            ])

        def get(self, path):
            assert path == "/brain/stats"
            return next(self.stats)

        def post(self, path, payload):
            self.posts.append((path, payload))
            return {"ok": True}

    client = Client()
    report = settle_brain(client)
    assert report["before"]["resident_terminals"] == 2048
    assert report["after"]["resident_terminals"] == 0
    assert client.posts == [
        ("/brain/sleep", {
            "min_use_count": 0,
            "stale_ticks": 9_223_372_036_854_775_807,
        }),
        ("/brain/checkpoint", {}),
    ]
    assert not hasattr(client, "timeout")


def test_neural_gate_restores_normal_timeout_after_long_maintenance():
    class Client:
        timeout = 90

        def __init__(self):
            self.observed_timeouts = []
            self.stats = iter([{"resident_terminals": 8}, {"resident_terminals": 0}])

        def get(self, path):
            return next(self.stats)

        def post(self, path, payload):
            self.observed_timeouts.append(self.timeout)
            return {"ok": True}

    client = Client()
    settle_brain(client)
    assert client.observed_timeouts == [900, 900]
    assert client.timeout == 90


def test_only_passing_or_explicitly_retained_gate_keeps_generated_brain():
    assert retain_attempt(True, False)
    assert retain_attempt(False, True)
    assert not retain_attempt(False, False)


def test_shared_metrics_are_normalized_to_admission_action_count():
    metrics = {"acted_directional_n": 7, "directional_accuracy": .5}
    rows = [
        {"actual": "updraft", "momentum_direction": 1},
        {"actual": "downshift", "momentum_direction": -1},
    ]
    add_baselines(metrics, rows)
    assert metrics["acted_observations"] == 7
    assert metrics["best_baseline_accuracy"] == 1.0
