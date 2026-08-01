from scripts.market_evolution_brain_gate import (
    available_port,
    feature_frame,
    genome_feature_frame,
    quantize,
    render_identity,
    settle_brain,
)
from scripts.market_evolution_service import Genome


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


def test_rendered_identity_applies_brain_genes_without_lowering_outcome_threshold(tmp_path):
    template = tmp_path / "template.toml"
    template.write_text('binding_emergence_threshold = 3\n[[pools]]\nkind = "SensoryInput"\nconcept_emergence_threshold = 5\n[[pools]]\nkind = "Action"\nconcept_emergence_threshold = 3\n')
    output = tmp_path / "identity.toml"
    render_identity(template, output, genome())
    text = output.read_text()
    assert "binding_emergence_threshold = 7" in text
    assert 'kind = "SensoryInput"\nconcept_emergence_threshold = 9' in text
    assert 'kind = "Action"\nconcept_emergence_threshold = 3' in text


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
