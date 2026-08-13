import json

from scripts.market_metric_fusion_audit import build_report


def _candidate(path, candidate_id, signature, folds, accuracy, profit, ece):
    payload = {
        "genome_id": candidate_id,
        "learner_kind": "extra_trees",
        "generation": 1,
        "result": {
            "evaluation_signature": signature,
            "evaluated_folds": folds,
            "requested_folds": 3,
            "summary": {
                "min_accuracy": accuracy,
                "min_profit_factor": profit,
                "max_ece": ece,
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_metric_audit_keeps_extrema_provenance_and_fold_scope(tmp_path):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (tmp_path / "state.json").write_text(
        json.dumps({"dataset_signature": "fresh"}), encoding="utf-8"
    )
    _candidate(candidates / "old.json", "old", "old-scope", 1, .70, 1.5, .20)
    _candidate(candidates / "fresh.json", "fresh", "fresh", 3, .60, 1.2, .10)
    _candidate(candidates / "tied.json", "tied", "fresh", 1, .60, 1.2, .10)

    report = build_report(tmp_path)

    assert report["historical_extrema"]["min_accuracy"]["candidate_id"] == "old"
    assert report["full_retention_extrema"]["min_accuracy"]["candidate_id"] == "fresh"
    assert report["current_signature_extrema"]["min_profit_factor"]["value"] == 1.2
    assert report["historical_extrema"]["max_ece"]["candidate_id"] == "fresh"
    assert report["signature_counts"] == {"fresh": 2, "old-scope": 1}
