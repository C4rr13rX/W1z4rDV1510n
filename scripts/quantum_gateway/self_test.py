from __future__ import annotations

import base64
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)

import gateway  # noqa: E402


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def test_branch_scoring_local() -> None:
    state = gateway.GatewayState(config_path="__missing__.json")
    payload = {
        "timestamp": {"unix": 123},
        "candidates": [
            {
                "event_kind": "BehavioralAtom",
                "intensity": 0.9,
                "expected_time_secs": 1.0,
                "base_rate": 0.2,
                "payload": {},
            },
            {
                "event_kind": "TopicEvent",
                "intensity": 0.1,
                "expected_time_secs": 20.0,
                "base_rate": 0.9,
                "payload": {},
            },
        ],
    }
    req = {
        "kind": "BRANCH_SCORING",
        "payload_b64": _b64e(json.dumps(payload).encode("utf-8")),
        "timeout_secs": 3,
    }
    resp, err = gateway.handle_quantum_submit(state, req)
    assert err is None
    out = json.loads(base64.b64decode(resp.payload_b64).decode("utf-8"))
    probs = out.get("probabilities")
    assert isinstance(probs, list) and len(probs) == 2
    assert abs(sum(probs) - 1.0) < 1e-6


def test_quantum_calibration_local() -> None:
    state = gateway.GatewayState(config_path="__missing__.json")
    payload = {
        "run_id": "r1",
        "best_energy": -1.2,
        "acceptance_ratio": 0.1,
        "energy_trace": [0.0, -0.1, -0.2],
        "config": {
            "trotter_slices": 16,
            "driver_strength": 0.5,
            "driver_final_strength": 0.2,
            "worldline_mix_prob": 0.1,
            "slice_temperature_scale": 1.0,
        },
    }
    req = {
        "kind": "QUANTUM_CALIBRATION",
        "payload_b64": _b64e(json.dumps(payload).encode("utf-8")),
        "timeout_secs": 3,
    }
    resp, err = gateway.handle_quantum_submit(state, req)
    assert err is None
    out = json.loads(base64.b64decode(resp.payload_b64).decode("utf-8"))
    assert "adjustments" in out


if __name__ == "__main__":
    test_branch_scoring_local()
    test_quantum_calibration_local()
    print("ok")
