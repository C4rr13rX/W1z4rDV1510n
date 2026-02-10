"""
Quantum Gateway
==============

This is a small HTTP service that implements the Rust-side QuantumHttpExecutor protocol:

POST /quantum/submit
{
  "kind": "BRANCH_SCORING" | "QUANTUM_CALIBRATION" | ...,
  "payload_b64": "<base64 of job payload bytes>",
  "timeout_secs": 30
}

Response:
{
  "payload_b64": "<base64 of response payload bytes>",
  "metadata": { "...": "..." }
}

Design goals (aligned with W1z4rDV1510n's architecture):
- Outcome protection: validate/normalize outputs, fall back safely, attach rich metadata.
- Provider orchestration: route to a configured provider or run locally when unavailable.
- Firepower: expose an extra /quantum/experiment endpoint for direct user experiments.

The gateway is dependency-free (stdlib only). Provider SDK integrations are optional and
loaded lazily when available.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from outcome_protection import (
    Cache,
    OutcomeProtector,
    normalize_probabilities,
    safe_softmax,
)
from providers import ProviderRegistry


@dataclasses.dataclass(frozen=True)
class QuantumEndpointRequest:
    kind: str
    payload_b64: str
    timeout_secs: int


@dataclasses.dataclass
class QuantumEndpointResponse:
    payload_b64: str
    metadata: Dict[str, str]


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=True)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class GatewayState:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._lock = threading.Lock()
        self._last_mtime = 0.0
        self.providers = ProviderRegistry([])
        self.cache = Cache(max_entries=20_000)
        self.protector = OutcomeProtector()
        self.auth_header = "Authorization"
        self.auth_prefix = "Bearer "
        self.auth_env = "W1Z4RDV1510N_QUANTUM_GATEWAY_TOKEN"
        self.reload()

    def reload(self) -> None:
        with self._lock:
            try:
                st = os.stat(self.config_path)
                mtime = st.st_mtime
            except FileNotFoundError:
                # Keep defaults; allow running with only env vars.
                return
            if mtime <= self._last_mtime:
                return
            raw = open(self.config_path, "rb").read()
            cfg = json.loads(raw.decode("utf-8"))
            self._last_mtime = mtime
            self.providers = ProviderRegistry(cfg.get("providers", []))
            auth = cfg.get("auth", {}) or {}
            self.auth_env = str(auth.get("env", self.auth_env))
            self.auth_header = str(auth.get("header", self.auth_header))
            self.auth_prefix = str(auth.get("prefix", self.auth_prefix))

    def check_auth(self, headers: Dict[str, str]) -> Optional[str]:
        token = os.environ.get(self.auth_env, "").strip()
        if not token:
            return None  # auth disabled
        expected = f"{self.auth_prefix}{token}"
        got = ""
        # HTTP headers are case-insensitive, but BaseHTTPRequestHandler exposes original keys.
        for k, v in headers.items():
            if k.lower() == self.auth_header.lower():
                got = v.strip()
                break
        if got != expected:
            return "unauthorized"
        return None


class QuantumGatewayHandler(BaseHTTPRequestHandler):
    server_version = "w1z4rdv1510n-quantum-gateway/0.1"

    def _read_json(self) -> Tuple[Optional[dict], Optional[str]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "invalid content-length"
        if length <= 0:
            return None, "empty body"
        if length > 10 * 1024 * 1024:
            return None, "request too large"
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except Exception:
            return None, "invalid json"

    def _send_json(self, status: int, obj: Any) -> None:
        raw = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802
        state: GatewayState = self.server.state  # type: ignore[attr-defined]
        state.reload()

        # Auth (optional)
        auth_err = state.check_auth({k: v for k, v in self.headers.items()})
        if auth_err:
            self._send_json(HTTPStatus.UNAUTHORIZED, {"error": auth_err})
            return

        if self.path == "/quantum/submit":
            body, err = self._read_json()
            if err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
                return
            resp, err = handle_quantum_submit(state, body)
            if err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
                return
            self._send_json(HTTPStatus.OK, dataclasses.asdict(resp))
            return

        if self.path == "/quantum/experiment":
            body, err = self._read_json()
            if err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
                return
            out, err = handle_experiment(state, body)
            if err:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": err})
                return
            self._send_json(HTTPStatus.OK, out)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep logs quiet by default; opt-in via env.
        if os.environ.get("W1Z4RDV1510N_QUANTUM_GATEWAY_LOG", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            super().log_message(fmt, *args)


def handle_quantum_submit(state: GatewayState, body: dict) -> Tuple[Optional[QuantumEndpointResponse], Optional[str]]:
    try:
        req = QuantumEndpointRequest(
            kind=str(body["kind"]),
            payload_b64=str(body["payload_b64"]),
            timeout_secs=int(body.get("timeout_secs", 30)),
        )
    except Exception:
        return None, "invalid request shape"

    try:
        payload = _b64d(req.payload_b64)
    except Exception:
        return None, "invalid payload_b64"

    started = _now_ms()
    job_hash = _sha256_hex(payload)

    if req.kind == "BRANCH_SCORING":
        out_payload, meta, err = run_branch_scoring(state, payload, req.timeout_secs)
        if err:
            return None, err
        meta = dict(meta)
        meta.setdefault("job_hash", job_hash)
        meta.setdefault("gateway_ms", str(_now_ms() - started))
        return QuantumEndpointResponse(payload_b64=_b64e(out_payload), metadata=meta), None

    if req.kind == "QUANTUM_CALIBRATION":
        out_payload, meta, err = run_quantum_calibration(state, payload)
        if err:
            return None, err
        meta = dict(meta)
        meta.setdefault("job_hash", job_hash)
        meta.setdefault("gateway_ms", str(_now_ms() - started))
        return QuantumEndpointResponse(payload_b64=_b64e(out_payload), metadata=meta), None

    return None, f"unsupported kind {req.kind}"


def run_branch_scoring(state: GatewayState, payload: bytes, timeout_secs: int) -> Tuple[bytes, Dict[str, str], Optional[str]]:
    # Payload is JSON from crates/core streaming branching runtime.
    try:
        req = json.loads(payload.decode("utf-8"))
        candidates = list(req.get("candidates") or [])
    except Exception:
        return b"", {}, "invalid BRANCH_SCORING payload"

    if not candidates:
        resp = {"scores": [], "probabilities": [], "metadata": {"used_remote": "false"}}
        return _json_bytes(resp), {"used_remote": "false"}, None

    # Baseline scores: negative "energy" proxy (higher is better) derived from available fields.
    # This mirrors the Rust local scoring but stays tolerant to missing fields.
    scores = []
    for c in candidates:
        intensity = float(c.get("intensity", 0.0))
        base_rate = float(c.get("base_rate", 0.0))
        expected = float(c.get("expected_time_secs", 0.0))
        time_penalty = min(max(expected / 120.0, 0.0), 1.0)
        energy = (1.0 - min(max(intensity, 0.0), 1.0)) + time_penalty * 0.25 + (1.0 - min(max(base_rate, 0.0), 1.0)) * 0.1
        scores.append(-energy)

    local_probs = safe_softmax(scores, temperature=0.7)
    local_probs = normalize_probabilities(local_probs)

    # Outcome protection: cache + provider orchestration + divergence checks.
    cache_key = ("BRANCH_SCORING", _sha256_hex(payload))
    cached = state.cache.get(cache_key)
    if cached is not None:
        resp = {
            "scores": scores,
            "probabilities": cached["probabilities"],
            "metadata": dict(cached.get("metadata") or {}),
        }
        resp["metadata"].setdefault("cache_hit", "true")
        return _json_bytes(resp), resp["metadata"], None

    selected = state.providers.select_for("branch_scoring")
    if not selected:
        resp = {
            "scores": scores,
            "probabilities": local_probs,
            "metadata": {"used_remote": "false", "reason": "no_providers_configured"},
        }
        return _json_bytes(resp), resp["metadata"], None

    # Try providers in priority order; fall back to local if they fail or look suspicious.
    for provider in selected:
        try:
            remote = provider.score_branches(req, timeout_secs=timeout_secs)
        except Exception as e:
            continue

        probs = remote.get("probabilities")
        meta = dict(remote.get("metadata") or {})
        if probs is None:
            continue
        try:
            probs = normalize_probabilities([float(x) for x in probs])
        except Exception:
            continue

        ok, reason = state.protector.accept_probabilities(local_probs, probs)
        meta.setdefault("used_remote", "true" if ok else "false")
        meta.setdefault("provider", provider.name)
        meta.setdefault("protector", reason)
        if ok:
            state.cache.put(cache_key, {"probabilities": probs, "metadata": meta})
            resp = {"scores": scores, "probabilities": probs, "metadata": meta}
            return _json_bytes(resp), meta, None

    resp = {
        "scores": scores,
        "probabilities": local_probs,
        "metadata": {"used_remote": "false", "reason": "all_providers_failed_or_rejected"},
    }
    return _json_bytes(resp), resp["metadata"], None


def run_quantum_calibration(state: GatewayState, payload: bytes) -> Tuple[bytes, Dict[str, str], Optional[str]]:
    # Payload is crates/core QuantumCalibrationRequest.
    try:
        req = json.loads(payload.decode("utf-8"))
    except Exception:
        return b"", {}, "invalid QUANTUM_CALIBRATION payload"

    # Simple heuristic: if acceptance is too low, reduce driver / raise temperature scale;
    # if acceptance is too high but energy isn't improving, increase driver.
    acceptance = req.get("acceptance_ratio")
    trace = list(req.get("energy_trace") or [])
    best_energy = float(req.get("best_energy", 0.0))

    adjust: Dict[str, Optional[float]] = {
        "driver_strength_scale": None,
        "driver_final_strength_scale": None,
        "slice_temperature_scale": None,
        "worldline_mix_delta": None,
    }

    # Trend: compare first and last samples (robust to short traces)
    trend = 0.0
    if len(trace) >= 2:
        trend = float(trace[-1]) - float(trace[0])

    if isinstance(acceptance, (int, float)):
        acc = float(acceptance)
        if acc < 0.15:
            adjust["driver_strength_scale"] = 0.9
            adjust["driver_final_strength_scale"] = 0.9
            adjust["slice_temperature_scale"] = 1.1
            adjust["worldline_mix_delta"] = 0.05
        elif acc > 0.6 and trend >= -1e-9:
            adjust["driver_strength_scale"] = 1.1
            adjust["driver_final_strength_scale"] = 1.1
            adjust["slice_temperature_scale"] = 0.95
            adjust["worldline_mix_delta"] = -0.02
        elif acc > 0.6 and trend < 0.0:
            adjust["slice_temperature_scale"] = 0.98
    else:
        # If acceptance isn't reported, keep changes conservative and bias toward mixing.
        adjust["worldline_mix_delta"] = 0.02

    meta = {
        "used_remote": "false",
        "calibration_best_energy": f"{best_energy:.6g}",
        "calibration_trend": f"{trend:.6g}",
        "calibration_trace_n": str(len(trace)),
    }
    resp = {"adjustments": adjust, "metadata": meta}
    return _json_bytes(resp), meta, None


def handle_experiment(state: GatewayState, body: dict) -> Tuple[Optional[dict], Optional[str]]:
    """
    Direct power-user endpoint.

    Shape:
    {
      "provider": "ionq" | "dwave" | ... (optional; auto-select if omitted),
      "task": { "type": "qubo" | "circuit_qasm" | "raw", ... },
      "timeout_secs": 60
    }
    """
    task = body.get("task")
    if not isinstance(task, dict):
        return None, "missing task"
    timeout_secs = int(body.get("timeout_secs", 60))
    provider_name = body.get("provider")

    if provider_name:
        provider = state.providers.get(str(provider_name))
        if provider is None:
            return None, f"unknown provider {provider_name}"
        out = provider.run_experiment(task, timeout_secs=timeout_secs)
        return out, None

    providers = state.providers.select_for("experiment")
    if not providers:
        return None, "no providers configured"
    last_err = None
    for p in providers:
        try:
            return p.run_experiment(task, timeout_secs=timeout_secs), None
        except Exception as e:
            last_err = str(e)
            continue
    return None, last_err or "all providers failed"


def main() -> None:
    config_path = os.environ.get(
        "W1Z4RDV1510N_QUANTUM_GATEWAY_CONFIG",
        os.path.join(os.path.dirname(__file__), "gateway_config.json"),
    )
    listen = os.environ.get("W1Z4RDV1510N_QUANTUM_GATEWAY_ADDR", "127.0.0.1:5050")
    host, port_s = listen.rsplit(":", 1)
    port = int(port_s)
    state = GatewayState(config_path=config_path)

    httpd = ThreadingHTTPServer((host, port), QuantumGatewayHandler)
    httpd.state = state  # type: ignore[attr-defined]
    print(f"quantum gateway listening on http://{listen} (config: {config_path})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()

