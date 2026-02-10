from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from .base import Provider
from .rest_base import RestProvider


class DWaveOceanProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose in ("experiment", "branch_scoring")

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        """
        Map branch candidates to a one-hot QUBO and sample.

        This turns "score candidates" into an optimization + sampling problem:
        - objective: pick exactly one candidate with maximum score
        - reads: distribution over solutions => distribution over candidates
        """
        try:
            import dimod  # type: ignore
            from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "dwave(ocean): missing dwave-ocean-sdk; pip install dwave-ocean-sdk"
            ) from e

        candidates = list((request_obj or {}).get("candidates") or [])
        if not candidates:
            return {"probabilities": [], "metadata": {"provider": "dwave", "used_remote": "false"}}

        # Use intensity as a score proxy (aligned with the Rust-side baseline).
        scores: List[float] = []
        for c in candidates:
            intensity = float((c or {}).get("intensity", 0.0))
            base_rate = float((c or {}).get("base_rate", 0.0))
            expected = float((c or {}).get("expected_time_secs", 0.0))
            time_penalty = min(max(expected / 120.0, 0.0), 1.0)
            energy = (1.0 - min(max(intensity, 0.0), 1.0)) + time_penalty * 0.25 + (1.0 - min(max(base_rate, 0.0), 1.0)) * 0.1
            scores.append(-energy)  # higher is better

        n = len(scores)
        max_abs = max((abs(s) for s in scores), default=1.0)
        penalty = float((self.cfg or {}).get("onehot_penalty", 0.0)) or (2.0 * max_abs + 1.0)

        # QUBO: minimize -sum(score_i * x_i) + penalty*(sum x_i - 1)^2
        linear = {}
        quadratic = {}
        for i, s in enumerate(scores):
            # From penalty*(x_i - 2*x_i) term => -penalty*x_i; plus objective -s*x_i
            linear[f"x{i}"] = (-s) + (-penalty)
        for i in range(n):
            for j in range(i + 1, n):
                quadratic[(f"x{i}", f"x{j}")] = 2.0 * penalty

        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        sampler = EmbeddingComposite(DWaveSampler(**((self.cfg or {}).get("sampler_kwargs") or {})))
        num_reads = int((self.cfg or {}).get("num_reads", 200))
        sampleset = sampler.sample(bqm, num_reads=max(num_reads, 10))

        counts = [0] * n
        total = 0
        for sample in sampleset.samples():
            chosen = None
            for i in range(n):
                if int(sample.get(f"x{i}", 0)) == 1:
                    chosen = i
                    break
            if chosen is None:
                continue
            counts[chosen] += 1
            total += 1

        if total <= 0:
            probs = [1.0 / n for _ in range(n)]
        else:
            probs = [c / total for c in counts]

        # Normalize defensively.
        s = sum(p for p in probs if math.isfinite(p) and p > 0.0)
        if s > 0.0:
            probs = [max(p, 0.0) / s for p in probs]

        meta = {
            "provider": "dwave",
            "used_remote": "true",
            "num_reads": str(num_reads),
            "onehot_penalty": f"{penalty:.6g}",
            "solver": str(getattr(getattr(sampler, "child", None), "solver", {})),
        }
        return {"probabilities": probs, "metadata": meta}

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        # For now, experiments are expected to be handled via Ocean directly, or via raw REST mode.
        raise RuntimeError("dwave(ocean): use /quantum/experiment with a 'raw_rest' task against SAPI, or extend this adapter")


def build_provider(cfg: dict):
    # If Ocean is installed and branch_scoring is enabled, provide a higher-level sampler.
    want_ocean = bool(cfg.get("use_ocean", True))
    if want_ocean:
        try:
            import dimod  # type: ignore  # noqa: F401
            import dwave  # type: ignore  # noqa: F401
            # Ocean exists -> return the higher-level provider.
            return DWaveOceanProvider(name="dwave", priority=int(cfg.get("priority", 50)), cfg=cfg)
        except Exception:
            pass

    # Fallback: raw REST wrapper for SAPI.
    return RestProvider(
        name="dwave",
        # D-Wave SAPI is region-specific; docs often use na-west-1 + /sapi/v2.
        base_url=str(cfg.get("base_url", "https://na-west-1.cloud.dwavesys.com/sapi/v2")),
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "DWAVE_API_TOKEN")),
        auth_header=str(cfg.get("auth_header", "X-Auth-Token")),
        auth_prefix=str(cfg.get("auth_prefix", "")),
        cfg=cfg,
    )
