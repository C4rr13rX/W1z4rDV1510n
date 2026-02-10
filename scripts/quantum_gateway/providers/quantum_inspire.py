from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # Quantum Inspire low-level API + SDK.
    # The exact API base URL depends on Quantum Inspire deployment; set base_url in config.
    return RestProvider(
        name="quantum_inspire",
        base_url=str(cfg.get("base_url", "")) or "https://api.quantum-inspire.com",
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "QUANTUM_INSPIRE_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        cfg=cfg,
    )

