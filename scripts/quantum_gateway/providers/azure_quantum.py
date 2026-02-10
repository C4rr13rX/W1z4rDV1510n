from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # Azure Quantum REST.
    # Auth is typically OAuth2 bearer token (Authorization: Bearer <token>).
    # Some scenarios may use x-ms-quantum-api-key; use cfg to override header/prefix/env.
    return RestProvider(
        name="azure_quantum",
        base_url=str(cfg.get("base_url", "")) or "https://quantum.azure.com",
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "AZURE_QUANTUM_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        cfg=cfg,
    )

