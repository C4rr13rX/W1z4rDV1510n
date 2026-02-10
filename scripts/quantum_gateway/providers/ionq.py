from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # IonQ Quantum Cloud API (REST)
    # Base URL: https://api.ionq.co/v0.4
    # Auth: Authorization: apiKey <token>
    return RestProvider(
        name="ionq",
        base_url=str(cfg.get("base_url", "https://api.ionq.co/v0.4")),
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "IONQ_API_KEY")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "apiKey ")),
        cfg=cfg,
    )

