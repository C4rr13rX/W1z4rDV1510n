from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # Google Quantum Engine (Quantum AI) uses OAuth2 bearer token.
    # The Cirq client (cirq-google) is the common integration; this module exposes raw_rest
    # for direct REST calls when a token is available.
    return RestProvider(
        name="google_engine",
        base_url=str(cfg.get("base_url", "https://quantumengine.googleapis.com/v1")),
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "GOOGLE_OAUTH_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        cfg=cfg,
    )

