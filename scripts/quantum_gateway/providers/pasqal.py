from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # Pasqal Cloud Services.
    # The official Python SDK is pasqal-cloud; authentication can be via SDK login or Auth0 tokens.
    # This gateway exposes raw_rest; set base_url in config to match your Pasqal endpoint.
    return RestProvider(
        name="pasqal",
        base_url=str(cfg.get("base_url", "")) or "https://cloud.pasqal.com",
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "PASQAL_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        cfg=cfg,
    )

