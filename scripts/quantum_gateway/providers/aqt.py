from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # AQT Public API (OpenAPI): https://arnica.aqt.eu/api/v1/docs
    return RestProvider(
        name="aqt",
        base_url=str(cfg.get("base_url", "https://arnica.aqt.eu/api/v1")),
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "AQT_API_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        cfg=cfg,
    )

