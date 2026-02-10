from __future__ import annotations

from .rest_base import RestProvider


def build_provider(cfg: dict):
    # qBraid REST API.
    # Header name varies by endpoint/version; configure auth_header/auth_prefix as needed.
    return RestProvider(
        name="qbraid",
        base_url=str(cfg.get("base_url", "")) or "https://api.qbraid.com",
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "QBRAID_API_KEY")),
        auth_header=str(cfg.get("auth_header", "X-API-KEY")),
        auth_prefix=str(cfg.get("auth_prefix", "")),
        cfg=cfg,
    )

