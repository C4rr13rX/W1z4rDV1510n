from __future__ import annotations

from typing import Any, Dict

from .base import Provider


class StrangeworksAzureProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError("strangeworks_azure: branch_scoring not implemented")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        try:
            import strangeworks_azure  # type: ignore  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "strangeworks_azure: missing dependency strangeworks-azure; pip install strangeworks-azure"
            ) from e
        raise RuntimeError("strangeworks_azure: SDK is available but task -> provider mapping not implemented; use the SDK directly or contribute an adapter")


def build_provider(cfg: dict):
    return StrangeworksAzureProvider(
        name="strangeworks_azure", priority=int(cfg.get("priority", 50)), cfg=cfg
    )

