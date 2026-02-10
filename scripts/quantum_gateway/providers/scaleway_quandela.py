from __future__ import annotations

from typing import Any, Dict

from .base import Provider


class ScalewayQuandelaProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError("scaleway_quandela: branch_scoring not implemented")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        # Scaleway QaS uses Scaleway credentials plus Perceval to access Quandela QPUs.
        try:
            import perceval as pcvl  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "scaleway_quandela: missing Perceval dependency; install Perceval per Quandela/Scaleway docs"
            ) from e
        raise RuntimeError("scaleway_quandela: Perceval is available but this gateway does not yet implement Scaleway credential wiring; use Scaleway docs or contribute an adapter")


def build_provider(cfg: dict):
    return ScalewayQuandelaProvider(
        name="scaleway_quandela", priority=int(cfg.get("priority", 50)), cfg=cfg
    )

