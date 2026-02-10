from __future__ import annotations

from typing import Any, Dict

from .base import Provider


class QuandelaPercevalProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError("quandela: branch_scoring not implemented")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        try:
            import perceval as pcvl  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "quandela: missing dependency perceval-quandela (Perceval); install Perceval per Quandela docs"
            ) from e
        raise RuntimeError("quandela: Perceval is available but this gateway does not yet implement a remote processor adapter; use Perceval directly or contribute an adapter")


def build_provider(cfg: dict):
    return QuandelaPercevalProvider(name="quandela", priority=int(cfg.get("priority", 50)), cfg=cfg)

