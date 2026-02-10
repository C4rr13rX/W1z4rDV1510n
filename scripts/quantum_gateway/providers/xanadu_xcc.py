from __future__ import annotations

from typing import Any, Dict

from .base import Provider


class XanaduXccProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError("xanadu_xcc: branch_scoring not implemented")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        try:
            import xc  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "xanadu_xcc: missing dependency xanadu-cloud-client; pip install xanadu-cloud-client"
            ) from e
        raise RuntimeError("xanadu_xcc: SDK is available but this gateway does not yet map tasks -> XCC jobs; use XCC directly or contribute an adapter")


def build_provider(cfg: dict):
    return XanaduXccProvider(name="xanadu_xcc", priority=int(cfg.get("priority", 50)), cfg=cfg)

