from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional


@dataclasses.dataclass
class Provider:
    name: str
    priority: int = 50
    cfg: Optional[dict] = None

    def supports(self, purpose: str) -> bool:
        # purpose: "branch_scoring" | "experiment"
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        # Default: allow experiments if configured; branch_scoring only if explicitly enabled.
        if purpose == "experiment":
            return True
        return False

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise NotImplementedError

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def stub(name: str, cfg: dict) -> "Provider":
        return _StubProvider(name=name, priority=int(cfg.get("priority", 50)), cfg=cfg)


class _StubProvider(Provider):
    def supports(self, purpose: str) -> bool:
        # Allow listing/selection but fail with a clear error.
        return True

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError(f"provider {self.name} is configured but not implemented in this gateway build")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError(f"provider {self.name} is configured but not implemented in this gateway build")

