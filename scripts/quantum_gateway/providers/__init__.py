from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from .base import Provider


_PROVIDER_MODULES = {
    # Unified / major platforms
    "braket": "providers.braket",
    "azure_quantum": "providers.azure_quantum",
    "ibm_runtime": "providers.ibm_runtime",
    "google_engine": "providers.google_engine",
    "strangeworks_azure": "providers.strangeworks_azure",
    "qbraid": "providers.qbraid",
    # Direct vendor APIs
    "ionq": "providers.ionq",
    "dwave": "providers.dwave",
    "pasqal": "providers.pasqal",
    "xanadu_xcc": "providers.xanadu_xcc",
    "aqt": "providers.aqt",
    "quandela": "providers.quandela",
    "scaleway_quandela": "providers.scaleway_quandela",
    "quantum_inspire": "providers.quantum_inspire",
}


def _load_provider(name: str, cfg: dict) -> Optional[Provider]:
    mod_name = _PROVIDER_MODULES.get(name)
    if not mod_name:
        return None
    mod = importlib.import_module(mod_name)
    build = getattr(mod, "build_provider", None)
    if build is None:
        return None
    provider = build(cfg)
    if provider is None:
        return None
    return provider


class ProviderRegistry:
    def __init__(self, provider_configs: List[dict]):
        self._providers: Dict[str, Provider] = {}
        for cfg in provider_configs or []:
            if not isinstance(cfg, dict):
                continue
            name = str(cfg.get("name", "")).strip()
            if not name:
                continue
            if not bool(cfg.get("enabled", True)):
                continue
            p = _load_provider(name, cfg)
            if p is None:
                # Keep an explicit stub so callers can show capability gaps clearly.
                self._providers[name] = Provider.stub(name, cfg)
            else:
                self._providers[name] = p

    def get(self, name: str) -> Optional[Provider]:
        return self._providers.get(name)

    def select_for(self, purpose: str) -> List[Provider]:
        # Priority: lower is earlier.
        providers = list(self._providers.values())
        providers.sort(key=lambda p: (p.priority, p.name))
        out: List[Provider] = []
        for p in providers:
            if p.supports(purpose):
                out.append(p)
        return out

