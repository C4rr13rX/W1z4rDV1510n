from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Dict, List, Optional, Tuple


def safe_softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    if not scores:
        return []
    t = max(float(temperature), 1e-9)
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    if z <= 0.0 or not math.isfinite(z):
        n = len(scores)
        return [1.0 / n for _ in range(n)]
    return [e / z for e in exps]


def normalize_probabilities(probs: List[float]) -> List[float]:
    if not probs:
        return []
    cleaned = [0.0 if (p is None or not math.isfinite(float(p))) else max(float(p), 0.0) for p in probs]
    s = sum(cleaned)
    if s <= 0.0:
        n = len(cleaned)
        return [1.0 / n for _ in range(n)]
    return [p / s for p in cleaned]


def kl_divergence(p: List[float], q: List[float]) -> float:
    # KL(p || q) with eps to avoid log(0). Not symmetric.
    eps = 1e-12
    if len(p) != len(q) or not p:
        return float("inf")
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(float(pi), 0.0)
        qi = max(float(qi), 0.0)
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return total


@dataclasses.dataclass
class CacheEntry:
    value: Any
    expires_at: float


class Cache:
    def __init__(self, max_entries: int = 10_000, ttl_secs: int = 60):
        self.max_entries = int(max_entries)
        self.ttl_secs = int(ttl_secs)
        self._store: Dict[Any, CacheEntry] = {}

    def get(self, key: Any) -> Optional[Any]:
        now = time.time()
        ent = self._store.get(key)
        if ent is None:
            return None
        if ent.expires_at < now:
            self._store.pop(key, None)
            return None
        return ent.value

    def put(self, key: Any, value: Any) -> None:
        if len(self._store) >= self.max_entries:
            # Drop an arbitrary key to stay bounded (O(1) worst case).
            try:
                self._store.pop(next(iter(self._store.keys())))
            except StopIteration:
                pass
        self._store[key] = CacheEntry(value=value, expires_at=time.time() + self.ttl_secs)


class OutcomeProtector:
    """
    Guardrails for remote quantum/provider outputs:
    - enforce shape invariants
    - normalize probabilities
    - reject wild divergences vs local baseline (prevents provider drift/outages from harming outcomes)
    """

    def __init__(self, max_kl: float = 1.5, max_l1: float = 0.85):
        self.max_kl = float(max_kl)
        self.max_l1 = float(max_l1)

    def accept_probabilities(self, local: List[float], remote: List[float]) -> Tuple[bool, str]:
        if not local or not remote or len(local) != len(remote):
            return False, "shape_mismatch"
        local = normalize_probabilities(local)
        remote = normalize_probabilities(remote)
        # Fast sanity: L1 distance too big => likely garbage.
        l1 = sum(abs(a - b) for a, b in zip(local, remote))
        if not math.isfinite(l1):
            return False, "l1_non_finite"
        if l1 > self.max_l1:
            return False, f"l1_reject:{l1:.3f}"
        kl = kl_divergence(remote, local)
        if not math.isfinite(kl):
            return False, "kl_non_finite"
        if kl > self.max_kl:
            return False, f"kl_reject:{kl:.3f}"
        return True, f"accepted:l1={l1:.3f},kl={kl:.3f}"

