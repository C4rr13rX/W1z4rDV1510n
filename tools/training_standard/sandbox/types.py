"""training_standard/sandbox/types.py — shared types."""
from __future__ import annotations

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True)
class CheckResult:
    ok: bool
    stderr: str = ""
    duration_ms: int = 0
    backend: str = ""

    @classmethod
    def passed(cls, *, backend: str, duration_ms: int = 0) -> "CheckResult":
        return cls(ok=True, backend=backend, duration_ms=duration_ms)

    @classmethod
    def failed(cls, stderr: str, *, backend: str, duration_ms: int = 0) -> "CheckResult":
        # Truncate huge compiler output so manifests stay readable.
        if len(stderr) > 4000:
            stderr = stderr[:2000] + "\n...[truncated]...\n" + stderr[-1500:]
        return cls(ok=False, stderr=stderr, backend=backend, duration_ms=duration_ms)


class Sandbox(Protocol):
    """Per-language syntactic/structural validation.

    `check` returns ok=True iff the code parses/compiles cleanly.  It
    does NOT execute user code — we only want syntactic + type-check
    validation, not behavioural verification.  Behavioural checks live
    in the benchmark layer (training_standard/runner.py), where the
    brain's response is scored against keyword / AST contracts.
    """

    def available(self) -> bool: ...
    def check(self, lang: str, code: str, *, timeout_s: float = 15.0) -> CheckResult: ...
