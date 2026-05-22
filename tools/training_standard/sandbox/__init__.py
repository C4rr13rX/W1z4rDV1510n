"""training_standard/sandbox — per-language code validation.

Every ingest/generate script that emits code-bearing rows MUST pass the
candidate code through this sandbox before writing the row.  The
default backend is Docker (one short-lived container per language,
read-only mount, no network); the "local" backend uses the host's
toolchain and exists only for development when Docker Desktop is
unavailable.

Usage:
    from tools.training_standard.sandbox import get_sandbox

    sb = get_sandbox()                  # auto-selects docker vs local
    result = sb.check("python", code)
    if not result.ok:
        # row rejected — log result.stderr and skip
        ...

Backend selection:
    W1Z4RD_SANDBOX=docker   (default; fails fast if Docker is off)
    W1Z4RD_SANDBOX=local    (dev override; uses host toolchains)
    W1Z4RD_SANDBOX=auto     (docker → local fallback, with warning)
"""
from __future__ import annotations

import os

from .types import CheckResult, Sandbox
from .docker_backend import DockerSandbox
from .local_backend import LocalSandbox


def get_sandbox(backend: str | None = None) -> Sandbox:
    """Return a sandbox instance per the W1Z4RD_SANDBOX env or override.

    Defaults to docker.  "auto" falls back to local if docker engine is
    unreachable — prints a warning so it's never silent.
    """
    name = (backend or os.getenv("W1Z4RD_SANDBOX", "docker")).strip().lower()
    if name == "docker":
        return DockerSandbox()
    if name == "local":
        return LocalSandbox()
    if name == "auto":
        ds = DockerSandbox()
        if ds.available():
            return ds
        import sys
        print(
            "[sandbox] WARNING: docker engine unreachable; falling back "
            "to LocalSandbox (dev mode).  Set W1Z4RD_SANDBOX=docker once "
            "Docker Desktop is up.",
            file=sys.stderr,
        )
        return LocalSandbox()
    raise ValueError(f"unknown sandbox backend {name!r}")


__all__ = ["get_sandbox", "Sandbox", "CheckResult"]
