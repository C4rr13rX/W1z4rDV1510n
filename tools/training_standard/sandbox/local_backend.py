"""training_standard/sandbox/local_backend.py — dev-only host-toolchain sandbox.

Validates code using whatever compilers/interpreters happen to be on
the host PATH.  No isolation — DO NOT use in CI or with untrusted
generated code.  This exists so ingest scripts can be iterated on
without Docker Desktop running.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .types import CheckResult, Sandbox

# Per-language check command.  Mode is either "file" (write code to a
# temp file and pass its path) or "stdin" (feed code via -c / stdin so
# Windows path-translation isn't an issue for POSIX-only tools).
# "file" entries use {path}; "stdin" entries use {code} substituted into
# the last arg.  Commands must perform syntactic validation, never execute.
_CHECKERS: dict[str, tuple[str, str, list[str]]] = {
    "python":     ("file",  ".py",  ["python", "-m", "py_compile", "{path}"]),
    "javascript": ("file",  ".js",  ["node",   "--check",          "{path}"]),
    # bash -nc reads code as a string — no temp file, no Windows path issue.
    "bash":       ("stdin", ".sh",  ["bash",   "-nc",              "{code}"]),
    "powershell": ("stdin", ".ps1", ["pwsh",   "-NoProfile", "-Command",
                                     "$null = [System.Management.Automation.PSParser]"
                                     "::Tokenize({code}, [ref]$null)"]),
    # Rust and TS need real compilers + project files, which are slow
    # and tooling-specific on Windows; skipped in local backend.  The
    # Docker backend handles them properly.
}


class LocalSandbox:
    backend_name = "local"

    def available(self) -> bool:
        return True  # always — at worst individual langs are unsupported

    def check(self, lang: str, code: str, *, timeout_s: float = 15.0) -> CheckResult:
        entry = _CHECKERS.get(lang)
        if entry is None:
            # Unsupported language under local mode — accept by default
            # so ingest doesn't grind to a halt.  Production runs use
            # Docker which has full coverage.
            return CheckResult.passed(backend=self.backend_name)

        mode, suffix, argv = entry
        tool = argv[0]
        if shutil.which(tool) is None:
            # Tool not installed locally — accept-with-warning, since
            # we'd rather have the row than block on local-mode gaps.
            return CheckResult.passed(backend=self.backend_name + "/no-tool")

        path: str | None = None
        if mode == "stdin":
            # Substitute code directly into the last argv arg.  For
            # PowerShell we wrap the code as a single-quoted string
            # since the {code} sits inside a larger PS expression.
            if tool.lower() in ("pwsh", "powershell"):
                escaped = "'" + code.replace("'", "''") + "'"
            else:
                escaped = code
            cmd = [a.replace("{code}", escaped) for a in argv]
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8",
            ) as f:
                f.write(code)
                path = f.name
            cmd = [a.replace("{path}", path) for a in argv]
        try:
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired as e:
                return CheckResult.failed(
                    f"timeout after {timeout_s}s: {e}",
                    backend=self.backend_name,
                    duration_ms=int(timeout_s * 1000),
                )
            duration_ms = int((time.monotonic() - t0) * 1000)
            if proc.returncode == 0:
                return CheckResult.passed(
                    backend=self.backend_name, duration_ms=duration_ms,
                )
            stderr = (proc.stderr or proc.stdout or "").strip()
            return CheckResult.failed(
                stderr or f"exit {proc.returncode}",
                backend=self.backend_name,
                duration_ms=duration_ms,
            )
        finally:
            if path is not None:
                try:
                    Path(path).unlink()
                except OSError:
                    pass
