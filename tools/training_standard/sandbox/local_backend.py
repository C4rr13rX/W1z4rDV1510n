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
    # `gofmt -e` parses and reports syntax errors without compiling, so a
    # bare function body validates without needing a package clause, a module
    # or the network. Real validation matters here: an unsupported language
    # falls through to CheckResult.passed(), so leaving Go out would admit
    # every Go row unchecked while reporting a clean ingest.
    "go":         ("file",  ".go",  ["gofmt",  "-e",               "{path}"]),
    # Rust and TS need real compilers + project files, which are slow
    # and tooling-specific on Windows; skipped in local backend.  The
    # Docker backend handles them properly.
}


def _go_batch_ok(snippets: list[str], timeout_s: float) -> bool:
    """Parse many Go functions in ONE gofmt call. True only if all are valid."""
    joiner = chr(10) * 2
    body = joiner.join(snippets)
    source = "package p" + chr(10) * 2 + body
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".go", delete=False, encoding="utf-8",
    ) as f:
        f.write(source)
        path = f.name
    try:
        proc = subprocess.run(
            ["gofmt", "-e", path], capture_output=True, text=True,
            timeout=timeout_s,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


class LocalSandbox:
    backend_name = "local"

    def available(self) -> bool:
        return True  # always — at worst individual langs are unsupported

    def check_batch(self, lang: str, codes: list[str], *,
                    timeout_s: float = 30.0) -> list[bool]:
        """Validate many snippets, spending one process when they are all fine.

        Per-row checking spends a process per row: measured 14.8 rows/s on the
        CodeSearchNet Go corpus, or 5.9 hours for its 317,832 functions, and
        essentially all of that is spawn overhead -- 100 functions parse in
        0.11 s when concatenated into one file.

        A batch is all-or-nothing (one bad function fails the whole file), so
        a failing batch is bisected down to the offending rows rather than
        discarding the batch. Real corpus Go is overwhelmingly valid, so the
        fast path is the common one and the bisect is rare.
        """
        if lang != "go" or shutil.which("gofmt") is None:
            return [self.check(lang, c, timeout_s=timeout_s).ok for c in codes]
        if not codes:
            return []
        if _go_batch_ok(codes, timeout_s):
            return [True] * len(codes)
        if len(codes) == 1:
            return [False]
        middle = len(codes) // 2
        return (self.check_batch(lang, codes[:middle], timeout_s=timeout_s)
                + self.check_batch(lang, codes[middle:], timeout_s=timeout_s))

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
            source = code
            if lang == "go":
                # gofmt parses a FILE, so a bare function -- which is what a
                # CodeSearchNet row holds -- fails with "expected 'package'"
                # before its real syntax is ever examined. Verified on the
                # host: an unwrapped valid function and an unwrapped broken
                # one both exit 2, so without this every Go row would be
                # rejected for the same spurious reason. Wrapped, a valid
                # function exits 0 and a broken one still exits 2, which is
                # the discrimination the check exists to provide.
                stripped = source.lstrip()
                if not stripped.startswith("package "):
                    source = "package p" + chr(10) * 2 + source
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8",
            ) as f:
                f.write(source)
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
