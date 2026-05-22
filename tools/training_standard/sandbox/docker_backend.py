"""training_standard/sandbox/docker_backend.py — Docker-isolated per-language validation.

One short-lived container per check.  Read-only mount of the candidate
file, --network=none, --memory=512m, --pids-limit=128, --read-only
rootfs with a tiny tmpfs at /tmp for compiler scratch.  Containers are
ephemeral (--rm) so there's no state between checks.

Image registry (one image per language, built on first use; reused
forever afterwards):

    Language     | Image tag (built on first run)
    -------------+--------------------------------
    python       | w1z4rd-sb-python:3.12-slim
    javascript   | w1z4rd-sb-node:20-alpine
    typescript   | w1z4rd-sb-ts:5-alpine
    rust         | w1z4rd-sb-rust:1.80-slim
    bash         | w1z4rd-sb-bash:5
    powershell   | mcr.microsoft.com/powershell:7.4-alpine
    go           | golang:1.22-alpine
    java         | eclipse-temurin:21-jdk-alpine
    cpp          | gcc:13
    csharp       | mcr.microsoft.com/dotnet/sdk:8.0-alpine

A Dockerfile per language ships under tools/training_standard/sandbox/dockerfiles/.
The build is lazy — first check for a given language triggers a build
if the image is missing.  Builds are cached locally by Docker.

Limits per check:
    --cpus=2 --memory=512m --pids-limit=128
    --network=none --read-only --tmpfs /tmp:size=64m
    timeout 15s default (override per call)

Validation philosophy: syntactic / type-check only, NEVER execute the
candidate's main / entrypoint.  Each language image has a small
`/validate.sh` that runs the language's check-mode (py_compile,
tsc --noEmit, cargo check, etc.) and exits with status only.
"""
from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Final

from .types import CheckResult, Sandbox


_DOCKERFILES_DIR: Final[Path] = Path(__file__).resolve().parent / "dockerfiles"


# Per-language: (image_tag, in-container path for the candidate, validate cmd).
_IMAGES: dict[str, tuple[str, str, list[str]]] = {
    "python":     ("w1z4rd-sb-python:3.12-slim", "/work/code.py",
                   ["python", "-m", "py_compile", "/work/code.py"]),
    "javascript": ("w1z4rd-sb-node:20-alpine",   "/work/code.js",
                   ["node", "--check", "/work/code.js"]),
    "typescript": ("w1z4rd-sb-ts:5-alpine",      "/work/code.ts",
                   ["sh", "-c", "tsc --noEmit --strict /work/code.ts"]),
    "rust":       ("w1z4rd-sb-rust:1.80-slim",   "/work/src/lib.rs",
                   # cargo check on a scratch project with the candidate
                   # as lib.rs.  Project files are baked into the image.
                   ["sh", "-c", "cd /work && cargo check --quiet 2>&1"]),
    "bash":       ("w1z4rd-sb-bash:5",            "/work/code.sh",
                   ["bash", "-n", "/work/code.sh"]),
    "powershell": ("mcr.microsoft.com/powershell:7.4-alpine", "/work/code.ps1",
                   ["pwsh", "-NoProfile", "-Command",
                    "$null = [System.Management.Automation.PSParser]::Tokenize("
                    "(Get-Content -Raw /work/code.ps1), [ref]$null)"]),
    "go":         ("golang:1.22-alpine",         "/work/main.go",
                   ["sh", "-c", "cd /work && go vet ./... 2>&1"]),
    "java":       ("eclipse-temurin:21-jdk-alpine", "/work/Code.java",
                   ["sh", "-c", "javac -d /tmp /work/Code.java"]),
    "cpp":        ("gcc:13",                     "/work/code.cpp",
                   ["sh", "-c", "g++ -std=c++20 -fsyntax-only /work/code.cpp"]),
    "csharp":     ("mcr.microsoft.com/dotnet/sdk:8.0-alpine", "/work/Program.cs",
                   ["sh", "-c", "cd /work && dotnet build -nologo --no-restore 2>&1 || dotnet build -nologo 2>&1"]),
}


class DockerSandbox:
    backend_name = "docker"

    def __init__(self) -> None:
        self._available: bool | None = None
        self._built: set[str] = set()

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            proc = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=5,
            )
            self._available = proc.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._available = False
        return self._available

    def _ensure_image(self, image_tag: str, lang: str) -> bool:
        """Pull or build the image if missing.  Returns True on success."""
        if image_tag in self._built:
            return True
        # Check if already present locally.
        proc = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0:
            self._built.add(image_tag)
            return True
        # If image_tag is a w1z4rd-sb-* local build, look for a Dockerfile.
        if image_tag.startswith("w1z4rd-sb-"):
            df = _DOCKERFILES_DIR / f"{lang}.Dockerfile"
            if not df.exists():
                return False
            build = subprocess.run(
                ["docker", "build", "-t", image_tag, "-f", str(df),
                 str(_DOCKERFILES_DIR)],
                capture_output=True, text=True, timeout=600,
            )
            if build.returncode != 0:
                return False
        else:
            # Public image — docker pull.
            pull = subprocess.run(
                ["docker", "pull", image_tag],
                capture_output=True, text=True, timeout=300,
            )
            if pull.returncode != 0:
                return False
        self._built.add(image_tag)
        return True

    def check(self, lang: str, code: str, *, timeout_s: float = 15.0) -> CheckResult:
        entry = _IMAGES.get(lang)
        if entry is None:
            # Unknown language — accept (we don't want unsupported langs
            # to silently nuke a corpus).  Caller can opt-in to strict
            # by checking lang in _IMAGES first.
            return CheckResult.passed(backend=self.backend_name + "/unknown-lang")

        if not self.available():
            return CheckResult.failed(
                "docker engine unreachable — start Docker Desktop or set "
                "W1Z4RD_SANDBOX=local for dev mode",
                backend=self.backend_name,
            )

        image_tag, in_path, argv = entry
        if not self._ensure_image(image_tag, lang):
            return CheckResult.failed(
                f"docker image {image_tag} could not be built/pulled",
                backend=self.backend_name,
            )

        # Write candidate to a host temp file, mount it read-only.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=Path(in_path).suffix, delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            host_path = f.name

        try:
            docker_argv = [
                "docker", "run", "--rm",
                "--network=none",
                "--read-only",
                "--tmpfs", "/tmp:size=64m,exec",
                "--cpus=2",
                "--memory=512m",
                "--pids-limit=128",
                "-v", f"{host_path}:{in_path}:ro",
                image_tag,
                *argv,
            ]
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    docker_argv, capture_output=True, text=True,
                    timeout=timeout_s + 5,  # docker overhead
                )
            except subprocess.TimeoutExpired:
                return CheckResult.failed(
                    f"validation timed out after {timeout_s}s",
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
            try:
                Path(host_path).unlink()
            except OSError:
                pass
