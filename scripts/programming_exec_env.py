"""Hermetic writable toolchain environment for programming benchmarks."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence


BENCHMARK_CACHE_ROOT = Path(__file__).resolve().parents[1] / "runtime" / "benchmark-tool-cache"


def isolated_tool_env(root: Path) -> dict[str, str]:
    """Keep compiler caches inside the disposable benchmark workspace."""
    directories = {
        "GOCACHE": root / ".cache" / "go-build",
        "GOMODCACHE": root / ".cache" / "go-mod",
        "DOTNET_CLI_HOME": root / ".dotnet",
        "NUGET_PACKAGES": root / ".nuget" / "packages",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({key: str(value) for key, value in directories.items()})
    environment.update({
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_NOLOGO": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE": "1",
        "NUGET_XMLDOC_MODE": "skip",
        "NUGET_CERT_REVOCATION_MODE": "offline",
        "RestoreIgnoreFailedSources": "true",
    })
    return environment


def benchmark_tool_env() -> dict[str, str]:
    """Reuse content-addressed compiler caches without touching user homes."""
    return isolated_tool_env(BENCHMARK_CACHE_ROOT)


def prepare_tool_command(command: Sequence[str], cwd: Path) -> list[str]:
    """Make compiler invocation hermetic before starting its process tree."""
    prepared = list(command)
    if not prepared or Path(prepared[0]).name.lower() not in {"dotnet", "dotnet.exe"}:
        return prepared
    config = cwd / "NuGet.Config"
    if not config.exists():
        config.write_text(
            '<?xml version="1.0" encoding="utf-8"?>'
            "<configuration><packageSources><clear />"
            "</packageSources></configuration>",
            encoding="utf-8",
        )
    if (
        len(prepared) > 1
        and prepared[1] in {"build", "run", "test"}
        and "--disable-build-servers" not in prepared
    ):
        prepared.append("--disable-build-servers")
    return prepared


def run_tool(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a compiler as a killable process tree so timed-out children release files."""
    prepared = prepare_tool_command(command, cwd)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        prepared,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ("taskkill", "/PID", str(process.pid), "/T", "/F"),
                capture_output=True,
                check=False,
            )
        else:
            import signal

            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(prepared, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(prepared, process.returncode, stdout, stderr)
