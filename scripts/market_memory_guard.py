#!/usr/bin/env python3
"""Conservative memory-pressure recovery for the local market-brain stack.

The recovery path never kills or restarts a process.  On Windows it can ask
the kernel to evict reclaimable resident pages from one uniquely identified,
healthy local brain-node process.  Neural state and private commitment remain
owned by that process; this is a paging-pressure relief valve, not a substitute
for bounded data structures or neuron-scoped sleep/checkpoint operations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def local_http_healthy(url: str, timeout: float = 3.0) -> bool:
    """Probe only an explicit loopback HTTP endpoint."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in LOCAL_HOSTS:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def unique_process(process_name: str, command_fragment: str) -> tuple[Any | None, int]:
    """Return one exact-name/command match; ambiguity always fails closed."""
    if psutil is None or not process_name or not command_fragment:
        return None, 0
    matches = []
    expected_name = process_name.casefold()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            info = process.info
            name = str(info.get("name") or "").casefold()
            command = " ".join(str(part) for part in (info.get("cmdline") or []))
            if name == expected_name and command_fragment in command:
                matches.append(process)
        except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
    return (matches[0] if len(matches) == 1 else None), len(matches)


def trim_windows_working_set(pid: int) -> bool:
    """Evict reclaimable pages without terminating or mutating process state."""
    if os.name != "nt" or pid <= 0:
        return False
    import ctypes

    process_set_quota = 0x0100
    process_query_information = 0x0400
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
    psapi.EmptyWorkingSet.restype = ctypes.c_bool
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_bool
    handle = kernel32.OpenProcess(
        process_set_quota | process_query_information, False, pid,
    )
    if not handle:
        return False
    try:
        return bool(psapi.EmptyWorkingSet(handle))
    finally:
        kernel32.CloseHandle(handle)


@dataclass
class VerifiedWorkingSetReclaimer:
    """Rate-limited reclamation for one uniquely identified local process."""

    process_name: str
    command_fragment: str
    health_url: str
    cooldown_seconds: float = 900.0
    health_timeout_seconds: float = 3.0
    platform_name: str = field(default_factory=lambda: os.name)
    clock: Callable[[], float] = time.monotonic
    finder: Callable[[str, str], tuple[Any | None, int]] = unique_process
    health_probe: Callable[[str, float], bool] = local_http_healthy
    trimmer: Callable[[int], bool] = trim_windows_working_set
    _last_attempt: float | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        return bool(
            self.platform_name == "nt" and self.process_name
            and self.command_fragment and self.health_url
        )

    def attempt(self) -> dict[str, Any] | None:
        """Try once after validation, returning audit evidence when attempted."""
        if not self.enabled:
            return None
        now = self.clock()
        if (self._last_attempt is not None
                and now - self._last_attempt < self.cooldown_seconds):
            return None
        self._last_attempt = now
        process, match_count = self.finder(self.process_name, self.command_fragment)
        if process is None:
            return {"outcome": "identity_rejected", "match_count": match_count}
        pid = int(process.pid)
        try:
            before = int(process.memory_info().rss)
        except Exception:
            return {"outcome": "inspection_failed", "pid": pid}
        if not self.health_probe(self.health_url, self.health_timeout_seconds):
            return {"outcome": "pre_health_failed", "pid": pid,
                    "working_set_before_bytes": before}
        if not self.trimmer(pid):
            return {"outcome": "trim_failed", "pid": pid,
                    "working_set_before_bytes": before}
        post_healthy = self.health_probe(self.health_url, self.health_timeout_seconds)
        try:
            after = int(process.memory_info().rss)
        except Exception:
            after = -1
        return {
            "outcome": "trimmed" if post_healthy else "post_health_failed",
            "pid": pid,
            "working_set_before_bytes": before,
            "working_set_after_bytes": after,
            "health_preserved": post_healthy,
        }
