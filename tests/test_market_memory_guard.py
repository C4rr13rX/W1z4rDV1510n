from types import SimpleNamespace

import scripts.market_memory_guard as memory_guard
from scripts.market_memory_guard import (
    VerifiedWorkingSetReclaimer,
    system_available_memory_gb,
)


class FakeProcess:
    pid = 42

    def __init__(self):
        self.reads = iter([8 * 1024**3, 256 * 1024**2])

    def memory_info(self):
        return SimpleNamespace(rss=next(self.reads))


def test_reclaimer_requires_unique_healthy_identity_and_preserves_health():
    process = FakeProcess()
    trimmed = []
    reclaimer = VerifiedWorkingSetReclaimer(
        "w1z4rd_node.exe", "api --addr 127.0.0.1:8090",
        "http://127.0.0.1:8090/health", platform_name="nt",
        clock=lambda: 100.0,
        finder=lambda *_args: (process, 1),
        health_probe=lambda *_args: True,
        trimmer=lambda pid: trimmed.append(pid) or True,
    )

    evidence = reclaimer.attempt()

    assert evidence == {
        "outcome": "trimmed",
        "pid": 42,
        "working_set_before_bytes": 8 * 1024**3,
        "working_set_after_bytes": 256 * 1024**2,
        "health_preserved": True,
    }
    assert trimmed == [42]
    assert reclaimer.attempt() is None  # cooldown prevents repeated eviction


def test_reclaimer_fails_closed_on_ambiguous_identity():
    reclaimer = VerifiedWorkingSetReclaimer(
        "w1z4rd_node.exe", "api --addr 127.0.0.1:8090",
        "http://127.0.0.1:8090/health", platform_name="nt",
        finder=lambda *_args: (None, 2),
        health_probe=lambda *_args: (_ for _ in ()).throw(
            AssertionError("ambiguous process must not be probed")
        ),
    )

    assert reclaimer.attempt() == {
        "outcome": "identity_rejected", "match_count": 2,
    }


def test_system_available_memory_gb_is_bounded_by_windows_commit_headroom(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(available=12 * 1024**3)
    )
    monkeypatch.setattr(
        memory_guard,
        "windows_commit_and_phys_available_bytes",
        lambda: (12 * 1024**3, 2 * 1024**3),
    )
    assert system_available_memory_gb(fake_psutil) == 2.0
