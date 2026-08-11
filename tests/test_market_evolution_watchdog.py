import json
from pathlib import Path
import subprocess
import sys

import scripts.market_evolution_watchdog as watchdog


def test_waiter_stays_alive_until_memory_is_available(tmp_path, monkeypatch):
    readings = iter([0.5, 1.25, 4.0])
    sleeps = []
    monkeypatch.setattr(watchdog.time, "sleep", sleeps.append)

    assert watchdog.wait_until_launchable(
        tmp_path, tmp_path / "STOP", 3.5, 2.0,
        memory_reader=lambda: next(readings),
    )
    status = json.loads((tmp_path / "supervisor_status.json").read_text())
    assert status["phase"] == "launch_ready"
    assert status["available_memory_gb"] == 4.0
    assert sleeps == [2.0, 2.0]


def test_waiter_honors_stop_without_launching(tmp_path):
    (tmp_path / "STOP").write_text("stop\n")
    assert not watchdog.wait_until_launchable(
        tmp_path, tmp_path / "STOP", 3.5, .01,
        memory_reader=lambda: 99.0,
    )


def test_claim_replaces_stale_pid_but_rejects_live_owner(tmp_path, monkeypatch):
    pid_path = tmp_path / "supervisor.pid"
    pid_path.write_text("123\n")
    monkeypatch.setattr(watchdog, "process_alive", lambda pid: False)
    assert watchdog.claim_supervisor(tmp_path) == pid_path
    pid_path.write_text("456\n")
    monkeypatch.setattr(watchdog, "process_alive", lambda pid: True)
    try:
        watchdog.claim_supervisor(tmp_path)
    except RuntimeError as exc:
        assert "456" in str(exc)
    else:
        raise AssertionError("live supervisor owner was not rejected")


def test_validated_profit_factor_requires_screened_three_fold_champion(tmp_path):
    champion = {
        "result": {
            "status": "screened",
            "evaluated_folds": 3,
            "summary": {"min_profit_factor": 1.23},
        }
    }
    (tmp_path / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
    assert watchdog.validated_profit_factor(tmp_path) == 1.23
    champion["result"]["evaluated_folds"] = 1
    (tmp_path / "champion.json").write_text(json.dumps(champion), encoding="utf-8")
    assert watchdog.validated_profit_factor(tmp_path) is None


def test_ghost_stack_stays_closed_below_validated_pf_threshold(tmp_path, monkeypatch):
    (tmp_path / "champion.json").write_text(json.dumps({
        "result": {"status": "screened", "evaluated_folds": 3,
                   "summary": {"min_profit_factor": 1.09}},
    }), encoding="utf-8")
    monkeypatch.setattr(watchdog, "ghost_stack_healthy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(watchdog.subprocess, "Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not launch")))
    last, healthy = watchdog.maybe_start_ghost_stack(
        tmp_path, tmp_path / "stack", 1.1, 0.0, now=100.0,
    )
    assert last == 0.0
    assert healthy is False
    admission = json.loads((tmp_path / "ghost_stack_admission.json").read_text())
    assert admission["admitted"] is False


def test_real_watchdog_restarts_exited_worker_until_explicit_stop(tmp_path):
    worker = tmp_path / "disposable_worker.py"
    worker.write_text(
        """from pathlib import Path
import sys
state = Path(sys.argv[sys.argv.index('--state-dir') + 1])
count_path = state / 'starts.txt'
count = int(count_path.read_text()) + 1 if count_path.exists() else 1
count_path.write_text(str(count))
if count >= 2:
    (state / 'STOP').write_text('test complete\\n')
""",
        encoding="utf-8",
    )
    script = Path(watchdog.__file__)
    result = subprocess.run(
        [
            sys.executable, str(script), "--python", sys.executable,
            "--service", str(worker), "--state-dir", str(tmp_path),
            "--min-free-memory-gb", "0", "--memory-poll-seconds", ".02",
            "--restart-delay-seconds", ".02", "--heartbeat-seconds", ".02",
        ],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "starts.txt").read_text() == "2"
    events = [json.loads(line) for line in
              (tmp_path / "supervisor_events.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events].count("worker_started") == 2
    assert events[-1]["event"] == "supervisor_stopped"
