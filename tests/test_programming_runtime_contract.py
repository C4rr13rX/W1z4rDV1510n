"""Cross-layer invariants for the persistent programming-brain runtime."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.programming_integrated_retention import mutation_enabled
from scripts.programming_curriculum_supervisor import (
    Phase,
    accept_last_good_guard,
    ensure_last_good_guard,
    phase_offsets,
    publish,
)
from scripts.programming_enterprise_retention import run_suite
from scripts.programming_exec_env import benchmark_tool_env, isolated_tool_env
from scripts.programming_corpus_recall import sample_window
from tools.training_standard.drive_corpora_brain import checkpoint_due


class ProgrammingRuntimeContractTests(unittest.TestCase):
    def test_wal_durable_training_still_bounds_snapshot_tail(self) -> None:
        self.assertFalse(checkpoint_due(4096, 4095))
        self.assertTrue(checkpoint_due(4096, 4096))
        self.assertFalse(checkpoint_due(0, 100_000))

    def test_read_only_retention_cannot_enter_training_branch(self) -> None:
        self.assertFalse(mutation_enabled(read_only=True))
        self.assertTrue(mutation_enabled(read_only=False))

    def test_supervisor_status_survives_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            import scripts.programming_curriculum_supervisor as supervisor
            real_replace = supervisor.os.replace
            attempts = 0

            def transient_lock(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("simulated reader lock")
                real_replace(source, destination)

            with patch.object(supervisor.os, "replace",
                              side_effect=transient_lock), \
                    patch.object(supervisor.time, "sleep"):
                publish(target, {"ram_next_row": 12, "durable_next_row": 10})
            self.assertEqual(phase_offsets(target), (12, 10))
            self.assertEqual(attempts, 3)

    def test_enterprise_suite_failure_is_preserved(self) -> None:
        result = run_suite("failure", ["-c", "import sys; sys.exit(7)"], 5)
        self.assertFalse(result["passed"])
        self.assertEqual(result["exit_code"], 7)

    def test_enterprise_suite_timeout_is_preserved(self) -> None:
        result = run_suite(
            "timeout", ["-c", "import time; time.sleep(1)"], 0.01,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"])

    def test_compiler_caches_are_confined_to_benchmark_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = isolated_tool_env(root)
            for key in ("GOCACHE", "GOMODCACHE", "DOTNET_CLI_HOME", "NUGET_PACKAGES"):
                self.assertTrue(Path(environment[key]).is_relative_to(root))
                self.assertTrue(Path(environment[key]).is_dir())

    def test_shared_compiler_cache_stays_inside_repository_runtime(self) -> None:
        environment = benchmark_tool_env()
        runtime = ROOT / "runtime" / "benchmark-tool-cache"
        for key in ("GOCACHE", "GOMODCACHE", "DOTNET_CLI_HOME", "NUGET_PACKAGES"):
                self.assertTrue(Path(environment[key]).is_relative_to(runtime))

    def test_semantic_stress_fails_on_any_missing_recall(self) -> None:
        source = (ROOT / "scripts" / "programming_semantic_stress.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('all(row["recalled"] for row in rows)', source)

    def test_phase_completion_gate_includes_strict_enterprise_retention(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("programming_enterprise_retention.py", source)
        self.assertIn('enterprise.get("tick_delta") != 0', source)

    def test_corpus_sampler_covers_both_ends_of_trained_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(
                "".join(
                    f'{{"prompt":"p{index}","response":"r{index}"}}\n'
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            probes, rows = sample_window(corpus, 2, 6, 3)
            self.assertEqual(rows, 6)
            self.assertEqual([row["prompt"] for row in probes], ["p2", "p4", "p7"])

    def test_direct_pretrain_is_chunked_between_retention_gates(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--limit-rows", str(min(args.gate_rows, phase.rows - ram))', source)
        self.assertIn("run_midphase_gate(args, phase, runtime, ram_after)", source)
        self.assertIn('"--no-checkpoint"', source)
        self.assertIn('"--gate-rows", type=int, default=16384', source)

    def test_dedicated_corpus_supervisor_preserves_live_inference_windows(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--batch-size", type=int, default=32', source)
        self.assertIn('"--inter-batch-yield-seconds", type=float, default=0.1', source)

    def test_chunk_snapshot_guard_survives_until_explicit_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.bin"
            snapshot.write_bytes(b"accepted-state")
            phase = Phase("phase-a", "script-a", runtime / "corpus.jsonl", 10)
            guard = ensure_last_good_guard(runtime, phase, 4)
            self.assertTrue(guard.exists())
            self.assertTrue(snapshot.samefile(guard))
            snapshot.unlink()
            snapshot.write_bytes(b"candidate-state")
            self.assertEqual(guard.read_bytes(), b"accepted-state")
            self.assertEqual(ensure_last_good_guard(runtime, phase, 6), guard)
            accept_last_good_guard(runtime)
            self.assertFalse(guard.exists())

    def test_attached_bounded_worker_is_gated_before_training_resumes(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument("--attach-phase", default="")', source)
        self.assertIn("run_midphase_gate(args, attach_phase, runtime, attached_ram)", source)


if __name__ == "__main__":
    unittest.main()
