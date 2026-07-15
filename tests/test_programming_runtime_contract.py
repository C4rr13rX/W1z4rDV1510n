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
from scripts.programming_curriculum_supervisor import phase_offsets, publish
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


if __name__ == "__main__":
    unittest.main()
