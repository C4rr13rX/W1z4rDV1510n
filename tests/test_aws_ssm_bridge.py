"""The admission watchdog's transport must exist and keep its stdout contract.

`admission_watchdog.probe()` shells out to `scripts/aws/ssm.py` and then scans
its stdout for the probe's own JSON line. The module was missing entirely, so
every poll returned `None` and the watchdog exited 2 on its first call without
having measured anything. These tests pin the two properties `probe()` relies
on: remote stdout is forwarded verbatim, and a transport failure produces no
stdout at all rather than a line the caller might parse as state.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws import ssm  # noqa: E402


class SsmBridgeTests(unittest.TestCase):
    def test_watchdog_call_site_matches_this_module(self) -> None:
        """The watchdog invokes this file by path; keep the two in step."""
        source = (ROOT / "scripts" / "aws"
                  / "admission_watchdog.py").read_text(encoding="utf-8")
        self.assertIn('str(SP / "ssm.py")', source)
        self.assertTrue((ROOT / "scripts" / "aws" / "ssm.py").is_file())

    def test_remote_stdout_is_forwarded_verbatim(self) -> None:
        """`probe()` matches on the probe's own JSON line, so do not reshape it."""
        script = ROOT / "scripts" / "aws" / "ssm.py"
        payload = '{"deferred": 28, "resolved": 59}'
        with mock.patch.object(ssm, "send_and_wait") as send:
            send.return_value = {
                "StandardOutputContent": f"noise\n{payload}\n",
                "StandardErrorContent": "",
            }
            with mock.patch("sys.stdout") as out:
                self.assertEqual(ssm.run(script, 300), 0)
        self.assertEqual(out.write.call_args[0][0], f"noise\n{payload}\n")
        # The script goes over as one command holding its whole body, exactly
        # as watch_programming_brain.py sends its probe.
        commands = send.call_args[0][2]
        self.assertEqual(commands, [script.read_text(encoding="utf-8")])
        self.assertEqual(send.call_args[0][3], 300)

    def test_transport_failure_is_not_a_curriculum_fault(self) -> None:
        """A dead network must skip the poll, never fabricate curriculum state."""
        script = ROOT / "scripts" / "aws" / "ssm.py"
        with mock.patch.object(ssm, "send_and_wait",
                               side_effect=RuntimeError("no route")):
            with mock.patch("sys.stdout") as out:
                self.assertEqual(ssm.main([str(script), "300"]), 1)
        out.write.assert_not_called()

    def test_missing_script_is_rejected_before_any_aws_call(self) -> None:
        with mock.patch.object(ssm, "send_and_wait") as send:
            self.assertEqual(ssm.main([str(ROOT / "does-not-exist.sh")]), 2)
        send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
