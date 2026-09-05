#!/usr/bin/env python3
"""Run a local shell script on the training host and print its remote stdout.

`admission_watchdog.py` shells out to this file once per poll:

    python ssm.py <script-path> <timeout-seconds>

It then scans stdout for the probe's JSON line. Without this module every
`probe()` returned `None`, so the watchdog printed "could not read a baseline"
and exited 2 on its very first call -- the watchdog CLAUDE.md names as the
defence against a curriculum that admits nothing was itself reporting a fault
it had not measured, which is the same silent-blindness class it exists to
catch.

The transport is the already-authorized SSM command channel used by
`watch_programming_brain.py`; nothing here opens a port or persists a
credential. Remote stdout is forwarded verbatim, because the caller matches on
the probe's own JSON line and any reformatting here would break it.
"""
from __future__ import annotations

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws.bootstrap_training_host import send_and_wait  # noqa: E402

#: The private training host. Kept identical to
#: `watch_programming_brain.DEFAULT_INSTANCE`; importing that module instead
#: would drag in its watcher dependencies for a one-line constant.
DEFAULT_INSTANCE = "i-0d7a6deeb0ead2dfc"
DEFAULT_PROFILE = "FountainServer"


def run(script: pathlib.Path, timeout: int) -> int:
    """Execute `script` remotely, forwarding its output to this process."""
    invocation = send_and_wait(
        os.environ.get("WIZARD_AWS_PROFILE", DEFAULT_PROFILE),
        os.environ.get("WIZARD_INSTANCE_ID", DEFAULT_INSTANCE),
        [script.read_text(encoding="utf-8")],
        timeout,
        comment="Wizard admission watchdog probe",
    )
    sys.stdout.write(str(invocation.get("StandardOutputContent") or ""))
    sys.stderr.write(str(invocation.get("StandardErrorContent") or ""))
    return 0


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    script = pathlib.Path(argv[0])
    if not script.is_file():
        print(f"no such script: {script}", file=sys.stderr)
        return 2
    timeout = int(argv[1]) if len(argv) == 2 else 300
    try:
        return run(script, timeout)
    # A transport failure is not a curriculum fault. Report it on stderr and
    # exit non-zero with no stdout: `probe()` then returns None and the
    # watchdog skips the poll rather than alarming on the network.
    except Exception as error:  # noqa: BLE001 - any transport failure is equal here
        print(f"ssm probe failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
