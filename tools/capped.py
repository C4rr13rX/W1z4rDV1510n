#!/usr/bin/env python3
"""
Run a command under a hard memory cap so it can never lock up the machine.

    python tools/capped.py --mb 2500 -- cargo test -p w1z4rd-brain --release
    python tools/capped.py --mb 2048 --timeout 600 --json peak.json -- target/release/examples/scorecard.exe

The command and everything it starts run inside a Windows Job Object whose
committed memory may not exceed --mb. When it tries to, the allocation fails
and the process dies -- the PC does not page itself to a standstill. The job
is closed with the wrapper, so no child outlives it, and it runs at
below-normal priority so the desktop stays responsive.

Prints one JSON line on stderr when the command ends:
    {"exit": 0, "peak_mb": 312.4, "cap_mb": 2500, "secs": 41.2, "hit_cap": false, "timed_out": false}
Exit status is the command's own, or 137 when the cap or --timeout killed it.

CARGO gets two more guards, automatically:
  - ONE BUILD AT A TIME, machine-wide. Several agents in several worktrees
    each building with -j 2 is several rustc processes at ~1.5 GB apiece; a
    cargo command waits for the build slot instead of stacking on top.
  - A STABLE TARGET DIR per agent. ContinuousRefinement cuts a fresh worktree
    (data/trees-<loop>/p<N>-<agent>) every pass, and a fresh target/ meant a
    ~9 minute cold build per agent per pass. Inside such a worktree
    CARGO_TARGET_DIR becomes <drive>:/cargo-targets/<loop>-<agent>, reused
    pass after pass. Use target_dir() to find the binaries.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import msvcrt
import os
import re
import subprocess
import tempfile
import sys
import time
from contextlib import nullcontext
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_PRIORITY_CLASS = 0x20
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
BELOW_NORMAL_PRIORITY_CLASS = 0x4000


class _Basic(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _Io(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in
                ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]


class _Extended(ctypes.Structure):
    _fields_ = [("Basic", _Basic),
                ("Io", _Io),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class MemoryCappedJob:
    """A Job Object holding this process and every child it starts."""

    def __init__(self, cap_mb: int):
        self.cap_mb = cap_mb
        self._k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k32.CreateJobObjectW.restype = wintypes.HANDLE
        self._k32.GetCurrentProcess.restype = wintypes.HANDLE
        self.handle = self._k32.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info = _Extended()
        info.Basic.LimitFlags = (JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                                 | JOB_OBJECT_LIMIT_PRIORITY_CLASS)
        info.Basic.PriorityClass = BELOW_NORMAL_PRIORITY_CLASS
        info.JobMemoryLimit = cap_mb * 1024 * 1024
        self._set(info)
        # Joining the job ourselves means every child is born inside it: no
        # window in which a grandchild escapes the cap.
        if not self._k32.AssignProcessToJobObject(wintypes.HANDLE(self.handle),
                                                  wintypes.HANDLE(self._k32.GetCurrentProcess())):
            raise ctypes.WinError(ctypes.get_last_error())

    def _set(self, info: _Extended) -> None:
        if not self._k32.SetInformationJobObject(wintypes.HANDLE(self.handle),
                                                 JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                                                 ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())

    def peak_mb(self) -> float:
        info = _Extended()
        self._k32.QueryInformationJobObject(wintypes.HANDLE(self.handle),
                                            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                                            ctypes.byref(info), ctypes.sizeof(info), None)
        return info.PeakJobMemoryUsed / (1024 * 1024)


def target_dir(root: Path) -> Path:
    """Where cargo builds for this checkout (see the module docstring)."""
    if os.environ.get("CARGO_TARGET_DIR"):
        return Path(os.environ["CARGO_TARGET_DIR"])
    m = re.fullmatch(r"p\d+-([a-z0-9_-]+)", root.name)
    if m:
        return Path(root.anchor) / "cargo-targets" / f"{root.parent.name.removeprefix('trees-')}-{m.group(1)}"
    return root / "target"


class BuildSlot:
    """The machine-wide right to run cargo, held for the life of the command."""

    PATH = Path(tempfile.gettempdir()) / "w1z4rd-cargo-build.lock"

    def __enter__(self):
        self._f = open(self.PATH, "a+b")
        waited = time.time()
        while True:
            try:
                msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(2)
        if time.time() - waited > 5:
            print(f"capped: waited {time.time() - waited:.0f}s for the build slot", file=sys.stderr)
        return self

    def __exit__(self, *exc):
        try:
            msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._f.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=2048, help="committed-memory cap for the whole tree")
    ap.add_argument("--timeout", type=float, default=0, help="wall-clock seconds; 0 = none")
    ap.add_argument("--json", help="also write the result line to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given")
    is_cargo = Path(cmd[0]).stem.lower() == "cargo"
    if is_cargo:
        os.environ["CARGO_TARGET_DIR"] = str(target_dir(ROOT))
    job = MemoryCappedJob(args.mb)
    with BuildSlot() if is_cargo else nullcontext():
        t0 = time.time()
        proc = subprocess.Popen(cmd)
        timed_out = False
        try:
            code = proc.wait(timeout=args.timeout or None)
        except subprocess.TimeoutExpired:
            timed_out = True
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            code = proc.wait()
    peak = job.peak_mb()
    hit_cap = peak >= args.mb * 0.97 and code != 0
    result = {"exit": code, "peak_mb": round(peak, 1), "cap_mb": args.mb,
              "secs": round(time.time() - t0, 1), "hit_cap": hit_cap, "timed_out": timed_out}
    line = json.dumps(result)
    print(line, file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(line + "\n")
    return 137 if (hit_cap or timed_out) else code


if __name__ == "__main__":
    raise SystemExit(main())
