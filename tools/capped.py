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
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from ctypes import wintypes

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
    job = MemoryCappedJob(args.mb)
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
