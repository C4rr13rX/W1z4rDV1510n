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

    LOCK_PATH = Path(tempfile.gettempdir()) / "w1z4rd-cargo-build.lock"
    HOLDER_PATH = Path(tempfile.gettempdir()) / "w1z4rd-cargo-build-holder.json"
    QUEUE_DIR = Path(tempfile.gettempdir()) / "w1z4rd-cargo-build-queue"
    # Legacy alias for compatibility with code expecting PATH
    PATH = LOCK_PATH

    def __init__(self, cmd: list[str] | None = None, tgt_dir: Path | None = None, prefix: str = "w1z4rd-cargo-build"):
        self.cmd = list(cmd) if cmd else ["cargo"]
        self.tgt_dir = str(tgt_dir) if tgt_dir else ""
        self.lock_path = Path(tempfile.gettempdir()) / f"{prefix}.lock"
        self.holder_path = Path(tempfile.gettempdir()) / f"{prefix}-holder.json"
        self.queue_dir = Path(tempfile.gettempdir()) / f"{prefix}-queue"
        self._f = None
        self._ticket_path = None

    def __enter__(self):
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        # High-resolution monotonic arrival ticket
        ticket_id = f"{time.time():018.6f}_{time.perf_counter_ns():020d}_{os.getpid()}"
        self._ticket_path = self.queue_dir / f"{ticket_id}.ticket"
        ticket_data = {
            "pid": os.getpid(),
            "cmd": self.cmd,
            "target_dir": self.tgt_dir,
            "arrived_at": time.time(),
        }
        self._ticket_path.write_text(json.dumps(ticket_data), encoding="utf-8")

        self._f = open(self.lock_path, "a+b")
        waited_start = time.time()
        last_report = 0.0

        while True:
            # Clean up dead tickets in queue_dir
            active_tickets = []
            for tp in sorted(self.queue_dir.glob("*.ticket")):
                try:
                    t_pid = int(tp.stem.split("_")[-1])
                    if not is_pid_alive(t_pid):
                        tp.unlink(missing_ok=True)
                        continue
                    active_tickets.append(tp)
                except Exception:
                    pass

            is_our_turn = bool(active_tickets and active_tickets[0] == self._ticket_path)

            if is_our_turn:
                try:
                    msvcrt.locking(self._f.fileno(), msvcrt.LK_NBLCK, 1)
                    holder_info = {
                        "pid": os.getpid(),
                        "cmd": self.cmd,
                        "target_dir": self.tgt_dir,
                        "acquired_at": time.time(),
                    }
                    try:
                        self.holder_path.write_text(json.dumps(holder_info), encoding="utf-8")
                    except Exception:
                        pass
                    break
                except OSError:
                    pass

            # Read holder details for diagnostics
            holder_pid = None
            holder_cmd = []
            holder_elapsed = 0.0
            if self.holder_path.exists():
                try:
                    hinfo = json.loads(self.holder_path.read_text(encoding="utf-8"))
                    h_pid = int(hinfo.get("pid", 0))
                    if is_pid_alive(h_pid):
                        holder_pid = h_pid
                        holder_cmd = hinfo.get("cmd", [])
                        if hinfo.get("acquired_at"):
                            holder_elapsed = time.time() - hinfo["acquired_at"]
                    else:
                        self.holder_path.unlink(missing_ok=True)
                except Exception:
                    pass

            now = time.time()
            if last_report == 0.0 or now - last_report >= 5.0:
                queue_pos = active_tickets.index(self._ticket_path) if self._ticket_path in active_tickets else 0
                cmd_str = " ".join(holder_cmd) if holder_cmd else "cargo"
                if holder_pid:
                    print(
                        f"capped: waiting for build slot held by PID {holder_pid} ({cmd_str}) "
                        f"for {holder_elapsed:.0f}s (queue position {queue_pos}/{len(active_tickets)})",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    print(
                        f"capped: waiting for build slot (queue position {queue_pos}/{len(active_tickets)})...",
                        file=sys.stderr,
                        flush=True,
                    )
                last_report = now

            time.sleep(0.5)

        total_waited = time.time() - waited_start
        if total_waited > 5:
            print(f"capped: waited {total_waited:.0f}s for the build slot", file=sys.stderr, flush=True)
        return self

    def __exit__(self, *exc):
        try:
            self.holder_path.unlink(missing_ok=True)
        except Exception:
            pass
        if self._ticket_path:
            self._ticket_path.unlink(missing_ok=True)
        try:
            msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
        except Exception:
            pass
        finally:
            self._f.close()


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with this PID is currently running."""
    if pid <= 0:
        return False
    try:
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
        if not h:
            return False
        exit_code = wintypes.DWORD()
        res = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
        ctypes.windll.kernel32.CloseHandle(h)
        STILL_ACTIVE = 259
        return bool(res != 0 and exit_code.value == STILL_ACTIVE)
    except Exception:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


class ActiveInvocation:
    """Guards against duplicate invocations from the same worktree with identical cmd and target_dir."""
    DIR = Path(tempfile.gettempdir()) / "w1z4rd-active-invocations"

    def __init__(self, cmd: list[str], worktree: Path, tgt_dir: Path):
        self.cmd = list(cmd)
        self.worktree = str(worktree.resolve())
        self.tgt_dir = str(tgt_dir.resolve())
        self.pid = os.getpid()
        self.path = self.DIR / f"{self.pid}.json"

    def check_duplicate(self) -> int | None:
        """Returns existing PID if a duplicate invocation is already running, else None."""
        self.DIR.mkdir(parents=True, exist_ok=True)
        for p in list(self.DIR.glob("*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                other_pid = int(data.get("pid", 0))
                if not is_pid_alive(other_pid):
                    p.unlink(missing_ok=True)
                    continue
                if other_pid == self.pid:
                    continue
                if (data.get("worktree") == self.worktree
                        and data.get("target_dir") == self.tgt_dir
                        and data.get("cmd") == self.cmd):
                    return other_pid
            except Exception:
                pass
        return None

    def __enter__(self):
        existing_pid = self.check_duplicate()
        if existing_pid is not None:
            cmd_str = " ".join(self.cmd)
            print(
                f"capped: refusing duplicate invocation; PID {existing_pid} is already running "
                f"'{cmd_str}' in {self.worktree} with target {self.tgt_dir}",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        self.DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": self.pid,
            "cmd": self.cmd,
            "worktree": self.worktree,
            "target_dir": self.tgt_dir,
            "started_at": time.time(),
        }
        self.path.write_text(json.dumps(record), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def compute_polite_allowance(requested_mb: int) -> tuple[int, int | None]:
    """Scale the job's memory cap and cargo parallelism to fit nicely within
    the PC's live available RAM, always leaving a >=7.5 GB polite reserve +
    12% under-cap margin so crypto strategies and desktop apps run unimpeded,
    while expanding up to `requested_mb` when free RAM is abundant."""
    class _MemStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    try:
        ms = _MemStatusEx()
        ms.dwLength = ctypes.sizeof(_MemStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            total_mb = int(ms.ullTotalPhys // (1024 * 1024))
            avail_mb = int(ms.ullAvailPhys // (1024 * 1024))
            reserve_mb = max(7680, int(total_mb * 0.25)) if total_mb >= 24576 else int(total_mb * 0.25)
            usable_mb = int(max(0, avail_mb - reserve_mb) * 0.88)
            effective_mb = min(requested_mb, max(2200, usable_mb))
            jobs_override = 1 if usable_mb < 3200 else None
            return effective_mb, jobs_override
    except Exception:
        pass
    return requested_mb, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=int, default=2048, help="committed-memory cap for the whole tree")
    ap.add_argument("--timeout", type=float, default=0, help="wall-clock seconds; 0 = none")
    ap.add_argument("--json", help="also write the result line to this file")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = list(args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd)
    if not cmd:
        ap.error("no command given")
    effective_cap_mb, jobs_override = compute_polite_allowance(args.mb)
    is_cargo = Path(cmd[0]).stem.lower() == "cargo"
    if is_cargo and jobs_override == 1:
        for idx, tok in enumerate(cmd[:-1]):
            if tok == "-j":
                cmd[idx + 1] = "1"
                break
    tgt_dir = target_dir(ROOT)
    if is_cargo:
        os.environ["CARGO_TARGET_DIR"] = str(tgt_dir)
    with ActiveInvocation(cmd, ROOT, tgt_dir):
        job = MemoryCappedJob(effective_cap_mb)
        with BuildSlot(cmd, tgt_dir) if is_cargo else nullcontext():
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
    hit_cap = peak >= effective_cap_mb * 0.97 and code != 0
    result = {"exit": code, "peak_mb": round(peak, 1), "cap_mb": effective_cap_mb,
              "requested_cap_mb": args.mb,
              "secs": round(time.time() - t0, 1), "hit_cap": hit_cap, "timed_out": timed_out}
    line = json.dumps(result)
    print(line, file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            f.write(line + "\n")
    return 137 if (hit_cap or timed_out) else code


if __name__ == "__main__":
    raise SystemExit(main())
