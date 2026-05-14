#!/usr/bin/env python3
"""
w1z4rd_supervisor.py — tiny watchdog that keeps the node alive and
restarts training if needed.

Why this exists
---------------
The node has crashed silently twice (Windows sleep/update + an
unrelated stall), each time taking ~hours-to-days of curriculum
progress with it.  This supervisor runs at higher priority than
normal user processes, polls the node's /health endpoint, restarts
the node if it goes unresponsive, and re-launches the configured
training script when it's not running but should be.

Config
------
Reads ``supervisor.toml`` in the project root (same dir as the
script's parent).  All fields have defaults; the operator overrides
any subset.  See ``supervisor.toml.example`` for the schema.

Startup integration
-------------------
``install_startup.cmd`` registers this as a Scheduled Task at user
logon so node + training survive reboots.  ``uninstall_startup.cmd``
removes it.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

# tomllib only in stdlib for 3.11+; fall back to toml or yaml if missing.
try:
    import tomllib  # type: ignore[attr-defined]
    _toml_loader = lambda b: tomllib.loads(b.decode("utf-8"))
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
        _toml_loader = lambda b: tomllib.loads(b.decode("utf-8"))
    except ImportError:
        _toml_loader = None


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = {
    "node": {
        "binary":          str(PROJECT_ROOT / "bin" / "w1z4rd_node.exe"),
        "working_dir":     str(PROJECT_ROOT),
        "data_dir":        "D:\\w1z4rdv1510n-data",
        "health_url":      "http://localhost:8090/health",
        "health_timeout":  3.0,
        # Allow this many consecutive failed health probes before restart.
        "health_misses_before_restart": 3,
        # Seconds between health probes.
        "poll_interval":   5.0,
        # After a node restart, wait this long for it to come back before
        # giving up on the cycle and trying again.
        "warmup_secs":     30.0,
    },
    "training": {
        # If false, never auto-launch training.  Set to true once you've
        # verified the curriculum works the way you want.
        "enabled":         True,
        # The command supervisor runs to start training.  Defaults to
        # the project's curriculum.  Replace with your own script if
        # desired (must respect SKIP_CLEAR=1 for resumability).
        "command":         "scripts/run_all_training.sh",
        # If true, the supervisor sets SKIP_CLEAR=1 on every relaunch
        # after the first run.  Combined with the per-phase markers
        # in the default curriculum, this prevents re-clearing the
        # pool every restart.
        "resume_on_restart": True,
        # Marker file written by the supervisor after the first launch
        # so it knows whether to set SKIP_CLEAR on subsequent launches.
        "first_run_marker": ".supervisor_training_started",
        # Wait this long after node comes up before launching training.
        "warmup_before_training": 10.0,
        # If training exits, wait this long before re-launching it.
        "restart_backoff_secs": 30.0,
    },
    "supervisor": {
        # Log file (rotated daily).
        "log_dir":        "D:\\w1z4rdv1510n-data\\training",
        "log_file_name":  "supervisor.log",
        # On Windows: HIGH_PRIORITY_CLASS so we react fast even when
        # the rest of the system is loaded.
        "boost_priority": True,
    },
    "django": {
        # The R3V3N!R / wizard-chat control tower.  Operationally
        # critical for the wizard chat UI; the node alone is not enough
        # because the rolling-context store and agent endpoints live in
        # Django.  Watchdog policy mirrors the node policy.
        "enabled":        True,
        "project_root":   "D:\\Projects\\CoolCryptoUtilities",
        "python":         "D:\\Projects\\CoolCryptoUtilities\\.venv\\Scripts\\python.exe",
        "working_dir":    "D:\\Projects\\CoolCryptoUtilities\\web",
        "launcher":       "run_waitress.py",
        "host":           "127.0.0.1",
        "port":           8000,
        "threads":        8,
        "health_url":     "http://127.0.0.1:8000/api/wizard-chat/status/",
        "health_timeout": 5.0,
        "health_misses_before_restart": 3,
        "warmup_secs":    15.0,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg_path = PROJECT_ROOT / "supervisor.toml"
    if cfg_path.exists() and _toml_loader is not None:
        try:
            user_cfg = _toml_loader(cfg_path.read_bytes())
            return _merge(DEFAULT_CONFIG, user_cfg)
        except Exception as exc:
            print(f"[supervisor] WARNING: could not parse {cfg_path}: {exc}",
                  file=sys.stderr)
    return DEFAULT_CONFIG


# ── Logging ─────────────────────────────────────────────────────────────────


class Logger:
    def __init__(self, log_dir: str, log_file: str):
        self.dir = pathlib.Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / log_file
        self._fh = None
        self._open_day = None

    def _ensure(self):
        today = dt.date.today()
        if self._open_day != today or self._fh is None:
            if self._fh:
                try: self._fh.close()
                except Exception: pass
            self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
            self._open_day = today

    def write(self, level: str, msg: str) -> None:
        self._ensure()
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {level:5s}  {msg}\n"
        self._fh.write(line)
        print(line, end="", file=sys.stdout)
        sys.stdout.flush()

    def info(self, msg: str) -> None:  self.write("INFO",  msg)
    def warn(self, msg: str) -> None:  self.write("WARN",  msg)
    def error(self, msg: str) -> None: self.write("ERROR", msg)


# ── Process helpers ─────────────────────────────────────────────────────────


def boost_priority() -> None:
    """Bump our own process to HIGH on Windows so we stay responsive
    when the system is under load.  No-op elsewhere."""
    if os.name != "nt":
        return
    try:
        import ctypes
        HIGH_PRIORITY_CLASS = 0x00000080
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(h, HIGH_PRIORITY_CLASS)
    except Exception:
        pass


def find_processes(name_substr: str, cmd_substr: str = "") -> list[int]:
    """Return PIDs of processes whose image name contains `name_substr`
    and (optionally) whose command line contains `cmd_substr`.  Uses
    WMIC on Windows, ps on POSIX."""
    pids: list[int] = []
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["wmic", "process", "where",
                 f"name like '%{name_substr}%'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=10,
            )
            for line in out.stdout.splitlines():
                parts = line.strip().split(",", 2)
                if len(parts) < 3: continue
                _node, cmdline, pid = parts
                if cmd_substr and cmd_substr.lower() not in cmdline.lower():
                    continue
                try: pids.append(int(pid))
                except ValueError: continue
        except Exception:
            pass
    else:
        try:
            out = subprocess.run(["ps", "-eo", "pid,command"],
                                  capture_output=True, text=True, timeout=10)
            for line in out.stdout.splitlines()[1:]:
                line = line.strip()
                if name_substr not in line: continue
                if cmd_substr and cmd_substr not in line: continue
                pid = line.split(None, 1)[0]
                try: pids.append(int(pid))
                except ValueError: continue
        except Exception:
            pass
    return pids


# ── Node management ────────────────────────────────────────────────────────


def node_healthy(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read())
        return data.get("status") == "OK"
    except (urllib.error.URLError, json.JSONDecodeError, ConnectionError, OSError):
        return False


def try_graceful_shutdown(url: str, log: Logger, timeout: float = 8.0) -> bool:
    """POST /shutdown so the node flushes multi-pool hot tiers + cross
    synapses + raw pool to disk before exiting.  Returns True if the node
    responded 200 (a 500-ms grace timer fires server-side after the
    response, so the process IS going to exit even if we don't see it
    confirm).  Returns False on any error so the caller can fall back to
    taskkill — better to lose RAM than leave a zombie.
    """
    try:
        req = urllib.request.Request(url, data=b"",
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = 200 <= r.status < 300
            if ok:
                log.info(f"graceful shutdown accepted at {url}")
            return ok
    except Exception as exc:
        log.warn(f"graceful shutdown POST to {url} failed: {exc}")
        return False


def kill_processes_by_image(image_name: str, log: Logger,
                              exclude_pid: int | None = None) -> int:
    """Force-kill every process whose image name matches *image_name*
    (case-insensitive on Windows).  Returns the number of processes
    killed.  Used to clear stuck/orphan instances before relaunching —
    without this, every restart cycle leaves the previous instance
    holding the data dir and the port, and the cycle never recovers.
    """
    killed = 0
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
            )
            pids: list[int] = []
            for line in out.stdout.splitlines():
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) >= 2 and parts[0].lower() == image_name.lower():
                    try:
                        pid = int(parts[1])
                    except ValueError:
                        continue
                    if exclude_pid is not None and pid == exclude_pid:
                        continue
                    pids.append(pid)
            for pid in pids:
                try:
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                    capture_output=True, timeout=5)
                    killed += 1
                except Exception as exc:
                    log.warn(f"taskkill PID={pid} failed: {exc}")
        else:
            out = subprocess.run(["pgrep", "-f", image_name],
                                  capture_output=True, text=True, timeout=10)
            for raw in out.stdout.splitlines():
                try:
                    pid = int(raw.strip())
                except ValueError:
                    continue
                if exclude_pid is not None and pid == exclude_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed += 1
                except Exception:
                    pass
    except Exception as exc:
        log.warn(f"kill_processes_by_image({image_name}) failed: {exc}")
    if killed:
        log.info(f"killed {killed} orphan {image_name} process(es) before relaunch")
    return killed


def start_node(cfg: dict, log: Logger) -> int | None:
    """Launch the node binary detached.  Returns PID on success.

    Kills any existing w1z4rd_node.exe instances first so port 8090 and
    the data dir lock are free.  Without this, repeated supervisor
    restart cycles pile up zombie nodes that all fail to bind.
    """
    node_cfg = cfg["node"]
    binary = node_cfg["binary"]
    image = pathlib.Path(binary).name
    # Try graceful shutdown first so the running node flushes its
    # multi-pool hot tiers + cross synapses + raw pool to disk before
    # exiting.  Without this, every taskkill cycle wipes whatever is in
    # RAM and only the most recent checkpoint survives.  The /shutdown
    # endpoint schedules a 500-ms exit timer server-side, then taskkill
    # mops up anything still hanging on (slow saves, file locks).
    health_url = node_cfg.get("health_url", "")
    if health_url:
        # Derive shutdown URL by replacing path suffix with /shutdown.
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(health_url)
        shutdown_url = urlunsplit((parts.scheme, parts.netloc, "/shutdown", "", ""))
        if try_graceful_shutdown(shutdown_url, log, timeout=8.0):
            # Give the node up to 5 s to flush + exit on its own.
            for _ in range(10):
                if not node_healthy(health_url, 1.0):
                    break
                time.sleep(0.5)
    kill_processes_by_image(image, log)
    # Brief settle so the kernel releases the port + file handles.
    time.sleep(1.0)

    if not pathlib.Path(binary).exists():
        log.error(f"node binary not found: {binary}")
        return None
    env = os.environ.copy()
    env["W1Z4RDV1510N_DATA_DIR"] = node_cfg["data_dir"]
    stdout_path = pathlib.Path(node_cfg["data_dir"]) / "training" / "node_stdout.log"
    stderr_path = pathlib.Path(node_cfg["data_dir"]) / "training" / "node_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # DETACHED_PROCESS + CREATE_NEW_PROCESS_GROUP so the node is
        # independent of this supervisor — killing the supervisor must
        # NOT kill the node.
        creation = 0
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        with open(stdout_path, "ab") as outf, open(stderr_path, "ab") as errf:
            proc = subprocess.Popen(
                [binary],
                cwd=node_cfg["working_dir"],
                env=env,
                stdout=outf, stderr=errf, stdin=subprocess.DEVNULL,
                creationflags=creation if os.name == "nt" else 0,
                close_fds=True,
            )
        log.info(f"started node, PID={proc.pid}")
        return proc.pid
    except Exception as exc:
        log.error(f"failed to launch node: {exc}")
        return None


# ── Django (R3V3N!R control tower) management ──────────────────────────────


def django_healthy(url: str, timeout: float) -> bool:
    """True iff Django answers a basic GET on its status endpoint
    within `timeout`.  We accept any 2xx — the endpoint format may
    evolve, but if the server is responding at all the panel is alive."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def _kill_listeners_on_port(port: int, log: Logger) -> int:
    """Kill any process currently listening on *port*.  Used before
    launching waitress so a stuck old runserver/waitress instance
    doesn't block the new one from binding.  Returns number killed."""
    killed = 0
    if os.name != "nt":
        return 0
    try:
        # netstat -ano | findstr LISTENING + port
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=10,
        )
        pids: set[int] = set()
        for raw in out.stdout.splitlines():
            line = raw.strip()
            if "LISTENING" not in line:
                continue
            # columns: Proto  LocalAddr  ForeignAddr  State  PID
            parts = line.split()
            if len(parts) < 5:
                continue
            local_addr = parts[1]
            if not local_addr.endswith(f":{port}"):
                continue
            try:
                pids.add(int(parts[-1]))
            except ValueError:
                continue
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                                capture_output=True, timeout=5)
                killed += 1
            except Exception as exc:
                log.warn(f"taskkill PID={pid} on port {port} failed: {exc}")
    except Exception as exc:
        log.warn(f"_kill_listeners_on_port({port}) failed: {exc}")
    if killed:
        log.info(f"killed {killed} stale listener(s) on port {port}")
    return killed


def start_django(cfg: dict, log: Logger) -> int | None:
    """Launch waitress (run_waitress.py) detached.  Returns PID on success.

    Kills any process currently listening on the configured port so a
    stuck old instance doesn't prevent the new waitress from binding.
    """
    dcfg = cfg["django"]
    if not dcfg.get("enabled"):
        return None
    python = dcfg["python"]
    launcher = dcfg["launcher"]
    working_dir = dcfg["working_dir"]
    if not pathlib.Path(python).exists():
        log.error(f"django python not found: {python}")
        return None
    if not (pathlib.Path(working_dir) / launcher).exists():
        log.error(f"django launcher not found: {working_dir}\\{launcher}")
        return None

    _kill_listeners_on_port(int(dcfg["port"]), log)
    time.sleep(1.0)

    env = os.environ.copy()
    env["WAITRESS_HOST"]    = str(dcfg["host"])
    env["WAITRESS_PORT"]    = str(dcfg["port"])
    env["WAITRESS_THREADS"] = str(dcfg["threads"])

    stdout_path = pathlib.Path(cfg["supervisor"]["log_dir"]) / "django_stdout.log"
    stderr_path = pathlib.Path(cfg["supervisor"]["log_dir"]) / "django_stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        creation = 0
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        with open(stdout_path, "ab") as outf, open(stderr_path, "ab") as errf:
            proc = subprocess.Popen(
                [python, launcher],
                cwd=working_dir,
                env=env,
                stdout=outf, stderr=errf, stdin=subprocess.DEVNULL,
                creationflags=creation if os.name == "nt" else 0,
                close_fds=True,
            )
        log.info(f"started django (waitress), PID={proc.pid}, port={dcfg['port']}")
        return proc.pid
    except Exception as exc:
        log.error(f"failed to launch django: {exc}")
        return None


# ── Training management ────────────────────────────────────────────────────


def training_running() -> bool:
    return bool(find_processes("bash", "run_all_training.sh") or
                find_processes("bash.exe", "run_all_training.sh"))


def find_git_bash() -> str | None:
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for c in candidates:
        if pathlib.Path(c).exists():
            return c
    try:
        out = subprocess.run(["where", "bash.exe"], capture_output=True,
                              text=True, timeout=5)
        for line in out.stdout.splitlines():
            line = line.strip()
            if line and pathlib.Path(line).exists():
                return line
    except Exception:
        pass
    return None


def start_training(cfg: dict, log: Logger) -> int | None:
    """Launch the configured training command detached."""
    tcfg = cfg["training"]
    if not tcfg.get("enabled"):
        return None

    marker = PROJECT_ROOT / tcfg["first_run_marker"]
    is_resume = marker.exists() and tcfg.get("resume_on_restart")

    env = os.environ.copy()
    if is_resume:
        env["SKIP_CLEAR"] = "1"
        log.info("relaunching training with SKIP_CLEAR=1 (resume mode)")
    else:
        log.info("first-run training launch (will clear pool)")

    cmd_line = tcfg["command"]
    if os.name == "nt":
        bash = find_git_bash()
        if not bash:
            log.error("Git Bash not found; cannot launch training")
            return None
        # cmd_line is a relative-to-project script path; build a
        # POSIX-style invocation through bash -c so it's portable.
        bash_cmd = (
            f"cd '{PROJECT_ROOT.as_posix()}' && "
            f"SKIP_CLEAR='{env.get('SKIP_CLEAR', '')}' "
            f"bash {cmd_line}"
        )
        args: list[str] = [bash, "-c", bash_cmd]
    else:
        args = ["bash", "-c", f"cd {PROJECT_ROOT} && bash {cmd_line}"]

    stdout_path = pathlib.Path(cfg["node"]["data_dir"]) / "training" / "run_all_full.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        creation = 0
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        with open(stdout_path, "ab") as outf:
            proc = subprocess.Popen(
                args, cwd=str(PROJECT_ROOT), env=env,
                stdout=outf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=creation if os.name == "nt" else 0,
                close_fds=True,
            )
        log.info(f"started training, PID={proc.pid}, command={cmd_line!r}")
        marker.touch(exist_ok=True)
        return proc.pid
    except Exception as exc:
        log.error(f"failed to launch training: {exc}")
        return None


# ── Supervisor loop ────────────────────────────────────────────────────────


def run_supervisor(cfg: dict, log: Logger, once: bool = False) -> int:
    if cfg["supervisor"]["boost_priority"]:
        boost_priority()
    log.info("supervisor starting")

    miss_count = 0
    django_miss_count = 0
    training_last_relaunch = 0.0

    while True:
        # --- Node liveness ---
        if node_healthy(cfg["node"]["health_url"], cfg["node"]["health_timeout"]):
            if miss_count > 0:
                log.info(f"node recovered after {miss_count} miss(es)")
            miss_count = 0
        else:
            miss_count += 1
            log.warn(f"node health probe failed ({miss_count}/"
                      f"{cfg['node']['health_misses_before_restart']})")
            if miss_count >= cfg["node"]["health_misses_before_restart"]:
                log.error("node appears dead; relaunching")
                start_node(cfg, log)
                miss_count = 0
                time.sleep(cfg["node"]["warmup_secs"])
                if once: return 0
                continue

        # --- Django liveness (R3V3N!R control tower) ---
        dcfg = cfg.get("django") or {}
        if dcfg.get("enabled"):
            if django_healthy(dcfg["health_url"], dcfg["health_timeout"]):
                if django_miss_count > 0:
                    log.info(f"django recovered after {django_miss_count} miss(es)")
                django_miss_count = 0
            else:
                django_miss_count += 1
                log.warn(f"django health probe failed ({django_miss_count}/"
                          f"{dcfg['health_misses_before_restart']})")
                if django_miss_count >= dcfg["health_misses_before_restart"]:
                    log.error("django appears dead; relaunching")
                    start_django(cfg, log)
                    django_miss_count = 0
                    time.sleep(dcfg["warmup_secs"])

        # --- Training liveness ---
        tcfg = cfg["training"]
        if tcfg.get("enabled") and not training_running():
            # Backoff: don't relaunch immediately if we just tried.
            now = time.time()
            if now - training_last_relaunch >= tcfg["restart_backoff_secs"]:
                log.warn("training not running; launching")
                start_training(cfg, log)
                training_last_relaunch = now
                # Give the curriculum a moment to advance past Phase 0.
                time.sleep(2)

        if once:
            return 0
        time.sleep(cfg["node"]["poll_interval"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                     help="Run one iteration and exit (for testing).")
    ap.add_argument("--print-config", action="store_true",
                     help="Print the resolved config and exit.")
    args = ap.parse_args()
    cfg = load_config()
    if args.print_config:
        print(json.dumps(cfg, indent=2))
        return 0
    log = Logger(cfg["supervisor"]["log_dir"], cfg["supervisor"]["log_file_name"])
    try:
        return run_supervisor(cfg, log, once=args.once)
    except KeyboardInterrupt:
        log.info("supervisor stopped by signal")
        return 0


if __name__ == "__main__":
    sys.exit(main())
