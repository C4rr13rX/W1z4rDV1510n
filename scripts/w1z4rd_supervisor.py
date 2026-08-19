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
        # RAM politeness floor, mirroring start_node.ps1. The tier
        # orchestrator evicts neurons to the SSD cold tier as available
        # system RAM approaches this, so the node never outgrows the box.
        #
        # start_node.ps1 has always set this, but start_node() below did
        # not -- a supervisor-launched node therefore ran with NO ceiling.
        # Observed 2026-08-17: 47 GB commit on a 32 GB machine, 11 GB
        # resident, pagefile at 34.8 GB, C: queue 6 at 1049% disk time.
        # Every /brain/tick and /brain/observe then blocked on a page-in
        # (one thread had burned 39 h of CPU in PageIn wait) while /health
        # kept answering instantly from a static struct -- so the watchdog
        # saw a healthy node while nothing could learn.
        "min_sys_avail_mb": 4096,
        # Autocheckpoint cadence (seconds), also mirroring start_node.ps1.
        "autocheckpoint_secs": 900,
        # Explicit brain directory (exported as W1Z4RD_NODE_BRAIN_DIR).
        # Empty = let the node resolve it, which prefers
        # <W1Z4RDV1510N_DATA_DIR>/brain. Set this whenever the brain lives
        # somewhere other than under data_dir, or a relaunch will start from
        # an empty fabric and orphan the running brain.
        "node_brain_dir": "",
        # Probe a route that actually exercises the write path. /health is
        # a static struct and stays fast even when the fabric is wedged.
        "write_health_url": "http://localhost:8090/brain/tick",
        "write_health_timeout": 45.0,
        "write_health_misses_before_restart": 4,
        # Seconds between write probes. Deliberately much larger than
        # poll_interval for two reasons:
        #   * /brain/tick MUTATES the fabric (it advances the tick counter),
        #     so probing every 5 s would add ~17k spurious ticks a day.
        #   * a blocked probe stalls this single-threaded loop for its full
        #     timeout, which would otherwise starve the Django and crypto
        #     checks of their 5 s cadence exactly when the box is unhealthy.
        # A wedge is a sustained condition, not a transient, so 5 min
        # sampling still catches it long before it matters.
        "write_health_interval": 300.0,
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
        # 8001 is the canonical R3V3N!R panel port: launch_revenir.ps1 and
        # scripts/operations/monitor_everything.ps1 both bind/probe 8001.
        # The old 8000 default meant every probe failed against a perfectly
        # healthy panel, so the supervisor "restarted" Django every 3 polls
        # and spawned a fresh CoolCryptoUtilities console each cycle.
        "port":           8001,
        "threads":        8,
        # Probe the SPA root, not /api/wizard-chat/status/ — the status
        # endpoint fans out to the node's /health + /brain and can take
        # 4-6 s under training load, pushing past health_timeout and
        # triggering the same false-positive restarts.
        "health_url":     "http://127.0.0.1:8001/",
        "health_timeout": 5.0,
        "health_misses_before_restart": 3,
        "warmup_secs":    15.0,
    },
    "crypto_stack": {
        # The R3V3N!R trading services. These are process-level watches, not
        # HTTP health checks: each entry is matched by a substring of the
        # python command line, and relaunched with its own argv if absent.
        #
        # Before this existed the supervisor watched only node + Django, so
        # when the production manager died nothing noticed — the heartbeat
        # sat 3+ days stale while the desktop icons still reported a
        # healthy-looking stack. launch_revenir.ps1 could bring them back,
        # but only if a human double-clicked it.
        "enabled":        True,
        "project_root":   "D:\\Projects\\CoolCryptoUtilities",
        "python":         "D:\\Projects\\CoolCryptoUtilities\\.venv\\Scripts\\python.exe",
        "log_dir":        "D:\\Projects\\CoolCryptoUtilities\\logs",
        # Public wallet address (not a secret) — mirrors launch_revenir.ps1.
        # default_env_user returns None outside the manage.py boot path,
        # which otherwise leaves PortfolioState unable to derive the wallet.
        "primary_wallet": "0x291c854811e92906a658fb94aa511bf919f968ad",
        # Grace period after a relaunch before that service is eligible to
        # be relaunched again (slow imports: TF, pandas, web3).
        "restart_backoff_secs": 180.0,
        "services": [
            {
                "name":  "production_manager",
                "match": "start_production",
                "args":  ["-u", "main.py", "--action", "start_production", "--stay-alive"],
                "log":   "prod_direct",
            },
            {
                "name":  "brain_feeder",
                "match": "run_brain_feeder",
                "args":  ["scripts/run_brain_feeder.py"],
                "log":   "feeder_direct",
                # Skip while a history supervisor is training, to avoid
                # fighting it for the node's inner lock.
                "skip_if_running": "brain_history_supervisor",
            },
        ],
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
        # Mirror to stdout only when there is one.  Under pythonw.exe (and
        # under a Scheduled Task with no console) sys.stdout is None or a
        # closed handle, so an unguarded print()/flush() raises and kills
        # the supervisor moments after the "supervisor starting" line --
        # exactly the silent death that left the log looking like it had
        # started fine and then nothing ever ran again.
        try:
            if sys.stdout is not None:
                print(line, end="", file=sys.stdout)
                sys.stdout.flush()
        except (OSError, ValueError):
            pass

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


def node_write_healthy(url: str, timeout: float) -> bool:
    """True iff the node can still service a WRITE-path request.

    /health returns a static struct and keeps answering in ~1 ms even when
    the fabric is wedged.  On 2026-08-17 the node had grown past physical
    RAM and every write route blocked indefinitely on page-ins, yet the
    watchdog saw a green /health for days while nothing could learn.

    /brain/tick is the cheapest route that actually takes the fabric lock,
    so a hang here is the signal the read-only probe cannot give us. Any
    HTTP answer counts as alive -- we are testing liveness, not the tick
    value. An empty URL disables the probe.
    """
    if not url:
        return True
    request = urllib.request.Request(
        url, data=b"{}", headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return True
    except urllib.error.HTTPError:
        # 4xx/5xx still proves the handler ran rather than blocking.
        return True
    except (urllib.error.URLError, ConnectionError, OSError):
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
    # Mirror start_node.ps1's politeness contract. Without these the node
    # has no RAM ceiling: it grew to 47 GB commit on a 32 GB box, paged
    # continuously, and every write route (/brain/tick, /brain/observe,
    # /brain/chat) blocked forever on page-ins while /health stayed fast.
    min_avail = node_cfg.get("min_sys_avail_mb")
    if min_avail:
        env["W1Z4RD_TIER_MIN_SYS_AVAIL_MB"] = str(int(min_avail))
    checkpoint_secs = node_cfg.get("autocheckpoint_secs")
    if checkpoint_secs:
        env["W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS"] = str(int(checkpoint_secs))
    # Pin the brain directory when configured. default_node_brain_dir()
    # otherwise prefers <W1Z4RDV1510N_DATA_DIR>/brain, which is NOT where the
    # live node checkpoints -- a relaunch would start from an empty fabric and
    # orphan the running brain.
    node_brain_dir = node_cfg.get("node_brain_dir")
    if node_brain_dir:
        env["W1Z4RD_NODE_BRAIN_DIR"] = str(node_brain_dir)
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


# ── Crypto stack (R3V3N!R trading services) ────────────────────────────────


def _python_cmdlines() -> list[str]:
    """Every running python process's command line (Windows via WMI, else ps).

    One snapshot per poll, so N service checks cost one subprocess call
    instead of N.
    """
    lines: list[str] = []
    if os.name == "nt":
        # Query WMI in-process via COM.  Shelling out to powershell.exe
        # looked simpler but fails under pythonw.exe/Scheduled Task: with no
        # console to inherit, the child gets invalid std handles and returns
        # nothing, so every service looked dead.  win32com is not guaranteed
        # present, so fall back to wmic (deprecated but still shipped) and
        # finally to CREATE_NO_WINDOW powershell.
        try:
            import win32com.client  # type: ignore

            wmi = win32com.client.GetObject("winmgmts:")
            query = ("SELECT CommandLine FROM Win32_Process WHERE "
                     "Name='python.exe' OR Name='pythonw.exe'")
            lines = [str(proc.CommandLine) for proc in wmi.ExecQuery(query)
                     if proc.CommandLine]
        except Exception:
            lines = []
        if not lines:
            CREATE_NO_WINDOW = 0x08000000
            for argv in (
                ["wmic", "process", "where",
                 "name='python.exe' or name='pythonw.exe'",
                 "get", "CommandLine", "/format:list"],
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR "
                 "Name='pythonw.exe'\" | ForEach-Object { $_.CommandLine }"],
            ):
                try:
                    out = subprocess.run(
                        argv, capture_output=True, text=True, timeout=30,
                        encoding="utf-8", errors="replace",
                        stdin=subprocess.DEVNULL,
                        creationflags=CREATE_NO_WINDOW,
                    )
                except Exception:
                    continue
                found = []
                for raw in out.stdout.splitlines():
                    line = raw.strip()
                    if not line:
                        continue
                    # wmic /format:list emits "CommandLine=<value>".
                    if line.startswith("CommandLine="):
                        line = line[len("CommandLine="):].strip()
                    if line:
                        found.append(line)
                if found:
                    lines = found
                    break
    else:
        try:
            out = subprocess.run(["ps", "-eo", "command"],
                                  capture_output=True, text=True, timeout=15)
            lines = [ln.strip() for ln in out.stdout.splitlines()[1:] if ln.strip()]
        except Exception:
            pass
    return lines



def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    """Parse a KEY=VALUE .env file. Returns {} when absent or unreadable.

    Deliberately minimal: no interpolation or export handling, because the
    crypto stack's .env is a flat list of literals and a partial parse is
    worse than none.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def start_crypto_service(cfg: dict, service: dict, log: Logger) -> int | None:
    """Launch one crypto-stack service detached, mirroring launch_revenir.ps1."""
    ccfg = cfg["crypto_stack"]
    python = ccfg["python"]
    if not pathlib.Path(python).exists():
        log.error(f"crypto python not found: {python}")
        return None
    project_root = ccfg["project_root"]

    env = os.environ.copy()
    # The crypto stack configures itself through CoolCryptoUtilities/.env --
    # GHOST_PAIR_LIMIT, ENABLE_LIVE_TRADING, GENOME_PAIR_SEED and the rest are
    # all read with plain os.getenv. Nothing in that project calls
    # load_dotenv, and this launcher only inherited the supervisor's own
    # environment, so every one of those knobs was silently inert: editing
    # .env changed nothing and the default was always used. Load it here so
    # the file means what it appears to mean.
    #
    # Real environment wins over the file, so an operator override on the
    # supervisor process is never clobbered by a stale checked-in value.
    for key, value in _read_env_file(pathlib.Path(project_root) / ".env").items():
        env.setdefault(key, value)
    env["PRIMARY_WALLET"] = str(ccfg.get("primary_wallet") or "")
    # Force re-hydration from the vault; a stale value here leaves the
    # manager without wallet-derived state.
    env["SECURE_ENV_HYDRATED"] = ""
    # Deliberately do NOT set SKIP_TF_CONFIGURE: let pipeline._load_tf try
    # the import once and log a single clear warning if it fails, rather
    # than silently disabling every TF-dependent subsystem.
    env.pop("SKIP_TF_CONFIGURE", None)

    log_dir = pathlib.Path(ccfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = service.get("log") or service["name"]
    try:
        creation = 0
        if os.name == "nt":
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            creation = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        with open(log_dir / f"{stem}.log", "ab") as outf, \
             open(log_dir / f"{stem}.err", "ab") as errf:
            proc = subprocess.Popen(
                [python, *service["args"]],
                cwd=project_root,
                env=env,
                stdout=outf, stderr=errf, stdin=subprocess.DEVNULL,
                creationflags=creation if os.name == "nt" else 0,
                close_fds=True,
            )
        log.info(f"started crypto service {service['name']}, PID={proc.pid}")
        return proc.pid
    except Exception as exc:
        log.error(f"failed to launch {service['name']}: {exc}")
        return None


def check_crypto_stack(cfg: dict, log: Logger, last_start: dict[str, float]) -> None:
    """Relaunch any crypto-stack service that isn't running."""
    ccfg = cfg.get("crypto_stack") or {}
    if not ccfg.get("enabled"):
        return
    services = ccfg.get("services") or []
    if not services:
        return
    cmdlines = _python_cmdlines()
    if not cmdlines:
        # A failed snapshot must not be read as "everything is dead" — that
        # would relaunch the whole stack on top of itself.
        log.warn("could not enumerate python processes; skipping crypto checks")
        return

    backoff = float(ccfg.get("restart_backoff_secs", 180.0))
    now = time.time()
    for service in services:
        needle = service["match"].lower()
        if any(needle in line.lower() for line in cmdlines):
            continue
        guard = (service.get("skip_if_running") or "").lower()
        if guard and any(guard in line.lower() for line in cmdlines):
            log.info(f"{service['name']} down but {service['skip_if_running']} "
                      f"is running; leaving it alone")
            continue
        if now - last_start.get(service["name"], 0.0) < backoff:
            continue
        log.warn(f"crypto service {service['name']} not running; launching")
        start_crypto_service(cfg, service, log)
        last_start[service["name"]] = now


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
    write_miss_count = 0
    # When /health first started answering after the last (re)start. The
    # write probe waits out warmup_secs from this point so it never fires
    # against a node that is still deserialising its brain.
    node_first_healthy_unix = 0.0
    # Last time the write probe actually ran, so it samples on its own
    # (much slower) cadence rather than once per poll.
    write_last_probe_unix = 0.0
    django_miss_count = 0
    training_last_relaunch = 0.0
    crypto_last_start: dict[str, float] = {}

    while True:
        # --- Node liveness ---
        if node_healthy(cfg["node"]["health_url"], cfg["node"]["health_timeout"]):
            if miss_count > 0:
                log.info(f"node recovered after {miss_count} miss(es)")
            miss_count = 0
            if not node_first_healthy_unix:
                node_first_healthy_unix = time.time()
        else:
            miss_count += 1
            node_first_healthy_unix = 0.0
            log.warn(f"node health probe failed ({miss_count}/"
                      f"{cfg['node']['health_misses_before_restart']})")
            if miss_count >= cfg["node"]["health_misses_before_restart"]:
                log.error("node appears dead; relaunching")
                start_node(cfg, log)
                miss_count = 0
                write_miss_count = 0
                node_first_healthy_unix = 0.0
                write_last_probe_unix = 0.0
                time.sleep(cfg["node"]["warmup_secs"])
                if once: return 0
                continue

        # --- Node WRITE path liveness ---
        # /health is a static struct, so it stays green while the fabric is
        # wedged. Probing a route that takes the fabric lock is the only way
        # to notice a node that is "up" but can no longer learn.
        #
        # Only probe once the node has been answering /health for a while:
        # during the 12-15 min startup deserialise the write path is
        # legitimately busy, and killing the node mid-load is exactly the
        # false positive warmup_secs exists to prevent.
        ncfg = cfg["node"]
        write_url = ncfg.get("write_health_url") or ""
        write_limit = int(ncfg.get("write_health_misses_before_restart") or 0)
        healthy_streak_secs = (
            time.time() - node_first_healthy_unix
            if node_first_healthy_unix else 0.0
        )
        write_interval = float(ncfg.get("write_health_interval") or 0.0)
        write_due = (time.time() - write_last_probe_unix) >= write_interval
        if (write_url and write_limit and miss_count == 0 and write_due
                and healthy_streak_secs >= float(ncfg.get("warmup_secs") or 0.0)):
            write_last_probe_unix = time.time()
            if node_write_healthy(
                write_url, float(ncfg.get("write_health_timeout") or 45.0)
            ):
                if write_miss_count > 0:
                    log.info(
                        f"node write path recovered after {write_miss_count} miss(es)"
                    )
                write_miss_count = 0
            else:
                write_miss_count += 1
                log.warn(
                    "node WRITE path probe failed "
                    f"({write_miss_count}/{write_limit}) — /health may still be "
                    "green while the fabric cannot learn"
                )
                if write_miss_count >= write_limit:
                    log.error("node write path wedged; relaunching")
                    start_node(cfg, log)
                    miss_count = 0
                    write_miss_count = 0
                    node_first_healthy_unix = 0.0
                    write_last_probe_unix = 0.0
                    time.sleep(ncfg["warmup_secs"])
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

        # --- Crypto stack liveness (production manager, brain feeder) ---
        try:
            check_crypto_stack(cfg, log, crypto_last_start)
        except Exception as exc:
            # Never let a trading-stack check kill the node watchdog.
            log.error(f"crypto stack check failed: {exc}")

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
    except Exception:
        # A watchdog that dies silently is worse than useless: the stack
        # looks supervised while nothing is watching it.  Record the full
        # traceback to the log (the only place visible under a Scheduled
        # Task) and exit non-zero so the task's RestartOnFailure brings us
        # back.
        import traceback
        log.error("supervisor crashed:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
