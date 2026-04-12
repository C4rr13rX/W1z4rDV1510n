#!/usr/bin/env python3
"""
start_local.py — Start the full W1z4rD chess training stack from any shell.

Works from Git Bash, PowerShell, or cmd.exe without path-conversion issues.
Equivalent to start_chess_local.bat but uses Python subprocess so the /MIN
flag ambiguity on Git Bash never arises.

Usage:
    python3 scripts/start_local.py
    python3 scripts/start_local.py --max-games 2000
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Windows-specific flags for detached background processes
# (no-op values used on non-Windows so the script parses everywhere)
DETACHED_PROCESS        = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
_WIN_FLAGS = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0


def log(msg: str) -> None:
    print(msg, flush=True)


def is_port_open(port: int) -> bool:
    """True if something is already LISTENING on the port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_detached(args: list, logfile: pathlib.Path, cwd: pathlib.Path = ROOT) -> subprocess.Popen:
    logfile.parent.mkdir(parents=True, exist_ok=True)
    fh = open(logfile, "w", encoding="utf-8")
    return subprocess.Popen(
        args,
        stdout=fh,
        stderr=fh,
        cwd=str(cwd),
        creationflags=_WIN_FLAGS,
        close_fds=True,
    )


def reset_logs() -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    board = {
        "game_count": 0, "game_id": "starting...", "result": "?",
        "ply": 0, "total_plies": 0, "side_to_move": "white", "pieces": [],
        "last_move": None, "predictions": [], "actual_move": None,
        "reveal_phase": True, "prediction_correct_top1": False,
        "square_heat": [[0.0] * 8 for _ in range(8)],
        "outcome_probs": {"1-0": 0.33, "1/2-1/2": 0.34, "0-1": 0.33},
        "model_breakdown": [], "model_ledgers": {},
        "running": {
            "move_top1": 0.0, "move_top3": 0.0, "outcome_acc": 0.0,
            "total_plies": 0, "game_count": 0,
        },
        "hourly_checkpoints": [], "training_mode": True, "iteration": 0,
    }
    (logs / "chess_live_board.json").write_text(json.dumps(board), encoding="utf-8")

    for fname in [
        "chess_live_run.jsonl", "chess_training_metrics.log", "chess_training_out.txt",
        "viz_server.log", "viz_server_err.log",
        "node.log", "node_err.log", "w1z4rd_api.log",
    ]:
        (logs / fname).write_text("", encoding="utf-8")

    # Clear node knowledge state for a fresh start
    ks = ROOT / "data" / "knowledge_state_local.json"
    if ks.exists():
        ks.unlink()
        log("  [reset] Cleared knowledge_state_local.json")

    log("  [reset] Logs cleared")


def _kill_python_scripts(*script_names: str) -> None:
    """Kill any python process whose command line contains one of the given script names."""
    import psutil
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or [])
            name = proc.info.get("name", "")
            if "python" not in name.lower():
                continue
            for script in script_names:
                if script in cmdline:
                    log(f"  [CLEANUP] Killing stale {script} (PID {proc.pid})")
                    proc.kill()
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Start W1z4rD chess training stack")
    parser.add_argument("--max-games", type=int, default=2000)
    parser.add_argument("--skip-reset", action="store_true", help="Don't clear logs")
    parser.add_argument("--node-only", action="store_true",
                        help="Start API + node + dashboard only; skip chess training and viz")
    parser.add_argument("--dashboard", action="store_true",
                        help="Also launch the dashboard GUI (always true with --node-only)")
    args = parser.parse_args()

    log("\n[W1Z4RD] Starting local chess training stack...\n")

    # ── psutil ───────────────────────────────────────────────────────────────
    try:
        import psutil  # noqa: F401
        log("[OK] psutil available")
    except ImportError:
        log("[SETUP] Installing psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil", "--quiet"])
        log("[OK] psutil installed")

    # ── Reset logs ───────────────────────────────────────────────────────────
    if not args.skip_reset:
        reset_logs()

    # ── Kill any stale training / viz python processes first ─────────────────
    _kill_python_scripts("chess_training_loop.py", "live_viz_server.py")
    time.sleep(1)

    # ── Kill stale processes on ports we need ────────────────────────────────
    killed_any = False
    for port in (8080, 8090, 8765):
        if is_port_open(port):
            log(f"[CLEANUP] Port {port} in use — killing occupant...")
            killed_any = True
            if sys.platform == "win32":
                # Use netstat + taskkill; run in cmd.exe via shell=True
                subprocess.run(
                    f'for /f "tokens=5" %p in (\'netstat -ano ^| findstr ":{port} "\') do taskkill /PID %p /F',
                    shell=True, capture_output=True,
                )

    if killed_any:
        # Wait long enough for the OS to release ports before we try to rebind.
        log("[CLEANUP] Waiting for ports to be released...")
        for _ in range(10):
            time.sleep(0.5)
            if not any(is_port_open(p) for p in (8080, 8090, 8765)):
                break
        log("[CLEANUP] Ports released")

    # ── 1. w1z4rd_api (port 8080) ────────────────────────────────────────────
    api_bin = ROOT / "bin" / "w1z4rd_api.exe"
    if not is_port_open(8080):
        if api_bin.exists():
            start_detached([str(api_bin)], ROOT / "logs" / "w1z4rd_api.log")
            log("[API] Started w1z4rd_api.exe on port 8080")
            time.sleep(2)
            if is_port_open(8080):
                log("[OK] w1z4rd_api listening on :8080")
            else:
                log("[WARN] w1z4rd_api may still be starting — continuing")
        else:
            log(f"[WARN] {api_bin} not found — skipping API")
    else:
        log("[OK] w1z4rd_api already on :8080")

    # ── 2. w1z4rd_node (port 8090, SENSOR/local mode) ───────────────────────
    node_bin    = ROOT / "bin" / "w1z4rd_node.exe"
    node_config = ROOT / "node_config_local.json"
    if not is_port_open(8090):
        if node_bin.exists() and node_config.exists():
            start_detached(
                [str(node_bin), "--config", str(node_config), "api", "--addr", "0.0.0.0:8090"],
                ROOT / "logs" / "node.log",
            )
            log("[NODE] Started w1z4rd_node.exe (SENSOR mode) on port 8090")
            time.sleep(3)
            if is_port_open(8090):
                log("[OK] w1z4rd_node listening on :8090")
            else:
                log("[WARN] Node may still be starting — check logs/node.log")
        else:
            log("[WARN] node binary or config missing — skipping node")
    else:
        log("[OK] w1z4rd_node already on :8090")

    # ── 3. Chess training loop (skipped in --node-only mode) ────────────────
    if not args.node_only:
        train_script = ROOT / "scripts" / "chess_training_loop.py"
        train_log    = ROOT / "logs" / "chess_training_out.txt"
        start_detached(
            [sys.executable, str(train_script),
             "--max-games", str(args.max_games),
             "--log-file", str(ROOT / "logs" / "chess_training_metrics.log"),
             "--summary-file", str(ROOT / "logs" / "chess_run_summary.txt")],
            train_log,
        )
        log(f"[TRAIN] Chess training loop started (max-games={args.max_games})")
        time.sleep(3)

    # ── 4. Viz server (port 8765, skipped in --node-only mode) ──────────────
    if not args.node_only:
        viz_script = ROOT / "scripts" / "live_viz_server.py"
        start_detached(
            [sys.executable, str(viz_script),
             "--board-file", str(ROOT / "logs" / "chess_live_board.json"),
             "--port", "8765"],
            ROOT / "logs" / "viz_server.log",
        )
        log("[VIZ] Viz server started on port 8765")
        time.sleep(2)

        if is_port_open(8765):
            log("[OK] Viz server listening on :8765")
            import webbrowser
            webbrowser.open("http://localhost:8765")
            log("[VIZ] Opened http://localhost:8765 in browser")
        else:
            log("[WARN] Viz server not yet up — check logs/viz_server.log")

    # ── 5. Dashboard GUI ─────────────────────────────────────────────────────
    if args.node_only or args.dashboard:
        dash_bin = ROOT / "bin" / "w1z4rd_dashboard.exe"
        if dash_bin.exists():
            import subprocess as _sp
            _sp.Popen(
                [str(dash_bin), "--node", "http://127.0.0.1:8090", "--api", "http://127.0.0.1:8080"],
                cwd=str(ROOT),
                creationflags=_WIN_FLAGS if sys.platform == "win32" else 0,
                close_fds=True,
            )
            log("[GUI] Launched w1z4rd_dashboard (node=:8090, api=:8080)")
        else:
            log("[WARN] w1z4rd_dashboard.exe not found in bin/")

    log("\n" + "=" * 60)
    log("  Stack running:")
    log("    Neuro API  : http://localhost:8080")
    log("    Local Node : http://localhost:8090")
    if not args.node_only:
        log("    Viz Server : http://localhost:8765")
        log("    Training   : logs/chess_training_out.txt")
        log("")
        log("  Monitor:  tail -f logs/chess_training_out.txt")
    log("  Stop all: taskkill /IM python3.exe /F  (Windows)")
    log("=" * 60 + "\n")


if __name__ == "__main__":
    main()
