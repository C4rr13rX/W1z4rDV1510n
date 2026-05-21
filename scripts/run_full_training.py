#!/usr/bin/env python3
"""scripts/run_full_training.py
=================================

End-to-end Stage-16 training orchestrator.  Does the whole pipeline
the way Stage 16 was validated:

  1. Launch (or reuse) the brain server on port 8095 with the
     canonical Stage-16 env variables:
       BRAIN_SPARSITY_ACTION = Constant(0.7711)
       BRAIN_MIN_ATOM_SCORE  = Constant(0.9412)
     (other ControlMode knobs at defaults)
  2. Wipe the data dir's brain.bin so we train on a fresh substrate
     (toggleable via --no-wipe).
  3. Run the toddler 32-pair dense-burst (8 reps) — this is what
     locks in the cross-pool baseline binding behaviour.
  4. Run the full curriculum via drive_corpora_brain in phase order
     with --burst + --sleep-between (Stage 16 defaults).
  5. POST /sleep one more time + /checkpoint to persist.
  6. Run scripts/brain_fluency_eval.py against the trained brain
     and print the summary block.

Usage:
  python scripts/run_full_training.py                    # full pipeline
  python scripts/run_full_training.py --no-wipe          # keep prior brain.bin
  python scripts/run_full_training.py --skip-toddler     # start from curriculum
  python scripts/run_full_training.py --skip-curriculum  # toddler + eval only
  python scripts/run_full_training.py --phase-max 5      # only phase <= 5
  python scripts/run_full_training.py --foreground       # spawn brain ourselves
                                                          and tear it down

By default we ASSUME the brain server is already running (so the
Django Wizard chat at D:/Projects/CoolCryptoUtilities/web doesn't
get interrupted).  Pass --foreground to spawn one and tear it down
when finished.

Wizard chat integration (D:/Projects/CoolCryptoUtilities/web/wizard_chat):
  * WIZARD_BRAIN_CHAT_URL defaults to http://localhost:8095
  * /chat is POST'd with {text, session_id, hops, min_strength}
  * Stage-16 brain /chat returns {reply, answer, decoder, predictions,
    grounding} — Wizard's views.py already maps `decoder=multi_pool` →
    confidence_tier=high and `decoder=char_chain` → confidence_tier=low,
    so nothing here needs to change for the frontend to see Stage-16
    answers.  This script just trains; the frontend reads the live
    brain state via the same port.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BRAIN_BIN    = PROJECT_ROOT / "bin" / "w1z4rd_brain_server.exe"
BRAIN_URL    = os.environ.get("W1Z4RD_BRAIN_URL", "http://127.0.0.1:8095")

# Stage-16 canonical wiring — discovered empirically by the GA in
# scripts/ga_brain_dynamical.py and verified by brain_fluency_eval
# returning toddler 32/32 + OOV 3/3 + K-12 16/16 + multi_fact 5/5
# + /integrate 32/32 (theoretical max).
CANONICAL_ENV = {
    "BRAIN_SPARSITY_ACTION": '{"Constant":0.7711}',
    "BRAIN_MIN_ATOM_SCORE":  '{"Constant":0.9412}',
}

# Default curriculum order.  Phase 0 = greetings (small, dense, should
# ground first).  Phase 1 = conversation + categorical (medium, the
# K-12 backbone).  Phases 5-35 = dialog, code, agent planning, long-context.
# drive_corpora_brain.py reads the registry and runs in phase order
# automatically; this list is just for the toddler-baseline step.
TODDLER_PAIRS = [
    ("dog","animal"),("cat","animal"),("cow","animal"),("horse","animal"),
    ("bird","animal"),("fish","animal"),
    ("apple","food"),("banana","food"),("bread","food"),("cake","food"),
    ("milk","food"),
    ("car","vehicle"),("truck","vehicle"),("bike","vehicle"),
    ("plane","vehicle"),("boat","vehicle"),
    ("red","color"),("blue","color"),("green","color"),("yellow","color"),
    ("ball","toy"),("doll","toy"),("kite","toy"),("drum","toy"),
    ("tree","nature"),("flower","nature"),("river","nature"),("mountain","nature"),
    ("hand","body"),("foot","body"),("eye","body"),("mouth","body"),
]


# ── Brain server lifecycle ────────────────────────────────────────────────


def _health_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout) as r:
            return bool(r.read())
    except Exception:
        return False


def _wait_for_brain(url: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok(url, timeout=2.0):
            return True
        time.sleep(1.0)
    return False


def spawn_brain_server(data_dir: pathlib.Path,
                         port: int = 8095) -> subprocess.Popen | None:
    if not BRAIN_BIN.exists():
        print(f"[run_full] brain binary not found at {BRAIN_BIN} — "
                f"build via `cargo build --release -p w1z4rdv1510n-node "
                f"--bin w1z4rd_brain_server` and copy to bin/", flush=True)
        return None
    env = os.environ.copy()
    env["W1Z4RD_BRAIN_PORT"]      = str(port)
    env["W1Z4RDV1510N_DATA_DIR"]  = str(data_dir)
    for k, v in CANONICAL_ENV.items():
        env.setdefault(k, v)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(
        [str(BRAIN_BIN)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    print(f"[run_full] spawned brain pid={proc.pid} port={port} "
            f"data={data_dir}", flush=True)
    return proc


def teardown_brain_server(proc: subprocess.Popen) -> None:
    try:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: pass
    except Exception: pass
    try: proc.kill()
    except Exception: pass
    if sys.platform == "win32" and proc.pid:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True, text=True)


# ── Curriculum steps ──────────────────────────────────────────────────────


def train_toddler(reps: int = 8) -> bool:
    """Stage-16 canonical toddler dense-burst — 32 pairs × N reps.
    This is what locks in cross-pool baseline behaviour before broader
    corpora introduce conflicts.  Uses the same /observe x2 + /tick
    contract as drive_corpora_brain."""
    import base64, json as _json
    def b64(s: str) -> str:
        return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")
    def post(path: str, body: dict, timeout: float = 10.0):
        raw = _json.dumps(body).encode()
        req = urllib.request.Request(
            f"{BRAIN_URL}{path}", data=raw, method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return _json.loads(r.read())
        except Exception:
            return None
    print(f"[run_full] toddler dense-burst ({len(TODDLER_PAIRS)} pairs × {reps} reps)…",
            flush=True)
    t0 = time.time()
    for p, r in TODDLER_PAIRS:
        for _ in range(reps):
            post("/observe", {"pool_id": 1, "frame": b64(p)})
            post("/observe", {"pool_id": 4, "frame": b64(r)})
            post("/tick", {})
    print(f"[run_full] toddler done in {time.time()-t0:.0f}s", flush=True)
    return True


def run_curriculum(repeats: int, phase_max: int | None) -> int:
    """Invoke drive_corpora_brain as a module so its phase-ordering
    + sleep-between-phases logic stays the single source of truth."""
    cmd = [
        sys.executable, "-m", "tools.training_standard.drive_corpora_brain",
        "--repeats", str(repeats),
        "--brain", BRAIN_URL,
        "--checkpoint",
    ]
    if phase_max is not None:
        cmd += ["--phase-max", str(phase_max)]
    print(f"[run_full] curriculum: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def run_fluency_eval() -> int:
    """Final empirical check — runs the canonical 61-probe fluency
    panel and prints the summary."""
    cmd = [sys.executable, "scripts/brain_fluency_eval.py"]
    print(f"[run_full] fluency_eval: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


# ── Main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=os.environ.get(
                     "W1Z4RDV1510N_DATA_DIR", r"D:\\w1z4rdv1510n-data"),
                     help="W1Z4RDV1510N_DATA_DIR for the brain.  Default reads "
                          "env var or falls back to D:\\w1z4rdv1510n-data.")
    p.add_argument("--no-wipe", action="store_true",
                     help="don't delete brain.bin before training (preserves "
                          "any prior knowledge — but Stage-16 100% requires a "
                          "fresh brain).")
    p.add_argument("--skip-toddler", action="store_true",
                     help="skip the canonical toddler dense-burst step")
    p.add_argument("--skip-curriculum", action="store_true",
                     help="skip the registry curriculum (only run toddler + eval)")
    p.add_argument("--skip-eval", action="store_true",
                     help="skip the final fluency_eval")
    p.add_argument("--phase-max", type=int, default=None,
                     help="curriculum: only run scripts with phase <= this value")
    p.add_argument("--repeats", type=int, default=4,
                     help="curriculum: pass to drive_corpora_brain (default 4 — "
                          "Stage-16 dense-burst converges fast)")
    p.add_argument("--toddler-reps", type=int, default=8,
                     help="reps per toddler pair (default 8)")
    p.add_argument("--foreground", action="store_true",
                     help="spawn the brain server ourselves and tear it down "
                          "when the run finishes.  Default: assume the brain "
                          "is already running (so the Django Wizard chat "
                          "endpoint stays up across training runs).")
    p.add_argument("--port", type=int, default=8095,
                     help="brain server port (default 8095)")
    args = p.parse_args(argv)

    global BRAIN_URL
    BRAIN_URL = f"http://127.0.0.1:{args.port}"

    data_dir = pathlib.Path(args.data_dir)
    proc: subprocess.Popen | None = None

    try:
        # 1. Wipe brain.bin so we train on a fresh substrate
        if not args.no_wipe:
            brain_bin = data_dir / "brain.bin"
            if brain_bin.exists():
                print(f"[run_full] wiping {brain_bin}", flush=True)
                brain_bin.unlink()

        # 2. Bring up brain if --foreground; else require it to already be up
        if args.foreground:
            data_dir.mkdir(parents=True, exist_ok=True)
            proc = spawn_brain_server(data_dir, port=args.port)
            if proc is None: return 2
            if not _wait_for_brain(BRAIN_URL, timeout=60.0):
                print(f"[run_full] brain didn't come up at {BRAIN_URL}",
                        flush=True)
                return 3
        else:
            if not _health_ok(BRAIN_URL):
                print(f"[run_full] brain not reachable at {BRAIN_URL}.\n"
                        f"  Either:\n"
                        f"    1) start it manually with the canonical env:\n"
                        f"         BRAIN_SPARSITY_ACTION='{{\"Constant\":0.7711}}' \\\n"
                        f"         BRAIN_MIN_ATOM_SCORE='{{\"Constant\":0.9412}}' \\\n"
                        f"         W1Z4RDV1510N_DATA_DIR='{data_dir}' \\\n"
                        f"         ./bin/w1z4rd_brain_server.exe\n"
                        f"    2) or run this script with --foreground so it "
                        f"manages the brain lifecycle itself.", flush=True)
                return 4
            print(f"[run_full] using existing brain at {BRAIN_URL}", flush=True)

        # 3. Toddler dense-burst (Stage-16 baseline)
        if not args.skip_toddler:
            train_toddler(reps=args.toddler_reps)

        # 4. Full curriculum via drive_corpora_brain
        if not args.skip_curriculum:
            rc = run_curriculum(args.repeats, args.phase_max)
            if rc != 0:
                print(f"[run_full] curriculum exited rc={rc} — continuing to eval",
                        flush=True)

        # 5. Final eval
        if not args.skip_eval:
            print("\n[run_full] ─── final fluency_eval ───", flush=True)
            run_fluency_eval()

        return 0
    finally:
        if proc is not None:
            print("[run_full] tearing down brain server", flush=True)
            teardown_brain_server(proc)


if __name__ == "__main__":
    sys.exit(main())
