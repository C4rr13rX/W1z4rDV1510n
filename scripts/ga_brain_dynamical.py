#!/usr/bin/env python3
"""
ga_brain_dynamical.py — dynamical-system GA over substrate WIRING.

Per the user's vision: genes are NOT static numbers like
MIN_ATOM_SCORE=0.65.  Genes are which substrate-internal signal drives
which knob, via feedback loops the system computes itself each tick.
The GA evolves WIRINGS, the substrate computes the actual values.

Each knob's gene is a ControlMode JSON spec, e.g.:
  {"Constant": 0.65}
  {"DrivenBy": {"signal": "InvSurprise", "scale": 0.7, "offset": 0.2,
                "min": 0.05, "max": 0.95}}

Available signals (the substrate's own observables, updated each tick):
  Surprise, InvSurprise              — EMA of unpredicted firing fraction
  FiringRate, InvFiringRate          — normalised currently_firing.len()
  DecodePrecisionEma, InvDecodePrecisionEma  — rolling avg of decode wins
  ConceptCountEma, TerminalCountEma  — log-normed substrate density

Genes:
  - BRAIN_SPARSITY_TEXT       : ControlMode for text pool k-WTA
  - BRAIN_SPARSITY_ACTION     : same for action pool
  - BRAIN_HET_LTD_DEFAULT     : ControlMode for heterosynaptic LTD
  - BRAIN_PREDICT_GATE_TEXT   : ControlMode for predict-gate text
  - BRAIN_PREDICT_GATE_ACTION : ControlMode for predict-gate action
  - BRAIN_MIN_ATOM_SCORE      : ControlMode for decode floor

Mutation operators:
  - swap_signal: pick a different ControlSignal for one knob
  - jiggle_scale: gaussian on scale parameter
  - jiggle_offset: gaussian on offset parameter
  - flip_mode: swap Constant ↔ DrivenBy
  - tighten_clamps: adjust min/max range

Fitness: same panel as before (toddler/K-12/multi_fact/OOV) with
guardrails preventing toddler/OOV regression.

Output:
  - data/foundation/ga_brain_dynamical.jsonl  one genome per line
  - data/foundation/ga_brain_dynamical_best.json  marker file the
    self-pinging watcher monitors

Run:
  python scripts/ga_brain_dynamical.py --workers 4 --time-limit 28800
                                       # 8 hours

The harness writes BEST_PROGRESS lines to stdout that an outer Monitor
can grep for and surface as notifications.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import random
import socket
import subprocess

# Defensive: force stdout/stderr to utf-8 on Windows so any Unicode
# characters in print statements don't crash the run.  The previous
# Stage 15.X stall was caused by a single `→` arrow in a
# BEST_PROGRESS message hitting cp1252's encoding limit.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    try: _sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception: pass
if hasattr(_sys.stderr, "reconfigure"):
    try: _sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception: pass
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
BRAIN_BIN    = PROJECT_ROOT / "bin" / "w1z4rd_brain_server.exe"
LOG_PATH     = PROJECT_ROOT / "data" / "foundation" / "ga_brain_dynamical.jsonl"
BEST_PATH    = PROJECT_ROOT / "data" / "foundation" / "ga_brain_dynamical_best.json"
CORPUS_PATH  = PROJECT_ROOT / "data" / "training" / "categorical_unified_001.jsonl"

# ───────── ControlSignal universe ─────────
SIGNALS = [
    "Surprise", "InvSurprise",
    "FiringRate", "InvFiringRate",
    "DecodePrecisionEma", "InvDecodePrecisionEma",
    "ConceptCountEma", "TerminalCountEma",
]

# Per-knob default ranges + clamps.  GA produces ControlModes within these.
KNOB_SPECS = {
    "BRAIN_SPARSITY_TEXT":       {"default_const": 1.0, "min": 0.05, "max": 1.0,
                                  "scale_range": (-0.9, 0.9), "offset_range": (0.0, 1.0)},
    "BRAIN_SPARSITY_ACTION":     {"default_const": 1.0, "min": 0.05, "max": 1.0,
                                  "scale_range": (-0.9, 0.9), "offset_range": (0.0, 1.0)},
    "BRAIN_HET_LTD_DEFAULT":     {"default_const": 0.0, "min": 0.0,  "max": 0.5,
                                  "scale_range": (-0.4, 0.4), "offset_range": (0.0, 0.3)},
    "BRAIN_PREDICT_GATE_TEXT":   {"default_const": 0.0, "min": 0.0,  "max": 0.9,
                                  "scale_range": (-0.7, 0.7), "offset_range": (0.0, 0.5)},
    "BRAIN_PREDICT_GATE_ACTION": {"default_const": 0.0, "min": 0.0,  "max": 0.9,
                                  "scale_range": (-0.7, 0.7), "offset_range": (0.0, 0.5)},
    "BRAIN_MIN_ATOM_SCORE":      {"default_const": 0.65, "min": 0.20, "max": 0.95,
                                  "scale_range": (-0.5, 0.5), "offset_range": (0.2, 0.85)},
    # Hebbian frequency-weight multiplier on use_count.ln().
    # Default 1.0 = canonical formula `1 + ln(use_count)`.  Higher
    # amplifies the gap between trained-many-times and trained-once
    # bindings — should let K-12 prompts with conflicting categorical
    # entries pick the right one when the right binding has been
    # reinforced even slightly more than competitors.
    "BRAIN_FREQ_WEIGHT":         {"default_const": 1.0, "min": 0.0, "max": 8.0,
                                  "scale_range": (-3.0, 3.0), "offset_range": (0.0, 4.0)},
}

POP_SIZE     = 12
ELITE_FRAC   = 0.30
MUTATION_RATE = 0.40

# ───────── ControlMode helpers ─────────


def make_constant(knob: str, value: float | None = None) -> dict:
    spec = KNOB_SPECS[knob]
    if value is None: value = spec["default_const"]
    return {"Constant": round(float(value), 4)}


def make_drivenby(knob: str, signal: str | None = None,
                  scale: float | None = None, offset: float | None = None) -> dict:
    spec = KNOB_SPECS[knob]
    if signal is None: signal = random.choice(SIGNALS)
    if scale is None:  scale  = random.uniform(*spec["scale_range"])
    if offset is None: offset = random.uniform(*spec["offset_range"])
    return {"DrivenBy": {
        "signal": signal,
        "scale":  round(float(scale), 4),
        "offset": round(float(offset), 4),
        "min":    spec["min"],
        "max":    spec["max"],
    }}


def random_controlmode(knob: str) -> dict:
    # 40% constant, 60% DrivenBy — encourages exploration of wirings.
    if random.random() < 0.4:
        spec = KNOB_SPECS[knob]
        return make_constant(knob, random.uniform(spec["min"], spec["max"]))
    return make_drivenby(knob)


# ───────── Genome ─────────


@dataclass
class Genome:
    knobs: dict        # knob -> ControlMode dict
    fitness: float = 0.0
    metrics: dict = field(default_factory=dict)
    eval_kind: str = "unscored"
    born_gen: int = 0

    def env(self) -> dict:
        out = {}
        for k, v in self.knobs.items():
            out[k] = json.dumps(v, separators=(",", ":"))
        return out


def seed_genome(gen: int = 0) -> Genome:
    """The known-good baseline: every knob at its default Constant.
    Guarantees the initial population has the toddler 32/32 + OOV 3/3
    genome from gen 0."""
    return Genome(
        knobs={k: make_constant(k) for k in KNOB_SPECS},
        born_gen=gen,
    )


def random_genome(gen: int = 0) -> Genome:
    return Genome(
        knobs={k: random_controlmode(k) for k in KNOB_SPECS},
        born_gen=gen,
    )


def mutate(g: Genome, gen: int) -> Genome:
    new_knobs = {k: dict(v) if isinstance(v, dict) else v for k, v in g.knobs.items()}
    # Deep copy nested DrivenBy
    for k in new_knobs:
        if "DrivenBy" in new_knobs[k]:
            new_knobs[k] = {"DrivenBy": dict(new_knobs[k]["DrivenBy"])}
    for knob in list(new_knobs.keys()):
        if random.random() > MUTATION_RATE: continue
        spec = KNOB_SPECS[knob]
        current = new_knobs[knob]
        op = random.choice(["flip_mode", "swap_signal", "jiggle_scale", "jiggle_offset", "jiggle_constant"])
        if op == "flip_mode":
            if "Constant" in current:
                new_knobs[knob] = make_drivenby(knob)
            else:
                new_knobs[knob] = make_constant(knob,
                    random.uniform(spec["min"], spec["max"]))
        elif op == "swap_signal" and "DrivenBy" in current:
            current["DrivenBy"]["signal"] = random.choice(SIGNALS)
        elif op == "jiggle_scale" and "DrivenBy" in current:
            cur_scale = current["DrivenBy"]["scale"]
            span = spec["scale_range"][1] - spec["scale_range"][0]
            new_scale = max(spec["scale_range"][0],
                            min(spec["scale_range"][1],
                                random.gauss(cur_scale, span * 0.15)))
            current["DrivenBy"]["scale"] = round(new_scale, 4)
        elif op == "jiggle_offset" and "DrivenBy" in current:
            cur_off = current["DrivenBy"]["offset"]
            span = spec["offset_range"][1] - spec["offset_range"][0]
            new_off = max(spec["offset_range"][0],
                          min(spec["offset_range"][1],
                              random.gauss(cur_off, span * 0.15)))
            current["DrivenBy"]["offset"] = round(new_off, 4)
        elif op == "jiggle_constant" and "Constant" in current:
            cur = current["Constant"]
            span = spec["max"] - spec["min"]
            new = max(spec["min"], min(spec["max"],
                random.gauss(cur, span * 0.15)))
            new_knobs[knob] = {"Constant": round(new, 4)}
    return Genome(knobs=new_knobs, born_gen=gen)


def crossover(a: Genome, b: Genome, gen: int) -> Genome:
    new_knobs = {}
    for k in KNOB_SPECS:
        src = a if random.random() < 0.5 else b
        if "DrivenBy" in src.knobs[k]:
            new_knobs[k] = {"DrivenBy": dict(src.knobs[k]["DrivenBy"])}
        else:
            new_knobs[k] = dict(src.knobs[k])
    return Genome(knobs=new_knobs, born_gen=gen)


# ───────── Worker (same shape as before) ─────────


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _post(url: str, body: dict, timeout: float = 30.0) -> dict | None:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=raw, method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.read(): return True
        except Exception:
            time.sleep(0.5)
    return False


def _port_is_free(port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        return False
    except Exception:
        return True


def _kill_port(port: int) -> None:
    for _ in range(3):
        killed = False
        try:
            out = subprocess.check_output(["netstat", "-ano", "-p", "TCP"],
                text=True, errors="ignore")
            for line in out.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(["taskkill", "/F", "/T", "/PID", pid],
                        capture_output=True, text=True)
                    killed = True
        except Exception: pass
        if not killed: return
        time.sleep(0.5)


def _server_alive(port: int) -> bool:
    """Quick health probe — distinct from _wait_for_server which retries."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return bool(r.read())
    except Exception:
        return False


def _train_toddler(port: int, reps: int = 8, deadline: float | None = None) -> bool:
    """Returns False if the brain server died mid-training."""
    pairs = [
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
    base = f"http://127.0.0.1:{port}"
    fail_streak = 0
    for p, r in pairs:
        if deadline and time.time() > deadline: return False
        for _ in range(reps):
            ok1 = _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(p.encode())}, timeout=5) is not None
            ok2 = _post(f"{base}/observe", {"pool_id": 4, "frame": _b64u(r.encode())}, timeout=5) is not None
            ok3 = _post(f"{base}/tick", {}, timeout=5) is not None
            if ok1 and ok2 and ok3:
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= 10:
                    # Server is dead; abandon this genome.
                    return False
    return True


def _train_corpus(port: int, corpus: list[dict], reps: int = 3,
                  deadline: float | None = None) -> bool:
    """Returns False if server died or deadline expired."""
    base = f"http://127.0.0.1:{port}"
    fail_streak = 0
    # Periodic health re-check every N requests so a dead server is
    # detected within a bounded number of failed posts.
    for i, row in enumerate(corpus):
        if deadline and time.time() > deadline: return False
        if i > 0 and i % 200 == 0:
            if not _server_alive(port):
                return False
        p = (row.get("prompt") or "").strip()
        r = (row.get("response") or "").strip()
        if not p or not r: continue
        for _ in range(reps):
            ok1 = _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(p.encode())}, timeout=5) is not None
            ok2 = _post(f"{base}/observe", {"pool_id": 4, "frame": _b64u(r.encode())}, timeout=5) is not None
            ok3 = _post(f"{base}/tick", {}, timeout=5) is not None
            if ok1 and ok2 and ok3:
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= 10:
                    return False
    return True


def _eval_panel(port: int) -> dict:
    base = f"http://127.0.0.1:{port}"
    toddler = [
        ("dog","animal"),("cat","animal"),("cow","animal"),("horse","animal"),
        ("bird","animal"),("fish","animal"),
        ("apple","food"),("banana","food"),("bread","food"),("cake","food"),("milk","food"),
        ("car","vehicle"),("truck","vehicle"),("bike","vehicle"),("plane","vehicle"),("boat","vehicle"),
        ("red","color"),("blue","color"),("green","color"),("yellow","color"),
        ("ball","toy"),("doll","toy"),("kite","toy"),("drum","toy"),
        ("tree","nature"),("flower","nature"),("river","nature"),("mountain","nature"),
        ("hand","body"),("foot","body"),("eye","body"),("mouth","body"),
    ]
    oov = ["xyzzy", "foobarbaz", "zzzzqqqq"]
    k12 = [
        ("rose","plant"),("oak","plant"),("hammer","tool"),("piano","musical_instrument"),
        ("triangle","shape"),("football","sport"),("tennis","sport"),
        ("nine","number"),("sad","emotion"),("happy","emotion"),
        ("doctor","occupation"),("school","place"),
    ]
    counts = {"toddler_hit": 0, "toddler_total": len(toddler),
              "oov_hit": 0, "oov_total": len(oov),
              "k12_hit": 0, "k12_total": len(k12),
              "int_contains": 0, "int_total": len(toddler)}
    for p, exp in toddler:
        r = _post(f"{base}/chat", {"text": p})
        if r and r.get("reply") == exp: counts["toddler_hit"] += 1
    for p, exp in k12:
        r = _post(f"{base}/chat", {"text": p})
        if r and r.get("reply") == exp: counts["k12_hit"] += 1
    for p in oov:
        r = _post(f"{base}/chat", {"text": p})
        if r and r.get("reply") == "" and r.get("grounding", {}).get("outside_grounding") is True:
            counts["oov_hit"] += 1
    for p, exp in toddler:
        _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(p.encode())})
        r = _post(f"{base}/integrate", {"query_pool": 1, "target_pool": 4})
        if r:
            ans_b64 = r.get("answer") or ""
            try:
                ans = base64.urlsafe_b64decode(ans_b64 + "==").decode("utf-8", "replace")
            except Exception: ans = ""
            if exp in ans: counts["int_contains"] += 1
    return counts


def _fitness(metrics: dict) -> tuple[float, dict]:
    t = metrics["toddler_hit"] / max(1, metrics["toddler_total"])
    o = metrics["oov_hit"]     / max(1, metrics["oov_total"])
    k = metrics["k12_hit"]     / max(1, metrics["k12_total"])
    i = metrics["int_contains"]/ max(1, metrics["int_total"])
    raw = 1.0*t + 1.0*o + 1.5*k + 1.0*i
    penalty = 0.0
    # Tightened guardrails: we already have toddler 100% and OOV 100%
    # at defaults, so any genome that loses ground gets penalised hard.
    if t < 1.0:    penalty += (1.0  - t) * 10.0
    if o < 1.0:    penalty += (1.0  - o) * 5.0
    if i < 0.85:   penalty += (0.85 - i) * 3.0
    return raw - penalty, {
        "toddler_frac": round(t,3), "oov_frac": round(o,3),
        "k12_frac": round(k,3),     "int_contains_frac": round(i,3),
        "raw": round(raw,3),        "penalty": round(penalty,3),
    }


def worker_evaluate(args: tuple) -> dict:
    """Per-genome budget: 25 minutes wall clock.  Past that we abort
    and return -100 fitness for this genome.  Without this bound, a
    genome that crashes its brain server mid-training would hang the
    GA indefinitely (62K HTTP calls × 5s timeout = days)."""
    genome, worker_id = args
    GENOME_BUDGET_S = 1500.0  # 25 min hard ceiling
    started = time.time()
    deadline = started + GENOME_BUDGET_S
    port = 8095 + worker_id
    data_dir = PROJECT_ROOT / "data" / f"ga_brain_w{worker_id}"
    data_dir.mkdir(parents=True, exist_ok=True)
    brain_bin = data_dir / "brain.bin"
    if brain_bin.exists(): brain_bin.unlink()

    env = os.environ.copy()
    env["W1Z4RD_BRAIN_PORT"] = str(port)
    env["W1Z4RDV1510N_DATA_DIR"] = str(data_dir)
    env.update(genome.env())

    _kill_port(port)
    for _ in range(30):
        if _port_is_free(port): break
        time.sleep(0.5)
    else:
        return {"genome": asdict(genome), "fitness": -100.0, "metrics": {},
                "eval_kind": "failed", "reason": "port stuck"}

    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen([str(BRAIN_BIN)], env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags)
    try:
        if not _wait_for_server(port, timeout=30):
            try: proc.kill()
            except Exception: pass
            _kill_port(port)
            return {"genome": asdict(genome), "fitness": -100.0, "metrics": {},
                    "eval_kind": "failed", "reason": "server didn't come up"}

        if not _train_toddler(port, deadline=deadline):
            return {"genome": asdict(genome), "fitness": -100.0, "metrics": {},
                    "eval_kind": "failed",
                    "reason": "server died during toddler train OR deadline expired"}
        corpus = []
        with CORPUS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try: corpus.append(json.loads(line))
                except Exception: continue
        if not _train_corpus(port, corpus, reps=3, deadline=deadline):
            return {"genome": asdict(genome), "fitness": -100.0, "metrics": {},
                    "eval_kind": "failed",
                    "reason": "server died during corpus train OR deadline expired"}
        if not _server_alive(port):
            return {"genome": asdict(genome), "fitness": -100.0, "metrics": {},
                    "eval_kind": "failed", "reason": "server died before eval"}
        metrics = _eval_panel(port)
        score, breakdown = _fitness(metrics)
        return {"genome": asdict(genome), "fitness": score,
                "metrics": {**metrics, **breakdown},
                "eval_kind": "full"}
    finally:
        try:
            proc.terminate()
            try: proc.wait(timeout=5)
            except Exception: pass
        except Exception: pass
        try: proc.kill()
        except Exception: pass
        if sys.platform == "win32" and proc.pid:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, text=True)
        _kill_port(port)


# ───────── Self-ping ─────────


def _read_best() -> dict | None:
    if not BEST_PATH.exists(): return None
    try: return json.loads(BEST_PATH.read_text())
    except Exception: return None


def _write_best(rec: dict) -> None:
    BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_PATH.write_text(json.dumps(rec, indent=2))


def _significant_improvement(prev_best: dict | None, candidate: dict) -> str | None:
    """Return a short reason string if `candidate` is meaningfully
    better than `prev_best`.  Used to drive Monitor pings.
    ASCII-only (Windows cp1252 console can't encode Unicode arrows)."""
    if prev_best is None: return "first-recorded best"
    cm = candidate.get("metrics", {})
    pm = prev_best.get("metrics", {})
    if cm.get("k12_frac", 0) >= pm.get("k12_frac", 0) + 0.08:
        return f"K-12 lifted {pm.get('k12_frac','?')} -> {cm.get('k12_frac','?')}"
    if cm.get("toddler_frac", 0) >= pm.get("toddler_frac", 0) + 0.05:
        return f"toddler lifted {pm.get('toddler_frac','?')} -> {cm.get('toddler_frac','?')}"
    if candidate.get("fitness", 0) >= prev_best.get("fitness", 0) + 0.5:
        return f"fitness {prev_best.get('fitness'):.2f} -> {candidate.get('fitness'):.2f}"
    return None


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--time-limit", type=int, default=28800,
        help="hard time limit in seconds (default 8h)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    random.seed(args.seed)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initial population:
    #   - 1 default-seed (every knob Constant at safe default)
    #   - 1 best-from-disk seed if available (resumes from prior runs)
    #   - N/3 near-default-seeds (mutated baselines)
    #   - N/3 near-best-seeds (mutated best wirings, for exploration
    #     around the known-good region)
    #   - rest random (broad exploration)
    pop: list[Genome] = [seed_genome(0)]
    prior_best = _read_best()
    best_genome: Genome | None = None
    if prior_best and isinstance(prior_best.get("genome"), dict):
        knobs = prior_best["genome"].get("knobs")
        if isinstance(knobs, dict):
            # Tolerate gene-set growth: backfill missing knobs with
            # their default Constant so a best.json from an earlier
            # gene-space lineage can still seed the new run.
            patched = {}
            for k in KNOB_SPECS:
                if k in knobs:
                    patched[k] = knobs[k]
                else:
                    patched[k] = make_constant(k)
            best_genome = Genome(knobs=patched, born_gen=0)
            pop.append(best_genome)
            print(f"[ga_dyn] seeding pop with best from disk: fit={prior_best.get('fitness','?')}"
                  f" (knobs={list(knobs.keys())}, backfilled={set(KNOB_SPECS)-set(knobs)})",
                  flush=True)
    n_near = max(2, args.pop // 3)
    for _ in range(n_near):
        pop.append(mutate(seed_genome(0), 0))
    if best_genome is not None:
        for _ in range(n_near):
            pop.append(mutate(best_genome, 0))
    while len(pop) < args.pop:
        pop.append(random_genome(0))
    pop = pop[:args.pop]

    fitness_cache: dict[str, dict] = {}
    def gkey(g: Genome) -> str:
        return json.dumps(g.knobs, sort_keys=True)

    print(f"[ga_dyn] start: workers={args.workers} pop={args.pop} "
          f"time_limit={args.time_limit}s log={LOG_PATH}", flush=True)
    start_time = time.time()
    gen = 0
    best_overall = _read_best()
    if best_overall:
        print(f"[ga_dyn] resuming with prior best fitness={best_overall.get('fitness','?')}",
              flush=True)
    while time.time() - start_time < args.time_limit:
        t_gen = time.time()
        print(f"\n=== gen {gen} (elapsed {int(time.time()-start_time)}s) ===", flush=True)

        results: list[dict] = []
        tasks = []
        for idx, g in enumerate(pop):
            k = gkey(g)
            if k in fitness_cache:
                cached = fitness_cache[k]
                rec = {"genome": asdict(g), "fitness": cached["fitness"],
                       "metrics": cached["metrics"], "eval_kind": "full",
                       "gen": gen, "cached": True}
                results.append(rec)
                m = rec["metrics"]
                print(f"  [gen{gen}] fit={rec['fitness']:.3f}  "
                      f"t={m.get('toddler_frac','?')} o={m.get('oov_frac','?')} "
                      f"k={m.get('k12_frac','?')} i={m.get('int_contains_frac','?')} "
                      f"[cached]", flush=True)
            else:
                tasks.append((g, idx % args.workers))

        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker_evaluate, t): t for t in tasks}
            for fut in as_completed(futs):
                try: rec = fut.result()
                except Exception as e:
                    rec = {"genome": asdict(futs[fut][0]), "fitness": -100.0,
                           "metrics": {}, "eval_kind": "failed", "reason": str(e)}
                rec["gen"] = gen
                results.append(rec)
                _append_log(rec)
                # Cache
                g_obj = Genome(**{k:v for k,v in rec["genome"].items()
                                  if k in ("knobs","fitness","metrics","eval_kind","born_gen")})
                fitness_cache[gkey(g_obj)] = {
                    "fitness": rec["fitness"], "metrics": rec.get("metrics", {})}
                m = rec.get("metrics", {})
                print(f"  [gen{gen}] fit={rec['fitness']:.3f}  "
                      f"t={m.get('toddler_frac','?')} o={m.get('oov_frac','?')} "
                      f"k={m.get('k12_frac','?')} i={m.get('int_contains_frac','?')} "
                      f"[{rec.get('eval_kind')}]", flush=True)
                # Self-ping if significantly better
                improvement = _significant_improvement(best_overall, rec)
                if improvement:
                    _write_best(rec)
                    best_overall = rec
                    # Emit a parseable marker line for Monitor to pick up
                    print(f"[BEST_PROGRESS] gen={gen} fit={rec['fitness']:.3f} "
                          f"t={m.get('toddler_frac','?')} o={m.get('oov_frac','?')} "
                          f"k={m.get('k12_frac','?')} i={m.get('int_contains_frac','?')} "
                          f"reason={improvement!r}", flush=True)

        results.sort(key=lambda r: -r["fitness"])
        best = results[0]
        print(f"=== gen {gen} done in {int(time.time()-t_gen)}s; best={best['fitness']:.3f}",
              flush=True)

        # Selection: elite + offspring
        n_elite = max(2, int(args.pop * ELITE_FRAC))
        elite_genomes: list[Genome] = []
        for r in results[:n_elite]:
            elite_genomes.append(Genome(**{k:v for k,v in r["genome"].items()
                if k in ("knobs","fitness","metrics","eval_kind","born_gen")}))
        new_pop = list(elite_genomes)
        while len(new_pop) < args.pop:
            if random.random() < 0.6 and len(elite_genomes) >= 2:
                a, b = random.sample(elite_genomes, 2)
                child = crossover(a, b, gen+1)
            else:
                child = mutate(random.choice(elite_genomes), gen+1)
            child = mutate(child, gen+1)
            new_pop.append(child)
        pop = new_pop
        gen += 1

    print(f"\n[ga_dyn] done after {int(time.time()-start_time)}s, "
          f"{gen} generations.  best logged to {BEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
