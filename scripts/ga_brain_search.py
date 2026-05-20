#!/usr/bin/env python3
"""
ga_brain_search.py — genetic-algorithm search over brain-substrate
genes that the brain server reads at startup via env vars.

Genes vary biologically-motivated knobs on the dynamical system
(sparsity, decay, emergence thresholds, decode floor).  No genome
encodes hardcoded outputs; every gene is a parameter the substrate's
emergence dynamics consume.

Architecture:
  - N parallel workers (default 4), each running its own brain server
    instance on a separate port + data dir.
  - Each worker pulls a genome from the queue, applies env-var
    overrides, runs train + eval, reports fitness back.
  - Main process maintains a population, generates offspring via
    crossover + mutation from top scorers, logs to JSONL.

Fitness = weighted sum across the fluency panel, with hard guardrails:
  - toddler EXACT (weight 1.0, guardrail >= 0.85)
  - OOV honesty (weight 1.0, guardrail = 3/3)
  - K-12 EXACT (weight 1.5) — the lift target
  - multi_fact (weight 1.0)
  - /integrate toddler "contains" (weight 1.0)

Guardrail violations subtract a heavy penalty so the GA never
optimises K-12 by sacrificing toddler or OOV honesty.

Two-tier evaluation:
  - SMOKE: 200-pair K-12-focused subset, ~1 min/genome.  Used to
    prune the population — only the top fraction proceeds to FULL.
  - FULL: full categorical_unified_001.jsonl (6,972 pairs), ~9 min.
    Final fitness comes from the FULL run.

Log: data/foundation/ga_brain_search.jsonl (one JSON line per genome).
Resumable: existing log is preserved; main process re-uses scored
genomes on restart instead of re-evaluating.

Run:
  python scripts/ga_brain_search.py --workers 4 --generations 20

Stop:
  Ctrl-C; the harness checkpoints current best to a meta JSON.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pathlib
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
BRAIN_BIN    = PROJECT_ROOT / "bin" / "w1z4rd_brain_server.exe"
LOG_PATH     = PROJECT_ROOT / "data" / "foundation" / "ga_brain_search.jsonl"
META_PATH    = PROJECT_ROOT / "data" / "foundation" / "ga_brain_search_meta.json"
CORPUS_PATH  = PROJECT_ROOT / "data" / "training" / "categorical_unified_001.jsonl"

# Gene definitions — every gene is a knob on the dynamical system.
# Format: (env_var_name, min, max, kind)  kind in {"float", "int", "log_float", "log_int"}
GENES = [
    # Sparsity per pool — primary lever against runaway concept-of-concept emergence.
    ("BRAIN_SPARSITY_TEXT",       0.05,   1.0,    "float"),
    ("BRAIN_SPARSITY_ACTION",     0.05,   1.0,    "float"),
    ("BRAIN_SPARSITY_DEFAULT",    0.05,   1.0,    "float"),
    # Decode-time floor — OOV honesty / partial-match threshold.
    ("BRAIN_MIN_ATOM_SCORE",      0.10,   0.90,   "float"),
    # Concept emergence threshold — how many co-fires before a concept crystallises.
    # Higher = more conservative, fewer concepts, slower runaway.
    ("BRAIN_EMERGENCE_TEXT",      3,      12,     "int"),
    ("BRAIN_EMERGENCE_ACTION",    3,      12,     "int"),
    # Recent-atoms window — fingerprint memory depth.
    ("BRAIN_WINDOW_TEXT",         4096,   131072, "log_int"),
    ("BRAIN_WINDOW_ACTION",       4096,   131072, "log_int"),
    # Decay / prune — synaptic homeostasis (SHY hypothesis).
    ("BRAIN_DECAY_DEFAULT",       1e-6,   1e-3,   "log_float"),
    ("BRAIN_PRUNE_FLOOR_DEFAULT", 1e-4,   1e-1,   "log_float"),
]

POP_SIZE     = 16
ELITE_FRAC   = 0.25
MUTATION_RATE = 0.30
MUTATION_SIGMA = 0.15      # fraction of gene range
CROSSOVER_RATE = 0.70

# Smoke eval subset: ~200 pairs touching the K-12 lift target.
SMOKE_PROMPT_SUBSET = [
    # toddler (must hold)
    "dog", "cat", "horse", "fish", "apple", "banana", "bread", "car", "bike",
    "red", "blue", "ball", "doll", "tree", "river", "hand", "foot", "eye",
    # K-12 (lift target)
    "rose", "oak", "hammer", "piano", "triangle",
    # OOV
    "xyzzy", "foobarbaz", "zzzzqqqq",
]

# ──────────────────────────────────────────────────────────────────


@dataclass
class Genome:
    values: dict       # env_var -> value
    fitness: float = 0.0
    metrics: dict = field(default_factory=dict)
    eval_kind: str = "unscored"     # "smoke" | "full" | "failed"
    born_gen: int = 0

    def env(self) -> dict:
        out = {}
        for k, v in self.values.items():
            out[k] = str(v) if not isinstance(v, float) else f"{v:.6g}"
        return out


def random_genome(gen: int = 0) -> Genome:
    vals = {}
    for name, lo, hi, kind in GENES:
        if kind == "float":
            vals[name] = round(random.uniform(lo, hi), 4)
        elif kind == "int":
            vals[name] = random.randint(int(lo), int(hi))
        elif kind == "log_float":
            vals[name] = round(10 ** random.uniform(
                __import__("math").log10(lo),
                __import__("math").log10(hi)), 8)
        elif kind == "log_int":
            import math
            vals[name] = int(round(10 ** random.uniform(
                math.log10(lo), math.log10(hi))))
    return Genome(values=vals, born_gen=gen)


def mutate(g: Genome, gen: int) -> Genome:
    import math
    vals = dict(g.values)
    for name, lo, hi, kind in GENES:
        if random.random() > MUTATION_RATE:
            continue
        cur = vals[name]
        if kind in ("float", "log_float"):
            span = (math.log10(hi) - math.log10(lo)) if kind == "log_float" else (hi - lo)
            sigma = span * MUTATION_SIGMA
            if kind == "log_float":
                lcur = math.log10(max(cur, lo))
                new = 10 ** max(math.log10(lo), min(math.log10(hi), random.gauss(lcur, sigma)))
            else:
                new = max(lo, min(hi, random.gauss(cur, sigma)))
            vals[name] = round(new, 4 if kind == "float" else 8)
        else:
            span = (math.log10(hi) - math.log10(lo)) if kind == "log_int" else (hi - lo)
            sigma = max(1, span * MUTATION_SIGMA)
            if kind == "log_int":
                lcur = math.log10(max(cur, lo))
                new = int(round(10 ** max(math.log10(lo), min(math.log10(hi), random.gauss(lcur, sigma)))))
            else:
                new = int(round(max(lo, min(hi, random.gauss(cur, sigma)))))
            vals[name] = new
    return Genome(values=vals, born_gen=gen)


def crossover(a: Genome, b: Genome, gen: int) -> Genome:
    vals = {}
    for name, *_ in GENES:
        vals[name] = a.values[name] if random.random() < 0.5 else b.values[name]
    return Genome(values=vals, born_gen=gen)


# ──────────────────────────────────────────────────────────────────
# Worker — runs one genome end-to-end.
# ──────────────────────────────────────────────────────────────────


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
                if r.read():
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def _kill_port(port: int) -> None:
    """Best-effort kill of a process listening on `port` on Windows."""
    try:
        out = subprocess.check_output(["netstat", "-ano", "-p", "TCP"], text=True, errors="ignore")
        for line in out.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                pid = line.split()[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid],
                    capture_output=True, text=True)
    except Exception:
        pass


def _train_toddler(port: int) -> int:
    """Dense-burst toddler 32 pairs × 4 reps.  Returns ticks."""
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
    total = 0
    for prompt, response in pairs:
        for _ in range(4):
            _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(prompt.encode())})
            _post(f"{base}/observe", {"pool_id": 4, "frame": _b64u(response.encode())})
            _post(f"{base}/tick", {})
            total += 1
    return total


def _train_corpus(port: int, corpus: list[dict], reps: int = 3) -> int:
    """Dense-burst training: each (prompt, response) pair observed `reps`
    times back-to-back as a co-fired (POOL_TEXT, POOL_ACTION) moment."""
    base = f"http://127.0.0.1:{port}"
    total = 0
    for row in corpus:
        prompt = (row.get("prompt") or "").strip()
        resp   = (row.get("response") or "").strip()
        if not prompt or not resp:
            continue
        for _ in range(reps):
            _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(prompt.encode())}, timeout=5)
            _post(f"{base}/observe", {"pool_id": 4, "frame": _b64u(resp.encode())},   timeout=5)
            _post(f"{base}/tick", {}, timeout=5)
            total += 1
    return total


def _eval_panel(port: int, full: bool = False) -> dict:
    """Returns dict of per-category EXACT hit counts.  Uses /chat for
    user-facing path and /integrate for substrate floor."""
    base = f"http://127.0.0.1:{port}"

    # Categories — toddler (32), OOV (3), and a K-12-focused subset.
    toddler = [
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
    oov = ["xyzzy", "foobarbaz", "zzzzqqqq"]
    k12 = [
        ("rose","plant"),("oak","plant"),("hammer","tool"),("piano","musical_instrument"),
        ("triangle","shape"),("football","sport"),("tennis","sport"),
        ("nine","number"),("sad","emotion"),("happy","emotion"),
        ("doctor","occupation"),("school","place"),
    ]

    counts = {"toddler_hit": 0, "toddler_total": len(toddler),
              "oov_hit": 0,     "oov_total": len(oov),
              "k12_hit": 0,     "k12_total": len(k12),
              "int_contains": 0, "int_total": len(toddler)}

    # /chat eval — toddler + K-12 EXACT
    for prompt, expected in toddler:
        r = _post(f"{base}/chat", {"text": prompt})
        if r and r.get("reply") == expected:
            counts["toddler_hit"] += 1
    for prompt, expected in k12:
        r = _post(f"{base}/chat", {"text": prompt})
        if r and r.get("reply") == expected:
            counts["k12_hit"] += 1
    # OOV honesty
    for prompt in oov:
        r = _post(f"{base}/chat", {"text": prompt})
        if r and r.get("reply") == "" and r.get("grounding", {}).get("outside_grounding") is True:
            counts["oov_hit"] += 1

    # /integrate floor (substrate health)
    for prompt, expected in toddler:
        _post(f"{base}/observe", {"pool_id": 1, "frame": _b64u(prompt.encode())})
        r = _post(f"{base}/integrate", {"query_pool": 1, "target_pool": 4})
        if r:
            ans_b64 = r.get("answer") or ""
            try:
                ans = base64.urlsafe_b64decode(ans_b64 + "==").decode("utf-8", "replace")
            except Exception:
                ans = ""
            if expected in ans:
                counts["int_contains"] += 1

    return counts


def _fitness(metrics: dict) -> tuple[float, dict]:
    """Weighted sum + hard guardrails.  Returns (score, breakdown)."""
    t = metrics["toddler_hit"] / max(1, metrics["toddler_total"])
    o = metrics["oov_hit"]     / max(1, metrics["oov_total"])
    k = metrics["k12_hit"]     / max(1, metrics["k12_total"])
    i = metrics["int_contains"]/ max(1, metrics["int_total"])
    raw = 1.0*t + 1.0*o + 1.5*k + 1.0*i
    # Guardrails — heavy penalty when toddler or OOV regress below floor.
    penalty = 0.0
    if t < 0.85:  penalty += (0.85 - t) * 5.0
    if o < 1.0:   penalty += (1.0 - o)  * 3.0
    if i < 0.85:  penalty += (0.85 - i) * 3.0
    return raw - penalty, {
        "toddler_frac": round(t,3), "oov_frac": round(o,3),
        "k12_frac": round(k,3),     "int_contains_frac": round(i,3),
        "raw": round(raw,3),        "penalty": round(penalty,3),
    }


def worker_evaluate(args: tuple) -> dict:
    """Spawn a brain server with the genome's env vars, train, eval,
    return metrics.  Runs in a subprocess so multiple genomes can be
    evaluated in parallel."""
    genome, worker_id, eval_kind = args
    port = 8095 + worker_id
    data_dir = PROJECT_ROOT / "data" / f"ga_brain_w{worker_id}"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Always start fresh — delete prior brain.bin.
    brain_bin = data_dir / "brain.bin"
    if brain_bin.exists(): brain_bin.unlink()

    env = os.environ.copy()
    env["W1Z4RD_BRAIN_PORT"] = str(port)
    env["W1Z4RDV1510N_DATA_DIR"] = str(data_dir)
    env.update(genome.env())

    _kill_port(port)
    time.sleep(0.5)
    proc = subprocess.Popen(
        [str(BRAIN_BIN)],
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_server(port, timeout=30):
            proc.kill()
            return {"genome": asdict(genome), "fitness": -100.0,
                    "metrics": {}, "eval_kind": "failed",
                    "reason": "server didn't come up"}

        # Train toddler always.
        _train_toddler(port)

        # If full eval, also train categorical_unified.
        if eval_kind == "full":
            corpus = []
            with CORPUS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        corpus.append(json.loads(line))
                    except Exception: continue
            _train_corpus(port, corpus, reps=3)

        metrics = _eval_panel(port, full=(eval_kind == "full"))
        score, breakdown = _fitness(metrics)
        return {"genome": asdict(genome), "fitness": score,
                "metrics": {**metrics, **breakdown},
                "eval_kind": eval_kind}
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            try: proc.kill()
            except Exception: pass
        _kill_port(port)


# ──────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────


def _load_existing() -> list[dict]:
    if not LOG_PATH.exists(): return []
    out = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except Exception: continue
    return out


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop", type=int, default=POP_SIZE)
    p.add_argument("--eval", choices=["smoke","full","both"], default="both",
        help="smoke = toddler-only training (fast); full = also train categorical; both = smoke for early gens then full for elites")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-after-gen", type=int, default=3,
        help="switch from smoke to full evaluation after this many generations")
    args = p.parse_args(argv)

    random.seed(args.seed)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Initial population.
    pop: list[Genome] = [random_genome(0) for _ in range(args.pop)]
    print(f"[ga_brain] starting: workers={args.workers} pop={args.pop} "
          f"generations={args.generations} log={LOG_PATH}", flush=True)

    for gen in range(args.generations):
        eval_kind = "full" if (args.eval == "full" or (args.eval == "both" and gen >= args.full_after_gen)) else "smoke"
        print(f"\n=== gen {gen}  (eval={eval_kind}) ===", flush=True)
        t_gen = time.time()

        # Distribute genomes across workers.
        tasks = [(g, idx % args.workers, eval_kind) for idx, g in enumerate(pop)]
        results: list[dict] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(worker_evaluate, t): t for t in tasks}
            for fut in as_completed(futs):
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"genome": asdict(futs[fut][0]), "fitness": -100.0,
                           "metrics": {}, "eval_kind": "failed", "reason": str(e)}
                rec["gen"] = gen
                results.append(rec)
                _append_log(rec)
                m = rec.get("metrics", {})
                print(f"  [gen{gen}] fitness={rec['fitness']:.3f}  "
                      f"t={m.get('toddler_frac','?')} o={m.get('oov_frac','?')} "
                      f"k={m.get('k12_frac','?')} i={m.get('int_contains_frac','?')} "
                      f"[{rec.get('eval_kind')}]", flush=True)

        # Sort by fitness desc.
        results.sort(key=lambda r: -r["fitness"])
        best = results[0]
        print(f"=== gen {gen} done in {time.time()-t_gen:.0f}s; best={best['fitness']:.3f} "
              f"k12={best.get('metrics',{}).get('k12_frac','?')}", flush=True)

        # Selection: keep elite, breed offspring.
        n_elite = max(2, int(args.pop * ELITE_FRAC))
        elites = [Genome(**{k:v for k,v in r["genome"].items() if k in ("values","fitness","metrics","eval_kind","born_gen")}) for r in results[:n_elite]]
        new_pop: list[Genome] = list(elites)
        while len(new_pop) < args.pop:
            if random.random() < CROSSOVER_RATE:
                a, b = random.sample(elites, 2) if len(elites) >= 2 else (elites[0], elites[0])
                child = crossover(a, b, gen+1)
            else:
                parent = random.choice(elites)
                child = mutate(parent, gen+1)
            # Always mutate at least a bit.
            child = mutate(child, gen+1)
            new_pop.append(child)
        pop = new_pop

        # Checkpoint best.
        META_PATH.write_text(json.dumps({
            "gen": gen,
            "best_fitness": best["fitness"],
            "best_genome": best["genome"],
            "best_metrics": best.get("metrics", {}),
        }, indent=2))

    print(f"\n[ga_brain] done; best logged to {META_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
