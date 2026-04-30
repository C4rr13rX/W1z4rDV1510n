#!/usr/bin/env python3
"""
ga_scoping_search.py — long-running phased GA that drills into accuracy.

Strategy ("organism growth"):
  Phase 0   — start at the seed genome (Recipe B from the prior run).
              Run a generation of mutated descendants around it with a
              moderate radius.
  Phase i+1 — take the best from phase i.  Shrink the mutation radius by
              `shrink_factor`.  Reserve `explore_frac` of the population
              for broad-radius exploration so we don't get stuck in the
              local basin.  Continue.

Cross-wiring feedback loops controlled by genome bools (handled in the
fitness harness):
  fb_eem_into_mp        — EEM matches on input → train (input, eqn keywords)
                          into multi-pool.  Closes EEM ↔ multi-pool loop.
  fb_mp_into_slowpool   — Trained (description, code) pair also pumped
                          through /media/train so the slow pool's char-chain
                          decoder learns the same vocabulary.  Closes
                          multi-pool ↔ slow-pool loop.
  fb_replay_low_conf    — When avg eval confidence drops below 0.20,
                          trigger /multi_pool/replay.  Closes the
                          confidence-monitoring → CLS-replay loop.
  fb_disagreement_ne    — When ≥25% of queries fall through to char_chain
                          or eem, release a NE spike so the *next* training
                          run gets elevated LR.  Closes the
                          route-disagreement → neuromodulator loop.

Logging:  every individual is appended to data/foundation/ga_scoping.jsonl
with full genome, metrics, and phase metadata.  Resumable: existing log
is preserved, new lines are appended.

Runtime:  default budget 4 hours.  Each phase is bounded by
`phase_budget_secs` (default 600s = 10 min).  Phases auto-roll-forward.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys
import time

from ga_node_fitness import run_one_genome, load_jsonl, NCBI_JSONL, CLASS_JSONL


# ─── Search space (knobs + cross-wiring toggles) ──────────────────────

PARAM_SPACE = {
    # Multi-pool params
    "lr":                       ("float", 0.05, 1.5),
    "passes":                   ("int",   3,    50),
    "hops":                     ("int",   1,    6),
    "min_strength":             ("float", 0.005, 0.3),

    # Connection toggles
    "cls_replay_after_epoch":   ("bool",  0,    1),
    "cls_lr_scale":             ("float", 0.02, 0.30),
    "eem_in_train_loop":        ("bool",  0,    1),
    "train_equation_pool":      ("bool",  0,    1),

    # Bayesian-routing knobs
    "mp_confidence_threshold":  ("float", 0.05, 0.80),
    "use_eem_fallback":         ("bool",  0,    1),

    # Atom-encoding architecture toggles
    "use_bigrams":              ("bool",  0,    1),
    "use_trigrams":             ("bool",  0,    1),
    "use_idf":                  ("bool",  0,    1),

    # Cross-wiring feedback loops (Phase 5)
    "fb_eem_into_mp":           ("bool",  0,    1),
    "fb_mp_into_slowpool":      ("bool",  0,    1),
    "fb_replay_low_conf":       ("bool",  0,    1),
    "fb_disagreement_ne":       ("bool",  0,    1),
}

# Recipe B from the prior wider GA — best combined 0.837.
RECIPE_B_SEED = {
    "lr": 0.167, "passes": 7, "hops": 1, "min_strength": 0.155,
    "cls_replay_after_epoch": 0, "cls_lr_scale": 0.026,
    "eem_in_train_loop": 0, "train_equation_pool": 1,
    "mp_confidence_threshold": 0.074, "use_eem_fallback": 1,
    "use_bigrams": 1, "use_trigrams": 1, "use_idf": 1,
    "fb_eem_into_mp":      0,
    "fb_mp_into_slowpool": 0,
    "fb_replay_low_conf":  0,
    "fb_disagreement_ne":  0,
}


# ─── Genome operators ────────────────────────────────────────────────

def random_genome(rng: random.Random) -> dict:
    g = {}
    for k, spec in PARAM_SPACE.items():
        kind = spec[0]
        if kind == "int":
            g[k] = rng.randint(spec[1], spec[2])
        elif kind == "bool":
            g[k] = rng.randint(0, 1)
        else:
            g[k] = rng.uniform(spec[1], spec[2])
    return g


def mutate(g: dict, radius: float, rng: random.Random) -> dict:
    """`radius` ∈ [0, 1].  Per-knob change probability AND magnitude scale.
    radius=0.05 → small jitter.  radius=0.50 → broad mutation."""
    out = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() > radius: continue
        kind = spec[0]
        if kind == "int":
            span = max(1, int((spec[2] - spec[1]) * radius))
            out[k] = max(spec[1], min(spec[2], out[k] + rng.randint(-span, span)))
        elif kind == "bool":
            # Bool mutation: flip with probability radius * 0.5 to avoid
            # constantly flipping — bools are often load-bearing once set.
            if rng.random() < radius * 0.5:
                out[k] = 1 - out[k]
        else:
            jitter = (spec[2] - spec[1]) * radius * rng.uniform(-1.0, 1.0)
            out[k] = max(spec[1], min(spec[2], float(out[k]) + jitter))
    return out


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def fmt_g(g: dict) -> str:
    parts = []
    for k, v in g.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ─── Scoping orchestrator ────────────────────────────────────────────

def evaluate(genome: dict, ncbi_records, class_records,
             ncbi_max: int, class_max: int) -> dict:
    try:
        return run_one_genome(
            genome, ncbi_records, class_records,
            ncbi_max=ncbi_max, class_max=class_max, log=False,
        )
    except Exception as e:
        return {"combined": 0.0, "error": str(e)[:120]}


def run_scoping(
    seed: dict,
    ncbi_records, class_records,
    *,
    pop_size: int      = 6,
    phase_budget_secs: int = 600,
    max_phases: int    = 100,
    max_runtime_secs: int = 4 * 3600,
    radius_start: float = 0.30,
    radius_min: float   = 0.05,
    shrink_factor: float = 0.85,
    explore_frac: float = 0.30,
    ncbi_max: int = 25,
    class_max: int = 15,
    seed_rng: int = 42,
    log_path: pathlib.Path = pathlib.Path("data/foundation/ga_scoping.jsonl"),
):
    """Phased GA with shrinking radius around the running best.

    Each phase the population is constructed as:
       - 1 elite (the running best)
       - (pop_size - 1) * (1 - explore_frac) descendants of the running
         best with mutation radius = current radius
       - the rest with broad mutation radius (radius_start) so we keep
         exploring the wider space.

    Runs until `max_runtime_secs` or `max_phases`.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    rng = random.Random(seed_rng)

    overall_start = time.time()
    best_genome = dict(seed)
    best_metrics = evaluate(best_genome, ncbi_records, class_records,
                            ncbi_max, class_max)
    best_score = best_metrics.get("combined", 0.0)
    print(f"[scoping] seed evaluated: combined={best_score:.4f}  "
          f"genome={fmt_g(best_genome)}", flush=True)
    log.write(json.dumps({
        "phase": -1, "ind": 0,
        "genome": best_genome, "metrics": best_metrics,
        "kind": "seed", "wall_secs_total": 0.0,
    }) + "\n")
    log.flush()

    radius = radius_start

    for phase in range(max_phases):
        if time.time() - overall_start > max_runtime_secs:
            print(f"[scoping] runtime budget exhausted; stopping at phase {phase}",
                  flush=True)
            break
        phase_start = time.time()

        # Build population around best with mixed exploit/explore.
        pop: list[tuple[dict, str]] = [(dict(best_genome), "elite")]
        n_explore = max(1, int(round(pop_size * explore_frac)))
        n_exploit = pop_size - 1 - n_explore + (n_explore - n_explore)
        # Actually compute clean: 1 elite, n_explore broad, rest exploit
        n_exploit = pop_size - 1 - n_explore
        for _ in range(n_exploit):
            pop.append((mutate(best_genome, radius, rng), "exploit"))
        for _ in range(n_explore):
            pop.append((mutate(best_genome, radius_start * 1.5, rng), "explore"))
        if rng.random() < 0.10:
            # 10% chance to inject a totally random genome
            pop.append((random_genome(rng), "random"))

        # Evaluate the population.
        scored: list[tuple[float, dict, dict, str]] = []
        for ind, (g, kind) in enumerate(pop):
            if time.time() - overall_start > max_runtime_secs: break
            t0 = time.time()
            m = evaluate(g, ncbi_records, class_records, ncbi_max, class_max)
            dt = time.time() - t0
            score = m.get("combined", 0.0)
            scored.append((score, m, g, kind))
            log.write(json.dumps({
                "phase": phase, "ind": ind, "kind": kind, "radius": radius,
                "genome": g, "metrics": m,
                "wall_secs": round(dt, 1),
                "wall_secs_total": round(time.time() - overall_start, 1),
            }) + "\n")
            log.flush()
            print(f"[scoping] ph{phase:02d} #{ind} ({kind:7s}) "
                  f"combined={score:.4f}  cpe={m.get('class_par_exact','?')}/{m.get('class_par_n','?')}  "
                  f"({dt:.0f}s)", flush=True)
            if time.time() - phase_start > phase_budget_secs:
                print(f"[scoping] phase {phase} budget hit, breaking inner loop",
                      flush=True)
                break

        if not scored: continue

        scored.sort(key=lambda x: -x[0])
        top_score, top_m, top_g, top_kind = scored[0]
        improved = top_score > best_score + 1e-4
        if improved:
            best_score = top_score
            best_genome = top_g
            best_metrics = top_m
            radius = max(radius_min, radius * shrink_factor)
            print(f"[scoping] === ph{phase} NEW BEST {top_score:.4f} "
                  f"(kind={top_kind}) — shrinking radius to {radius:.3f}",
                  flush=True)
        else:
            # No improvement: bump radius a bit to escape local basin.
            radius = min(radius_start, radius * 1.10)
            print(f"[scoping] === ph{phase} no improvement (best stays "
                  f"{best_score:.4f}); radius -> {radius:.3f}", flush=True)

        log.write(json.dumps({
            "phase": phase, "phase_summary": True,
            "best_score": best_score,
            "best_genome": best_genome,
            "best_metrics": best_metrics,
            "radius_after": radius,
            "wall_secs_total": round(time.time() - overall_start, 1),
        }) + "\n")
        log.flush()

    log.close()
    return best_genome, best_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--phase-budget-secs", type=int, default=600)
    ap.add_argument("--max-phases", type=int, default=200)
    ap.add_argument("--max-runtime-secs", type=int, default=4 * 3600)
    ap.add_argument("--ncbi-max", type=int, default=25)
    ap.add_argument("--class-max", type=int, default=15)
    ap.add_argument("--seed-rng", type=int, default=42)
    ap.add_argument("--log", type=str,
                    default="data/foundation/ga_scoping.jsonl")
    args = ap.parse_args()

    ncbi  = load_jsonl(NCBI_JSONL)
    cls_  = load_jsonl(CLASS_JSONL)
    print(f"[scoping] corpora: ncbi={len(ncbi)} classes={len(cls_)}", flush=True)
    print(f"[scoping] seed=Recipe B (combined~0.837 prior best)", flush=True)
    print(f"[scoping] runtime budget: {args.max_runtime_secs/3600:.1f} h", flush=True)

    best_g, best_m = run_scoping(
        RECIPE_B_SEED, ncbi, cls_,
        pop_size=args.pop,
        phase_budget_secs=args.phase_budget_secs,
        max_phases=args.max_phases,
        max_runtime_secs=args.max_runtime_secs,
        ncbi_max=args.ncbi_max,
        class_max=args.class_max,
        seed_rng=args.seed_rng,
        log_path=pathlib.Path(args.log),
    )
    print("\n" + "=" * 70, flush=True)
    print(f"[scoping] FINAL combined={best_m.get('combined', 0):.4f}", flush=True)
    print(f"          metrics: {best_m}", flush=True)
    print(f"          genome:  {fmt_g(best_g)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
