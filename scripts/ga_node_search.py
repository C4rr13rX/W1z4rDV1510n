#!/usr/bin/env python3
"""
ga_node_search.py — adaptive GA over the entire-node configuration knobs
plus the connection-graph toggles between architectural pieces.

Search space (the "individual pieces and how they connect"):

  Multi-pool (the first layer):
    lr            float  0.05..1.5    cross-pool learning rate per pass
    passes        int    5..50        passes per pair
    hops          int    1..6         query hops within target pool
    min_strength  float  0.005..0.2   activation cutoff in queries

  Connection toggles:
    cls_replay_after_epoch  0|1   run /multi_pool/replay between epochs
    cls_lr_scale            float 0.02..0.30   slow-pool replay LR
    eem_in_train_loop       0|1   call /equations/apply on every train Q
    train_equation_pool     0|1   train (description -> class_id) into 'equation' pool

Fitness = combined Lev-similarity score across NCBI Q-A pairs and the
science/engineering class corpus.  See ga_node_fitness.run_one_genome for
the per-piece breakdown.

Adaptive population: small pop (default 4) × few gens (default 5) so the
whole search completes in ~tens of minutes, not hours.  Mutation strength
adapts to fitness (low when close to optimum, climbs after stagnation).
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import time

# Re-use the harness module
from ga_node_fitness import run_one_genome, load_jsonl, NCBI_JSONL, CLASS_JSONL


# ─── Search space ───────────────────────────────────────────────────────

PARAM_SPACE = {
    "lr":                       ("float", 0.05, 1.5),
    "passes":                   ("int",   5,    50),
    "hops":                     ("int",   1,    6),
    "min_strength":             ("float", 0.005, 0.2),
    "cls_replay_after_epoch":   ("bool",  0,    1),
    "cls_lr_scale":             ("float", 0.02, 0.30),
    "eem_in_train_loop":        ("bool",  0,    1),
    "train_equation_pool":      ("bool",  0,    1),
    # ── Bayesian-sensor routing knobs ────────────────────────────
    "mp_confidence_threshold":  ("float", 0.05, 0.80),
    "use_eem_fallback":         ("bool",  0,    1),
    # ── Atom-encoding architecture toggle ───────────────────────
    "use_bigrams":              ("bool",  0,    1),
}

DEFAULT_GENOME = {
    "lr": 0.5, "passes": 30, "hops": 3, "min_strength": 0.05,
    "cls_replay_after_epoch": 1, "cls_lr_scale": 0.1,
    "eem_in_train_loop": 0, "train_equation_pool": 1,
    "mp_confidence_threshold": 0.30, "use_eem_fallback": 1,
    "use_bigrams": 0,
}


# ─── Genome operators ──────────────────────────────────────────────────

def random_genome(rng: random.Random) -> dict:
    g = {}
    for k, spec in PARAM_SPACE.items():
        kind = spec[0]
        if kind == "int":
            g[k] = rng.randint(spec[1], spec[2])
        elif kind == "bool":
            g[k] = rng.randint(0, 1)
        else:  # float
            g[k] = rng.uniform(spec[1], spec[2])
    return g


def mutate(g: dict, ms: float, rng: random.Random) -> dict:
    out = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() > ms: continue
        kind = spec[0]
        if kind == "int":
            span = max(1, int((spec[2] - spec[1]) * 0.5))
            out[k] = max(spec[1], min(spec[2], out[k] + rng.randint(-span, span)))
        elif kind == "bool":
            out[k] = 1 - out[k]
        else:
            jitter = (spec[2] - spec[1]) * rng.uniform(-0.4, 0.4)
            out[k] = max(spec[1], min(spec[2], float(out[k]) + jitter))
    return out


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def fmt(g: dict) -> str:
    parts = []
    for k, v in g.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ─── GA loop ───────────────────────────────────────────────────────────

def run_ga(
    pop_size: int,
    max_gens: int,
    target: float,
    seed: int,
    ncbi_max: int,
    class_max: int,
    log_path: pathlib.Path,
):
    rng = random.Random(seed)
    ncbi_records = load_jsonl(NCBI_JSONL)
    class_records = load_jsonl(CLASS_JSONL)
    print(f"[GA] corpora: ncbi={len(ncbi_records)} class={len(class_records)}")
    print(f"[GA] eval subsets: ncbi_max={ncbi_max} class_max={class_max}")
    print(f"[GA] pop={pop_size} max_gens={max_gens} target={target}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")

    pop = [dict(DEFAULT_GENOME)]
    while len(pop) < pop_size:
        pop.append(mutate(DEFAULT_GENOME, 0.5, rng) if rng.random() < 0.5
                   else random_genome(rng))

    best_ever = None
    best_score = -1.0
    no_improve = 0

    for gen in range(max_gens):
        scored = []
        gen_t0 = time.time()
        for i, g in enumerate(pop):
            print(f"\n[GA] gen {gen} ind {i}/{len(pop)}: {fmt(g)}")
            t0 = time.time()
            try:
                m = run_one_genome(
                    g, ncbi_records, class_records,
                    ncbi_max=ncbi_max, class_max=class_max, log=False,
                )
            except Exception as e:
                print(f"[GA]   ERROR: {e}")
                m = {"combined": 0.0,
                     "ncbi_mem_mean": 0.0, "ncbi_mem_exact": 0, "ncbi_mem_n": 0,
                     "ncbi_gen_mean": 0.0, "ncbi_gen_exact": 0, "ncbi_gen_n": 0,
                     "class_mem_mean": 0.0, "class_mem_exact": 0, "class_mem_n": 0,
                     "class_par_mean": 0.0, "class_par_exact": 0, "class_par_n": 0,
                     "equation_hit_rate": 0.0, "train_time_s": 0.0}
            dt = time.time() - t0
            rc = m.get("route_counts") or {}
            rc_str = f"mp={rc.get('multi_pool',0)} cc={rc.get('char_chain',0)} eem={rc.get('eem',0)}"
            print(f"[GA]   combined={m['combined']:.3f}  "
                  f"ncbi mem={m['ncbi_mem_mean']:.2f}({m['ncbi_mem_exact']}/{m['ncbi_mem_n']}) "
                  f"gen={m['ncbi_gen_mean']:.2f}({m['ncbi_gen_exact']}/{m['ncbi_gen_n']})  "
                  f"class mem={m['class_mem_mean']:.2f}({m['class_mem_exact']}/{m['class_mem_n']}) "
                  f"par={m['class_par_mean']:.2f}({m['class_par_exact']}/{m['class_par_n']})  "
                  f"eq={m['equation_hit_rate']:.2f} routes[{rc_str}]  ({dt:.0f}s)")
            log.write(json.dumps({
                "gen": gen, "ind": i, "genome": g, "metrics": m,
                "wall_secs": round(dt, 1),
            }) + "\n")
            log.flush()
            scored.append((m["combined"], m, g))

        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        improved = best[0] > best_score + 1e-4
        if improved:
            best_score = best[0]
            best_ever = best
            no_improve = 0
        else:
            no_improve += 1

        gen_dt = time.time() - gen_t0
        ms = max(0.05, 0.5 * (1 - best_score))
        if no_improve >= 2: ms = min(0.7, ms + 0.2)
        avg = sum(s for s, _, _ in scored) / len(scored)
        m = best[1]
        print(f"\n[GA] === gen {gen} summary === best={best[0]:.3f} "
              f"avg={avg:.3f} mut={ms:.2f}  "
              f"({gen_dt:.0f}s for {len(pop)} indivs)")
        print(f"[GA]   best genome: {fmt(best[2])}")
        print(f"[GA]   ncbi mem {m['ncbi_mem_exact']}/{m['ncbi_mem_n']} "
              f"gen {m['ncbi_gen_exact']}/{m['ncbi_gen_n']}  "
              f"class mem {m['class_mem_exact']}/{m['class_mem_n']} "
              f"par {m['class_par_exact']}/{m['class_par_n']} "
              f"eq {m['equation_hit_rate']:.3f}")

        if best[0] >= target:
            print(f"[GA] *** target {target} reached at gen {gen} ***")
            break

        elites = [g for _, _, g in scored[:max(2, pop_size // 2)]]
        children = list(elites)
        if no_improve >= 2:
            children.append(random_genome(rng))
        while len(children) < pop_size:
            a, b = rng.sample(elites, 2)
            children.append(mutate(crossover(a, b, rng), ms, rng))
        pop = children

    log.close()
    if best_ever is not None:
        print("\n" + "=" * 60)
        print(f"[GA] best ever: combined={best_ever[0]:.3f}")
        print(f"     metrics: {best_ever[1]}")
        print(f"     genome:  {fmt(best_ever[2])}")
        print("=" * 60)
    return best_ever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop",       type=int,   default=4)
    ap.add_argument("--max-gens",  type=int,   default=5)
    ap.add_argument("--target",    type=float, default=0.95)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--ncbi-max",  type=int,   default=40)
    ap.add_argument("--class-max", type=int,   default=15)
    ap.add_argument("--log",       type=str,
                    default="data/foundation/ga_node_search.log.jsonl")
    args = ap.parse_args()
    run_ga(args.pop, args.max_gens, args.target, args.seed,
           args.ncbi_max, args.class_max, pathlib.Path(args.log))


if __name__ == "__main__":
    main()
