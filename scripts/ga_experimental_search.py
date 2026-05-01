#!/usr/bin/env python3
"""
ga_experimental_search.py — wider-exploration GA aimed at the 0.95 target.

Rationale (from the prior scoping run that plateaued at 0.840):
  - Memorization (NCBI + class) is at 1.0 / 0.70 of the score; no headroom.
  - The 0.95 target requires lifting NCBI generalization (currently 0.21)
    and class paraphrase (0.72 mean / 9-of-15 exact) into the 0.80+ band.
  - The previous fitness biased toward memorization (weight 0.70), so any
    mutation that injected useful training noise — including the four
    cross-wiring loops — got pruned because mem dipped slightly while gen
    only ticked up.  We re-weight here to push generalization.

What this search does differently:

  1. **Re-weighted fitness** — `experimental_combined`:
       0.20 ncbi_mem + 0.20 class_mem
     + 0.25 ncbi_gen + 0.20 class_par
     + 0.05 ncbi_mem_exact + 0.05 class_mem_exact
     + 0.05 equation_hit_rate
     Generalization carries 0.45 of the weight (was 0.30) so the GA can
     trade a little memorization for a lot of generalization.

  2. **Wider exploration floor** — radius_min=0.15, shrink=0.95.  The search
     never collapses to a tight basin.

  3. **Stratified diversity** — every phase guarantees:
       - 1 elite (running best)
       - 1 explore (broad mutation around best)
       - 1 random anchor (fully fresh genome)
       - 1 individual per cross-wiring toggle ON (4 individuals)
       - rest exploit-mutated around best
     so the four feedback loops always get some test budget regardless of
     whether the elite has them ON.

  4. **Multi-anchor champions** — tracks the all-time best for `combined`,
     `ncbi_gen_mean`, AND `class_par_mean` separately.  Every 5 phases,
     the elite rotates to whichever champion the search hasn't visited
     recently.  Lets the GA escape combined-only local maxima.

  5. **Plateau breaker** — if no improvement in `combined` for `restart_after`
     phases, replace the elite with a random member of the top quartile
     across the whole log (not just current phase).

  6. **Larger eval corpus** — ncbi_max=50, class_max=25 (was 25/15).  More
     eval pairs = cleaner signal, less noise around the 0.04 toggle deltas.

  7. **Target-aware** — stops at experimental_combined ≥ 0.95.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import time

from ga_node_fitness import run_one_genome, load_jsonl, NCBI_JSONL, CLASS_JSONL


# ─── Search space (same 17 dims) ─────────────────────────────────────

PARAM_SPACE = {
    "lr":                       ("float", 0.05, 1.5),
    "passes":                   ("int",   3,    50),
    "hops":                     ("int",   1,    6),
    "min_strength":             ("float", 0.005, 0.3),
    "cls_replay_after_epoch":   ("bool",  0,    1),
    "cls_lr_scale":             ("float", 0.02, 0.30),
    "eem_in_train_loop":        ("bool",  0,    1),
    "train_equation_pool":      ("bool",  0,    1),
    "mp_confidence_threshold":  ("float", 0.05, 0.80),
    "use_eem_fallback":         ("bool",  0,    1),
    "use_bigrams":              ("bool",  0,    1),
    "use_trigrams":             ("bool",  0,    1),
    "use_idf":                  ("bool",  0,    1),
    "fb_eem_into_mp":           ("bool",  0,    1),
    "fb_mp_into_slowpool":      ("bool",  0,    1),
    "fb_replay_low_conf":       ("bool",  0,    1),
    "fb_disagreement_ne":       ("bool",  0,    1),
}

CROSS_WIRING_KEYS = [
    "fb_eem_into_mp", "fb_mp_into_slowpool",
    "fb_replay_low_conf", "fb_disagreement_ne",
]

# Best genome from the prior 200-phase scoping run (combined=0.840).
RECIPE_BPLUS_SEED = {
    "lr": 0.230, "passes": 10, "hops": 1, "min_strength": 0.155,
    "cls_replay_after_epoch": 1, "cls_lr_scale": 0.0212,
    "eem_in_train_loop": 0, "train_equation_pool": 1,
    "mp_confidence_threshold": 0.05, "use_eem_fallback": 1,
    "use_bigrams": 1, "use_trigrams": 1, "use_idf": 1,
    "fb_eem_into_mp":      0,
    "fb_mp_into_slowpool": 0,
    "fb_replay_low_conf":  0,
    "fb_disagreement_ne":  0,
}


# ─── Re-weighted fitness ─────────────────────────────────────────────

def experimental_combined(m: dict) -> float:
    """Re-weight memorization down, generalization up.

    The original combined was 0.30 mem each + 0.15 gen each + 0.05 exact each.
    Mem is already saturated at 1.0 in our best — so its 0.70 weight is dead
    weight that punishes any exploration.  Shift toward gen+par+eq.
    """
    nmm = m.get("ncbi_mem_mean", 0.0)
    cmm = m.get("class_mem_mean", 0.0)
    ngm = m.get("ncbi_gen_mean", 0.0)
    cpm = m.get("class_par_mean", 0.0)
    nme = m.get("ncbi_mem_exact", 0) / max(1, m.get("ncbi_mem_n", 1))
    cme = m.get("class_mem_exact", 0) / max(1, m.get("class_mem_n", 1))
    eq  = m.get("equation_hit_rate", 0.0)
    return (0.20 * nmm + 0.20 * cmm
          + 0.25 * ngm + 0.20 * cpm
          + 0.05 * nme + 0.05 * cme
          + 0.05 * eq)


# ─── Genome operators ────────────────────────────────────────────────

def random_genome(rng: random.Random) -> dict:
    g = {}
    for k, spec in PARAM_SPACE.items():
        kind = spec[0]
        if   kind == "int":   g[k] = rng.randint(spec[1], spec[2])
        elif kind == "bool":  g[k] = rng.randint(0, 1)
        else:                 g[k] = rng.uniform(spec[1], spec[2])
    return g


def mutate(g: dict, radius: float, rng: random.Random,
           protect_bools: bool = False) -> dict:
    out = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() > radius: continue
        kind = spec[0]
        if kind == "int":
            span = max(1, int((spec[2] - spec[1]) * radius))
            out[k] = max(spec[1], min(spec[2], out[k] + rng.randint(-span, span)))
        elif kind == "bool":
            # Cross-wiring toggles get a smaller flip probability when
            # protect_bools=True so that a genome that has them set ON
            # doesn't immediately lose them.
            flip_p = radius * (0.25 if protect_bools and k in CROSS_WIRING_KEYS
                               else 0.5)
            if rng.random() < flip_p:
                out[k] = 1 - out[k]
        else:
            jitter = (spec[2] - spec[1]) * radius * rng.uniform(-1.0, 1.0)
            out[k] = max(spec[1], min(spec[2], float(out[k]) + jitter))
    return out


def force_toggle_on(g: dict, key: str) -> dict:
    out = dict(g)
    out[key] = 1
    return out


def fmt_g(g: dict) -> str:
    parts = []
    for k, v in g.items():
        if isinstance(v, float): parts.append(f"{k}={v:.3f}")
        else:                    parts.append(f"{k}={v}")
    return ", ".join(parts)


# ─── Search ──────────────────────────────────────────────────────────

def run_experimental(
    seed: dict,
    ncbi_records, class_records,
    *,
    pop_size:           int   = 8,
    phase_budget_secs:  int   = 900,
    max_phases:         int   = 200,
    max_runtime_secs:   int   = 4 * 3600,
    radius_start:       float = 0.40,
    radius_min:         float = 0.15,
    shrink_factor:      float = 0.95,
    explore_frac:       float = 0.50,
    random_inject_p:    float = 0.25,
    restart_after:      int   = 15,
    target:             float = 0.95,
    ncbi_max:           int   = 50,
    class_max:          int   = 25,
    seed_rng:           int   = 1337,
    log_path:           pathlib.Path = pathlib.Path("data/foundation/ga_experimental.jsonl"),
):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    rng = random.Random(seed_rng)
    overall_start = time.time()

    def evaluate(g: dict) -> dict:
        try:
            return run_one_genome(g, ncbi_records, class_records,
                                  ncbi_max=ncbi_max, class_max=class_max,
                                  log=False)
        except Exception as e:
            return {"combined": 0.0, "error": str(e)[:160]}

    # Seed evaluation.
    seed_m = evaluate(seed)
    seed_score = experimental_combined(seed_m)
    print(f"[xp] seed evaluated: experimental={seed_score:.4f} "
          f"(orig combined={seed_m.get('combined',0):.4f})", flush=True)
    log.write(json.dumps({
        "phase": -1, "ind": 0, "kind": "seed",
        "genome": seed, "metrics": seed_m,
        "experimental": seed_score,
        "wall_secs_total": 0.0,
    }) + "\n")
    log.flush()

    best_genome = dict(seed)
    best_metrics = seed_m
    best_score = seed_score

    # Multi-anchor champions.
    champions = {
        "combined":      (best_score, best_genome, best_metrics),
        "ncbi_gen":      (seed_m.get("ncbi_gen_mean", 0.0), best_genome, seed_m),
        "class_par":     (seed_m.get("class_par_mean", 0.0), best_genome, seed_m),
    }

    # All-time top-quartile pool for plateau restarts.
    all_inds: list[tuple[float, dict, dict]] = [(seed_score, seed, seed_m)]

    # Pre-warm champions and all_inds from any prior log records, so plateau
    # pivots have access to specialists discovered in earlier sessions.
    if log_path.exists():
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("phase_summary"): continue
                    g = rec.get("genome")
                    m = rec.get("metrics")
                    if not g or not m: continue
                    xp = rec.get("experimental", experimental_combined(m))
                    all_inds.append((xp, g, m))
                    ngm = m.get("ncbi_gen_mean", 0.0)
                    cpm = m.get("class_par_mean", 0.0)
                    if ngm > champions["ncbi_gen"][0]:
                        champions["ncbi_gen"] = (ngm, g, m)
                    if cpm > champions["class_par"][0]:
                        champions["class_par"] = (cpm, g, m)
                    if xp > champions["combined"][0]:
                        champions["combined"] = (xp, g, m)
            print(f"[xp] pre-warmed from log: ncbi_gen_champ={champions['ncbi_gen'][0]:.3f}  "
                  f"class_par_champ={champions['class_par'][0]:.3f}  "
                  f"all_inds={len(all_inds)}", flush=True)
        except Exception as e:
            print(f"[xp] could not pre-warm champions: {e}", flush=True)

    radius = radius_start
    phases_since_improvement = 0
    rotate_counter = 0

    for phase in range(max_phases):
        if time.time() - overall_start > max_runtime_secs:
            print(f"[xp] runtime budget exhausted at phase {phase}", flush=True)
            break
        if best_score >= target:
            print(f"[xp] *** target {target} reached at phase {phase} ***",
                  flush=True)
            break
        phase_start = time.time()

        # Build population — stratified diversity.
        pop: list[tuple[dict, str]] = [(dict(best_genome), "elite")]

        # Always include broad explore around best.
        n_broad = max(1, int(round(pop_size * explore_frac)))
        for _ in range(n_broad):
            pop.append((mutate(best_genome, radius_start * 1.5, rng,
                               protect_bools=True), "explore"))

        # Always include one individual per cross-wiring toggle ON.
        for fb in CROSS_WIRING_KEYS:
            base = mutate(best_genome, radius, rng, protect_bools=True)
            pop.append((force_toggle_on(base, fb), f"force_{fb[3:]}"))

        # Random anchor with given probability.
        if rng.random() < random_inject_p:
            pop.append((random_genome(rng), "random"))

        # Fill remaining slots with exploit mutations.
        while len(pop) < pop_size + len(CROSS_WIRING_KEYS):
            pop.append((mutate(best_genome, radius, rng,
                               protect_bools=True), "exploit"))

        # Multi-anchor rotation: every 5 phases, swap elite to the best
        # champion on a metric the search hasn't been tracking.
        rotate_counter += 1
        if rotate_counter >= 5:
            rotate_counter = 0
            metric = rng.choice(["ncbi_gen", "class_par"])
            ch_score, ch_g, ch_m = champions[metric]
            if ch_score > 0:
                pop.append((dict(ch_g), f"anchor_{metric}"))
                print(f"[xp] phase {phase} anchoring on {metric}-champion "
                      f"({metric}={ch_score:.3f})", flush=True)

        scored: list[tuple[float, dict, dict, str]] = []
        for ind, (g, kind) in enumerate(pop):
            if time.time() - overall_start > max_runtime_secs: break
            t0 = time.time()
            m = evaluate(g)
            dt = time.time() - t0
            xp_score = experimental_combined(m)
            scored.append((xp_score, m, g, kind))
            all_inds.append((xp_score, g, m))

            log.write(json.dumps({
                "phase": phase, "ind": ind, "kind": kind, "radius": radius,
                "genome": g, "metrics": m,
                "experimental": xp_score,
                "wall_secs": round(dt, 1),
                "wall_secs_total": round(time.time() - overall_start, 1),
            }) + "\n")
            log.flush()

            print(f"[xp] ph{phase:03d} #{ind:02d} ({kind:18s}) "
                  f"xp={xp_score:.4f} (orig={m.get('combined',0):.3f}) "
                  f"gen={m.get('ncbi_gen_mean',0):.2f} "
                  f"par={m.get('class_par_mean',0):.2f}/"
                  f"{m.get('class_par_exact',0)}/{m.get('class_par_n',0)} "
                  f"({dt:.0f}s)", flush=True)

            # Update champions.
            if m.get("ncbi_gen_mean", 0.0) > champions["ncbi_gen"][0]:
                champions["ncbi_gen"] = (m["ncbi_gen_mean"], g, m)
            if m.get("class_par_mean", 0.0) > champions["class_par"][0]:
                champions["class_par"] = (m["class_par_mean"], g, m)

            if time.time() - phase_start > phase_budget_secs:
                print(f"[xp] phase {phase} budget hit", flush=True)
                break

        if not scored: continue

        scored.sort(key=lambda x: -x[0])
        top_score, top_m, top_g, top_kind = scored[0]
        improved = top_score > best_score + 1e-4
        if improved:
            best_score = top_score
            best_genome = top_g
            best_metrics = top_m
            champions["combined"] = (best_score, best_genome, best_metrics)
            radius = max(radius_min, radius * shrink_factor)
            phases_since_improvement = 0
            print(f"[xp] === ph{phase} NEW BEST xp={top_score:.4f} "
                  f"(kind={top_kind}) — radius -> {radius:.3f}", flush=True)
        else:
            phases_since_improvement += 1
            radius = min(radius_start, radius * 1.05)
            print(f"[xp] === ph{phase} no improvement ({phases_since_improvement} "
                  f"phases stuck); xp_best={best_score:.4f}; radius -> {radius:.3f}",
                  flush=True)

        # Plateau break: pivot the elite-search to the metric specialist
        # for whichever weak metric (gen / par) is currently lowest in the
        # running best.  The specialist may have a *lower* xp_score (their
        # mem dropped) but their basin in genome space is unexplored under
        # the current elite — so search around them now.
        #
        # 30% of the time we still fall back to a random top-quartile pick
        # to keep nearby variants in the loop.
        if phases_since_improvement >= restart_after:
            bm = best_metrics
            ngm = bm.get("ncbi_gen_mean", 0.0)
            cpm = bm.get("class_par_mean", 0.0)
            weak = "ncbi_gen" if ngm < cpm else "class_par"
            ch_score, ch_g, _ = champions[weak]
            use_specialist = rng.random() < 0.70 and ch_g is not None
            if use_specialist:
                print(f"[xp] === ph{phase} PLATEAU PIVOT to {weak} "
                      f"specialist ({weak}={ch_score:.3f}, was {ngm if weak=='ncbi_gen' else cpm:.3f})",
                      flush=True)
                best_genome = dict(ch_g)
            else:
                all_inds.sort(key=lambda x: -x[0])
                quartile = all_inds[:max(4, len(all_inds) // 4)]
                new_score, new_g, new_m = rng.choice(quartile)
                print(f"[xp] === ph{phase} PLATEAU RESTART -> top-quartile "
                      f"genome (xp={new_score:.4f})", flush=True)
                best_genome = dict(new_g)
            # Don't reset best_score — we still track the global best xp.
            radius = radius_start
            phases_since_improvement = 0

        log.write(json.dumps({
            "phase": phase, "phase_summary": True,
            "best_score": best_score,
            "best_genome": best_genome,
            "best_metrics": best_metrics,
            "champions": {
                k: {"score": v[0], "genome": v[1]}
                for k, v in champions.items()
            },
            "radius_after": radius,
            "phases_stuck": phases_since_improvement,
            "wall_secs_total": round(time.time() - overall_start, 1),
        }) + "\n")
        log.flush()

    log.close()
    return best_genome, best_metrics, champions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--phase-budget-secs", type=int, default=900)
    ap.add_argument("--max-phases", type=int, default=200)
    ap.add_argument("--max-runtime-secs", type=int, default=4 * 3600)
    ap.add_argument("--ncbi-max", type=int, default=50)
    ap.add_argument("--class-max", type=int, default=25)
    ap.add_argument("--seed-rng", type=int, default=1337)
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--restart-after", type=int, default=15)
    ap.add_argument("--explore-frac", type=float, default=0.50)
    ap.add_argument("--random-inject-p", type=float, default=0.25)
    ap.add_argument("--radius-start", type=float, default=0.40)
    ap.add_argument("--radius-min", type=float, default=0.15)
    ap.add_argument("--shrink-factor", type=float, default=0.95)
    ap.add_argument("--log", type=str,
                    default="data/foundation/ga_experimental.jsonl")
    args = ap.parse_args()

    ncbi  = load_jsonl(NCBI_JSONL)
    cls_  = load_jsonl(CLASS_JSONL)
    print(f"[xp] corpora: ncbi={len(ncbi)} classes={len(cls_)}", flush=True)
    print(f"[xp] eval subsets: ncbi_max={args.ncbi_max} class_max={args.class_max}",
          flush=True)

    # Resume: if a prior log exists, load the best xp_score genome from it
    # and use that as the seed instead of RECIPE_BPLUS_SEED.
    seed_genome = dict(RECIPE_BPLUS_SEED)
    log_p = pathlib.Path(args.log)
    if log_p.exists() and log_p.stat().st_size > 0:
        try:
            best_xp = -1.0
            best_g_resume = None
            with open(log_p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("phase_summary"): continue
                    xp = rec.get("experimental", -1.0)
                    if xp > best_xp and rec.get("genome"):
                        best_xp = xp
                        best_g_resume = rec["genome"]
            if best_g_resume is not None and best_xp > 0:
                seed_genome = best_g_resume
                print(f"[xp] resuming from prior log best xp={best_xp:.4f}",
                      flush=True)
            else:
                print(f"[xp] prior log empty/no scores; starting from Recipe B+",
                      flush=True)
        except Exception as e:
            print(f"[xp] could not resume from log: {e}; starting from Recipe B+",
                  flush=True)
    else:
        print(f"[xp] seed=Recipe B+ (prior best, orig combined~0.840)", flush=True)
    print(f"[xp] target experimental_combined={args.target}", flush=True)

    best_g, best_m, champs = run_experimental(
        seed_genome, ncbi, cls_,
        pop_size=args.pop,
        phase_budget_secs=args.phase_budget_secs,
        max_phases=args.max_phases,
        max_runtime_secs=args.max_runtime_secs,
        radius_start=args.radius_start,
        radius_min=args.radius_min,
        shrink_factor=args.shrink_factor,
        explore_frac=args.explore_frac,
        random_inject_p=args.random_inject_p,
        restart_after=args.restart_after,
        target=args.target,
        ncbi_max=args.ncbi_max, class_max=args.class_max,
        seed_rng=args.seed_rng,
        log_path=pathlib.Path(args.log),
    )
    print("\n" + "=" * 70, flush=True)
    print(f"[xp] FINAL experimental={experimental_combined(best_m):.4f}",
          flush=True)
    print(f"     orig combined={best_m.get('combined',0):.4f}", flush=True)
    print(f"     metrics: {best_m}", flush=True)
    print(f"     genome:  {fmt_g(best_g)}", flush=True)
    print(f"     champions:", flush=True)
    for k, (s, _, _) in champs.items():
        print(f"        {k:12s} = {s:.4f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
