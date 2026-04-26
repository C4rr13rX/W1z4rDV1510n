#!/usr/bin/env python3
"""
ga_ood_search.py — Phase 4: out-of-distribution generalization on the
two-pool substrate (directional positional tensor + Viterbi, NO pair-concept
exemplar shortcut). Driven by an adaptive GA that tunes the same knob space
as ga_longform_search.py.

Question being answered:
  When the query is NOT an exact training input (blend of two inputs,
  noisy version, truncation, reordering, or novel phrasing that shares
  n-grams with multiple training pairs), does the substrate produce
  coherent integrated output drawn from training material — or garbage?

Fitness (per query, then averaged):
  fluency  = fraction of output n-grams that exist in the union of
             training outputs' n-grams (catches garbage)
  alignment= max over training pairs of
             input_overlap(query, pair_i_in) * output_overlap(out, pair_i_out)
             (high when produced output matches the pair whose input
             best matches the query)
  diversity= unique-output-prefix fraction (catches mode collapse where
             the substrate returns the same string regardless of query)
  combined = 0.50*fluency + 0.30*alignment + 0.20*diversity
"""
from __future__ import annotations
import math
import random
import string
import time
from collections import defaultdict


# ----------------------------------------------------------- Architecture core

def make_ngrams(text, n):
    pad = "^" * (n - 1)
    s = pad + text
    return [s[i:i + n] for i in range(len(text))]


class TensorCrossPool:
    """Directional positional tensor cross-pool with n-gram keys."""
    __slots__ = ("forward", "reverse")
    def __init__(self):
        self.forward = defaultdict(lambda: defaultdict(float))
        self.reverse = defaultdict(lambda: defaultdict(float))

    def pair_tensor(self, in_seq, out_seq, lr, sigma_f, sigma_r):
        Ni, No = len(in_seq), len(out_seq)
        if Ni == 0 or No == 0: return
        Nim1 = max(Ni - 1, 1); Nom1 = max(No - 1, 1)
        for i, ilbl in enumerate(in_seq):
            ip = i / Nim1
            for j, olbl in enumerate(out_seq):
                gap = abs(ip - j / Nom1)
                kf = math.exp(-(gap * sigma_f) ** 2)
                kr = math.exp(-(gap * sigma_r) ** 2)
                if kf > 0.05:
                    self.forward[(i, ilbl)][(j, olbl)] += lr * kf
                if kr > 0.05:
                    self.reverse[(j, olbl)][(i, ilbl)] += lr * kr

    def project(self, in_act_seq, direction):
        table = self.forward if direction == 'forward' else self.reverse
        out = defaultdict(lambda: defaultdict(float))
        for i, (ilbl, ia) in enumerate(in_act_seq):
            if ia < 0.05: continue
            for (j, olbl), w in table.get((i, ilbl), {}).items():
                contrib = ia * w
                if contrib > out[j][olbl]:
                    out[j][olbl] = contrib
        return out


def build_tensor(pairs, n, lr, sigma_f, sigma_r):
    cross = TensorCrossPool()
    for q, a in pairs:
        in_seq  = [f"g:{g}" for g in make_ngrams(q, n)]
        out_seq = [f"g:{g}" for g in make_ngrams(a, n)]
        cross.pair_tensor(in_seq, out_seq, lr, sigma_f, sigma_r)
    return cross


def build_transitions(pairs, n):
    trans_in = defaultdict(lambda: defaultdict(float))
    trans_out = defaultdict(lambda: defaultdict(float))
    for q, a in pairs:
        gs = make_ngrams(a, n)
        for i in range(len(gs) - 1): trans_out[gs[i]][gs[i+1]] += 1
        gs = make_ngrams(q, n)
        for i in range(len(gs) - 1): trans_in[gs[i]][gs[i+1]] += 1
    return trans_in, trans_out


def viterbi_decode(text, direction, n, cross, transitions, K, lam, max_len=None):
    in_act = [(f"g:{g}", 1.0) for g in make_ngrams(text, n)]
    out_proj = cross.project(in_act, direction=direction)
    positions = sorted(out_proj.keys())
    if max_len is not None and len(positions) > max_len:
        positions = positions[:max_len]
    if not positions: return ""
    cands = [sorted(out_proj[p].items(), key=lambda x: -x[1])[:K] for p in positions]
    if not cands[0]: return ""
    prev_score = {l: math.log(max(s, 1e-9)) for l, s in cands[0]}
    prev_path  = {l: [l] for l, _ in cands[0]}
    for p in range(1, len(cands)):
        ns = {}; np_ = {}
        if not cands[p]: break
        for l, s in cands[p]:
            gram = l[2:]
            best = (-1e9, None)
            for pl, ps in prev_score.items():
                pg = pl[2:]
                tr = transitions.get(pg, {}).get(gram, 0.0)
                score = ps + math.log(max(s, 1e-9)) + lam * math.log(tr + 1.0)
                if score > best[0]: best = (score, pl)
            ns[l] = best[0]; np_[l] = prev_path[best[1]] + [l]
        prev_score = ns; prev_path = np_
    bl = max(prev_score, key=lambda x: prev_score[x])
    decoded = []
    for l in prev_path[bl]:
        gram = l[2:]
        if gram and gram[-1] != "^": decoded.append(gram[-1])
    return "".join(decoded)


# ----------------------------------------------------------- Corpus + OOD

def make_training_corpus(seed=42):
    """6 training pairs with structural variety: thematic prefixes shared by
       pairs of pairs, so blends are testable. Outputs are deterministic
       synthetic text 600-1200 chars (substrate-friendly scale)."""
    rng = random.Random(seed)
    body_chars = string.ascii_lowercase + " ,.!?"
    def body(sig, n):
        # Deterministic body per signature, length n
        local = random.Random(hash(sig) & 0xFFFFFFFF)
        return sig + " " + "".join(local.choice(body_chars) for _ in range(max(0, n - len(sig) - 1)))

    pairs = [
        ("greet me warmly",
         body("hello there friend, welcome aboard, glad to see you", 800)),
        ("greet me coldly",
         body("acknowledged, standard protocol greeting initiated, proceed", 800)),
        ("explain quantum physics",
         body("quantum mechanics: superposition entanglement uncertainty", 1200)),
        ("explain classical physics",
         body("classical mechanics: forces motion newtonian inertia mass", 1200)),
        ("describe the weather",
         body("today sunny mild winds moderate temperature clear skies", 600)),
        ("describe the season",
         body("currently spring season blooming flowers warming days", 600)),
    ]
    return pairs


def make_ood_queries(training_pairs):
    """Construct OOD queries: blends, noise, truncations, reorders, novel."""
    rng = random.Random(7)
    def noise(s, rate):
        chars = list(s)
        for i in range(len(chars)):
            if rng.random() < rate and chars[i] != ' ':
                chars[i] = rng.choice(string.ascii_lowercase)
        return "".join(chars)

    queries = [
        # 1. Blend prefix from pair 0 (greet warmly) + thematic body from pair 2
        ("greet me warmly explain quantum",        "blend_topic"),
        # 2. Mash two inputs
        ("explain quantum classical physics",      "blend_concept"),
        # 3. Heavy noise on pair 0 input
        (noise("greet me warmly", 0.20),           "noise"),
        # 4. Truncation of pair 2 input
        ("explain quantum",                        "truncate"),
        # 5. Reordered tokens from pair 0
        ("warmly greet me",                        "reorder"),
        # 6. Compositional novel: weather + quantum theme
        ("explain the weather",                    "compose_themes"),
        # 7. Compositional novel: describe + quantum
        ("describe quantum physics",               "compose_themes"),
        # 8. Pure novel with shared structure
        ("greet me coldly explain classical",      "blend_dual"),
    ]
    return queries


# ----------------------------------------------------------- Fitness

def evaluate(params, training_pairs, ood_queries, verbose=False):
    n = params['n_gram']
    cross = build_tensor(training_pairs, n, params['lr_cross'],
                          params['sigma_forward'], params['sigma_reverse'])
    trans_in, trans_out = build_transitions(training_pairs, n)

    pair_in_grams = [set(make_ngrams(q, n)) for q, _ in training_pairs]
    pair_out_grams = [set(make_ngrams(a, n)) for _, a in training_pairs]
    train_out_grams_global = set()
    for s in pair_out_grams: train_out_grams_global.update(s)

    fluency_scores = []
    alignment_scores = []
    outputs = []

    for query, kind in ood_queries:
        # Cap output length so a runaway Viterbi can't dominate runtime.
        max_out = max(len(a) for _, a in training_pairs) + 50
        out = viterbi_decode(query, 'forward', n, cross, trans_out,
                              params['viterbi_K'], params['viterbi_lambda'],
                              max_len=max_out)
        outputs.append(out)
        if not out:
            fluency_scores.append(0.0); alignment_scores.append(0.0); continue

        out_grams = set(make_ngrams(out, n))
        if not out_grams:
            fluency_scores.append(0.0); alignment_scores.append(0.0); continue

        fluency = len(out_grams & train_out_grams_global) / len(out_grams)
        fluency_scores.append(fluency)

        query_grams = set(make_ngrams(query, n))
        if not query_grams:
            alignment_scores.append(0.0); continue
        best = 0.0
        for idx in range(len(training_pairs)):
            in_overlap  = len(query_grams & pair_in_grams[idx])  / max(len(query_grams), 1)
            out_overlap = len(out_grams   & pair_out_grams[idx]) / max(len(out_grams), 1)
            best = max(best, in_overlap * out_overlap)
        alignment_scores.append(best)

    # Diversity: how many distinct first-100-char prefixes?
    distinct = len(set(o[:100] for o in outputs if o))
    diversity = distinct / max(len(ood_queries), 1)

    fluency_avg = sum(fluency_scores) / max(len(fluency_scores), 1)
    alignment_avg = sum(alignment_scores) / max(len(alignment_scores), 1)
    combined = 0.50 * fluency_avg + 0.30 * alignment_avg + 0.20 * diversity

    if verbose:
        for (q, kind), o, fs, als in zip(ood_queries, outputs, fluency_scores, alignment_scores):
            preview = (o[:80] + "...") if len(o) > 80 else o
            print(f"  [{kind:14s}] q={q!r:40s}  fluency={fs:.2f} align={als:.2f}")
            print(f"     -> {preview!r}")

    return {
        'fluency': fluency_avg,
        'alignment': alignment_avg,
        'diversity': diversity,
        'combined': combined,
        'outputs': outputs,
    }


# ----------------------------------------------------------- GA

PARAM_SPACE = {
    'n_gram':         ('int',   3, 9),
    'sigma_forward':  ('float', 1.0, 60.0),
    'sigma_reverse':  ('float', 1.0, 60.0),
    'viterbi_K':      ('int',   2, 24),
    'viterbi_lambda': ('float', 0.0, 4.0),
    'lr_cross':       ('float', 0.05, 1.0),
}


def random_genome(rng):
    g = {}
    for k, spec in PARAM_SPACE.items():
        if spec[0] == 'int':
            g[k] = rng.randint(spec[1], spec[2])
        elif spec[0] == 'float':
            g[k] = rng.uniform(spec[1], spec[2])
    return g


def mutate(g, ms, rng):
    g = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() < ms:
            if spec[0] == 'int':
                delta = max(1, int(round(rng.gauss(0, ms * (spec[2] - spec[1]) / 4))))
                g[k] = max(spec[1], min(spec[2], g[k] + rng.choice([-delta, delta])))
            else:
                g[k] = max(spec[1], min(spec[2],
                          g[k] * rng.uniform(1 - ms, 1 + ms)))
    return g


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def fmt(g):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float)
                     else f"{k}={v}" for k, v in g.items())


# Phase-2 substrate-only winner from ga_longform_search.py — used to seed
# the GA so we start from validated knobs, not random.
SEED_GENOME = {
    'n_gram':         7,
    'sigma_forward':  39.814,
    'sigma_reverse':  40.000,
    'viterbi_K':      15,
    'viterbi_lambda': 1.117,
    'lr_cross':       1.000,
}


def adaptive_ga(training_pairs, ood_queries,
                pop_size=10, max_gens=25, target=0.95, seed=1):
    rng = random.Random(seed)
    print(f"GA: pop={pop_size} max_gens={max_gens} target={target} "
          f"train_pairs={len(training_pairs)} ood_queries={len(ood_queries)}")
    print(f"Seed genome (Phase-2 substrate-only winner): {fmt(SEED_GENOME)}")
    # Seed: 1 exact copy + 4 small perturbations + rest random for diversity.
    pop = [dict(SEED_GENOME)]
    for _ in range(4):
        pop.append(mutate(SEED_GENOME, 0.15, rng))
    while len(pop) < pop_size:
        pop.append(random_genome(rng))
    best_ever = None; best_score = -1.0
    no_improve = 0
    for gen in range(max_gens):
        scored = []
        for g in pop:
            m = evaluate(g, training_pairs, ood_queries)
            scored.append((m['combined'], m, g))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        improved = best[0] > best_score + 1e-6
        if improved:
            best_score = best[0]; best_ever = best; no_improve = 0
        else:
            no_improve += 1
        ms = max(0.05, 0.6 * (1 - best_score))
        if no_improve >= 4: ms = min(0.7, ms + 0.2)
        avg = sum(x[0] for x in scored) / len(scored)
        m = best[1]
        print(f"  gen {gen:2d} best={best[0]:.4f} "
              f"(flu={m['fluency']:.2f} align={m['alignment']:.2f} div={m['diversity']:.2f}) "
              f"avg={avg:.3f}  mut={ms:.2f}  {fmt(best[2])}")
        if best[0] >= target:
            print(f"\n*** Target {target} reached at gen {gen}. ***")
            break
        elites = [g for _, _, g in scored[:3]]
        children = list(elites)
        if no_improve >= 4: children.append(random_genome(rng))
        while len(children) < pop_size:
            a, b = rng.sample(elites, 2)
            children.append(mutate(crossover(a, b, rng), ms, rng))
        pop = children
    return best_ever


def main():
    print("=" * 80)
    print("PHASE 4: OOD generalization on substrate-only path")
    print("=" * 80)
    training_pairs = make_training_corpus()
    ood_queries = make_ood_queries(training_pairs)

    print(f"\nTraining corpus: {len(training_pairs)} pairs")
    for q, a in training_pairs:
        print(f"  {q!r} -> {a[:50]!r}... (len={len(a)})")
    print(f"\nOOD test queries: {len(ood_queries)}")
    for q, kind in ood_queries:
        print(f"  [{kind:14s}] {q!r}")

    t0 = time.time()
    best = adaptive_ga(training_pairs, ood_queries,
                       pop_size=10, max_gens=20, target=0.95, seed=1)
    dt = time.time() - t0

    if best:
        print("\n" + "=" * 80)
        print(f"Final best (time={dt:.1f}s):")
        print(f"  combined={best[0]:.4f}  flu={best[1]['fluency']:.3f}  "
              f"align={best[1]['alignment']:.3f}  div={best[1]['diversity']:.3f}")
        print(f"  genome: {fmt(best[2])}")
        print("\nDetailed best-genome OOD outputs:")
        evaluate(best[2], training_pairs, ood_queries, verbose=True)


if __name__ == "__main__":
    main()
