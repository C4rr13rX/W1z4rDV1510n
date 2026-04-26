#!/usr/bin/env python3
"""
ga_longform_search.py — adaptive GA over the full architectural knob space
on a long-form asymmetric corpus (300-char inputs paired with 45,000-char
outputs, ratio 1:150).

Strategy stack under test:
  Layer 1: pair-concept exemplar retrieval (whole-pair concept neurons,
           formed at end-of-pair from the buffered atom sequence)
  Layer 2: directional positional tensor cross-pool with n-gram keys
  Layer 3: Viterbi decode over within-pool n-gram transitions

GA knobs (all variable):
  n_gram               int    1..7
  sigma_forward        float  0.5..40
  sigma_reverse        float  0.5..40
  viterbi_K            int    2..16
  viterbi_lambda       float  0.0..3.0
  pair_concept_threshold float  0.0..1.0  (when >, use exemplar; else fall back to tensor)
  pair_concept_extras_penalty  float 0.0..0.5
  match_score_min      float  0.0..1.0  (reject below — empty answer)
  use_pair_concepts    bool

Adaptive feedback:
  - mutation_strength shrinks as best fitness rises
  - elite preservation (top 3)
  - random-restart slot if no improvement in 4 generations
  - terminates when best_fitness >= 0.999 or generation budget hit
"""
from __future__ import annotations
import math
import random
import string
import time
from collections import defaultdict


# --------------------------------------------------------------- Long-form data

def make_corpus(seed=42):
    """Produce variable-length, partially overlapping pairs that mimic
       real-world data: some shared input prefixes, some unique inputs,
       outputs ranging from 1k to 50k chars, ratios from 5:1 to 500:1."""
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + " ,.!?-:'\n"
    def rnd(n):
        return "".join(rng.choice(alphabet) for _ in range(n))

    shared_prefix_a = "Tell me about "
    shared_prefix_b = "Explain how "
    pairs = []
    # (input_len, output_len, shared_prefix_or_None)
    spec = [
        (60,    1_000,  None),
        (120,   5_000,  shared_prefix_a),
        (200,  10_000,  shared_prefix_a),
        (300,  45_000,  None),
        (450,  20_000,  shared_prefix_b),
        (520,  30_000,  None),
        (90,   50_000,  shared_prefix_b),
        (250,   2_500,  None),
    ]
    for in_len, out_len, prefix in spec:
        if prefix is not None:
            body = rnd(max(in_len - len(prefix), 1))
            inp = prefix + body
        else:
            inp = rnd(in_len)
        out = rnd(out_len)
        pairs.append((inp, out))
    return pairs


# --------------------------------------------------------------- Architecture

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


class PairConceptLayer:
    """Whole-pair concept neurons, one per training pair, in each pool.
       Members are the n-gram set of the pair's text — robust to shared
       prefixes since high-order n-grams stay unique even when individual
       chars overlap heavily."""
    def __init__(self, n=4):
        self.n = n
        self.in_grams = []     # list of frozenset of n-grams per pair
        self.out_grams = []
        self.in_seq   = []
        self.out_seq  = []

    def train(self, pairs):
        for q, a in pairs:
            self.in_grams.append(frozenset(make_ngrams(q, self.n)))
            self.out_grams.append(frozenset(make_ngrams(a, self.n)))
            self.in_seq.append(q)
            self.out_seq.append(a)

    def query(self, text, direction, extras_penalty):
        """Score each pair-concept by Jaccard-like n-gram overlap."""
        stim = frozenset(make_ngrams(text, self.n))
        seqs = self.in_grams if direction == 'forward' else self.out_grams
        best_idx = None; best_score = -2.0
        if not stim:
            return None, -2.0
        for idx, members in enumerate(seqs):
            if not members: continue
            hits = len(stim & members)
            score = hits / len(members)
            extras = len(stim - members)
            score = score - extras_penalty * (extras / max(len(members), 1))
            if score > best_score:
                best_score = score; best_idx = idx
        return best_idx, best_score


# --------------------------------------------------------------- Train + Query

def build_tensor(pairs, n, lr, sigma_f, sigma_r):
    cross = TensorCrossPool()
    for q, a in pairs:
        in_seq = [f"g:{g}" for g in make_ngrams(q, n)]
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


def viterbi_decode(text, direction, n, cross, transitions, K, lam):
    in_act = [(f"g:{g}", 1.0) for g in make_ngrams(text, n)]
    out_proj = cross.project(in_act, direction=direction)
    positions = sorted(out_proj.keys())
    if not positions: return ""
    cands = [sorted(out_proj[p].items(), key=lambda x: -x[1])[:K] for p in positions]
    prev_score = {l: math.log(max(s, 1e-9)) for l, s in cands[0]}
    prev_path = {l: [l] for l, _ in cands[0]}
    for p in range(1, len(cands)):
        ns = {}; np = {}
        for l, s in cands[p]:
            gram = l[2:]
            best = (-1e9, None)
            for pl, ps in prev_score.items():
                pg = pl[2:]
                tr = transitions.get(pg, {}).get(gram, 0.0)
                score = ps + math.log(max(s, 1e-9)) + lam * math.log(tr + 1.0)
                if score > best[0]: best = (score, pl)
            ns[l] = best[0]; np[l] = prev_path[best[1]] + [l]
        prev_score = ns; prev_path = np
    bl = max(prev_score, key=lambda x: prev_score[x])
    decoded = []
    for l in prev_path[bl]:
        gram = l[2:]
        if gram and gram[-1] != "^": decoded.append(gram[-1])
    return "".join(decoded)


def query(text, direction, params, layer, cross, trans_in, trans_out):
    if params['use_pair_concepts']:
        idx, score = layer.query(text, direction, params['pair_concept_extras_penalty'])
        if score >= params['pair_concept_threshold']:
            return layer.out_seq[idx] if direction == 'forward' else layer.in_seq[idx]
    transitions = trans_out if direction == 'forward' else trans_in
    return viterbi_decode(text, direction, params['n_gram'], cross,
                           transitions, params['viterbi_K'], params['viterbi_lambda'])


# --------------------------------------------------------------- Fitness

def lev_sim(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    la, lb = len(a), len(b)
    # Cap distance computation to avoid O(N*M) on huge strings:
    # for our 45k outputs, exact Levenshtein is too slow. Use position-wise
    # match ratio + length penalty as a proxy.
    if max(la, lb) > 500:
        # match positions
        L = min(la, lb)
        match = sum(1 for i in range(L) if a[i] == b[i])
        return match / max(la, lb)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


def evaluate(params, pairs):
    """Lazy: skip the expensive tensor + Viterbi build when pair-concept
       layer covers every query at the requested threshold."""
    layer = PairConceptLayer(n=params['pair_concept_n'])
    layer.train(pairs)

    needs_tensor = (not params['use_pair_concepts'])
    if not needs_tensor:
        for q, a in pairs:
            _, sf = layer.query(q, 'forward', params['pair_concept_extras_penalty'])
            if sf < params['pair_concept_threshold']:
                needs_tensor = True; break
            _, sr = layer.query(a, 'reverse', params['pair_concept_extras_penalty'])
            if sr < params['pair_concept_threshold']:
                needs_tensor = True; break

    if needs_tensor:
        n = params['n_gram']
        cross = build_tensor(pairs, n, params['lr_cross'],
                              params['sigma_forward'], params['sigma_reverse'])
        trans_in, trans_out = build_transitions(pairs, n)
    else:
        cross = TensorCrossPool()
        trans_in, trans_out = {}, {}

    fwd_scores, rev_scores = [], []
    exact_fwd = 0; exact_rev = 0
    for q, a in pairs:
        pf = query(q, 'forward', params, layer, cross, trans_in, trans_out)
        pr = query(a, 'reverse', params, layer, cross, trans_in, trans_out)
        fwd_scores.append(lev_sim(pf, a))
        rev_scores.append(lev_sim(pr, q))
        if pf == a: exact_fwd += 1
        if pr == q: exact_rev += 1
    return {
        'fwd': sum(fwd_scores)/len(fwd_scores),
        'rev': sum(rev_scores)/len(rev_scores),
        'combined': (sum(fwd_scores)+sum(rev_scores)) / (2 * len(pairs)),
        'exact_fwd': exact_fwd,
        'exact_rev': exact_rev,
        'n_pairs': len(pairs),
        'tensor_built': needs_tensor,
    }


# --------------------------------------------------------------- GA

PARAM_SPACE = {
    'n_gram':                       ('int',   1, 7),
    'pair_concept_n':               ('int',   2, 8),
    'sigma_forward':                ('float', 0.5, 40.0),
    'sigma_reverse':                ('float', 0.5, 40.0),
    'viterbi_K':                    ('int',   2, 16),
    'viterbi_lambda':               ('float', 0.0, 3.0),
    'pair_concept_threshold':       ('float', 0.0, 1.0),
    'pair_concept_extras_penalty':  ('float', 0.0, 0.5),
    'lr_cross':                     ('float', 0.05, 1.0),
    'use_pair_concepts':            ('bool',  0, 1),
}


def random_genome(rng):
    g = {}
    for k, spec in PARAM_SPACE.items():
        if spec[0] == 'int':
            g[k] = rng.randint(spec[1], spec[2])
        elif spec[0] == 'float':
            g[k] = rng.uniform(spec[1], spec[2])
        else:
            g[k] = rng.random() < 0.5
    return g


def mutate(g, mutation_strength, rng):
    g = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() < mutation_strength:
            if spec[0] == 'int':
                delta = max(1, int(round(rng.gauss(0, mutation_strength * (spec[2] - spec[1]) / 4))))
                g[k] = max(spec[1], min(spec[2], g[k] + rng.choice([-delta, delta])))
            elif spec[0] == 'float':
                g[k] = max(spec[1], min(spec[2],
                          g[k] * rng.uniform(1 - mutation_strength, 1 + mutation_strength)))
            else:
                if rng.random() < mutation_strength:
                    g[k] = not g[k]
    return g


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def fmt(g):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float)
                     else f"{k}={v}" for k, v in g.items())


def adaptive_ga(pairs, pop_size=12, max_gens=40, target=0.999, seed=0,
                lock_params=None):
    rng = random.Random(seed)
    print(f"GA: pop={pop_size} max_gens={max_gens} target={target} pairs={len(pairs)} "
          f"in_len={len(pairs[0][0])} out_len={len(pairs[0][1])} lock={lock_params}")
    def apply_lock(g):
        if lock_params:
            for k, v in lock_params.items(): g[k] = v
        return g
    pop = [apply_lock(random_genome(rng)) for _ in range(pop_size)]
    best_ever = None; best_score = -1.0
    no_improve = 0
    for gen in range(max_gens):
        scored = []
        for g in pop:
            m = evaluate(g, pairs)
            scored.append((m['combined'], m, g))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        improved = best[0] > best_score + 1e-6
        if improved:
            best_score = best[0]; best_ever = best; no_improve = 0
        else:
            no_improve += 1
        # Adaptive mutation: tighter as we improve, looser if stuck
        mutation_strength = max(0.05, 0.6 * (1 - best_score))
        if no_improve >= 4:
            mutation_strength = min(0.7, mutation_strength + 0.2)
        avg = sum(x[0] for x in scored) / len(scored)
        m = best[1]
        print(f"  gen {gen:2d} best={best[0]:.4f} (fwd={m['fwd']:.3f} rev={m['rev']:.3f}) "
              f"avg={avg:.3f}  exact={m['exact_fwd']}/{m['exact_rev']}  mut={mutation_strength:.2f}  "
              f"{fmt(best[2])}")
        if best[0] >= target:
            print(f"\n*** Target {target} reached at gen {gen}. ***")
            break
        # Selection: top 3 elites, rest from crossover+mutate
        elites = [g for _, _, g in scored[:3]]
        children = list(elites)
        if no_improve >= 4:
            children.append(apply_lock(random_genome(rng)))
        while len(children) < pop_size:
            a, b = rng.sample(elites, 2)
            children.append(apply_lock(mutate(crossover(a, b, rng), mutation_strength, rng)))
        pop = children
    return best_ever


def main():
    print("=" * 80)
    print("Long-form GA: 5 pairs, 300-char in, 45,000-char out (ratio 1:150)")
    print("=" * 80)
    pairs = make_corpus(seed=42)
    in_lens = [len(q) for q, _ in pairs]
    out_lens = [len(a) for _, a in pairs]
    print(f"Corpus: {len(pairs)} pairs.  in_len {min(in_lens)}..{max(in_lens)}  "
          f"out_len {min(out_lens)}..{max(out_lens)}")
    print(f"Ratios: " + ", ".join(f"{ol/il:.0f}:1" for il, ol in zip(in_lens, out_lens)))
    # Sanity: pair-concept layer alone (deterministic, no GA needed for this layer).
    layer = PairConceptLayer(n=4); layer.train(pairs)
    fwd_ok = 0; rev_ok = 0
    for q, a in pairs:
        i_f, sf = layer.query(q, 'forward', 0.05)
        i_r, sr = layer.query(a, 'reverse', 0.05)
        if i_f is not None and pairs[i_f][1] == a: fwd_ok += 1
        if i_r is not None and pairs[i_r][0] == q: rev_ok += 1
    print(f"\nPair-concept exemplar layer alone: fwd={fwd_ok}/{len(pairs)} rev={rev_ok}/{len(pairs)}")

    # Phase 1: full architecture (pair-concepts allowed).
    print("\n" + "=" * 80); print("PHASE 1: full architecture (pair-concepts allowed)"); print("=" * 80)
    t0 = time.time()
    best = adaptive_ga(pairs, pop_size=12, max_gens=30, target=0.999, seed=1)
    dt = time.time() - t0
    if best:
        print(f"\nPhase 1 best (time={dt:.1f}s):  combined={best[0]:.4f}  "
              f"fwd={best[1]['fwd']:.3f}  rev={best[1]['rev']:.3f}  "
              f"exact={best[1]['exact_fwd']}/{best[1]['n_pairs']} {best[1]['exact_rev']}/{best[1]['n_pairs']}")

    # Phase 2: substrate only (use_pair_concepts locked False).
    print("\n" + "=" * 80); print("PHASE 2: SUBSTRATE-ONLY (no pair-concept exemplars)"); print("=" * 80)
    # Smaller corpus for substrate phase — Viterbi over 50k tokens is brutal.
    sub_pairs = [(q, a[:1500]) for q, a in pairs]  # cap outputs to 1.5k to keep wall-clock sane
    t0 = time.time()
    best_sub = adaptive_ga(sub_pairs, pop_size=10, max_gens=20, target=0.999, seed=2,
                           lock_params={'use_pair_concepts': False})
    dt = time.time() - t0
    if best_sub:
        print(f"\nPhase 2 best (time={dt:.1f}s):  combined={best_sub[0]:.4f}  "
              f"fwd={best_sub[1]['fwd']:.3f}  rev={best_sub[1]['rev']:.3f}  "
              f"exact={best_sub[1]['exact_fwd']}/{best_sub[1]['n_pairs']} {best_sub[1]['exact_rev']}/{best_sub[1]['n_pairs']}")
        print(f"  genome: {fmt(best_sub[2])}")

    # Phase 3: noise robustness on the Phase-1 best genome.
    print("\n" + "=" * 80); print("PHASE 3: NOISE ROBUSTNESS (pair-concept layer at Phase-1 best)"); print("=" * 80)
    if best:
        params = best[2]
        layer = PairConceptLayer(n=params['pair_concept_n']); layer.train(pairs)
        rng = random.Random(7)
        def corrupt(s, rate):
            chars = list(s)
            for i in range(len(chars)):
                if rng.random() < rate:
                    chars[i] = random.choice(string.ascii_letters)
            return "".join(chars)
        def truncate(s, frac):
            return s[: max(1, int(len(s) * frac))]
        for label, mutator in [
            ("clean",       lambda s: s),
            ("5% noise",    lambda s: corrupt(s, 0.05)),
            ("15% noise",   lambda s: corrupt(s, 0.15)),
            ("30% noise",   lambda s: corrupt(s, 0.30)),
            ("50% truncate", lambda s: truncate(s, 0.50)),
            ("25% truncate", lambda s: truncate(s, 0.25)),
        ]:
            fwd_ok = 0
            for i, (q, a) in enumerate(pairs):
                qm = mutator(q)
                idx, sc = layer.query(qm, 'forward', params['pair_concept_extras_penalty'])
                if idx == i: fwd_ok += 1
            print(f"  {label:14s}  fwd_top1={fwd_ok}/{len(pairs)}")


if __name__ == "__main__":
    main()
