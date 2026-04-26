#!/usr/bin/env python3
"""
ga_two_pool_search.py — offline GA parameter sweep for the two-pool architecture.

Builds a faithful Python model of:
  * streaming atom consumption with sliding buffer
  * within-pool STDP-style edges (decaying with positional gap)
  * branching-entropy concept promotion (Tanaka-Ishii criterion)
  * concept re-entry (concept id replaces member-tail in buffer)
  * symmetric cross-pool Hebbian synapses
  * within-pool propagation + cross-pool projection + decoder

Then runs a genetic algorithm searching the parameter space for genomes
that maximise forward + reverse char-level recall fidelity on the actual
conversation training set.

Output: ranked param values, concept counts, sample fwd/rev decodes.
No modifications to Rust.  Pure exploration.
"""
from __future__ import annotations
import math
import random
import time
from collections import defaultdict, deque

CONVERSATIONS = [
    ("hello",                   "Hello, I am W1z4rD. Ask me anything."),
    ("hi",                      "Hi there. How can I help?"),
    ("hey",                     "Hey. Ready to learn something new?"),
    ("good morning",            "Good morning. Hope your day is great."),
    ("how are you",             "I am doing well, thanks for asking."),
    ("who are you",             "I am W1z4rD, a distributed neural AI."),
    ("what is your name",       "My name is W1z4rD."),
    ("what can you do",         "I can answer questions and learn from data."),
    ("are you an ai",           "Yes, I am W1z4rD, a Hebbian neural AI."),
    ("goodbye",                 "Goodbye. Take care."),
    ("thanks",                  "You are welcome."),
]

# ---------------------------------------------------------------- Pool model

class Pool:
    """Faithful simplified model of the Rust NeuronPool."""
    def __init__(self, p):
        self.p = p
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: list[str] = []
        self.is_concept: list[bool] = []
        self.members: dict[int, list[int]] = {}
        self.edges: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.activation: dict[int, float] = defaultdict(float)
        # Streaming statistics (sequences truncated to length <= seq_lookback)
        self.successors: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.seq_count: dict[tuple, int] = defaultdict(int)
        self.promoted: dict[tuple, int] = {}
        self.buffer: deque = deque(maxlen=p['K'])

    def _ensure(self, label, is_concept=False, members=None):
        if label in self.label_to_id:
            return self.label_to_id[label]
        i = len(self.id_to_label)
        self.label_to_id[label] = i
        self.id_to_label.append(label)
        self.is_concept.append(is_concept)
        if is_concept and members is not None:
            self.members[i] = list(members)
        return i

    def consume_chars(self, text: str):
        L = self.p['seq_lookback']
        for ch in text:
            i = self._ensure(f"a:{ch}")
            buf_list = list(self.buffer)
            # within-pool STDP from each buffer item to i (decay with gap)
            for off, prev in enumerate(reversed(buf_list)):
                w = self.p['lr_within'] * math.exp(-off / 2.0)
                self.edges[prev][i] += w
            # update sequence statistics for prefixes ending at buffer tail
            for ll in range(1, min(L, len(buf_list)) + 1):
                prefix = tuple(buf_list[-ll:])
                self.successors[prefix][i] += 1
                self.seq_count[prefix + (i,)] += 1
            self.buffer.append(i)
            self.activation[i] = 1.0
            self._maybe_reentry()

    def end_pair(self):
        """Run concept-promotion pass after a full pair has been streamed."""
        self._maybe_promote()

    def _maybe_promote(self):
        # candidates: any sequence with enough occurrences and length >= 2
        items = [(s, c) for s, c in self.seq_count.items()
                 if len(s) >= 2 and c >= self.p['theta_n']]
        items.sort(key=lambda x: (len(x[0]), -x[1]))
        for seq, _total in items:
            if seq in self.promoted:
                continue
            succs = self.successors.get(seq, {})
            stot = sum(succs.values())
            if stot < max(2, self.p['theta_n'] // 2):
                continue
            h = 0.0
            for c in succs.values():
                pp = c / stot
                if pp > 0:
                    h -= pp * math.log2(pp)
            if h >= self.p['theta_branch']:
                lab = "C:" + "/".join(self.id_to_label[x] for x in seq)
                cid = self._ensure(lab, is_concept=True, members=list(seq))
                self.promoted[seq] = cid
                # bind concept to members both directions
                for m in seq:
                    self.edges[m][cid] += self.p['lr_within']
                    self.edges[cid][m] += self.p['lr_within'] * 0.5

    def _maybe_reentry(self):
        buf = list(self.buffer)
        for L in range(min(self.p['seq_lookback'], len(buf)), 1, -1):
            tail = tuple(buf[-L:])
            if tail in self.promoted:
                cid = self.promoted[tail]
                for _ in range(L):
                    self.buffer.pop()
                self.buffer.append(cid)
                return

    def reset_runtime(self):
        self.activation = defaultdict(float)
        self.buffer.clear()

    def stimulate_chars(self, text: str, base=1.0):
        for ch in text:
            i = self._ensure(f"a:{ch}")
            if base > self.activation[i]:
                self.activation[i] = base
        # also fire any concept whose member-tail matches a suffix of text
        ids = [self.label_to_id.get(f"a:{c}") for c in text]
        ids = [x for x in ids if x is not None]
        for L in range(len(ids), 1, -1):
            for start in range(0, len(ids) - L + 1):
                t = tuple(ids[start:start + L])
                if t in self.promoted:
                    self.activation[self.promoted[t]] = max(self.activation[self.promoted[t]], 1.0)

    def propagate(self, steps=3, threshold=0.02):
        for _ in range(steps):
            new = defaultdict(float)
            for src, a in list(self.activation.items()):
                if a < threshold:
                    continue
                if a * self.p['propagate_decay'] > new[src]:
                    new[src] = a * self.p['propagate_decay']
                outs = self.edges.get(src)
                if not outs:
                    continue
                wmax = max(outs.values()) or 1.0
                for dst, w in outs.items():
                    contrib = a * (w / wmax) * self.p['propagate_gain']
                    if contrib > threshold and contrib > new[dst]:
                        new[dst] = contrib
            self.activation = new


class CrossPool:
    def __init__(self):
        self.w: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def pair(self, src_act: dict[str, float], dst_act: dict[str, float], lr: float):
        for s, sv in src_act.items():
            if sv < 0.1:
                continue
            for d, dv in dst_act.items():
                if dv < 0.1:
                    continue
                delta = lr * sv * dv
                self.w[s][d] += delta
                self.w[d][s] += delta

    def project(self, src_act: dict[str, float], threshold=0.02):
        out = defaultdict(float)
        for s, sv in src_act.items():
            if sv < threshold:
                continue
            for d, w in self.w.get(s, {}).items():
                contrib = sv * w
                if contrib > out[d]:
                    out[d] = contrib
        return out


# --------------------------------------------------------------- Train + Eval

def train(pairs, params, seed=0):
    random.seed(seed)
    pin = Pool(params)
    pout = Pool(params)
    cross = CrossPool()
    for _ in range(params['passes']):
        for q, a in pairs:
            pin.reset_runtime()
            pout.reset_runtime()
            pin.consume_chars(q)
            pout.consume_chars(a)
            pin.end_pair()
            pout.end_pair()
            in_act = {pin.id_to_label[i]: v for i, v in pin.activation.items()}
            out_act = {pout.id_to_label[i]: v for i, v in pout.activation.items()}
            cross.pair(in_act, out_act, params['lr_cross'])
    return pin, pout, cross


def query(source: Pool, cross: CrossPool, target: Pool, text: str, expect_len: int):
    source.reset_runtime()
    target.reset_runtime()
    source.stimulate_chars(text)
    source.propagate(steps=2, threshold=0.02)
    src_act = {source.id_to_label[i]: v for i, v in source.activation.items()}
    proj = cross.project(src_act)
    for lab, v in proj.items():
        i = target._ensure(lab)
        if v > target.activation[i]:
            target.activation[i] = v
    target.propagate(steps=3, threshold=0.02)
    return decode(target, expect_len)


def decode(pool: Pool, length: int):
    if not pool.activation:
        return ""
    ranked = sorted(pool.activation.items(), key=lambda x: -x[1])
    # If the top-K has a concept, walk its members first
    for i, _v in ranked[:5]:
        if pool.is_concept[i]:
            out = []
            stack = list(pool.members.get(i, []))
            while stack and len(out) < length:
                m = stack.pop(0)
                lab = pool.id_to_label[m]
                if lab.startswith("a:"):
                    out.append(lab[2:])
                elif pool.is_concept[m]:
                    stack = list(pool.members.get(m, [])) + stack
            if "".join(out):
                return "".join(out)[:length]
    # fall back to greedy walk along strongest edge
    cur = ranked[0][0]
    out = []
    visited = set()
    for _ in range(length * 2):
        if cur in visited:
            break
        visited.add(cur)
        lab = pool.id_to_label[cur]
        if lab.startswith("a:"):
            out.append(lab[2:])
            if len(out) >= length:
                break
        elif pool.is_concept[cur]:
            for m in pool.members.get(cur, []):
                ml = pool.id_to_label[m]
                if ml.startswith("a:"):
                    out.append(ml[2:])
                if len(out) >= length:
                    break
        cands = [(d, w * (pool.activation.get(d, 0.001) + 0.01))
                 for d, w in pool.edges.get(cur, {}).items() if d not in visited]
        if not cands:
            break
        cur = max(cands, key=lambda x: x[1])[0]
    return "".join(out)[:length]


def lev_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


def fitness(params, pairs, seeds=(0,)):
    fwd_scores = []
    rev_scores = []
    last_meta = {}
    for s in seeds:
        pin, pout, cross = train(pairs, params, seed=s)
        for q, a in pairs:
            fwd_scores.append(lev_sim(query(pin, cross, pout, q, len(a)), a))
            rev_scores.append(lev_sim(query(pout, cross, pin, a, len(q)), q))
        last_meta = {
            'concepts_in': len(pin.promoted),
            'concepts_out': len(pout.promoted),
            'cross_edges': sum(len(v) for v in cross.w.values()),
        }
    fwd = sum(fwd_scores) / len(fwd_scores)
    rev = sum(rev_scores) / len(rev_scores)
    return (fwd + rev) / 2.0, fwd, rev, last_meta


# ---------------------------------------------------------------- GA

PARAM_SPACE = {
    'K':                ('int',   3,    12),
    'theta_n':          ('int',   2,    10),
    'theta_branch':     ('float', 0.10, 2.00),
    'lr_within':        ('float', 0.05, 0.60),
    'lr_cross':         ('float', 0.05, 0.60),
    'propagate_decay':  ('float', 0.40, 0.95),
    'propagate_gain':   ('float', 0.40, 1.80),
    'seq_lookback':     ('int',   2,    6),
    'passes':           ('int',   5,    50),
}


def random_genome():
    g = {}
    for k, spec in PARAM_SPACE.items():
        if spec[0] == 'int':
            g[k] = random.randint(spec[1], spec[2])
        else:
            g[k] = random.uniform(spec[1], spec[2])
    return g


def mutate(g):
    g = dict(g)
    for k, spec in PARAM_SPACE.items():
        if random.random() < 0.25:
            if spec[0] == 'int':
                g[k] = max(spec[1], min(spec[2], g[k] + random.choice([-2, -1, 1, 2])))
            else:
                g[k] = max(spec[1], min(spec[2], g[k] * random.uniform(0.6, 1.6)))
    return g


def crossover(a, b):
    return {k: (a[k] if random.random() < 0.5 else b[k]) for k in a}


def fmt_g(g):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in g.items())


def main():
    POP = 14
    GEN = 12
    print(f"GA: pop={POP} gens={GEN} pairs={len(CONVERSATIONS)}")
    pop = [random_genome() for _ in range(POP)]
    t0 = time.time()
    history = []
    for gen in range(GEN):
        scored = []
        for g in pop:
            f, fwd, rev, meta = fitness(g, CONVERSATIONS)
            scored.append((f, fwd, rev, meta, g))
        scored.sort(key=lambda x: -x[0])
        top = scored[0]
        avg = sum(x[0] for x in scored) / len(scored)
        print(f"gen {gen:2d}  best={top[0]:.3f} (fwd={top[1]:.2f} rev={top[2]:.2f})  "
              f"avg={avg:.3f}  cin={top[3]['concepts_in']} cout={top[3]['concepts_out']}  xe={top[3]['cross_edges']}")
        history.append(top)
        elites = [x[4] for x in scored[:4]]
        children = []
        while len(children) < POP - len(elites):
            a, b = random.sample(elites, 2)
            children.append(mutate(crossover(a, b)))
        pop = elites + children
    final = []
    for g in pop:
        f, fwd, rev, meta = fitness(g, CONVERSATIONS, seeds=(0, 1, 2))  # robust eval
        final.append((f, fwd, rev, meta, g))
    final.sort(key=lambda x: -x[0])
    f, fwd, rev, meta, best = final[0]
    dt = time.time() - t0
    print(f"\n=== best score={f:.3f} fwd={fwd:.3f} rev={rev:.3f}  time={dt:.1f}s ===")
    print(fmt_g(best))
    print(f"  concepts_in={meta['concepts_in']} concepts_out={meta['concepts_out']} cross_edges={meta['cross_edges']}")
    pin, pout, cross = train(CONVERSATIONS, best, seed=0)
    print("\n--- FORWARD (in -> out) ---")
    for q, a in CONVERSATIONS:
        print(f"  {q!r:30s} -> {query(pin, cross, pout, q, len(a))!r}")
    print("\n--- REVERSE (out -> in) ---")
    for q, a in CONVERSATIONS:
        print(f"  {a[:30]!r:32s} -> {query(pout, cross, pin, a, len(q))!r}")
    print("\n--- TOP 5 GENOMES (final) ---")
    for f, fwd, rev, meta, g in final[:5]:
        print(f"  score={f:.3f} fwd={fwd:.3f} rev={rev:.3f} cin={meta['concepts_in']} cout={meta['concepts_out']}  {fmt_g(g)}")


if __name__ == "__main__":
    main()
