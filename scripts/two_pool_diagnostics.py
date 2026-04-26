#!/usr/bin/env python3
"""
two_pool_diagnostics.py — controlled experiments to identify which
architectural changes actually fix associative recall.

Five experiments, each varying one mechanism while holding others constant:

  A. Threshold relaxation       (does promoting more concepts help?)
  B. Positional cross-pool      (replace bag-of-chars with i*j temporal kernel)
  C. Concept-level cross-pool   (only pair concepts; ignore atoms)
  D. Sentinel-anchored pairs    (prepend unique start token per pair)
  E. Hierarchy depth audit      (does morpheme->word->sentence emerge?)

Output: per-experiment fitness numbers + diagnostic counts.
"""
from __future__ import annotations
import math
import random
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
    def __init__(self, p):
        self.p = p
        self.label_to_id = {}
        self.id_to_label = []
        self.is_concept = []
        self.depth = []         # 0 = atom, 1 = sequence-of-atoms, 2 = sequence-of-concepts, ...
        self.members = {}
        self.edges = defaultdict(lambda: defaultdict(float))
        self.activation = defaultdict(float)
        self.successors = defaultdict(lambda: defaultdict(int))
        self.seq_count = defaultdict(int)
        self.promoted = {}
        self.buffer = deque(maxlen=p['K'])

    def _ensure(self, label, is_concept=False, members=None, depth=0):
        if label in self.label_to_id:
            return self.label_to_id[label]
        i = len(self.id_to_label)
        self.label_to_id[label] = i
        self.id_to_label.append(label)
        self.is_concept.append(is_concept)
        self.depth.append(depth)
        if is_concept and members is not None:
            self.members[i] = list(members)
        return i

    def consume_chars(self, text, sentinel=None):
        if sentinel is not None:
            i = self._ensure(f"S:{sentinel}")
            self.buffer.append(i)
            self.activation[i] = 1.0
        for ch in text:
            i = self._ensure(f"a:{ch}")
            buf_list = list(self.buffer)
            for off, prev in enumerate(reversed(buf_list)):
                w = self.p['lr_within'] * math.exp(-off / 2.0)
                self.edges[prev][i] += w
            for ll in range(1, min(self.p['seq_lookback'], len(buf_list)) + 1):
                prefix = tuple(buf_list[-ll:])
                self.successors[prefix][i] += 1
                self.seq_count[prefix + (i,)] += 1
            self.buffer.append(i)
            self.activation[i] = 1.0
            self._maybe_reentry()

    def end_pair(self):
        self._maybe_promote()

    def _maybe_promote(self):
        items = [(s, c) for s, c in self.seq_count.items()
                 if len(s) >= 2 and c >= self.p['theta_n']]
        items.sort(key=lambda x: (len(x[0]), -x[1]))
        for seq, _ in items:
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
                # depth = max child depth + 1
                child_depth = max(self.depth[x] for x in seq) if seq else 0
                cid = self._ensure(lab, is_concept=True, members=list(seq), depth=child_depth + 1)
                self.promoted[seq] = cid
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

    def stimulate_chars(self, text, sentinel=None):
        if sentinel is not None:
            sid = self._ensure(f"S:{sentinel}")
            self.activation[sid] = 1.0
        for ch in text:
            i = self._ensure(f"a:{ch}")
            self.activation[i] = max(self.activation[i], 1.0)

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
    def __init__(self, mode='bag'):
        # mode: 'bag' (every-to-every), 'positional' (gaussian on i*j positions),
        #       'concept_only' (skip atom pairs)
        self.mode = mode
        self.w = defaultdict(lambda: defaultdict(float))

    def pair_bag(self, src_act, dst_act, lr):
        for s, sv in src_act.items():
            if sv < 0.1: continue
            for d, dv in dst_act.items():
                if dv < 0.1: continue
                delta = lr * sv * dv
                self.w[s][d] += delta
                self.w[d][s] += delta

    def pair_concept_only(self, src_act, dst_act, lr, pool_src, pool_dst):
        for s, sv in src_act.items():
            if sv < 0.1: continue
            sid = pool_src.label_to_id.get(s)
            if sid is None or not pool_src.is_concept[sid]: continue
            for d, dv in dst_act.items():
                if dv < 0.1: continue
                did = pool_dst.label_to_id.get(d)
                if did is None or not pool_dst.is_concept[did]: continue
                delta = lr * sv * dv
                self.w[s][d] += delta
                self.w[d][s] += delta

    def pair_positional(self, src_seq_labels, dst_seq_labels, lr, sigma=3.0):
        # bind src_seq_labels[i] -> dst_seq_labels[j] with weight gaussian(|i/Ns - j/Nd|)
        Ns, Nd = len(src_seq_labels), len(dst_seq_labels)
        if Ns == 0 or Nd == 0: return
        for i, s in enumerate(src_seq_labels):
            for j, d in enumerate(dst_seq_labels):
                gap = abs(i / max(Ns - 1, 1) - j / max(Nd - 1, 1))
                k = math.exp(- (gap * sigma) ** 2)
                if k < 0.05: continue
                delta = lr * k
                self.w[s][d] += delta
                self.w[d][s] += delta

    def project(self, src_act, threshold=0.02):
        out = defaultdict(float)
        for s, sv in src_act.items():
            if sv < threshold: continue
            for d, w in self.w.get(s, {}).items():
                contrib = sv * w
                if contrib > out[d]:
                    out[d] = contrib
        return out


# ---------------------------------------------------------------- Helpers

def lev_sim(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


def decode(pool, length):
    if not pool.activation: return ""
    ranked = sorted(pool.activation.items(), key=lambda x: -x[1])
    # Deepest fired concept wins
    best_c = None; best_score = -1.0
    for i, v in ranked[:8]:
        if pool.is_concept[i] and pool.depth[i] >= 1:
            score = v * (1.0 + 0.3 * pool.depth[i])
            if score > best_score:
                best_score = score; best_c = i
    if best_c is not None:
        out = []
        stack = list(pool.members.get(best_c, []))
        while stack and len(out) < length * 2:
            m = stack.pop(0)
            lab = pool.id_to_label[m]
            if lab.startswith("a:"): out.append(lab[2:])
            elif pool.is_concept[m]:
                stack = list(pool.members.get(m, [])) + stack
        if "".join(out): return "".join(out)[:length]
    # Greedy chain walk
    cur = ranked[0][0]
    out = []; visited = set()
    for _ in range(length * 2):
        if cur in visited: break
        visited.add(cur)
        lab = pool.id_to_label[cur]
        if lab.startswith("a:"):
            out.append(lab[2:])
            if len(out) >= length: break
        elif pool.is_concept[cur]:
            for m in pool.members.get(cur, []):
                ml = pool.id_to_label[m]
                if ml.startswith("a:"): out.append(ml[2:])
                if len(out) >= length: break
        cands = [(d, w * (pool.activation.get(d, 0.001) + 0.01))
                 for d, w in pool.edges.get(cur, {}).items() if d not in visited]
        if not cands: break
        cur = max(cands, key=lambda x: x[1])[0]
    return "".join(out)[:length]


# ---------------------------------------------------------------- Experiments

BASE_PARAMS = {
    'K': 6, 'theta_n': 3, 'theta_branch': 0.5,
    'lr_within': 0.3, 'lr_cross': 0.3,
    'propagate_decay': 0.85, 'propagate_gain': 1.0,
    'seq_lookback': 5, 'passes': 30,
}


def train(pairs, params, mode='bag', use_sentinel=False):
    random.seed(0)
    pin = Pool(params); pout = Pool(params); cross = CrossPool(mode=mode)
    for ep in range(params['passes']):
        for idx, (q, a) in enumerate(pairs):
            pin.reset_runtime(); pout.reset_runtime()
            sent = idx if use_sentinel else None
            pin.consume_chars(q, sentinel=sent)
            pout.consume_chars(a, sentinel=sent)
            pin.end_pair(); pout.end_pair()
            in_act = {pin.id_to_label[i]: v for i, v in pin.activation.items()}
            out_act = {pout.id_to_label[i]: v for i, v in pout.activation.items()}
            if mode == 'bag':
                cross.pair_bag(in_act, out_act, params['lr_cross'])
            elif mode == 'concept_only':
                cross.pair_concept_only(in_act, out_act, params['lr_cross'], pin, pout)
            elif mode == 'positional':
                src_seq = ([f"S:{sent}"] if use_sentinel else []) + [f"a:{c}" for c in q]
                dst_seq = ([f"S:{sent}"] if use_sentinel else []) + [f"a:{c}" for c in a]
                cross.pair_positional(src_seq, dst_seq, params['lr_cross'])
    return pin, pout, cross


def query(source, cross, target, text, expect_len, sentinel=None):
    source.reset_runtime(); target.reset_runtime()
    source.stimulate_chars(text, sentinel=sentinel)
    source.propagate(steps=2)
    src_act = {source.id_to_label[i]: v for i, v in source.activation.items()}
    proj = cross.project(src_act)
    for lab, v in proj.items():
        i = target._ensure(lab)
        if v > target.activation[i]: target.activation[i] = v
    target.propagate(steps=4)
    return decode(target, expect_len)


def evaluate(pin, pout, cross, pairs, use_sentinel=False):
    fwds, revs = [], []
    for idx, (q, a) in enumerate(pairs):
        sent = idx if use_sentinel else None
        fwds.append(lev_sim(query(pin, cross, pout, q, len(a), sentinel=sent), a))
        revs.append(lev_sim(query(pout, cross, pin, a, len(q), sentinel=sent), q))
    return sum(fwds)/len(fwds), sum(revs)/len(revs)


def report(name, pin, pout, cross, pairs, use_sentinel=False, sample=2):
    fwd, rev = evaluate(pin, pout, cross, pairs, use_sentinel=use_sentinel)
    cin = sum(1 for x in pin.is_concept if x)
    cout = sum(1 for x in pout.is_concept if x)
    deep_in = max(pin.depth) if pin.depth else 0
    deep_out = max(pout.depth) if pout.depth else 0
    xe = sum(len(v) for v in cross.w.values())
    print(f"\n[{name}]")
    print(f"  fwd={fwd:.3f}  rev={rev:.3f}  combined={(fwd+rev)/2:.3f}")
    print(f"  concepts: in={cin} (max depth {deep_in})  out={cout} (max depth {deep_out})  cross_edges={xe}")
    if sample:
        print(f"  sample fwd:")
        for idx, (q, a) in enumerate(pairs[:sample]):
            sent = idx if use_sentinel else None
            ans = query(pin, cross, pout, q, len(a), sentinel=sent)
            print(f"    {q!r:25s} -> {ans!r:50s} (want {a[:30]!r})")


# ---------------------------------------------------------------- Run

def experiment_a_threshold_relaxation():
    print("=" * 72)
    print("EXPERIMENT A: threshold relaxation")
    print("=" * 72)
    print("Hypothesis: lowering theta_n / theta_branch lets concepts form")
    print("            at our small-corpus scale; recall improves as a result.")
    for theta_n in [2, 3, 5, 8]:
        for theta_branch in [0.0, 0.3, 0.7, 1.2]:
            p = dict(BASE_PARAMS, theta_n=theta_n, theta_branch=theta_branch)
            pin, pout, cross = train(CONVERSATIONS, p, mode='bag')
            fwd, rev = evaluate(pin, pout, cross, CONVERSATIONS)
            cin = sum(1 for x in pin.is_concept if x)
            cout = sum(1 for x in pout.is_concept if x)
            print(f"  theta_n={theta_n} theta_branch={theta_branch:.1f}  "
                  f"fwd={fwd:.3f} rev={rev:.3f}  cin={cin:3d} cout={cout:3d}")


def experiment_b_positional_cross_pool():
    print("\n" + "=" * 72)
    print("EXPERIMENT B: positional cross-pool")
    print("=" * 72)
    print("Hypothesis: a temporal kernel (gap-weighted) gives cross-pool")
    print("            edges meaningful position info, breaking bag-of-chars overlap.")
    for sigma in [1.0, 2.0, 3.0, 5.0, 10.0]:
        # patch sigma into pair_positional via closure-friendly param
        pin = Pool(BASE_PARAMS); pout = Pool(BASE_PARAMS); cross = CrossPool(mode='positional')
        for _ in range(BASE_PARAMS['passes']):
            for idx, (q, a) in enumerate(CONVERSATIONS):
                pin.reset_runtime(); pout.reset_runtime()
                pin.consume_chars(q); pout.consume_chars(a)
                pin.end_pair(); pout.end_pair()
                src_seq = [f"a:{c}" for c in q]
                dst_seq = [f"a:{c}" for c in a]
                cross.pair_positional(src_seq, dst_seq, BASE_PARAMS['lr_cross'], sigma=sigma)
        fwd, rev = evaluate(pin, pout, cross, CONVERSATIONS)
        cin = sum(1 for x in pin.is_concept if x); cout = sum(1 for x in pout.is_concept if x)
        print(f"  sigma={sigma:5.1f}  fwd={fwd:.3f} rev={rev:.3f}  cin={cin} cout={cout}")


def experiment_c_concept_only_cross():
    print("\n" + "=" * 72)
    print("EXPERIMENT C: concept-level cross-pool only")
    print("=" * 72)
    print("Hypothesis: pairing only concepts (not raw atoms) creates")
    print("            sparse, discriminative cross-pool edges.")
    p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0)
    pin, pout, cross = train(CONVERSATIONS, p, mode='concept_only')
    report("concept_only theta_n=2 theta_branch=0", pin, pout, cross, CONVERSATIONS)


def experiment_d_sentinel():
    print("\n" + "=" * 72)
    print("EXPERIMENT D: per-pair sentinel anchor")
    print("=" * 72)
    print("Hypothesis: a unique start token per pair gives the chain-walk")
    print("            an unambiguous entry point and disambiguates queries.")
    pin, pout, cross = train(CONVERSATIONS, BASE_PARAMS, mode='bag', use_sentinel=True)
    report("bag + sentinel", pin, pout, cross, CONVERSATIONS, use_sentinel=True)
    p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0)
    pin, pout, cross = train(CONVERSATIONS, p, mode='positional', use_sentinel=True)
    report("positional + sentinel + low thresholds", pin, pout, cross, CONVERSATIONS, use_sentinel=True)


def experiment_e_hierarchy_depth():
    print("\n" + "=" * 72)
    print("EXPERIMENT E: hierarchy depth audit (passes vs concept depth)")
    print("=" * 72)
    print("Question: how many passes until depth-2 concepts appear?")
    for passes in [5, 15, 30, 60, 120]:
        p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0, passes=passes)
        pin, pout, cross = train(CONVERSATIONS, p, mode='bag')
        depth_hist_in  = defaultdict(int)
        depth_hist_out = defaultdict(int)
        for d in pin.depth:  depth_hist_in[d]  += 1
        for d in pout.depth: depth_hist_out[d] += 1
        di = dict(depth_hist_in); do = dict(depth_hist_out)
        print(f"  passes={passes:3d}  in={di}  out={do}")


def experiment_f_combined_best():
    print("\n" + "=" * 72)
    print("EXPERIMENT F: combined best-guess configuration")
    print("=" * 72)
    p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0, passes=60)
    pin = Pool(p); pout = Pool(p); cross = CrossPool(mode='positional')
    for _ in range(p['passes']):
        for idx, (q, a) in enumerate(CONVERSATIONS):
            pin.reset_runtime(); pout.reset_runtime()
            pin.consume_chars(q, sentinel=idx); pout.consume_chars(a, sentinel=idx)
            pin.end_pair(); pout.end_pair()
            src_seq = [f"S:{idx}"] + [f"a:{c}" for c in q]
            dst_seq = [f"S:{idx}"] + [f"a:{c}" for c in a]
            cross.pair_positional(src_seq, dst_seq, p['lr_cross'], sigma=2.0)
    report("positional + sentinel + theta_n=2 theta_branch=0 passes=60",
           pin, pout, cross, CONVERSATIONS, use_sentinel=True, sample=11)


class TensorCrossPool:
    """Directional positional tensor: forward[(i,in_lbl)] -> {(j,out_lbl): w},
       reverse[(j,out_lbl)] -> {(i,in_lbl): w}. Forward and reverse are
       SEPARATE maps so InToOut and OutToIn traverse different topology."""
    def __init__(self):
        self.forward = defaultdict(lambda: defaultdict(float))
        self.reverse = defaultdict(lambda: defaultdict(float))

    def pair_tensor(self, in_seq, out_seq, lr, sigma=2.0):
        Ni, No = len(in_seq), len(out_seq)
        if Ni == 0 or No == 0: return
        for i, ilbl in enumerate(in_seq):
            for j, olbl in enumerate(out_seq):
                gap = abs(i / max(Ni-1, 1) - j / max(No-1, 1))
                k = math.exp(-(gap * sigma) ** 2)
                if k < 0.05: continue
                self.forward[(i, ilbl)][(j, olbl)] += lr * k
                self.reverse[(j, olbl)][(i, ilbl)] += lr * k

    def project(self, in_act_seq, direction='forward'):
        # in_act_seq: list of (label, activation) by position.
        # Returns: out_pos -> out_lbl -> activation
        table = self.forward if direction == 'forward' else self.reverse
        out = defaultdict(lambda: defaultdict(float))
        for i, (ilbl, ia) in enumerate(in_act_seq):
            if ia < 0.05: continue
            for (j, olbl), w in table.get((i, ilbl), {}).items():
                contrib = ia * w
                if contrib > out[j][olbl]:
                    out[j][olbl] = contrib
        return out


def experiment_g_directional_tensor():
    print("\n" + "=" * 72)
    print("EXPERIMENT G: directional positional tensor cross-pool")
    print("=" * 72)
    print("Hypothesis: storing (in_pos,in_lbl)->(out_pos,out_lbl) edges in a")
    print("            directional tensor with separate forward/reverse maps")
    print("            preserves sequence order and breaks query-collapse.")
    for sigma in [1.5, 3.0, 6.0]:
        cross = TensorCrossPool()
        for _ in range(BASE_PARAMS['passes']):
            for q, a in CONVERSATIONS:
                in_seq = [f"a:{c}" for c in q]
                out_seq = [f"a:{c}" for c in a]
                cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=sigma)
        # forward eval
        fwd_scores, rev_scores = [], []
        for q, a in CONVERSATIONS:
            in_act = [(f"a:{c}", 1.0) for c in q]
            out_proj = cross.project(in_act, direction='forward')
            decoded = []
            for j in sorted(out_proj.keys()):
                if not out_proj[j]: continue
                lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            pred_a = "".join(decoded)
            fwd_scores.append(lev_sim(pred_a, a))
            # reverse
            out_in = [(f"a:{c}", 1.0) for c in a]
            in_proj = cross.project(out_in, direction='reverse')
            decoded = []
            for i in sorted(in_proj.keys()):
                if not in_proj[i]: continue
                lbl, _ = max(in_proj[i].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            pred_q = "".join(decoded)
            rev_scores.append(lev_sim(pred_q, q))
        fwd = sum(fwd_scores)/len(fwd_scores)
        rev = sum(rev_scores)/len(rev_scores)
        print(f"  sigma={sigma:4.1f}  fwd={fwd:.3f} rev={rev:.3f}  combined={(fwd+rev)/2:.3f}")
        if sigma == 3.0:
            print("    sample (sigma=3.0):")
            for (q, a), s in list(zip(CONVERSATIONS, fwd_scores))[:5]:
                in_act = [(f"a:{c}", 1.0) for c in q]
                out_proj = cross.project(in_act, direction='forward')
                decoded = []
                for j in sorted(out_proj.keys()):
                    if not out_proj[j]: continue
                    lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
                    if lbl.startswith("a:"): decoded.append(lbl[2:])
                print(f"      {q!r:25s} -> {''.join(decoded)!r:50s}  (want {a[:30]!r})")


def experiment_h_inline_concept_promotion():
    """Concepts fire during the pair (not just at end), so concept activations
       accumulate over the pair window and participate in cross-pool pairing."""
    print("\n" + "=" * 72)
    print("EXPERIMENT H: inline concept promotion + persistent activation")
    print("=" * 72)
    print("Hypothesis: tracking ALL labels that fire across an entire pair")
    print("            (including transient concept activations) and pairing")
    print("            that union with the other pool's union produces")
    print("            sparse, discriminative concept-to-concept cross-edges.")

    p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0, passes=30)

    class InlinePool(Pool):
        def __init__(self, params):
            super().__init__(params)
            self.pair_active_history = []  # labels active across whole pair

        def consume_chars(self, text, sentinel=None):
            self.pair_active_history = []
            super().consume_chars(text, sentinel)
            # capture per-step active labels by snapshotting after each char
            # (we approximate by re-running with snapshot collection)
            # For simplicity: after super(), promote, then snapshot all activations.
            self.end_pair()
            for lbl_i, _ in enumerate(self.id_to_label):
                if lbl_i in self.activation and self.activation[lbl_i] > 0.1:
                    self.pair_active_history.append(self.id_to_label[lbl_i])
            # Also walk the buffer history: any label that ended up in the
            # buffer during this pair will be in label_to_id with id <= current
            # (good enough for diagnostic).

    pin = InlinePool(p); pout = InlinePool(p); cross_b = CrossPool(mode='bag')
    for _ in range(p['passes']):
        for q, a in CONVERSATIONS:
            pin.reset_runtime(); pout.reset_runtime()
            pin.consume_chars(q); pout.consume_chars(a)
            in_active = {lbl: 1.0 for lbl in pin.pair_active_history}
            out_active = {lbl: 1.0 for lbl in pout.pair_active_history}
            cross_b.pair_bag(in_active, out_active, p['lr_cross'])
    report("inline+persistent (bag)", pin, pout, cross_b, CONVERSATIONS, sample=3)

    # Combined: inline + tensor
    print("\n  Combined: inline-promotion + tensor cross-pool")
    pin = InlinePool(p); pout = InlinePool(p); cross_t = TensorCrossPool()
    for _ in range(p['passes']):
        for q, a in CONVERSATIONS:
            pin.reset_runtime(); pout.reset_runtime()
            pin.consume_chars(q); pout.consume_chars(a)
            in_seq = pin.pair_active_history
            out_seq = pout.pair_active_history
            cross_t.pair_tensor(in_seq, out_seq, p['lr_cross'], sigma=3.0)
    fwd_scores, rev_scores = [], []
    for q, a in CONVERSATIONS:
        in_act = [(f"a:{c}", 1.0) for c in q]
        out_proj = cross_t.project(in_act, direction='forward')
        decoded = []
        for j in sorted(out_proj.keys()):
            if not out_proj[j]: continue
            lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
            if lbl.startswith("a:"): decoded.append(lbl[2:])
        fwd_scores.append(lev_sim("".join(decoded), a))
        out_in = [(f"a:{c}", 1.0) for c in a]
        in_proj = cross_t.project(out_in, direction='reverse')
        decoded = []
        for i in sorted(in_proj.keys()):
            if not in_proj[i]: continue
            lbl, _ = max(in_proj[i].items(), key=lambda x: x[1])
            if lbl.startswith("a:"): decoded.append(lbl[2:])
        rev_scores.append(lev_sim("".join(decoded), q))
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    cin = sum(1 for x in pin.is_concept if x); cout = sum(1 for x in pout.is_concept if x)
    print(f"    fwd={fwd:.3f} rev={rev:.3f} combined={(fwd+rev)/2:.3f}  cin={cin} cout={cout}")
    print("    samples:")
    for q, a in CONVERSATIONS[:5]:
        in_act = [(f"a:{c}", 1.0) for c in q]
        out_proj = cross_t.project(in_act, direction='forward')
        decoded = []
        for j in sorted(out_proj.keys()):
            if not out_proj[j]: continue
            lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
            if lbl.startswith("a:"): decoded.append(lbl[2:])
        print(f"      {q!r:25s} -> {''.join(decoded)!r:50s}  (want {a[:30]!r})")


def experiment_i_pair_signature():
    """Pair-level signature: each input atom-sequence creates a unique fixed
       activation pattern; cross-pool maps signature->signature."""
    print("\n" + "=" * 72)
    print("EXPERIMENT I: pair-signature key-value memory")
    print("=" * 72)
    print("Hypothesis: when each input pair produces a unique activation")
    print("            signature in pool_in (achieved by atom-bigram features),")
    print("            cross-pool becomes a signature->signature lookup that")
    print("            faithfully retrieves the trained pair.")

    def signature(text, n=3):
        # n-gram set as a bag of features
        sig = set()
        for i in range(len(text) - n + 1):
            sig.add(text[i:i+n])
        return sig

    # Train: store mapping from input sigs to output sigs, weighted
    fwd_table = defaultdict(lambda: defaultdict(float))
    rev_table = defaultdict(lambda: defaultdict(float))
    for q, a in CONVERSATIONS:
        sq = signature(q); sa = signature(a)
        for fi in sq:
            for fo in sa:
                fwd_table[fi][fo] += 1.0
                rev_table[fo][fi] += 1.0

    # Query: project query sig through fwd_table, rank candidate output pairs
    # by total feature overlap with the projected output sig.
    fwd_scores, rev_scores = [], []
    for q, a in CONVERSATIONS:
        sq = signature(q)
        # projected output features
        proj = defaultdict(float)
        for fi in sq:
            for fo, w in fwd_table.get(fi, {}).items():
                proj[fo] += w
        # rank training outputs by overlap
        best_out = None; best_score = -1.0
        for _, candidate_a in CONVERSATIONS:
            sc = signature(candidate_a)
            score = sum(proj.get(f, 0.0) for f in sc)
            if score > best_score:
                best_score = score; best_out = candidate_a
        fwd_scores.append(lev_sim(best_out, a))

        sa = signature(a)
        proj = defaultdict(float)
        for fo in sa:
            for fi, w in rev_table.get(fo, {}).items():
                proj[fi] += w
        best_in = None; best_score = -1.0
        for candidate_q, _ in CONVERSATIONS:
            sq2 = signature(candidate_q)
            score = sum(proj.get(f, 0.0) for f in sq2)
            if score > best_score:
                best_score = score; best_in = candidate_q
        rev_scores.append(lev_sim(best_in, q))
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    print(f"  trigram signature retrieval:  fwd={fwd:.3f} rev={rev:.3f}  combined={(fwd+rev)/2:.3f}")


def experiment_j_sigma_sweep():
    """Fine sigma sweep on the tensor cross-pool to find the optimum."""
    print("\n" + "=" * 72)
    print("EXPERIMENT J: fine sigma sweep on directional tensor cross-pool")
    print("=" * 72)
    sigmas = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 50.0]
    for sigma in sigmas:
        cross = TensorCrossPool()
        for _ in range(BASE_PARAMS['passes']):
            for q, a in CONVERSATIONS:
                in_seq = [f"a:{c}" for c in q]
                out_seq = [f"a:{c}" for c in a]
                cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=sigma)
        fwd_scores, rev_scores = [], []
        for q, a in CONVERSATIONS:
            in_act = [(f"a:{c}", 1.0) for c in q]
            out_proj = cross.project(in_act, direction='forward')
            decoded = []
            for j in sorted(out_proj.keys()):
                if not out_proj[j]: continue
                lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            fwd_scores.append(lev_sim("".join(decoded), a))
            out_in = [(f"a:{c}", 1.0) for c in a]
            in_proj = cross.project(out_in, direction='reverse')
            decoded = []
            for i in sorted(in_proj.keys()):
                if not in_proj[i]: continue
                lbl, _ = max(in_proj[i].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            rev_scores.append(lev_sim("".join(decoded), q))
        fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
        edges = sum(len(v) for v in cross.forward.values())
        print(f"  sigma={sigma:5.1f}  fwd={fwd:.3f} rev={rev:.3f}  combined={(fwd+rev)/2:.3f}  edges={edges}")


def experiment_k_passes_sweep():
    """Hold tensor cross-pool fixed and sweep passes."""
    print("\n" + "=" * 72)
    print("EXPERIMENT K: passes vs recall fidelity (tensor cross-pool, sigma=6)")
    print("=" * 72)
    for passes in [1, 5, 10, 20, 30, 60, 120]:
        cross = TensorCrossPool()
        for _ in range(passes):
            for q, a in CONVERSATIONS:
                in_seq = [f"a:{c}" for c in q]
                out_seq = [f"a:{c}" for c in a]
                cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=6.0)
        fwd_scores, rev_scores = [], []
        for q, a in CONVERSATIONS:
            in_act = [(f"a:{c}", 1.0) for c in q]
            out_proj = cross.project(in_act, direction='forward')
            decoded = []
            for j in sorted(out_proj.keys()):
                if not out_proj[j]: continue
                lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            fwd_scores.append(lev_sim("".join(decoded), a))
            out_in = [(f"a:{c}", 1.0) for c in a]
            in_proj = cross.project(out_in, direction='reverse')
            decoded = []
            for i in sorted(in_proj.keys()):
                if not in_proj[i]: continue
                lbl, _ = max(in_proj[i].items(), key=lambda x: x[1])
                if lbl.startswith("a:"): decoded.append(lbl[2:])
            rev_scores.append(lev_sim("".join(decoded), q))
        fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
        print(f"  passes={passes:3d}  fwd={fwd:.3f} rev={rev:.3f}  combined={(fwd+rev)/2:.3f}")


def make_ngrams(text, n):
    out = []
    pad = "^" * (n - 1)
    s = pad + text
    for i in range(len(text)):
        out.append(s[i:i + n])
    return out


def experiment_l_ngram_tensor():
    """Use n-gram features as keys instead of single chars.
       Each input position has a trigram context; same for output."""
    print("\n" + "=" * 72)
    print("EXPERIMENT L: n-gram feature keys in tensor cross-pool")
    print("=" * 72)
    print("Hypothesis: trigram-keyed entries are more discriminative than")
    print("            single-char keys, closing the gap to the 0.81 ceiling.")
    for n in [2, 3, 4, 5]:
        for sigma in [4.0, 8.0, 16.0]:
            cross = TensorCrossPool()
            for q, a in CONVERSATIONS:
                in_seq = [f"g{n}:{g}" for g in make_ngrams(q, n)]
                out_seq = [f"g{n}:{g}" for g in make_ngrams(a, n)]
                cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=sigma)
            fwd_scores, rev_scores = [], []
            for q, a in CONVERSATIONS:
                in_act = [(f"g{n}:{g}", 1.0) for g in make_ngrams(q, n)]
                out_proj = cross.project(in_act, direction='forward')
                # decode: at each output position pick best n-gram, take last char
                decoded = []
                for j in sorted(out_proj.keys()):
                    if not out_proj[j]: continue
                    lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
                    if lbl.startswith(f"g{n}:"):
                        gram = lbl[len(f"g{n}:"):]
                        if gram and gram[-1] != "^":
                            decoded.append(gram[-1])
                pred_a = "".join(decoded)
                fwd_scores.append(lev_sim(pred_a, a))

                out_in = [(f"g{n}:{g}", 1.0) for g in make_ngrams(a, n)]
                in_proj = cross.project(out_in, direction='reverse')
                decoded = []
                for i in sorted(in_proj.keys()):
                    if not in_proj[i]: continue
                    lbl, _ = max(in_proj[i].items(), key=lambda x: x[1])
                    if lbl.startswith(f"g{n}:"):
                        gram = lbl[len(f"g{n}:"):]
                        if gram and gram[-1] != "^":
                            decoded.append(gram[-1])
                pred_q = "".join(decoded)
                rev_scores.append(lev_sim(pred_q, q))
            fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
            print(f"  n={n} sigma={sigma:5.1f}  fwd={fwd:.3f} rev={rev:.3f}  combined={(fwd+rev)/2:.3f}")
    # Sample with best config (n=3, sigma=8)
    print("\n  samples (n=3, sigma=8):")
    cross = TensorCrossPool()
    for q, a in CONVERSATIONS:
        in_seq = [f"g3:{g}" for g in make_ngrams(q, 3)]
        out_seq = [f"g3:{g}" for g in make_ngrams(a, 3)]
        cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=8.0)
    for q, a in CONVERSATIONS[:5]:
        in_act = [(f"g3:{g}", 1.0) for g in make_ngrams(q, 3)]
        out_proj = cross.project(in_act, direction='forward')
        decoded = []
        for j in sorted(out_proj.keys()):
            if not out_proj[j]: continue
            lbl, _ = max(out_proj[j].items(), key=lambda x: x[1])
            if lbl.startswith("g3:"):
                gram = lbl[3:]
                if gram and gram[-1] != "^": decoded.append(gram[-1])
        print(f"    {q!r:25s} -> {''.join(decoded)!r:50s}  (want {a[:30]!r})")


def experiment_m_viterbi_decoder():
    """Decode by Viterbi over output position: choose label sequence that
       maximizes (cross_pool projection score) * (within-pool STDP coherence)."""
    print("\n" + "=" * 72)
    print("EXPERIMENT M: Viterbi decoder using within-pool STDP edges")
    print("=" * 72)
    # Build a within-pool transition table for pool_out from training
    transitions = defaultdict(lambda: defaultdict(float))
    for q, a in CONVERSATIONS:
        for i in range(len(a) - 1):
            transitions[a[i]][a[i + 1]] += 1.0

    cross = TensorCrossPool()
    for q, a in CONVERSATIONS:
        in_seq = [f"a:{c}" for c in q]
        out_seq = [f"a:{c}" for c in a]
        cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=8.0)

    fwd_scores = []
    for q, a in CONVERSATIONS:
        in_act = [(f"a:{c}", 1.0) for c in q]
        out_proj = cross.project(in_act, direction='forward')
        positions = sorted(out_proj.keys())
        if not positions:
            fwd_scores.append(0.0); continue
        # Build candidate set per position: top-K labels
        K = 5
        cands_per_pos = []
        for p in positions:
            ranked = sorted(out_proj[p].items(), key=lambda x: -x[1])[:K]
            cands_per_pos.append(ranked)
        # Viterbi
        prev_score = {lbl: math.log(max(s, 1e-9)) for lbl, s in cands_per_pos[0]}
        prev_path = {lbl: [lbl] for lbl, _ in cands_per_pos[0]}
        for p in range(1, len(cands_per_pos)):
            new_score = {}; new_path = {}
            for lbl, s in cands_per_pos[p]:
                # find best predecessor
                best = (-1e9, None, None)
                for plbl, pscore in prev_score.items():
                    pchar = plbl[2:] if plbl.startswith("a:") else plbl
                    cchar = lbl[2:] if lbl.startswith("a:") else lbl
                    trans = transitions.get(pchar, {}).get(cchar, 0.0)
                    score = pscore + math.log(max(s, 1e-9)) + 0.5 * math.log(trans + 1.0)
                    if score > best[0]:
                        best = (score, plbl, plbl)
                new_score[lbl] = best[0]
                new_path[lbl] = prev_path[best[1]] + [lbl]
            prev_score = new_score; prev_path = new_path
        # pick best ending
        best_lbl = max(prev_score, key=lambda x: prev_score[x])
        decoded = []
        for lbl in prev_path[best_lbl]:
            if lbl.startswith("a:"): decoded.append(lbl[2:])
        fwd_scores.append(lev_sim("".join(decoded), a))
    print(f"  fwd (Viterbi)={sum(fwd_scores)/len(fwd_scores):.3f}")
    # Show samples
    print("  samples:")
    for (q, a), s in list(zip(CONVERSATIONS, fwd_scores))[:5]:
        in_act = [(f"a:{c}", 1.0) for c in q]
        out_proj = cross.project(in_act, direction='forward')
        positions = sorted(out_proj.keys())
        if not positions: continue
        K = 5
        cands_per_pos = []
        for p in positions:
            ranked = sorted(out_proj[p].items(), key=lambda x: -x[1])[:K]
            cands_per_pos.append(ranked)
        prev_score = {lbl: math.log(max(ss, 1e-9)) for lbl, ss in cands_per_pos[0]}
        prev_path = {lbl: [lbl] for lbl, _ in cands_per_pos[0]}
        for p in range(1, len(cands_per_pos)):
            new_score = {}; new_path = {}
            for lbl, ss in cands_per_pos[p]:
                best = (-1e9, None)
                for plbl, pscore in prev_score.items():
                    pchar = plbl[2:] if plbl.startswith("a:") else plbl
                    cchar = lbl[2:] if lbl.startswith("a:") else lbl
                    trans = transitions.get(pchar, {}).get(cchar, 0.0)
                    score = pscore + math.log(max(ss, 1e-9)) + 0.5 * math.log(trans + 1.0)
                    if score > best[0]: best = (score, plbl)
                new_score[lbl] = best[0]
                new_path[lbl] = prev_path[best[1]] + [lbl]
            prev_score = new_score; prev_path = new_path
        best_lbl = max(prev_score, key=lambda x: prev_score[x])
        decoded = "".join(lbl[2:] for lbl in prev_path[best_lbl] if lbl.startswith("a:"))
        print(f"    {q!r:25s} -> {decoded!r:50s} (sim={s:.2f}, want {a[:30]!r})")


def experiment_n_combined_ngram_viterbi():
    """N-gram tensor + Viterbi decode using within-pool n-gram transitions."""
    print("\n" + "=" * 72)
    print("EXPERIMENT N: n-gram tensor + Viterbi decode")
    print("=" * 72)
    n = 3
    sigma_fwd = 8.0
    sigma_rev = 20.0
    # within-pool n-gram transitions
    trans_out = defaultdict(lambda: defaultdict(float))
    trans_in  = defaultdict(lambda: defaultdict(float))
    for q, a in CONVERSATIONS:
        gs = make_ngrams(a, n)
        for i in range(len(gs) - 1):
            trans_out[gs[i]][gs[i + 1]] += 1.0
        gs = make_ngrams(q, n)
        for i in range(len(gs) - 1):
            trans_in[gs[i]][gs[i + 1]] += 1.0

    cross = TensorCrossPool()
    for q, a in CONVERSATIONS:
        in_seq = [f"g:{g}" for g in make_ngrams(q, n)]
        out_seq = [f"g:{g}" for g in make_ngrams(a, n)]
        cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=sigma_fwd)

    def viterbi_decode(in_text, direction, expected_len, transitions):
        in_act = [(f"g:{g}", 1.0) for g in make_ngrams(in_text, n)]
        out_proj = cross.project(in_act, direction=direction)
        positions = sorted(out_proj.keys())
        if not positions: return ""
        K = 8
        cands_per_pos = []
        for p in positions:
            ranked = sorted(out_proj[p].items(), key=lambda x: -x[1])[:K]
            cands_per_pos.append(ranked)
        prev_score = {lbl: math.log(max(s, 1e-9)) for lbl, s in cands_per_pos[0]}
        prev_path = {lbl: [lbl] for lbl, _ in cands_per_pos[0]}
        for p in range(1, len(cands_per_pos)):
            new_score = {}; new_path = {}
            for lbl, s in cands_per_pos[p]:
                gram = lbl[2:] if lbl.startswith("g:") else lbl
                best = (-1e9, None)
                for plbl, pscore in prev_score.items():
                    pgram = plbl[2:] if plbl.startswith("g:") else plbl
                    trans = transitions.get(pgram, {}).get(gram, 0.0)
                    score = pscore + math.log(max(s, 1e-9)) + 1.0 * math.log(trans + 1.0)
                    if score > best[0]: best = (score, plbl)
                new_score[lbl] = best[0]
                new_path[lbl] = prev_path[best[1]] + [lbl]
            prev_score = new_score; prev_path = new_path
        best_lbl = max(prev_score, key=lambda x: prev_score[x])
        decoded = []
        for lbl in prev_path[best_lbl]:
            if lbl.startswith("g:"):
                gram = lbl[2:]
                if gram and gram[-1] != "^": decoded.append(gram[-1])
        return "".join(decoded)[:expected_len]

    fwd_scores, rev_scores = [], []
    for q, a in CONVERSATIONS:
        fwd_scores.append(lev_sim(viterbi_decode(q, 'forward', len(a), trans_out), a))
        rev_scores.append(lev_sim(viterbi_decode(a, 'reverse', len(q), trans_in), q))
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    print(f"  fwd={fwd:.3f} rev={rev:.3f} combined={(fwd+rev)/2:.3f}")
    print("  forward samples:")
    for q, a in CONVERSATIONS[:6]:
        decoded = viterbi_decode(q, 'forward', len(a), trans_out)
        print(f"    {q!r:25s} -> {decoded!r}")
    print("  reverse samples:")
    for q, a in CONVERSATIONS[:6]:
        decoded = viterbi_decode(a, 'reverse', len(q), trans_in)
        print(f"    {a[:30]!r:32s} -> {decoded!r}")


def experiment_o_concept_aware_tensor():
    """Tensor entries for both atoms AND concepts that fired during the pair.
       Concepts get a single virtual position (start of pair)."""
    print("\n" + "=" * 72)
    print("EXPERIMENT O: concept-aware tensor (atoms + concepts as entries)")
    print("=" * 72)
    p = dict(BASE_PARAMS, theta_n=2, theta_branch=0.0, passes=10)

    pin = Pool(p); pout = Pool(p); cross = TensorCrossPool()
    for _ in range(p['passes']):
        for q, a in CONVERSATIONS:
            pin.reset_runtime(); pout.reset_runtime()
            pin.consume_chars(q); pout.consume_chars(a)
            pin.end_pair(); pout.end_pair()
            in_seq = [f"a:{c}" for c in q]
            out_seq = [f"a:{c}" for c in a]
            # Concepts that fired in pool_in / pool_out get virtual entries
            in_concepts = [pin.id_to_label[i] for i, v in pin.activation.items()
                           if pin.is_concept[i] and v > 0.1]
            out_concepts = [pout.id_to_label[i] for i, v in pout.activation.items()
                            if pout.is_concept[i] and v > 0.1]
            full_in = in_seq + in_concepts
            full_out = out_seq + out_concepts
            cross.pair_tensor(full_in, full_out, p['lr_cross'], sigma=8.0)
    fwd_scores, rev_scores = [], []
    for q, a in CONVERSATIONS:
        pin.reset_runtime(); pout.reset_runtime()
        pin.stimulate_chars(q)
        pin.propagate(steps=2)
        in_act = []
        # use atoms in order
        for c in q:
            in_act.append((f"a:{c}", 1.0))
        # plus active concepts
        for i, v in pin.activation.items():
            if pin.is_concept[i] and v > 0.1:
                in_act.append((pin.id_to_label[i], v))
        out_proj = cross.project(in_act, direction='forward')
        decoded = []
        for j in sorted(out_proj.keys()):
            if not out_proj[j]: continue
            # filter to atoms only when decoding to chars
            atoms_only = {k: v for k, v in out_proj[j].items() if k.startswith("a:")}
            if not atoms_only: continue
            lbl, _ = max(atoms_only.items(), key=lambda x: x[1])
            decoded.append(lbl[2:])
        fwd_scores.append(lev_sim("".join(decoded), a))

        pin.reset_runtime(); pout.reset_runtime()
        pout.stimulate_chars(a); pout.propagate(steps=2)
        out_in = [(f"a:{c}", 1.0) for c in a]
        for i, v in pout.activation.items():
            if pout.is_concept[i] and v > 0.1:
                out_in.append((pout.id_to_label[i], v))
        in_proj = cross.project(out_in, direction='reverse')
        decoded = []
        for i in sorted(in_proj.keys()):
            if not in_proj[i]: continue
            atoms_only = {k: v for k, v in in_proj[i].items() if k.startswith("a:")}
            if not atoms_only: continue
            lbl, _ = max(atoms_only.items(), key=lambda x: x[1])
            decoded.append(lbl[2:])
        rev_scores.append(lev_sim("".join(decoded), q))
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    print(f"  atoms+concepts in tensor:  fwd={fwd:.3f} rev={rev:.3f} combined={(fwd+rev)/2:.3f}")


def experiment_p_pair_concepts():
    """Each pair forms a unique pair-concept neuron in each pool.
       Cross-pool gets a strong concept->concept edge.
       Query path: stimulate atoms -> fire pair-concept (member-vote) ->
                   route through cross-pool -> walk members verbatim."""
    print("\n" + "=" * 72)
    print("EXPERIMENT P: pair-concept neurons + concept-to-concept cross-pool")
    print("=" * 72)
    print("Hypothesis: a unique pair-concept per pair gives an unambiguous")
    print("            firing signature; cross-pool routes concept->concept;")
    print("            walking the target concept's members emits the trained")
    print("            output verbatim. Fitness should approach 1.0.")

    class PairPool:
        def __init__(self):
            self.label_to_id = {}
            self.id_to_label = []
            self.is_concept = []
            self.members = {}
            self.activation = defaultdict(float)

        def ensure(self, label, is_concept=False, members=None):
            if label in self.label_to_id: return self.label_to_id[label]
            i = len(self.id_to_label)
            self.label_to_id[label] = i
            self.id_to_label.append(label)
            self.is_concept.append(is_concept)
            if members is not None: self.members[i] = list(members)
            return i

    pin = PairPool(); pout = PairPool()
    pair_in_concepts = []   # parallel lists indexed by pair index
    pair_out_concepts = []
    for idx, (q, a) in enumerate(CONVERSATIONS):
        in_atom_ids = [pin.ensure(f"a:{c}") for c in q]
        out_atom_ids = [pout.ensure(f"a:{c}") for c in a]
        in_concept_label = f"P:{idx}:{q}"
        out_concept_label = f"P:{idx}:{a}"
        in_cid = pin.ensure(in_concept_label, is_concept=True, members=in_atom_ids)
        out_cid = pout.ensure(out_concept_label, is_concept=True, members=out_atom_ids)
        pair_in_concepts.append(in_cid)
        pair_out_concepts.append(out_cid)

    # Query: stimulate atoms in source, score each pair-concept by fraction of
    # its members that are in the stimulated set.  Pick highest. Cross-pool
    # routes concept_in[idx] -> concept_out[idx]. Walk members.
    def query_forward(q):
        stim = set(pin.label_to_id[f"a:{c}"] for c in q if f"a:{c}" in pin.label_to_id)
        best_idx = None; best_score = -1.0
        for idx in range(len(CONVERSATIONS)):
            cid = pair_in_concepts[idx]
            members = pin.members[cid]
            if not members: continue
            hits = sum(1 for m in members if m in stim)
            score = hits / len(members)
            # also penalize for members of stim NOT in concept (handles "hellp" robustly)
            extras = sum(1 for m in stim if m not in members)
            score = score - 0.05 * extras
            if score > best_score:
                best_score = score; best_idx = idx
        if best_idx is None: return ""
        out_cid = pair_out_concepts[best_idx]
        return "".join(pout.id_to_label[m][2:] for m in pout.members[out_cid]
                       if pout.id_to_label[m].startswith("a:"))

    def query_reverse(a):
        stim = set(pout.label_to_id[f"a:{c}"] for c in a if f"a:{c}" in pout.label_to_id)
        best_idx = None; best_score = -1.0
        for idx in range(len(CONVERSATIONS)):
            cid = pair_out_concepts[idx]
            members = pout.members[cid]
            if not members: continue
            hits = sum(1 for m in members if m in stim)
            score = hits / len(members)
            extras = sum(1 for m in stim if m not in members)
            score = score - 0.05 * extras
            if score > best_score:
                best_score = score; best_idx = idx
        if best_idx is None: return ""
        in_cid = pair_in_concepts[best_idx]
        return "".join(pin.id_to_label[m][2:] for m in pin.members[in_cid]
                       if pin.id_to_label[m].startswith("a:"))

    fwd_scores, rev_scores = [], []
    exact_fwd = 0; exact_rev = 0
    for q, a in CONVERSATIONS:
        pf = query_forward(q)
        pr = query_reverse(a)
        fwd_scores.append(lev_sim(pf, a))
        rev_scores.append(lev_sim(pr, q))
        if pf == a: exact_fwd += 1
        if pr == q: exact_rev += 1
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    print(f"  fwd={fwd:.3f} rev={rev:.3f} combined={(fwd+rev)/2:.3f}")
    print(f"  exact matches: fwd={exact_fwd}/{len(CONVERSATIONS)}  rev={exact_rev}/{len(CONVERSATIONS)}")
    print("  forward samples:")
    for q, a in CONVERSATIONS[:6]:
        print(f"    {q!r:25s} -> {query_forward(q)!r}")
    print("  reverse samples:")
    for q, a in CONVERSATIONS[:6]:
        print(f"    {a[:30]!r:32s} -> {query_reverse(a)!r}")


def experiment_q_pair_concept_robustness():
    """Stress-test pair-concept retrieval with perturbed queries:
       capitalization changes, typos, prefix-only, partial input."""
    print("\n" + "=" * 72)
    print("EXPERIMENT Q: pair-concept robustness (typos, partial inputs)")
    print("=" * 72)
    class PairPool:
        def __init__(self):
            self.label_to_id = {}; self.id_to_label = []
            self.is_concept = []; self.members = {}
        def ensure(self, label, is_concept=False, members=None):
            if label in self.label_to_id: return self.label_to_id[label]
            i = len(self.id_to_label)
            self.label_to_id[label] = i
            self.id_to_label.append(label)
            self.is_concept.append(is_concept)
            if members is not None: self.members[i] = list(members)
            return i

    pin = PairPool(); pout = PairPool()
    pair_in_concepts = []; pair_out_concepts = []
    for idx, (q, a) in enumerate(CONVERSATIONS):
        in_atom_ids = [pin.ensure(f"a:{c}") for c in q]
        out_atom_ids = [pout.ensure(f"a:{c}") for c in a]
        in_cid = pin.ensure(f"P:{idx}:{q}", is_concept=True, members=in_atom_ids)
        out_cid = pout.ensure(f"P:{idx}:{a}", is_concept=True, members=out_atom_ids)
        pair_in_concepts.append(in_cid); pair_out_concepts.append(out_cid)

    def retrieve(query):
        stim = set(pin.label_to_id[f"a:{c}"] for c in query if f"a:{c}" in pin.label_to_id)
        best_idx = None; best_score = -1.0
        for idx in range(len(CONVERSATIONS)):
            cid = pair_in_concepts[idx]
            members = pin.members[cid]
            if not members: continue
            hits = sum(1 for m in members if m in stim)
            score = hits / len(members)
            extras = sum(1 for m in stim if m not in members)
            score = score - 0.05 * extras
            if score > best_score:
                best_score = score; best_idx = idx
        if best_idx is None: return None, 0.0
        out_cid = pair_out_concepts[best_idx]
        return "".join(pout.id_to_label[m][2:] for m in pout.members[out_cid]
                       if pout.id_to_label[m].startswith("a:")), best_score

    perturbations = [
        ("hello", "hello"),    # exact
        ("Hello", "hello"),     # capitalisation
        ("helo",  "hello"),     # typo (drop one char)
        ("hellp", "hello"),     # typo (substitution)
        ("hi",    "hi"),
        ("hii",   "hi"),
        ("hey there", "hey"),   # extra chars
        ("how r u", "how are you"),  # phonetic
        ("good morn",  "good morning"),
        ("thnks",      "thanks"),
        ("xyz",         None),  # garbage
    ]
    for q, expected_match in perturbations:
        ans, score = retrieve(q)
        match = (CONVERSATIONS[0][1] if expected_match == "hello" else
                 dict(CONVERSATIONS).get(expected_match) if expected_match else None)
        ok = (ans == match) if match else "n/a"
        print(f"  query={q!r:18s} -> top-score={score:.2f}  ans={ans[:40]!r}  expected match: {expected_match!r}  ok={ok}")


def experiment_r_pair_concept_with_tensor_fallback():
    """Combine pair-concept retrieval (high-confidence exact-match) with
       n-gram tensor + Viterbi as a graceful fallback for off-distribution
       queries, eliminating the binary 'either match or garbage' edge."""
    print("\n" + "=" * 72)
    print("EXPERIMENT R: pair-concept + n-gram tensor fallback")
    print("=" * 72)
    # Pair-concept layer
    class PairPool:
        def __init__(self):
            self.label_to_id = {}; self.id_to_label = []
            self.is_concept = []; self.members = {}
        def ensure(self, label, is_concept=False, members=None):
            if label in self.label_to_id: return self.label_to_id[label]
            i = len(self.id_to_label)
            self.label_to_id[label] = i; self.id_to_label.append(label)
            self.is_concept.append(is_concept)
            if members is not None: self.members[i] = list(members)
            return i
    pin = PairPool(); pout = PairPool()
    pair_in = []; pair_out = []
    for idx, (q, a) in enumerate(CONVERSATIONS):
        ia = [pin.ensure(f"a:{c}") for c in q]
        oa = [pout.ensure(f"a:{c}") for c in a]
        ic = pin.ensure(f"P:{idx}", is_concept=True, members=ia)
        oc = pout.ensure(f"P:{idx}", is_concept=True, members=oa)
        pair_in.append(ic); pair_out.append(oc)

    # Tensor fallback
    cross = TensorCrossPool()
    for q, a in CONVERSATIONS:
        in_seq = [f"g:{g}" for g in make_ngrams(q, 3)]
        out_seq = [f"g:{g}" for g in make_ngrams(a, 3)]
        cross.pair_tensor(in_seq, out_seq, BASE_PARAMS['lr_cross'], sigma=8.0)
    trans_out = defaultdict(lambda: defaultdict(float))
    trans_in  = defaultdict(lambda: defaultdict(float))
    for q, a in CONVERSATIONS:
        gs = make_ngrams(a, 3)
        for i in range(len(gs) - 1): trans_out[gs[i]][gs[i+1]] += 1
        gs = make_ngrams(q, 3)
        for i in range(len(gs) - 1): trans_in[gs[i]][gs[i+1]] += 1

    def viterbi_decode(text, direction, transitions):
        in_act = [(f"g:{g}", 1.0) for g in make_ngrams(text, 3)]
        out_proj = cross.project(in_act, direction=direction)
        positions = sorted(out_proj.keys())
        if not positions: return ""
        K = 8
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
                    score = ps + math.log(max(s, 1e-9)) + math.log(tr + 1.0)
                    if score > best[0]: best = (score, pl)
                ns[l] = best[0]; np[l] = prev_path[best[1]] + [l]
            prev_score = ns; prev_path = np
        bl = max(prev_score, key=lambda x: prev_score[x])
        decoded = []
        for l in prev_path[bl]:
            gram = l[2:]
            if gram and gram[-1] != "^": decoded.append(gram[-1])
        return "".join(decoded)

    def query(text, direction, threshold=0.6):
        stim_pool = pin if direction == 'forward' else pout
        target_pool = pout if direction == 'forward' else pin
        target_concepts = pair_out if direction == 'forward' else pair_in
        stim = set(stim_pool.label_to_id[f"a:{c}"] for c in text
                   if f"a:{c}" in stim_pool.label_to_id)
        best_idx = None; best_score = -1.0
        source_concepts = pair_in if direction == 'forward' else pair_out
        for idx in range(len(CONVERSATIONS)):
            cid = source_concepts[idx]
            members = stim_pool.members[cid]
            if not members: continue
            hits = sum(1 for m in members if m in stim)
            score = hits / len(members)
            extras = sum(1 for m in stim if m not in members)
            score = score - 0.05 * extras
            if score > best_score: best_score = score; best_idx = idx
        if best_idx is None or best_score < threshold:
            # fall back to tensor
            return viterbi_decode(text, direction,
                                   trans_out if direction == 'forward' else trans_in)
        cid = target_concepts[best_idx]
        return "".join(target_pool.id_to_label[m][2:] for m in target_pool.members[cid]
                       if target_pool.id_to_label[m].startswith("a:"))

    fwd_scores, rev_scores = [], []
    exact_fwd = 0; exact_rev = 0
    for q, a in CONVERSATIONS:
        pf = query(q, 'forward'); pr = query(a, 'reverse')
        fwd_scores.append(lev_sim(pf, a)); rev_scores.append(lev_sim(pr, q))
        if pf == a: exact_fwd += 1
        if pr == q: exact_rev += 1
    fwd = sum(fwd_scores)/len(fwd_scores); rev = sum(rev_scores)/len(rev_scores)
    print(f"  fwd={fwd:.3f} rev={rev:.3f} combined={(fwd+rev)/2:.3f}")
    print(f"  exact matches: fwd={exact_fwd}/{len(CONVERSATIONS)}  rev={exact_rev}/{len(CONVERSATIONS)}")
    print("  forward samples:")
    for q, a in CONVERSATIONS:
        print(f"    {q!r:25s} -> {query(q, 'forward')!r}")
    print("  perturbed query handling:")
    for q in ["Hello", "helo", "hellp", "hi!", "hey there", "good morn", "xyz"]:
        ans = query(q, 'forward')
        print(f"    {q!r:18s} -> {ans!r}")


def main():
    experiment_a_threshold_relaxation()
    experiment_b_positional_cross_pool()
    experiment_c_concept_only_cross()
    experiment_d_sentinel()
    experiment_e_hierarchy_depth()
    experiment_f_combined_best()
    experiment_g_directional_tensor()
    experiment_h_inline_concept_promotion()
    experiment_i_pair_signature()
    experiment_j_sigma_sweep()
    experiment_k_passes_sweep()
    experiment_l_ngram_tensor()
    experiment_m_viterbi_decoder()
    experiment_n_combined_ngram_viterbi()
    experiment_o_concept_aware_tensor()
    experiment_p_pair_concepts()
    experiment_q_pair_concept_robustness()
    experiment_r_pair_concept_with_tensor_fallback()


if __name__ == "__main__":
    main()
