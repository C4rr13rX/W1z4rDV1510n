#!/usr/bin/env python3
"""
ga_neuro_config_search.py — adaptive GA over the ACTUAL NeuroConfig knobs
that exist in crates/core/src/neuro.rs, against a faithful Python port of
the existing Rust two_pool_train_pair / two_pool_query algorithm.

What's faithfully ported (matches crates/core/src/neuro.rs):
  - NeuronPool: per-neuron activation, trace, fatigue, use_count,
    excitatory/inhibitory synapse lists kept sorted by target.
  - train_weighted: per-symbol activation = (1 - fatigue), trace += 0.1,
    HEBBIAN_WINDOW = 20, MAX_PAIRS = 8000, scale = excitatory_scale * lr_scale.
    Asymmetric STDP via hebbian_pair: base = hebbian_lr * (a.act+a.trace)
    * (b.act+b.trace) * scale; LTP forward (×stdp_pre_post=1.0), LTD backward
    (×stdp_post_pre=-0.3) only on existing synapses.
  - propagate_weighted: dense activation vector, fan-out normalization
    src_act * weight * 0.5 / sqrt(fan_out), inhibitory subtracts
    src * weight * 0.3, per-hop decay ×0.85, kWTA top-k =
    max(sdr_sparsity * n, sdr_k_min) with sub-threshold zeroed.
  - Mini-column promotion: minicolumn_counts keyed by sorted-3-label
    signature; when count >= minicolumn_threshold a concept neuron is
    spawned with those atoms as members. update_minicolumns sets stability
    EMA (decay=stability_decay), inhibition EMA (decay=inhibition_decay);
    when stability >= activation_threshold AND inhibition < collapse_threshold,
    members are scaled by minicolumn_inhibit_scale (0.35) before propagation.
  - two_pool_train_pair: pool_in.train_weighted, pool_out.train_weighted,
    cross_pool.pair(in_label, out_label, lr) for all (in_active × out_active)
    above 0.05 activation.  Symmetric label-keyed.
  - two_pool_query: seed atoms in source, propagate 2 hops, cross-project
    via forward/reverse, propagate hops>=3 in target, predecessor-pull
    start selection, greedy chain-walk along excitatory edges into the
    active-atom set.

GA optimises every knob that exists in NeuroConfig:
  hebbian_lr, decay, composite_threshold, inhibitory_scale, excitatory_scale,
  fatigue_increment, fatigue_decay, wta_k_per_zone (unused in two-pool path
  but kept), stdp_scale (unused), stdp_pre_post, stdp_post_pre, max_weight,
  sdr_sparsity, sdr_k_min, minicolumn_threshold, minicolumn_inhibit_scale,
  minicolumn_inhibition_decay, minicolumn_stability_decay,
  minicolumn_activation_threshold, minicolumn_collapse_threshold,
  homeostatic_target, homeostatic_period, homeostatic_lr.

Plus runtime args used by the two_pool path:
  query_hops, query_min_activation, train_lr.

Adaptive feedback: mutation strength shrinks with best fitness, +0.2 if
stagnant for 4 gens, random-restart slot when stuck.
"""
from __future__ import annotations
import math
import os
import random
import string
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field


# ─── Politeness: stay out of the way of other processes ────────────────────
# After a system lockup we want to be a good neighbor.  Lower this process
# priority on Windows (so other apps preempt easily) and keep a single thread.
# An adaptive yield reads system CPU pressure each generation and inserts a
# small sleep when load is high — backs off automatically, scales back up
# when the box is idle.
_HAS_PSUTIL = False
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore


def be_polite():
    """Lower process priority on Windows; cap thread libs to 1 thread.
    Also pin PYTHONHASHSEED=0 so set iteration order is reproducible across
    runs (otherwise the cross.pair() iteration order in train_pair varies
    and A/B baselines drift between invocations)."""
    # Single-thread numeric libs (we don't import numpy here, but harmless).
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    if sys.platform == "win32":
        try:
            import ctypes
            BELOW_NORMAL = 0x00004000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, BELOW_NORMAL)
            print("[polite] process priority lowered to BELOW_NORMAL",
                  flush=True)
        except Exception as e:
            print(f"[polite] could not lower priority: {e}", flush=True)
    else:
        try:
            os.nice(10)
            print("[polite] os.nice(10) applied", flush=True)
        except Exception as e:
            print(f"[polite] could not nice: {e}", flush=True)


def adaptive_yield():
    """Sleep proportionally to system CPU load — 0 when idle, up to ~0.2s
    when load is high.  Called between GA generations and between A/B
    configs so other processes can grab the CPU."""
    if not _HAS_PSUTIL:
        time.sleep(0.05)  # tiny unconditional yield
        return
    try:
        # Non-blocking sample (uses last delta).  Other processes' cpu pct
        # = total - our pct.  We don't have a clean view, so use total load
        # as proxy.  When >70% busy, sleep more.
        load = psutil.cpu_percent(interval=None) / 100.0
        # Map [0..1] to [0.02 .. 0.4]s.  At low load this is a courtesy
        # blink; at high load we actively step back.
        delay = 0.02 + 0.38 * max(0.0, min(1.0, load))
        time.sleep(delay)
    except Exception:
        time.sleep(0.05)


# ──────────────────────────────────────────────────────── Faithful Rust port

@dataclass
class NeuroConfig:
    hebbian_lr:                       float = 0.05
    decay:                            float = 0.99
    composite_threshold:              int   = 25
    inhibitory_scale:                 float = 0.5
    excitatory_scale:                 float = 1.0
    fatigue_increment:                float = 0.02
    fatigue_decay:                    float = 0.98
    wta_k_per_zone:                   int   = 4
    stdp_scale:                       float = 0.02
    stdp_pre_post:                    float = 1.0
    stdp_post_pre:                    float = -0.3
    max_weight:                       float = 4.0
    sdr_sparsity:                     float = 0.02
    sdr_k_min:                        int   = 50
    minicolumn_threshold:             int   = 18
    minicolumn_max:                   int   = 512
    minicolumn_inhibit_scale:         float = 0.35
    minicolumn_inhibition_decay:      float = 0.85
    minicolumn_stability_decay:       float = 0.9
    minicolumn_activation_threshold:  float = 0.65
    minicolumn_collapse_threshold:    float = 0.55
    homeostatic_target:               float = 0.10
    homeostatic_period:               int   = 500
    homeostatic_lr:                   float = 0.04


class Neuron:
    __slots__ = ("id", "label", "activation", "trace", "fatigue", "use_count",
                 "members", "excit", "inhib", "act_ema")
    def __init__(self, nid, label):
        self.id         = nid
        self.label      = label
        self.activation = 0.0
        self.trace      = 0.0
        self.fatigue    = 0.0
        self.use_count  = 0
        self.members    = []          # list of member neuron ids (composites)
        self.excit      = {}          # target_id -> weight (sorted via dict-insertion not strictly needed)
        self.inhib      = {}
        self.act_ema    = 0.0


class MiniColumn:
    __slots__ = ("signature", "members", "concept_id", "stability",
                 "inhibition", "collapsed")
    def __init__(self, signature, members, concept_id):
        self.signature  = signature
        self.members    = list(members)   # member neuron ids
        self.concept_id = concept_id
        self.stability  = 0.0
        self.inhibition = 0.0
        self.collapsed  = False


HEBBIAN_WINDOW = 20
MAX_PAIRS      = 8_000


class NeuronPool:
    """Faithful port of the Rust NeuronPool features used by two_pool_*."""
    def __init__(self, cfg: NeuroConfig):
        self.cfg               = cfg
        self.neurons           = []        # list[Neuron]
        self.label_to_id       = {}        # label -> id
        self.step              = 0
        self.minicolumn_counts = defaultdict(int)   # signature -> count
        self.minicolumns       = []
        self.minicolumn_index  = {}        # signature -> index into self.minicolumns
        # Atoms-only flag per neuron (for active_labels filtering / cross-pool)
        # (Composites have non-empty members; atoms are leaves.)

    # ── Neuron lifecycle ──────────────────────────────────────────────────
    def get_or_create(self, label):
        nid = self.label_to_id.get(label)
        if nid is not None:
            return nid
        nid = len(self.neurons)
        self.neurons.append(Neuron(nid, label))
        self.label_to_id[label] = nid
        return nid

    # ── Hebbian + STDP (faithful) ────────────────────────────────────────
    def _add_synapse(self, src, tgt, delta, inhibitory):
        max_w = self.cfg.max_weight
        n = self.neurons[src]
        d = n.inhib if inhibitory else n.excit
        d[tgt] = min(d.get(tgt, 0.0) + delta, max_w)

    def _weaken_synapse(self, src, tgt, delta_pos, inhibitory):
        n = self.neurons[src]
        d = n.inhib if inhibitory else n.excit
        if tgt in d:
            d[tgt] = max(0.0, d[tgt] - delta_pos)
            if d[tgt] <= 0.0:
                del d[tgt]

    def hebbian_pair(self, a, b, scale, inhibitory=False):
        n_a = self.neurons[a]
        n_b = self.neurons[b]
        base = (self.cfg.hebbian_lr
                * (n_a.activation + n_a.trace)
                * (n_b.activation + n_b.trace)
                * scale)
        if base <= 0.0: return
        # LTP forward
        d_fwd = base * self.cfg.stdp_pre_post
        if d_fwd > 0.0:
            self._add_synapse(a, b, d_fwd, inhibitory)
        # LTD backward (only if synapse exists when negative)
        d_bwd = base * self.cfg.stdp_post_pre
        if d_bwd > 0.0:
            self._add_synapse(b, a, d_bwd, inhibitory)
        elif d_bwd < 0.0:
            self._weaken_synapse(b, a, min(-d_bwd, base * 0.5), inhibitory)

    # ── Train ─────────────────────────────────────────────────────────────
    def train_weighted(self, symbols, lr_scale, inhibitory=False):
        if not symbols or lr_scale <= 0.0: return
        self.step += 1
        ids = []
        for label in symbols:
            nid = self.get_or_create(label)
            n = self.neurons[nid]
            n.activation = max(0.0, 1.0 - n.fatigue)
            n.trace += 0.1 * lr_scale
            n.use_count += 1
            ids.append(nid)
        # Mini-column member suppression BEFORE propagation (same as Rust).
        self._update_minicolumns(ids)
        # Hebbian pairing windowed
        scale = self.cfg.excitatory_scale * lr_scale
        pair_count = 0
        for i in range(len(ids)):
            end = min(i + 1 + HEBBIAN_WINDOW, len(ids))
            for j in range(i + 1, end):
                self.hebbian_pair(ids[i], ids[j], scale, inhibitory)
                pair_count += 1
                if pair_count >= MAX_PAIRS:
                    break
            if pair_count >= MAX_PAIRS:
                break
        # Mini-column counting + promotion
        self._accumulate_minicolumn_counts(ids)
        self._maybe_promote_minicolumns()
        # Fatigue dynamics
        for nid in ids:
            n = self.neurons[nid]
            n.fatigue = min(1.0, n.fatigue + self.cfg.fatigue_increment)
            n.act_ema = n.act_ema * 0.99 + n.activation * 0.01
        for n in self.neurons:
            n.fatigue *= self.cfg.fatigue_decay
            n.trace *= 0.95
        # Homeostatic scaling on schedule
        if (self.cfg.homeostatic_period > 0
            and self.step % self.cfg.homeostatic_period == 0):
            self._homeostatic_scale()

    def _homeostatic_scale(self):
        target = self.cfg.homeostatic_target
        rate   = self.cfg.homeostatic_lr
        for n in self.neurons:
            if n.act_ema <= 0.0: continue
            ratio = target / n.act_ema
            scale = 1.0 + rate * (ratio - 1.0)
            for k in list(n.excit.keys()):
                n.excit[k] = max(0.0, min(self.cfg.max_weight,
                                          n.excit[k] * scale))

    # ── Mini-columns ─────────────────────────────────────────────────────
    def _accumulate_minicolumn_counts(self, ids):
        # Count co-occurring 3-label signatures within HEBBIAN_WINDOW.
        if len(ids) < 3: return
        seen_sigs = set()
        for i in range(len(ids) - 2):
            tri = tuple(sorted(ids[i:i+3]))
            if tri in seen_sigs: continue
            seen_sigs.add(tri)
            sig = "mini::" + "|".join(self.neurons[t].label for t in tri)
            self.minicolumn_counts[sig] += 1

    def _maybe_promote_minicolumns(self):
        if len(self.minicolumns) >= self.cfg.minicolumn_max: return
        thr = self.cfg.minicolumn_threshold
        # Promote any signature that crossed threshold and isn't a column yet.
        promoted = 0
        for sig, count in list(self.minicolumn_counts.items()):
            if count < thr: continue
            if sig in self.minicolumn_index: continue
            members = []
            for label_part in sig[len("mini::"):].split("|"):
                nid = self.label_to_id.get(label_part)
                if nid is None: break
                members.append(nid)
            if len(members) < 3: continue
            concept_id = self.get_or_create(sig)
            self.neurons[concept_id].members = list(members)
            mc = MiniColumn(sig, members, concept_id)
            self.minicolumn_index[sig] = len(self.minicolumns)
            self.minicolumns.append(mc)
            promoted += 1
            if promoted >= 4: break  # rate-limit
            if len(self.minicolumns) >= self.cfg.minicolumn_max: break

    def _update_minicolumns(self, recent_ids):
        # Match each column's signature against the recent batch.
        if not self.minicolumns: return
        recent_set = set(recent_ids)
        s_decay = self.cfg.minicolumn_stability_decay
        i_decay = self.cfg.minicolumn_inhibition_decay
        act_thr = self.cfg.minicolumn_activation_threshold
        col_thr = self.cfg.minicolumn_collapse_threshold
        inhib_scale = self.cfg.minicolumn_inhibit_scale
        for mc in self.minicolumns:
            hits = sum(1 for m in mc.members if m in recent_set)
            evidence = hits / max(len(mc.members), 1)
            conflict = 0.0 if hits == len(mc.members) else (1.0 - evidence)
            mc.stability  = mc.stability  * s_decay + evidence * (1.0 - s_decay)
            mc.inhibition = mc.inhibition * i_decay + conflict * (1.0 - i_decay)
            mc.collapsed  = mc.inhibition >= col_thr
            active = (mc.stability >= act_thr) and (not mc.collapsed)
            col_act = max(0.0, min(1.0, mc.stability * (1.0 - mc.inhibition)))
            concept = self.neurons[mc.concept_id]
            concept.activation = max(concept.activation, col_act)
            if active:
                for mid in mc.members:
                    self.neurons[mid].activation *= inhib_scale

    # ── Propagate ────────────────────────────────────────────────────────
    def propagate_weighted(self, seed_activations, hops, min_activation,
                           prop_no_clamp=False, prop_max_accum=False):
        n = len(self.neurons)
        if n == 0: return {}
        k_active = (max(int(self.cfg.sdr_sparsity * n), self.cfg.sdr_k_min)
                    if self.cfg.sdr_sparsity > 0.0 else 0)
        activation = [0.0] * n
        for label, init in seed_activations.items():
            nid = self.label_to_id.get(label)
            if nid is not None:
                activation[nid] = max(0.0, init if prop_no_clamp
                                            else min(1.0, init))
        nxt = [0.0] * n
        for _ in range(hops):
            for i in range(n):
                nxt[i] = activation[i]
            for src, src_act in enumerate(activation):
                if src_act < 0.001: continue
                neuron = self.neurons[src]
                if neuron.excit:
                    fan_out = max(1.0, math.sqrt(len(neuron.excit)))
                    for tgt, w in neuron.excit.items():
                        delta = src_act * w * 0.5 / fan_out
                        if prop_max_accum:
                            if delta > nxt[tgt]: nxt[tgt] = delta
                        elif prop_no_clamp:
                            nxt[tgt] = nxt[tgt] + delta
                        else:
                            nxt[tgt] = min(1.0, nxt[tgt] + delta)
                if neuron.inhib:
                    for tgt, w in neuron.inhib.items():
                        nxt[tgt] = max(0.0, nxt[tgt] - src_act * abs(w) * 0.3)
            for i in range(n):
                nxt[i] *= 0.85
            if k_active > 0 and k_active < n:
                self._apply_kwta(nxt, k_active)
            activation, nxt = nxt, activation
        out = {}
        for i, a in enumerate(activation):
            if a >= min_activation:
                out[self.neurons[i].label] = a
        return out

    @staticmethod
    def _apply_kwta(activation, k):
        vals = [v for v in activation if v > 0.0]
        if len(vals) <= k: return
        vals.sort(reverse=True)
        threshold = vals[k - 1]
        for i in range(len(activation)):
            if activation[i] < threshold:
                activation[i] = 0.0

    def active_labels(self, min_activation):
        return {n.label for n in self.neurons if n.activation >= min_activation}

    def composite_constituents(self, label):
        nid = self.label_to_id.get(label)
        if nid is None: return [label]
        n = self.neurons[nid]
        if not n.members: return [label]
        out = []
        for m in n.members:
            out.extend(self.composite_constituents(self.neurons[m].label))
        return out

    # ── Force-promote a full ordered atom sequence to a concept neuron ──
    # Mirrors crates/core/src/neuro.rs:1897 promote_full_sequence.
    # Used by the two-pool training endpoint where the (input, output)
    # pair is a known training unit — concept members preserve atom order
    # so chain-walk can recall the exact sequence.
    def promote_full_sequence(self, atom_labels):
        if len(atom_labels) < 2: return None
        ids = [self.get_or_create(a) for a in atom_labels]
        # Stable concept label via hash of ordered ids.
        sig = "concept::" + "|".join(self.neurons[i].label for i in ids[:64])
        if len(ids) > 64:
            sig += f"::len{len(ids)}"
        cid = self.get_or_create(sig)
        self.neurons[cid].members = list(ids)
        self.neurons[cid].activation = 1.0
        self.neurons[cid].use_count += 1
        # Member → concept dendrites (so partial member firing partly
        # activates the concept — exactly the Rust hebbian_pair call).
        scale = self.cfg.excitatory_scale
        for mid in ids:
            self.hebbian_pair(mid, cid, scale, False)
        return sig


class CrossPoolSynapses:
    def __init__(self):
        self.weights = {}   # (in_label, out_label) -> weight
        self.cap = 200_000
        self.max_w = 4.0

    def pair(self, in_label, out_label, lr):
        k = (in_label, out_label)
        self.weights[k] = min(self.weights.get(k, 0.0) + lr, self.max_w)
        if self.cap > 0 and len(self.weights) > self.cap:
            target = int(self.cap * 0.95)
            kept = sorted(self.weights.items(), key=lambda x: -x[1])[:target]
            self.weights = dict(kept)

    def forward(self, in_label):
        return [(b, w) for (a, b), w in self.weights.items() if a == in_label]

    def reverse(self, out_label):
        return [(a, w) for (a, b), w in self.weights.items() if b == out_label]


# ────────────────────────────────────────────────────────── Two-pool wrapper

class TwoPool:
    def __init__(self, cfg: NeuroConfig):
        self.cfg = cfg
        self.pool_in  = NeuronPool(cfg)
        self.pool_out = NeuronPool(cfg)
        self.cross    = CrossPoolSynapses()

    def train_pair(self, in_atoms, out_atoms, lr,
                   use_full_sequence=False, concept_only_cross=False,
                   reset_activations_per_pair=False):
        if not in_atoms or not out_atoms: return
        # Clear activations from any prior pair so cross.pair() doesn't
        # see concepts left over from previous training rounds (which
        # otherwise saturates every concept↔concept cross-edge to max_w).
        if reset_activations_per_pair:
            for n in self.pool_in.neurons:
                n.activation = 0.0
                n.trace = 0.0
            for n in self.pool_out.neurons:
                n.activation = 0.0
                n.trace = 0.0
        self.pool_in.train_weighted(in_atoms, 1.0, False)
        self.pool_out.train_weighted(out_atoms, 1.0, False)
        if use_full_sequence:
            self.pool_in.promote_full_sequence(in_atoms)
            self.pool_out.promote_full_sequence(out_atoms)
        in_active  = self.pool_in.active_labels(0.05)
        out_active = self.pool_out.active_labels(0.05)
        # Filter to concepts only — character-level cross-pool drowns the
        # clean concept-level signal once char weights saturate at max_w.
        if concept_only_cross and use_full_sequence:
            in_active  = {l for l in in_active  if l.startswith("concept::")}
            out_active = {l for l in out_active if l.startswith("concept::")}
        if len(in_active) * len(out_active) > 200_000:
            in_sorted  = sorted(in_active,
                                key=lambda l: -self.pool_in.neurons[self.pool_in.label_to_id[l]].activation)[:300]
            out_sorted = sorted(out_active,
                                key=lambda l: -self.pool_out.neurons[self.pool_out.label_to_id[l]].activation)[:600]
            in_active, out_active = in_sorted, out_sorted
        for a in in_active:
            for b in out_active:
                self.cross.pair(a, b, lr)

    def query(self, atoms, direction, hops, min_activation,
              cross_seed_no_clamp=False, src_concept_argmax=False,
              tgt_concept_argmax=False, prop_no_clamp=False,
              prop_max_accum=False, src_concept_overlap=False,
              tgt_concept_overlap=False, src_concept_lev=False,
              src_concept_tfidf=False):
        if not atoms: return None
        src_pool = self.pool_in if direction == "fwd" else self.pool_out
        tgt_pool = self.pool_out if direction == "fwd" else self.pool_in
        atom_set = set(atoms)

        # 1. Seed source, 2-hop within-source propagate.
        src_seeds = {a: 1.0 for a in atoms}
        src_active = src_pool.propagate_weighted(
            src_seeds, 2, min(min_activation, 0.02),
            prop_no_clamp=prop_no_clamp, prop_max_accum=prop_max_accum)
        if not src_active: return None

        # 1b. Pick the source concept that best matches the input.
        # Strategies (priority: lev > tfidf > overlap > argmax):
        #   src_concept_lev:    lev_sim of atom sequence vs concept members
        #                       — handles paraphrase inputs (single-char diffs)
        #   src_concept_tfidf:  weight each member-overlap by 1/concept_freq
        #                       — handles inputs with mostly-shared chars
        #   src_concept_overlap: |members ∩ atom_set| / |members| — fast
        #   src_concept_argmax: most-active concept after propagation
        chose_concept = None
        if src_concept_lev:
            best, best_score = None, -1.0
            for lbl in list(src_active.keys()):
                if not lbl.startswith("concept::"): continue
                cid = src_pool.label_to_id.get(lbl)
                if cid is None: continue
                members = src_pool.neurons[cid].members
                if not members: continue
                # Reconstruct ordered atom-label sequence of the concept.
                mseq = [src_pool.neurons[m].label for m in members]
                score = _seq_sim(atoms, mseq)
                if score > best_score:
                    best_score, best = score, lbl
            chose_concept = best
        elif src_concept_tfidf:
            # Compute concept frequency per atom over the candidate set.
            candidates = [(l, src_pool.label_to_id.get(l))
                          for l in src_active.keys()
                          if l.startswith("concept::")]
            candidates = [(l, cid) for l, cid in candidates
                          if cid is not None
                          and src_pool.neurons[cid].members]
            atom_concept_freq = {}
            for _, cid in candidates:
                ms = {src_pool.neurons[m].label
                      for m in src_pool.neurons[cid].members}
                for a in ms:
                    atom_concept_freq[a] = atom_concept_freq.get(a, 0) + 1
            best, best_score = None, -1.0
            for lbl, cid in candidates:
                members = src_pool.neurons[cid].members
                score = 0.0
                for m in members:
                    ml = src_pool.neurons[m].label
                    if ml in atom_set:
                        f = atom_concept_freq.get(ml, 1)
                        score += 1.0 / f
                score /= max(1, len(members))
                if score > best_score:
                    best_score, best = score, lbl
            chose_concept = best
        elif src_concept_overlap:
            best, best_score = None, -1.0
            for lbl in list(src_active.keys()):
                if not lbl.startswith("concept::"): continue
                cid = src_pool.label_to_id.get(lbl)
                if cid is None: continue
                members = src_pool.neurons[cid].members
                if not members: continue
                hits = sum(1 for m in members
                           if src_pool.neurons[m].label in atom_set)
                score = hits / len(members)
                if score > best_score:
                    best_score, best = score, lbl
            chose_concept = best
        elif src_concept_argmax:
            concepts = {l: a for l, a in src_active.items()
                        if l.startswith("concept::")}
            if concepts:
                chose_concept = max(concepts, key=concepts.get)
        if chose_concept is not None and chose_concept in src_active:
            src_active = {chose_concept: src_active[chose_concept]}

        # 2. Cross-project.
        tgt_seeds = {}
        for src_lbl, src_act in src_active.items():
            edges = (self.cross.forward(src_lbl) if direction == "fwd"
                     else self.cross.reverse(src_lbl))
            for tgt_lbl, w in edges:
                accum = tgt_seeds.get(tgt_lbl, 0.0) + src_act * w
                if not cross_seed_no_clamp:
                    accum = min(1.0, accum)
                tgt_seeds[tgt_lbl] = accum
        if not tgt_seeds: return None
        # When unclamped, downstream propagate expects ≤1 inputs, so renorm
        # by max so highest tgt seed = 1.0 and discrimination is preserved.
        if cross_seed_no_clamp:
            mx = max(tgt_seeds.values())
            if mx > 1e-9:
                for k in tgt_seeds:
                    tgt_seeds[k] /= mx

        # 2b. Pick a single target concept seed.  tgt_concept_overlap is
        # only meaningful for queries where the *answer* atoms are known,
        # which they aren't at query time — so we just fall through to
        # the activation-based argmax (or do nothing).
        if tgt_concept_argmax:
            concept_seeds = {l: v for l, v in tgt_seeds.items()
                             if l.startswith("concept::")}
            if concept_seeds:
                top_lbl = max(concept_seeds, key=concept_seeds.get)
                tgt_seeds = {top_lbl: concept_seeds[top_lbl]}

        # 3. Propagate within target.
        tgt_acts = tgt_pool.propagate_weighted(
            tgt_seeds, max(hops, 3), min_activation,
            prop_no_clamp=prop_no_clamp, prop_max_accum=prop_max_accum)
        if not tgt_acts: return None

        # 4a. Concept walk if a composite fired.
        best_concept = None
        for lbl, act in tgt_acts.items():
            parts = tgt_pool.composite_constituents(lbl)
            if len(parts) >= 2:
                if best_concept is None or act > best_concept[1]:
                    best_concept = (lbl, act)
        if best_concept:
            parts = tgt_pool.composite_constituents(best_concept[0])
            chars = "".join(_label_to_char(p) for p in parts
                            if _label_to_char(p) is not None)
            if chars: return chars

        # 4b. Predecessor-pull start + chain-walk.
        active_atoms = []
        for lbl, act in tgt_acts.items():
            ch = _label_to_char(lbl)
            if ch is None: continue
            nid = tgt_pool.label_to_id.get(lbl)
            if nid is None: continue
            active_atoms.append((nid, act))
        if not active_atoms: return None
        active_set = {nid for nid, _ in active_atoms}
        act_lookup = {nid: a for nid, a in active_atoms}
        scores = []
        for nid, act in active_atoms:
            pull = 0.0
            for other_id, other_act in active_atoms:
                if other_id == nid: continue
                w = tgt_pool.neurons[other_id].excit.get(nid, 0.0)
                pull += other_act * w
            scores.append((nid, act - 0.5 * pull))
        scores.sort(key=lambda x: -x[1])
        start = scores[0][0]
        max_len = 200
        visited = set()
        chain = []
        cur = start
        for _ in range(max_len):
            if cur in visited: break
            visited.add(cur)
            chain.append(cur)
            best_next, best_score = None, 0.0
            for tgt, w in tgt_pool.neurons[cur].excit.items():
                if tgt in visited: continue
                if tgt not in active_set: continue
                tact = act_lookup.get(tgt, 0.0)
                s = w * (tact + 0.05)
                if s > best_score:
                    best_score, best_next = s, tgt
            if best_next is None: break
            cur = best_next
        if not chain: return None
        out = []
        for nid in chain:
            ch = _label_to_char(tgt_pool.neurons[nid].label)
            if ch is not None: out.append(ch)
        return "".join(out) if out else None


def _label_to_char(label):
    """Decode 'a:<chr>' atom labels back to their character."""
    if not label.startswith("a:"): return None
    rest = label[2:]
    return rest if len(rest) == 1 else None


def _seq_sim(a_seq, b_seq):
    """Levenshtein similarity over two label sequences.  Uses len-cap so
    that very long sequences don't blow up cost (capped at 200 atoms)."""
    if not a_seq and not b_seq: return 1.0
    if not a_seq or not b_seq:  return 0.0
    a = a_seq[:200]; b = b_seq[:200]
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


def text_to_atoms(text):
    """Each character becomes a unique atom label.  Matches the
       'neuron label = source_prefix + raw data bits' rule with single-char
       payloads."""
    return [f"a:{ch}" for ch in text]


# ──────────────────────────────────────────────────────────────── Corpus

def make_corpus(seed=42, scale="small"):
    """Mixed prose-like content. Two scales:
       small  — discovery (fast iteration): 30..1500 char outputs
       large  — validation: 30..45000 char outputs (tens of thousands)
    """
    bases = [
        ("greet warmly",
         "hello there friend welcome aboard glad to see you "),
        ("greet coldly",
         "acknowledged standard protocol greeting initiated proceed "),
        ("explain quantum physics",
         "quantum mechanics superposition entanglement uncertainty principle "),
        ("explain classical physics",
         "classical mechanics forces motion newtonian inertia momentum "),
        ("describe weather sunny",
         "today sunny mild winds moderate temperature clear blue sky "),
        ("describe spring season",
         "currently spring season blooming flowers warming days mild rain "),
        ("describe ocean waves",
         "deep blue rolling waves crashing against shoreline salty mist "),
        ("describe forest morning",
         "ancient towering pines dappled sunlight rustling leaves cool air "),
    ]
    if scale == "large":
        spec = [30, 200, 1500, 10_000, 25_000, 45_000, 500, 5_000]
    else:
        spec = [30, 80, 200, 400, 800, 1500, 200, 600]
    pairs = []
    for (q_base, a_base), out_len in zip(bases, spec):
        a = (a_base * (out_len // len(a_base) + 1))[:out_len]
        in_len = max(15, min(250, len(q_base) * 2))
        q = (q_base + " ") * (in_len // (len(q_base)+1) + 1)
        q = q[:in_len].rstrip()
        if not q: q = q_base
        pairs.append((q, a))
    return pairs


# ─────────────────────────────────────────────────────────────── Fitness

def lev_sim(a, b):
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    la, lb = len(a), len(b)
    if max(la, lb) > 500:
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


def evaluate(genome, pairs):
    cfg = NeuroConfig(**{k: v for k, v in genome.items()
                         if k in NeuroConfig.__dataclass_fields__})
    tp = TwoPool(cfg)
    train_lr = genome['train_lr']
    passes   = genome['passes']
    use_seq  = bool(genome.get('use_full_sequence', False))
    cpc_only = bool(genome.get('concept_only_cross', False))
    reset_pp = bool(genome.get('reset_activations_per_pair', False))
    no_clamp = bool(genome.get('cross_seed_no_clamp', False))
    src_arg  = bool(genome.get('src_concept_argmax', False))
    tgt_arg  = bool(genome.get('tgt_concept_argmax', False))
    p_noclmp = bool(genome.get('prop_no_clamp', False))
    p_maxacc = bool(genome.get('prop_max_accum', False))
    src_ovlp = bool(genome.get('src_concept_overlap', False))
    src_lev  = bool(genome.get('src_concept_lev', False))
    src_tfidf= bool(genome.get('src_concept_tfidf', False))
    for q, a in pairs:
        in_atoms  = text_to_atoms(q)
        out_atoms = text_to_atoms(a)
        for pi in range(passes):
            tp.train_pair(in_atoms, out_atoms, train_lr,
                          use_full_sequence=use_seq,
                          concept_only_cross=cpc_only,
                          reset_activations_per_pair=(reset_pp and pi == 0))

    fwd_scores, rev_scores = [], []
    exact_fwd = 0; exact_rev = 0
    qhops = genome['query_hops']
    qmin  = genome['query_min_activation']
    for q, a in pairs:
        pf = tp.query(text_to_atoms(q), "fwd", qhops, qmin,
                      cross_seed_no_clamp=no_clamp,
                      src_concept_argmax=src_arg,
                      tgt_concept_argmax=tgt_arg,
                      prop_no_clamp=p_noclmp,
                      prop_max_accum=p_maxacc,
                      src_concept_overlap=src_ovlp,
                      src_concept_lev=src_lev,
                      src_concept_tfidf=src_tfidf) or ""
        pr = tp.query(text_to_atoms(a), "rev", qhops, qmin,
                      cross_seed_no_clamp=no_clamp,
                      src_concept_argmax=src_arg,
                      tgt_concept_argmax=tgt_arg,
                      prop_no_clamp=p_noclmp,
                      prop_max_accum=p_maxacc,
                      src_concept_overlap=src_ovlp,
                      src_concept_lev=src_lev,
                      src_concept_tfidf=src_tfidf) or ""
        fwd_scores.append(lev_sim(pf, a))
        rev_scores.append(lev_sim(pr, q))
        if pf == a: exact_fwd += 1
        if pr == q: exact_rev += 1
    return {
        'fwd':       sum(fwd_scores) / len(fwd_scores),
        'rev':       sum(rev_scores) / len(rev_scores),
        'combined':  (sum(fwd_scores) + sum(rev_scores)) / (2 * len(pairs)),
        'exact_fwd': exact_fwd,
        'exact_rev': exact_rev,
        'n_pairs':   len(pairs),
    }


# ────────────────────────────────────────────────────────────────── GA

PARAM_SPACE = {
    'hebbian_lr':                       ('float', 0.005, 0.5),
    'decay':                            ('float', 0.80, 0.999),
    'composite_threshold':              ('int',   5, 60),
    'inhibitory_scale':                 ('float', 0.0, 2.0),
    'excitatory_scale':                 ('float', 0.1, 4.0),
    'fatigue_increment':                ('float', 0.0, 0.2),
    'fatigue_decay':                    ('float', 0.80, 0.999),
    'wta_k_per_zone':                   ('int',   1, 16),
    'stdp_scale':                       ('float', 0.001, 0.5),
    'stdp_pre_post':                    ('float', 0.1, 4.0),
    'stdp_post_pre':                    ('float', -2.0, 1.0),
    'max_weight':                       ('float', 1.0, 10.0),
    'sdr_sparsity':                     ('float', 0.005, 0.3),
    'sdr_k_min':                        ('int',   2, 200),
    'minicolumn_threshold':             ('int',   2, 60),
    'minicolumn_inhibit_scale':         ('float', 0.0, 1.0),
    'minicolumn_inhibition_decay':      ('float', 0.5, 0.99),
    'minicolumn_stability_decay':       ('float', 0.5, 0.99),
    'minicolumn_activation_threshold':  ('float', 0.1, 0.95),
    'minicolumn_collapse_threshold':    ('float', 0.1, 0.95),
    'homeostatic_target':               ('float', 0.01, 0.5),
    'homeostatic_period':               ('int',   50, 5000),
    'homeostatic_lr':                   ('float', 0.0, 0.2),
    # Two-pool runtime args
    'train_lr':                         ('float', 0.05, 1.5),
    'passes':                           ('int',   1, 80),
    'query_hops':                       ('int',   2, 12),
    'query_min_activation':             ('float', 0.005, 0.3),
    # Architectural toggles
    'use_full_sequence':                ('bool',  0, 1),
    'concept_only_cross':               ('bool',  0, 1),
    'cross_seed_no_clamp':              ('bool',  0, 1),
    'src_concept_argmax':               ('bool',  0, 1),
    'tgt_concept_argmax':               ('bool',  0, 1),
    'prop_no_clamp':                    ('bool',  0, 1),
    'prop_max_accum':                   ('bool',  0, 1),
    'reset_activations_per_pair':       ('bool',  0, 1),
    'src_concept_overlap':              ('bool',  0, 1),
    'src_concept_lev':                  ('bool',  0, 1),
    'src_concept_tfidf':                ('bool',  0, 1),
}

# Seed: defaults from NeuroConfig + sensible runtime args.
DEFAULT_GENOME = {
    'hebbian_lr': 0.05, 'decay': 0.99, 'composite_threshold': 25,
    'inhibitory_scale': 0.5, 'excitatory_scale': 1.0,
    'fatigue_increment': 0.02, 'fatigue_decay': 0.98, 'wta_k_per_zone': 4,
    'stdp_scale': 0.02, 'stdp_pre_post': 1.0, 'stdp_post_pre': -0.3,
    'max_weight': 4.0, 'sdr_sparsity': 0.02, 'sdr_k_min': 50,
    'minicolumn_threshold': 18, 'minicolumn_inhibit_scale': 0.35,
    'minicolumn_inhibition_decay': 0.85, 'minicolumn_stability_decay': 0.9,
    'minicolumn_activation_threshold': 0.65, 'minicolumn_collapse_threshold': 0.55,
    'homeostatic_target': 0.10, 'homeostatic_period': 500, 'homeostatic_lr': 0.04,
    'train_lr': 0.5, 'passes': 30, 'query_hops': 4, 'query_min_activation': 0.05,
    'use_full_sequence': 1, 'concept_only_cross': 1, 'cross_seed_no_clamp': 0,
    'src_concept_argmax': 1, 'tgt_concept_argmax': 1,
    'prop_no_clamp': 0, 'prop_max_accum': 1,
    'reset_activations_per_pair': 1,
    'src_concept_overlap': 0,
    'src_concept_lev': 1,
    'src_concept_tfidf': 0,
}


def random_genome(rng):
    g = {}
    for k, spec in PARAM_SPACE.items():
        if spec[0] == 'int':
            g[k] = rng.randint(spec[1], spec[2])
        elif spec[0] == 'bool':
            g[k] = rng.randint(0, 1)
        else:
            g[k] = rng.uniform(spec[1], spec[2])
    return g


def mutate(g, ms, rng):
    g = dict(g)
    for k, spec in PARAM_SPACE.items():
        if rng.random() < ms:
            if spec[0] == 'int':
                delta = max(1, int(round(rng.gauss(0, ms * (spec[2]-spec[1])/4))))
                g[k] = max(spec[1], min(spec[2], g[k] + rng.choice([-delta, delta])))
            elif spec[0] == 'bool':
                if rng.random() < 0.3: g[k] = 1 - g[k]
            else:
                lo, hi = spec[1], spec[2]
                if lo < 0 < hi:
                    span = hi - lo
                    g[k] = max(lo, min(hi, g[k] + rng.gauss(0, ms * span * 0.25)))
                else:
                    g[k] = max(lo, min(hi, g[k] * rng.uniform(1 - ms, 1 + ms)))
    return g


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def fmt(g):
    parts = []
    for k, v in g.items():
        parts.append(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}")
    return ", ".join(parts)


def adaptive_ga(pairs, pop_size=8, max_gens=20, target=0.999, seed=1):
    rng = random.Random(seed)
    print(f"GA: pop={pop_size} max_gens={max_gens} target={target} pairs={len(pairs)}")
    print(f"Seed (NeuroConfig defaults): {fmt(DEFAULT_GENOME)}")
    # Population: 1 default + 3 small perturbations + rest random.
    pop = [dict(DEFAULT_GENOME)]
    for _ in range(3):
        pop.append(mutate(DEFAULT_GENOME, 0.15, rng))
    while len(pop) < pop_size:
        pop.append(random_genome(rng))

    best_ever = None; best_score = -1.0; no_improve = 0
    for gen in range(max_gens):
        scored = []
        t0 = time.time()
        for g in pop:
            try:
                m = evaluate(g, pairs)
            except Exception as e:
                m = {'combined': 0.0, 'fwd': 0.0, 'rev': 0.0,
                     'exact_fwd': 0, 'exact_rev': 0, 'n_pairs': len(pairs),
                     'err': str(e)[:60]}
            scored.append((m['combined'], m, g))
        scored.sort(key=lambda x: -x[0])
        best = scored[0]
        gen_time = time.time() - t0
        improved = best[0] > best_score + 1e-6
        if improved:
            best_score = best[0]; best_ever = best; no_improve = 0
        else:
            no_improve += 1
        ms = max(0.05, 0.5 * (1 - best_score))
        if no_improve >= 4: ms = min(0.7, ms + 0.2)
        avg = sum(x[0] for x in scored) / len(scored)
        m = best[1]
        print(f"  gen {gen:2d} best={best[0]:.4f} (fwd={m['fwd']:.3f} "
              f"rev={m['rev']:.3f}) exact={m['exact_fwd']}/{m['exact_rev']} "
              f"avg={avg:.3f} mut={ms:.2f} t={gen_time:.0f}s")
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
        # Polite yield between generations — sleeps longer when system
        # is busy so other apps stay responsive; near-zero when idle.
        adaptive_yield()
    return best_ever


def main():
    be_polite()
    print("=" * 80)
    print("GA over actual NeuroConfig knobs (faithful Python port of Rust two_pool)")
    print("=" * 80)

    # ── A/B grid: 4 configs ──
    print("\n--- A/B GRID: cross-pool wiring options ---")
    probe = make_corpus(scale="small")
    pin = [len(q) for q,_ in probe]; pout = [len(a) for _,a in probe]
    print(f"Corpus: {len(probe)} pairs.  in {min(pin)}..{max(pin)}  "
          f"out {min(pout)}..{max(pout)}")
    print(f"{'config':<48} {'combined':>9} {'fwd':>6} "
          f"{'rev':>6} {'exact':>10} {'t':>4}")
    # (label, use_seq, cpc, no_clamp, src_arg, tgt_arg, p_noclmp, p_maxacc, reset_pp, src_ovlp)
    configs = [
        ("CURRENT (chars only)",                       0, 0, 0, 0, 0, 0, 0, 0, 0),
        ("+ promote_full_sequence",                    1, 0, 0, 0, 0, 0, 0, 0, 0),
        ("+ reset + argmax + max_accum",               1, 1, 0, 1, 1, 0, 1, 1, 0),
        ("+ src_concept_overlap (replaces argmax)",    1, 1, 0, 0, 0, 0, 0, 1, 1),
        ("+ overlap + max_accum + tgt_argmax",         1, 1, 0, 0, 1, 0, 1, 1, 1),
        ("+ overlap + sum + tgt_argmax",               1, 1, 0, 0, 1, 1, 0, 1, 1),
        ("+ overlap + sum + tgt_argmax + no_clamp",    1, 1, 1, 0, 1, 1, 0, 1, 1),
    ]
    for (label, use_seq, cpc, no_clamp, src_arg, tgt_arg,
         p_noclmp, p_maxacc, reset_pp, src_ovlp) in configs:
        g = dict(DEFAULT_GENOME)
        g['use_full_sequence']           = use_seq
        g['concept_only_cross']          = cpc
        g['cross_seed_no_clamp']         = no_clamp
        g['src_concept_argmax']          = src_arg
        g['tgt_concept_argmax']          = tgt_arg
        g['prop_no_clamp']               = p_noclmp
        g['prop_max_accum']              = p_maxacc
        g['reset_activations_per_pair']  = reset_pp
        g['src_concept_overlap']         = src_ovlp
        t0 = time.time()
        m = evaluate(g, probe)
        print(f"{label:<48} {m['combined']:>9.4f} "
              f"{m['fwd']:>6.3f} {m['rev']:>6.3f} "
              f"{m['exact_fwd']}/{m['exact_rev']:<7} {time.time()-t0:>3.0f}s")
        adaptive_yield()

    print("\n--- DISCOVERY GA on small corpus ---")
    pairs = probe
    in_lens  = [len(q) for q, _ in pairs]
    out_lens = [len(a) for _, a in pairs]
    print(f"Corpus: {len(pairs)} pairs.  in {min(in_lens)}..{max(in_lens)}  "
          f"out {min(out_lens)}..{max(out_lens)}")
    print(f"Ratios: " + ", ".join(f"{ol/max(il,1):.0f}:1"
                                  for il, ol in zip(in_lens, out_lens)))
    t0 = time.time()
    best = adaptive_ga(pairs, pop_size=8, max_gens=20, target=0.999, seed=1)
    dt = time.time() - t0
    if best:
        print(f"\nDiscovery best (time={dt:.0f}s):")
        print(f"  combined={best[0]:.4f}  fwd={best[1]['fwd']:.3f}  "
              f"rev={best[1]['rev']:.3f}")
        print(f"  exact: fwd={best[1]['exact_fwd']}/{best[1]['n_pairs']}  "
              f"rev={best[1]['exact_rev']}/{best[1]['n_pairs']}")
        print(f"  genome: {fmt(best[2])}")
        if best[0] >= 0.5:
            print("\n--- VALIDATION: large corpus (30..45,000 chars) ---")
            big = make_corpus(scale="large")
            big_in  = [len(q) for q, _ in big]
            big_out = [len(a) for _, a in big]
            print(f"Corpus: {len(big)} pairs.  in {min(big_in)}..{max(big_in)}  "
                  f"out {min(big_out)}..{max(big_out)}")
            t1 = time.time()
            m = evaluate(best[2], big)
            print(f"\nValidation (time={time.time()-t1:.0f}s):")
            print(f"  combined={m['combined']:.4f}  fwd={m['fwd']:.3f}  "
                  f"rev={m['rev']:.3f}")
            print(f"  exact: fwd={m['exact_fwd']}/{m['n_pairs']}  "
                  f"rev={m['exact_rev']}/{m['n_pairs']}")


if __name__ == "__main__":
    main()
