"""Diagnostic: run two_pool with the 'best' config and see what's happening
during query — which decode branch fires, what concepts are in src/tgt active.
"""
import sys
sys.path.insert(0, "scripts")

from ga_neuro_config_search import (
    NeuroConfig, TwoPool, text_to_atoms,
    DEFAULT_GENOME, make_corpus, _label_to_char,
)


def run():
    pairs = make_corpus(scale="small")
    print(f"Corpus: {len(pairs)} pairs")
    for i, (q, a) in enumerate(pairs):
        print(f"  [{i}] q[{len(q):>3}]={q[:30]!r}  a[{len(a):>3}]={a[:30]!r}")

    g = dict(DEFAULT_GENOME)
    cfg = NeuroConfig(**{k: v for k, v in g.items()
                         if k in NeuroConfig.__dataclass_fields__})
    tp = TwoPool(cfg)
    train_lr = g['train_lr']
    passes   = g['passes']
    use_seq  = bool(g['use_full_sequence'])
    cpc_only = bool(g['concept_only_cross'])

    print(f"\nTraining with passes={passes}, use_seq={use_seq}, "
          f"cpc_only={cpc_only}")
    for q, a in pairs:
        in_atoms  = text_to_atoms(q)
        out_atoms = text_to_atoms(a)
        for _ in range(passes):
            tp.train_pair(in_atoms, out_atoms, train_lr,
                          use_full_sequence=use_seq,
                          concept_only_cross=cpc_only)

    print(f"\npool_in: {len(tp.pool_in.neurons)} neurons "
          f"(concepts: {sum(1 for n in tp.pool_in.neurons if n.label.startswith('concept::'))})")
    print(f"pool_out: {len(tp.pool_out.neurons)} neurons "
          f"(concepts: {sum(1 for n in tp.pool_out.neurons if n.label.startswith('concept::'))})")

    qhops = g['query_hops']
    qmin  = g['query_min_activation']

    for i, (q, a) in enumerate(pairs):
        # Manually replay query() to inspect intermediates.
        atoms = text_to_atoms(q)
        src_seeds = {at: 1.0 for at in atoms}
        src_active = tp.pool_in.propagate_weighted(src_seeds, 2,
                                                   min(qmin, 0.02))
        # Top-5 src_active labels by activation
        sorted_src = sorted(src_active.items(), key=lambda x: -x[1])[:5]
        src_concepts = [(l, v) for l, v in src_active.items()
                        if l.startswith("concept::")]

        # cross-project (no clamp, no argmax — see raw signal)
        tgt_seeds = {}
        for src_lbl, src_act in src_active.items():
            for tgt_lbl, w in tp.cross.forward(src_lbl):
                tgt_seeds[tgt_lbl] = min(1.0,
                    tgt_seeds.get(tgt_lbl, 0.0) + src_act * w)
        tgt_concept_seeds = sorted(
            [(l, v) for l, v in tgt_seeds.items() if l.startswith("concept::")],
            key=lambda x: -x[1])[:3]

        # propagate within target
        tgt_acts = tp.pool_out.propagate_weighted(
            tgt_seeds, max(qhops, 3), qmin)
        # Find concept fires
        tgt_concept_acts = sorted(
            [(l, v) for l, v in (tgt_acts or {}).items()
             if l.startswith("concept::")],
            key=lambda x: -x[1])[:3]

        # Try concept walk
        best_concept = None
        for lbl, act in (tgt_acts or {}).items():
            parts = tp.pool_out.composite_constituents(lbl)
            if len(parts) >= 2:
                if best_concept is None or act > best_concept[1]:
                    best_concept = (lbl, act)

        pred_concept = None
        if best_concept:
            parts = tp.pool_out.composite_constituents(best_concept[0])
            chars = "".join(_label_to_char(p) for p in parts
                            if _label_to_char(p) is not None)
            pred_concept = chars

        # Run the actual query method (concept-only off, no_clamp off)
        pred_actual = tp.query(text_to_atoms(q), "fwd", qhops, qmin) or ""

        print(f"\n=== pair {i} ===")
        print(f"  q[:40] = {q[:40]!r}")
        print(f"  a[:40] = {a[:40]!r}")
        print(f"  src_active size={len(src_active)} top5={[(l[:30], round(v,3)) for l,v in sorted_src]}")
        print(f"  src concepts (n={len(src_concepts)}): "
              f"{[(l[:50], round(v,3)) for l,v in src_concepts[:3]]}")
        print(f"  tgt concept seeds top3: "
              f"{[(l[:50], round(v,3)) for l,v in tgt_concept_seeds]}")
        print(f"  tgt concept fires top3: "
              f"{[(l[:50], round(v,3)) for l,v in tgt_concept_acts]}")
        print(f"  best_concept = {best_concept[0][:60] if best_concept else None!r}")
        print(f"  pred via concept walk[:40] = {pred_concept[:40] if pred_concept else None!r}")
        print(f"  pred via tp.query[:40]    = {pred_actual[:40]!r}")
        print(f"  match exact = {pred_actual == a}")


if __name__ == "__main__":
    run()
