"""Integration tests for the two_pool architectural fixes — exercises
realistic training/query lifecycle patterns before we port to Rust.

  IT1: Train, query each pair fwd & rev, verify exact recall.
  IT2: Repeated queries on same model — state must not drift.
  IT3: Incremental training — add pair P9 after P0..P7 trained, verify
       P0..P7 still recall and P9 recalls correctly.
  IT4: Out-of-distribution query — input that's not any trained pair.
       Output should be SOMETHING (not crash, not None) and ideally
       biased toward the closest match.
  IT5: Empty input / single-char input — verify no crash.
  IT6: Mixed lengths — train pairs with very different in/out lengths.
  IT7: Identical inputs with different outputs — last training wins.
       (Sanity: just confirms behaviour is deterministic and doesn't merge.)
  IT8: Determinism across runs (with PYTHONHASHSEED=0) — same train/query
       order produces same result.
"""
import os, sys, time
sys.path.insert(0, "scripts")

# Force deterministic hashing for IT8.  Must happen before module imports.
os.environ.setdefault("PYTHONHASHSEED", "0")

from ga_neuro_config_search import (
    NeuroConfig, TwoPool, text_to_atoms, DEFAULT_GENOME,
    be_polite,
)


def winning_genome():
    g = dict(DEFAULT_GENOME)
    g['use_full_sequence']           = 1
    g['concept_only_cross']          = 1
    g['reset_activations_per_pair']  = 1
    g['prop_max_accum']              = 1
    g['tgt_concept_argmax']          = 1
    g['src_concept_lev']             = 1
    g['src_concept_overlap']         = 0
    g['src_concept_tfidf']           = 0
    return g


def build_two_pool(g):
    cfg = NeuroConfig(**{k: v for k, v in g.items()
                         if k in NeuroConfig.__dataclass_fields__})
    return TwoPool(cfg), g


def train_pairs(tp, g, pairs):
    train_lr = g['train_lr']
    passes   = g['passes']
    use_seq  = bool(g['use_full_sequence'])
    cpc_only = bool(g['concept_only_cross'])
    reset_pp = bool(g['reset_activations_per_pair'])
    for q, a in pairs:
        in_atoms  = text_to_atoms(q)
        out_atoms = text_to_atoms(a)
        for pi in range(passes):
            tp.train_pair(in_atoms, out_atoms, train_lr,
                          use_full_sequence=use_seq,
                          concept_only_cross=cpc_only,
                          reset_activations_per_pair=(reset_pp and pi == 0))


def query_one(tp, g, text, direction):
    return tp.query(
        text_to_atoms(text), direction,
        g['query_hops'], g['query_min_activation'],
        cross_seed_no_clamp=bool(g.get('cross_seed_no_clamp', False)),
        src_concept_argmax=bool(g.get('src_concept_argmax', False)),
        tgt_concept_argmax=bool(g.get('tgt_concept_argmax', False)),
        prop_no_clamp=bool(g.get('prop_no_clamp', False)),
        prop_max_accum=bool(g.get('prop_max_accum', False)),
        src_concept_overlap=bool(g.get('src_concept_overlap', False)),
        src_concept_lev=bool(g.get('src_concept_lev', False)),
        src_concept_tfidf=bool(g.get('src_concept_tfidf', False)),
    ) or ""


# ───────────────────────────────────────────────────────────── result helpers

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
totals = {'pass': 0, 'fail': 0}


def check(name, cond, detail=""):
    label = PASS if cond else FAIL
    totals['pass' if cond else 'fail'] += 1
    print(f"  [{label}] {name}{(' — ' + detail) if detail else ''}")
    return cond


# ───────────────────────────────────────────────────────────────────── tests

def it1_recall(tp, g, pairs):
    print("\nIT1: Train, query each pair fwd & rev -> verify exact recall")
    fwd_ok = rev_ok = 0
    for q, a in pairs:
        pf = query_one(tp, g, q, "fwd")
        pr = query_one(tp, g, a, "rev")
        if pf == a: fwd_ok += 1
        if pr == q: rev_ok += 1
    n = len(pairs)
    check(f"forward recall {fwd_ok}/{n}", fwd_ok == n)
    check(f"reverse recall {rev_ok}/{n}", rev_ok == n)


def it2_repeat(tp, g, pairs):
    print("\nIT2: Repeat query 3x — state must not drift")
    for q, a in pairs:
        results = [query_one(tp, g, q, "fwd") for _ in range(3)]
        check(f"q[:20]={q[:20]!r:25} stable",
              results[0] == results[1] == results[2],
              f"got {set(r[:20] for r in results)}")


def it3_incremental(g, pairs8, new_pair):
    print("\nIT3: Incremental — train 8 pairs, then add 1 more, "
          "verify all 9 recall correctly")
    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs8)
    # Verify initial recall
    for q, a in pairs8:
        if query_one(tp, g, q, "fwd") != a:
            check("initial recall before incremental", False,
                  f"failed for {q[:20]!r}"); return
    # Add new pair
    train_pairs(tp, g, [new_pair])
    # Verify all 9 still work
    all_ok = True
    for q, a in pairs8 + [new_pair]:
        pf = query_one(tp, g, q, "fwd")
        pr = query_one(tp, g, a, "rev")
        if pf != a or pr != q:
            all_ok = False
            print(f"    drifted on {q[:25]!r}: fwd={pf[:25]!r} rev={pr[:25]!r}")
    check("all 9 pairs recall after incremental train", all_ok)


def it4_ood(tp, g, pairs):
    print("\nIT4: Out-of-distribution — untrained queries should return "
          "the closest trained answer (not garbage)")
    trained_answers = {a for _, a in pairs}
    # Cases: each is a paraphrase / near-neighbor of a trained Q.  The
    # expected answer is the trained answer of the *closest* trained Q.
    # We accept either the exact answer or a "reasonable" (= trained) one.
    cases = [
        # near "greet warmly"
        ("greet warmly!",                           "hello there friend welcome aboard"),
        ("greet warmly ",                           "hello there friend welcome aboard"),
        # near "explain quantum physics"
        ("explain quantum physic",                  "quantum mechanics superposition entanglement uncertainty"),
        # near "describe ocean waves"
        ("describe ocean wave",                     "deep blue rolling waves crashing against shores"),
        # near "describe spring season"
        ("describe spring",                         "currently blooming flowers warming temperatures growth"),
    ]
    for q, expected in cases:
        try:
            pf = query_one(tp, g, q, "fwd")
            ok_exact     = (pf == expected)
            ok_nearest   = (pf in trained_answers)
            check(f"near-neighbor {q[:30]!r:32} -> matches trained",
                  ok_nearest,
                  f"expected {expected[:30]!r}, got {pf[:30]!r}")
            check(f"near-neighbor {q[:30]!r:32} -> exact-closest",
                  ok_exact,
                  f"expected {expected[:30]!r}, got {pf[:30]!r}")
        except Exception as e:
            check(f"near-neighbor {q[:30]!r} no crash", False, str(e)[:60])

    # Far-from-training: should still return SOMETHING from trained answers
    far_inputs = ["asdfqwer xyz totally unrelated", "hello"]
    for q in far_inputs:
        try:
            pf = query_one(tp, g, q, "fwd")
            check(f"far OOD {q[:25]!r:30} returns string, no crash",
                  isinstance(pf, str) and len(pf) > 0,
                  f"got len {len(pf)}")
        except Exception as e:
            check(f"far OOD {q[:25]!r} no crash", False, str(e)[:60])


def it5_edges(tp, g):
    print("\nIT5: Edge inputs — empty / single char")
    for inp in ["", "a", "z"]:
        try:
            pf = query_one(tp, g, inp, "fwd")
            check(f"input={inp!r:6} no crash",
                  isinstance(pf, str))
        except Exception as e:
            check(f"input={inp!r} no crash", False, str(e)[:60])


def it6_mixed_lengths(g):
    print("\nIT6: Mixed lengths — short query -> long answer; long -> short")
    pairs = [
        ("hi", "this is a much longer answer about greetings and salutations"),
        ("a long detailed query about the weather today",  "warm"),
        ("short", "tiny"),
        ("medium length question",
         "medium length answer of similar size"),
    ]
    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs)
    fwd_ok = rev_ok = 0
    for q, a in pairs:
        pf = query_one(tp, g, q, "fwd")
        pr = query_one(tp, g, a, "rev")
        if pf == a: fwd_ok += 1
        if pr == q: rev_ok += 1
    check(f"mixed-length fwd {fwd_ok}/{len(pairs)}",
          fwd_ok == len(pairs))
    check(f"mixed-length rev {rev_ok}/{len(pairs)}",
          rev_ok == len(pairs))


def it7_identical_inputs(g):
    print("\nIT7: Identical inputs different outputs — last write wins or "
          "concept signature distinguishes")
    pairs = [
        ("same query input here", "first answer text"),
        ("same query input here", "second different answer"),
    ]
    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs)
    pf = query_one(tp, g, "same query input here", "fwd")
    # Either answer is acceptable (collision is expected behavior); check
    # only that it's one of the trained answers.
    check("identical input returns one of the trained answers",
          pf in [pairs[0][1], pairs[1][1]],
          f"got {pf[:30]!r}")


def it8_determinism(g, pairs):
    print("\nIT8: Determinism — same train/query order produces same result")
    results_a = []
    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs)
    for q, _ in pairs:
        results_a.append(query_one(tp, g, q, "fwd"))
    results_b = []
    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs)
    for q, _ in pairs:
        results_b.append(query_one(tp, g, q, "fwd"))
    check("two identical runs agree on every query",
          results_a == results_b,
          f"diffs: {sum(1 for x, y in zip(results_a, results_b) if x != y)}")


# ─────────────────────────────────────────────────────────────────── runner

def main():
    be_polite()

    pairs8 = [
        ("greet warmly", "hello there friend welcome aboard"),
        ("greet coldly", "acknowledged standard protocol greeting"),
        ("explain quantum physics",
         "quantum mechanics superposition entanglement uncertainty"),
        ("explain classical physics",
         "classical mechanics newton forces gravity motion"),
        ("describe weather sunny",
         "today bright skies warm temperatures gentle breeze"),
        ("describe spring season",
         "currently blooming flowers warming temperatures growth"),
        ("describe ocean waves",
         "deep blue rolling waves crashing against shores"),
        ("describe forest morning",
         "ancient towering pines dappled sunlight quiet birds"),
    ]

    g = winning_genome()
    print("=" * 74)
    print("INTEGRATION TESTS — winning genome (lev concept-selection)")
    print("=" * 74)
    print(f"PYTHONHASHSEED={os.environ.get('PYTHONHASHSEED')}")

    tp, _ = build_two_pool(g)
    train_pairs(tp, g, pairs8)

    it1_recall(tp, g, pairs8)
    it2_repeat(tp, g, pairs8)
    new_pair = ("describe new topic added later",
                "novel content emerging from incremental training")
    it3_incremental(g, pairs8, new_pair)
    it4_ood(tp, g, pairs8)
    it5_edges(tp, g)
    it6_mixed_lengths(g)
    it7_identical_inputs(g)
    it8_determinism(g, pairs8)

    print("\n" + "=" * 74)
    print(f"RESULTS: {totals['pass']} passed, {totals['fail']} failed")
    print("=" * 74)
    sys.exit(0 if totals['fail'] == 0 else 1)


if __name__ == "__main__":
    main()
