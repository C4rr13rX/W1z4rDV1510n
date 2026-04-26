"""Stress test for the two_pool architectural fixes.

Builds harder corpora than the 8-pair baseline and runs the winning genome
to see whether 1.0 holds up:

  T1: 8-pair baseline (sanity check)
  T2: 24 pairs with shared prefixes ("describe X", "explain X", "summarize X")
  T3: 16 paraphrase pairs (q strings differ by 1-3 chars; outputs distinct)
  T4: 32 pairs from a synthetic encyclopedia with overlapping topics
"""
import sys, time
sys.path.insert(0, "scripts")

from ga_neuro_config_search import (
    DEFAULT_GENOME, evaluate, be_polite, adaptive_yield,
)


def t1_baseline():
    from ga_neuro_config_search import make_corpus
    return make_corpus(scale="small")


def t2_shared_prefix():
    """Many concepts whose chars overlap heavily — tests overlap re-ranker."""
    topics = [
        "weather sunny",     "weather rainy",     "weather snowy",
        "physics quantum",   "physics classical", "physics relativity",
        "biology cells",     "biology genetics",  "biology evolution",
        "history ancient",   "history medieval",  "history modern",
        "math algebra",      "math calculus",     "math topology",
        "music classical",   "music jazz",        "music electronic",
        "cooking italian",   "cooking japanese",  "cooking french",
        "sports baseball",   "sports football",   "sports tennis",
    ]
    answers = [
        "today bright skies warm temperatures",
        "today wet grey overcast precipitation",
        "today white frozen flakes accumulating",
        "superposition entanglement uncertainty principle",
        "newton mechanics forces motion gravity",
        "einstein spacetime curvature lorentz invariance",
        "membrane organelles mitosis differentiation",
        "dna chromosomes mendel inheritance traits",
        "darwin selection adaptation speciation lineages",
        "egypt pyramids pharaohs nile civilization",
        "knights castles feudalism crusades guilds",
        "industrial revolution democracy globalization",
        "variables equations linear quadratic systems",
        "derivatives integrals limits differential",
        "manifolds homotopy continuous deformation",
        "symphony orchestra concerto sonata baroque",
        "improvisation swing blues bebop ensemble",
        "synthesizer drum machine sequencer beats",
        "pasta tomato basil mozzarella olive oil",
        "sushi miso tempura matcha umami dashi",
        "baguette croissant coq au vin escargot",
        "diamond pitcher batter homerun innings",
        "quarterback touchdown linebacker scrimmage",
        "racquet serve volley baseline backhand",
    ]
    pairs = list(zip(["describe " + t for t in topics], answers))
    return pairs


def t3_paraphrase():
    """Inputs differ by tiny edits; outputs are wholly distinct.  Tests
    whether overlap is robust to small input changes."""
    base = "explain {x} please now"
    topics = [
        "fire", "rain", "snow", "wind",
        "fish", "bird", "tree", "moon",
        "star", "rock", "leaf", "wave",
        "soup", "cake", "salt", "oil",
    ]
    answers = [
        "combustion oxygen heat releasing energy chemical reaction",
        "atmospheric water condensation precipitation falling drops",
        "frozen water crystals hexagonal lattice falling slowly",
        "moving air pressure differential atmospheric circulation",
        "aquatic vertebrate gills scales swim bladder fins",
        "feathered vertebrate beak wings hollow bones flight",
        "perennial plant trunk branches photosynthesis bark",
        "natural satellite earth tidal locking phases lunar",
        "luminous celestial body nuclear fusion plasma sphere",
        "mineral aggregate solid lithified geological formation",
        "photosynthetic plant organ chlorophyll vascular tissue",
        "oscillating water surface energy propagation periodic",
        "liquid food broth simmered vegetables herbs spices",
        "sweet baked dessert flour eggs sugar leavening",
        "sodium chloride crystalline ionic seasoning preservative",
        "viscous liquid lipid hydrophobic refined extraction",
    ]
    return list(zip([base.format(x=t) for t in topics], answers))


def t4_encyclopedia():
    """32 pairs from 4 distinct topic clusters, each cluster sharing
    many chars internally but differing across clusters."""
    pairs = []
    pairs += [
        ("animal cat domestic pet feline",
         "small carnivorous mammal whiskers retractable claws purring"),
        ("animal dog domestic pet canine",
         "loyal mammal barking olfactory acute pack social companion"),
        ("animal cow domestic farm bovine",
         "ruminant mammal four chambered stomach grass milk producing"),
        ("animal pig domestic farm porcine",
         "intelligent mammal omnivorous social truffle hunting traits"),
        ("animal horse domestic riding equine",
         "ungulate mammal hooves galloping mane tail working partner"),
        ("animal sheep domestic wool ovine",
         "ruminant mammal wool fleece lanolin shepherd flock grazing"),
        ("animal goat domestic milk caprine",
         "agile climber browser horns beard cheese fiber sustenance"),
        ("animal rabbit domestic pet lagomorph",
         "burrowing mammal long ears swift hind legs vegetation diet"),
    ]
    pairs += [
        ("element hydrogen lightest universe",
         "atomic number one proton electron isotopes water synthesis"),
        ("element helium noble inert gas",
         "atomic number two stable shell balloons cryogenic coolant"),
        ("element carbon backbone organic",
         "atomic number six allotropes diamond graphite covalent bonds"),
        ("element oxygen breathable diatomic",
         "atomic number eight respiration combustion oxidation reactive"),
        ("element nitrogen abundant atmosphere",
         "atomic number seven inert triple bond fertilizer protein"),
        ("element iron ferrous metal magnetic",
         "atomic number twenty six oxide rust core hemoglobin alloy"),
        ("element gold precious metal lustrous",
         "atomic number seventy nine ductile conductive jewelry coinage"),
        ("element silver precious metal conductive",
         "atomic number forty seven malleable utensils mirrors photography"),
    ]
    pairs += [
        ("planet mercury inner closest sun",
         "smallest rocky body extreme temperatures thin exosphere cratered"),
        ("planet venus second hottest atmosphere",
         "thick carbon dioxide runaway greenhouse retrograde rotation"),
        ("planet earth third habitable life",
         "liquid water nitrogen oxygen biosphere magnetic field moon"),
        ("planet mars fourth red iron oxide",
         "thin atmosphere polar caps olympus mons valles marineris dust"),
        ("planet jupiter fifth largest gas giant",
         "great red spot hydrogen helium sixty seven moons strong magnet"),
        ("planet saturn sixth ringed gas giant",
         "complex ice rings titan enceladus hexagonal pole low density"),
        ("planet uranus seventh ice giant",
         "axial tilt sideways rotation methane atmosphere blue green hue"),
        ("planet neptune eighth ice giant deepest",
         "supersonic winds dark spot triton retrograde discovered math"),
    ]
    pairs += [
        ("language python interpreted dynamic",
         "indentation whitespace duck typing rich standard library readable"),
        ("language rust compiled systems memory",
         "ownership borrow checker zero cost abstractions safe concurrency"),
        ("language javascript browser web async",
         "prototype inheritance event loop callbacks promises ecosystem"),
        ("language haskell functional pure lazy",
         "type inference monads algebraic data types referentially transparent"),
        ("language go simple concurrent compiled",
         "goroutines channels interfaces gc small standard library fast"),
        ("language lisp homoiconic macros recursion",
         "parentheses code as data symbolic computation atom cons cells"),
        ("language sql declarative relational queries",
         "joins predicates aggregates indexes acid normalization schemas"),
        ("language assembly low level processor",
         "registers opcodes mnemonics instruction set hardware proximity"),
    ]
    return pairs


def run_one(name, pairs, genome):
    in_lens  = [len(q) for q, _ in pairs]
    out_lens = [len(a) for _, a in pairs]
    print(f"\n=== {name} ===")
    print(f"  pairs={len(pairs)}  in {min(in_lens)}..{max(in_lens)}  "
          f"out {min(out_lens)}..{max(out_lens)}")
    t0 = time.time()
    m = evaluate(genome, pairs)
    dt = time.time() - t0
    print(f"  combined={m['combined']:.4f}  fwd={m['fwd']:.3f}  "
          f"rev={m['rev']:.3f}  exact={m['exact_fwd']}/{m['exact_rev']}  "
          f"of {m['n_pairs']}  ({dt:.1f}s)")
    return m


def main():
    be_polite()
    base = dict(DEFAULT_GENOME)
    base['use_full_sequence']           = 1
    base['concept_only_cross']          = 1
    base['reset_activations_per_pair']  = 1
    base['prop_max_accum']              = 1
    base['tgt_concept_argmax']          = 1

    variants = [
        ("overlap (set intersection)",
            {'src_concept_overlap': 1, 'src_concept_lev': 0, 'src_concept_tfidf': 0}),
        ("tfidf (1/freq weighted overlap)",
            {'src_concept_overlap': 0, 'src_concept_lev': 0, 'src_concept_tfidf': 1}),
        ("lev (sequence sim)",
            {'src_concept_overlap': 0, 'src_concept_lev': 1, 'src_concept_tfidf': 0}),
    ]

    corpora = [
        ("T1: 8-pair baseline",          t1_baseline()),
        ("T2: 24-pair shared prefixes",  t2_shared_prefix()),
        ("T3: 16-pair paraphrase",       t3_paraphrase()),
        ("T4: 32-pair encyclopedia",     t4_encyclopedia()),
    ]

    print("=" * 76)
    print("STRESS TEST — concept selection strategies × hard corpora")
    print("=" * 76)
    for vname, vtoggles in variants:
        print(f"\n--- variant: {vname} ---")
        g = dict(base); g.update(vtoggles)
        for cname, cpairs in corpora:
            run_one(cname, cpairs, g)
            adaptive_yield()


if __name__ == "__main__":
    main()
