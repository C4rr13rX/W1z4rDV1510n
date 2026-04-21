#!/usr/bin/env python3
"""
build_cognitive_corpus.py -- Stages 27 & 28

Stage 27 -- Cognitive / IQ / logical reasoning
  * Hand-crafted corpus: number sequences, analogies, syllogisms, logic
    puzzles, spatial reasoning, lateral thinking, critical thinking
  * OEIS API: top integer sequences with descriptions and formulas
  * Wikipedia: articles on reasoning, logic, cognitive biases, heuristics

Stage 28 -- Sorting & searching algorithms
  * Every major sorting algorithm: description, step-by-step walkthrough,
    complexity analysis, stability/in-place/adaptive properties, when to
    use vs avoid, interview Q&A
  * Implementations in Python, C++, and Rust for each algorithm
  * Searching algorithms: binary search, interpolation search, etc.
  * Common interview patterns and pitfalls

Usage:
  python scripts/build_cognitive_corpus.py --node localhost:8090
  python scripts/build_cognitive_corpus.py --stages 27
  python scripts/build_cognitive_corpus.py --stages 28
"""

import argparse, json, re, sys, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
DEFAULT_NODE     = 'localhost:8090'
UA = 'W1z4rDV1510n-Cognitive/1.0 (adamedsall@gmail.com; educational AI)'
STAGES = {
    27: 'Cognitive / IQ reasoning -- sequences, logic, analogies, lateral thinking',
    28: 'Sorting & searching algorithms -- all major algorithms, multi-language impls',
}

# -- HTTP -----------------------------------------------------------------------

def _make_session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=3))
    s.mount('http://',  HTTPAdapter(max_retries=3))
    s.headers['User-Agent'] = UA
    return s

def _train(text, node, session):
    try:
        r = session.post(f'http://{node}/media/train',
                         data=json.dumps({'modality': 'text', 'text': text}),
                         headers={'Content-Type': 'application/json'}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f'  [WARN] train: {e}', flush=True)
        return False

# ==============================================================================
# STAGE 27 -- COGNITIVE CORPUS
# ==============================================================================

COGNITIVE_PROBLEMS = [
    # -- Number sequences ------------------------------------------------------
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 2, 6, 12, 20, 30, 42, ?',
        'reasoning': (
            'Examine first differences: 4, 6, 8, 10, 12 -- increasing by 2 each time.\n'
            'Second differences are constant: 2. This is a second-order arithmetic sequence.\n'
            'Next first difference: 12 + 2 = 14. Next term: 42 + 14 = 56.\n'
            'Formula: n(n+1) for n=1,2,3... -> 2,6,12,20,30,42,56. These are pronic (oblong) numbers.'
        ),
        'answer': '56',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 4, 9, 16, 25, 36, ?',
        'reasoning': 'These are perfect squares: 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36, 7^2=49.',
        'answer': '49',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 3, 6, 11, 18, 27, 38, ?',
        'reasoning': (
            'First differences: 3, 5, 7, 9, 11 -- odd numbers increasing by 2.\n'
            'Next difference: 13. Next term: 38 + 13 = 51.\n'
            'Formula: n^2 + 2 for n=1,2,3...'
        ),
        'answer': '51',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 2, 4, 8, 16, 32, ?',
        'reasoning': 'Powers of 2: each term multiplied by 2. 32 x 2 = 64.',
        'answer': '64',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 1, 2, 3, 5, 8, 13, 21, ?',
        'reasoning': (
            'Fibonacci sequence: each term is the sum of the two preceding terms.\n'
            '13 + 21 = 34. The ratio of consecutive terms converges to φ ~= 1.618 (golden ratio).'
        ),
        'answer': '34',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 2, 3, 5, 7, 11, 13, 17, 19, 23, ?',
        'reasoning': 'Prime numbers in order. The next prime after 23 is 29 (not divisible by 2, 3, or 5).',
        'answer': '29',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 3, 6, 10, 15, 21, 28, ?',
        'reasoning': (
            'Triangular numbers: T(n) = n(n+1)/2. Differences: 2,3,4,5,6,7,8.\n'
            'T(8) = 8x9/2 = 36. These count objects arranged in equilateral triangles.'
        ),
        'answer': '36',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 8, 27, 64, 125, 216, ?',
        'reasoning': 'Perfect cubes: 1^3=1, 2^3=8, 3^3=27, 4^3=64, 5^3=125, 6^3=216, 7^3=343.',
        'answer': '343',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 0, 1, 3, 7, 15, 31, 63, ?',
        'reasoning': (
            'Each term = 2^n - 1 for n=0,1,2,3... Or: double previous term and add 1.\n'
            '63x2+1=127. These are Mersenne numbers: 2^7-1=127.'
        ),
        'answer': '127',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 1, 2, 6, 24, 120, 720, ?',
        'reasoning': 'Factorials: 1!=1, 2!=2, 3!=6, 4!=24, 5!=120, 6!=720, 7!=5040.',
        'answer': '5040',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 4, 7, 12, 19, 28, 39, ?',
        'reasoning': (
            'First differences: 3, 5, 7, 9, 11 -- odd numbers.\n'
            'Next difference: 13. 39 + 13 = 52. Formula: n^2 + 3.'
        ),
        'answer': '52',
    },
    {
        'type': 'Number Sequence',
        'problem': 'What comes next: 100, 50, 25, 12.5, 6.25, ?',
        'reasoning': 'Each term divided by 2 (geometric sequence, ratio 1/2). 6.25 / 2 = 3.125.',
        'answer': '3.125',
    },
    # -- Verbal Analogies ------------------------------------------------------
    {
        'type': 'Verbal Analogy',
        'problem': 'Doctor : Hospital :: Teacher : ?',
        'reasoning': 'A doctor works in a hospital. A teacher works in a school. Relationship: professional -> workplace.',
        'answer': 'School',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Hot : Cold :: Light : ?',
        'reasoning': 'Hot and cold are antonyms. Light and heavy are antonyms.',
        'answer': 'Heavy',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Glove : Hand :: Shoe : ?',
        'reasoning': 'A glove covers a hand. A shoe covers a foot. Relationship: covering -> body part.',
        'answer': 'Foot',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Carpenter : Wood :: Mason : ?',
        'reasoning': 'A carpenter works with wood. A mason works with stone/brick.',
        'answer': 'Stone / Brick',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Symphony : Composer :: Painting : ?',
        'reasoning': 'A symphony is created by a composer. A painting is created by a painter/artist.',
        'answer': 'Painter / Artist',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Proud : Humble :: Brave : ?',
        'reasoning': 'Proud and humble are antonyms. Brave and cowardly are antonyms.',
        'answer': 'Cowardly',
    },
    {
        'type': 'Verbal Analogy',
        'problem': 'Book : Library :: Painting : ?',
        'reasoning': 'Books are stored in a library. Paintings are stored in a gallery/museum.',
        'answer': 'Gallery / Museum',
    },
    # -- Logical Deduction / Syllogisms ----------------------------------------
    {
        'type': 'Syllogism',
        'problem': 'All mammals are warm-blooded. Whales are mammals. Therefore?',
        'reasoning': (
            'Valid syllogism (Barbara form: All M are P; All S are M; Therefore All S are P).\n'
            'Premise 1: All mammals -> warm-blooded.\n'
            'Premise 2: Whales -> mammals.\n'
            'Conclusion follows necessarily: whales are warm-blooded.'
        ),
        'answer': 'Whales are warm-blooded.',
    },
    {
        'type': 'Syllogism',
        'problem': 'Some doctors are musicians. All musicians are creative. What can we conclude?',
        'reasoning': (
            'Premise 1: Some doctors are musicians (partial overlap).\n'
            'Premise 2: All musicians -> creative.\n'
            'Since SOME doctors are musicians, those doctors are creative.\n'
            'Conclusion: Some doctors are creative. (Cannot say ALL doctors are creative.)'
        ),
        'answer': 'Some doctors are creative.',
    },
    {
        'type': 'Syllogism',
        'problem': 'No reptiles have fur. All snakes are reptiles. Therefore?',
        'reasoning': (
            'Premise 1: No reptiles have fur.\n'
            'Premise 2: All snakes are reptiles.\n'
            'If X is a snake -> X is a reptile -> X has no fur.\n'
            'Conclusion: No snakes have fur.'
        ),
        'answer': 'No snakes have fur.',
    },
    {
        'type': 'Logic Puzzle',
        'problem': (
            'Three friends -- Alice, Bob, Carol -- each like a different sport: '
            'tennis, swimming, cycling. Alice does not like tennis. '
            'Bob does not like swimming or cycling. Who likes what?'
        ),
        'reasoning': (
            'Step 1: Bob does not like swimming or cycling -> Bob likes tennis.\n'
            'Step 2: Alice does not like tennis (given), and tennis is taken by Bob.\n'
            '        Alice likes swimming or cycling.\n'
            'Step 3: Carol gets whichever Alice does not take. No more constraints, '
            '        so Alice could like swimming or cycling. But since no further '
            '        distinction is given, both are valid unless another clue exists.\n'
            'Definite: Bob -> tennis. Alice and Carol split swimming and cycling.'
        ),
        'answer': 'Bob: tennis. Alice and Carol: one each takes swimming and cycling.',
    },
    {
        'type': 'Truth/Liar Puzzle',
        'problem': (
            'On an island, knights always tell the truth and knaves always lie. '
            'A says: "At least one of us is a knave." What are A and B?'
        ),
        'reasoning': (
            'Assume A is a knight (tells truth). Statement "at least one of us is a knave" '
            'would be true -> B is a knave. Consistent.\n'
            'Assume A is a knave (lies). Statement would be false -> neither is a knave, '
            'but A is a knave -- contradiction.\n'
            'Therefore A is a knight (truth-teller) and B is a knave.'
        ),
        'answer': 'A is a knight; B is a knave.',
    },
    {
        'type': 'Logic Puzzle',
        'problem': (
            'Five houses in a row are colored: red, blue, green, yellow, white. '
            'The green house is immediately to the left of the white house. '
            'The red house is in the middle. The blue house is the first on the left. '
            'What is the order?'
        ),
        'reasoning': (
            'Constraint: blue is first (position 1).\n'
            'Constraint: red is middle (position 3).\n'
            'Constraint: green is immediately left of white -> green-white are consecutive.\n'
            'Positions 1=blue, 3=red. Remaining: 2,4,5 for yellow, green, white.\n'
            'Green-white must be consecutive: options are (2,3), (3,4), or (4,5).\n'
            '(2,3) fails because 3=red. (3,4) fails same reason.\n'
            'Therefore: green=4, white=5. Yellow=2.\n'
            'Order: blue, yellow, red, green, white.'
        ),
        'answer': 'Blue, Yellow, Red, Green, White.',
    },
    # -- Matrix / Pattern Completion --------------------------------------------
    {
        'type': 'Matrix Pattern',
        'problem': (
            'In a 3x3 matrix:\n'
            'Row 1: ○ □ △\n'
            'Row 2: *  # ▲\n'
            'Row 3: ○ # ?\n'
            'What fills the ?'
        ),
        'reasoning': (
            'Column pattern: col 1 alternates hollow/filled: ○*○ -> hollow.\n'
            'Column pattern: col 2: □## -> filled.\n'
            'Column pattern: col 3: △▲? -- follows filled/hollow rule from column.\n'
            'Row pattern: row 3 has hollow ○ and filled # -> mix.\n'
            'Shape rule: col 3 has triangle. Shape stays, fill toggles between rows.\n'
            'Row 1: △ hollow; Row 2: ▲ filled; Row 3 col 1 is hollow again -> △ hollow.'
        ),
        'answer': '△ (hollow triangle)',
    },
    {
        'type': 'Matrix Pattern',
        'problem': (
            'Find the missing number:\n'
            '| 2  4  8  |\n'
            '| 3  9  27 |\n'
            '| 4  16  ? |'
        ),
        'reasoning': (
            'Row 1: 2, 2^2, 2^3 -- powers of the row-start value.\n'
            'Row 2: 3, 3^2, 3^3 -- same pattern.\n'
            'Row 3: 4, 4^2, 4^3 = 4, 16, 64.\n'
            'Pattern: column n contains the row-start raised to power n.'
        ),
        'answer': '64',
    },
    {
        'type': 'Matrix Pattern',
        'problem': (
            '| 6   10  14 |\n'
            '| 15  19  23 |\n'
            '| 28  ?   36 |'
        ),
        'reasoning': (
            'Row differences: all rows differ by +4 across columns.\n'
            'Column differences: col 1: 6,15,28 -> diff 9,13 -> diff of diffs = 4.\n'
            'Col 2: 10,19,? -> same pattern -> diff 9, then 13 -> 19+13=32.'
        ),
        'answer': '32',
    },
    # -- Spatial Reasoning -----------------------------------------------------
    {
        'type': 'Spatial Reasoning',
        'problem': (
            'A cube has each face painted a different color: red (top), blue (bottom), '
            'green (front), yellow (back), orange (left), white (right). '
            'If you rotate the cube 90deg to the right (right face goes down), '
            'which color is now on top?'
        ),
        'reasoning': (
            'Start: top=red, bottom=blue, front=green, back=yellow, left=orange, right=white.\n'
            'Rotation 90deg right means the right face goes down, left goes up.\n'
            'New positions: top=orange (was left), bottom=white (was right), '
            'left=blue (was bottom), right=red (was top).\n'
            'Front and back unchanged: front=green, back=yellow.'
        ),
        'answer': 'Orange is on top.',
    },
    {
        'type': 'Spatial Reasoning',
        'problem': (
            'How many cubes are needed to build a 4x4x4 solid cube? '
            'If the outside is painted, how many small cubes have exactly two painted faces?'
        ),
        'reasoning': (
            'Total cubes: 4x4x4 = 64.\n'
            'Two painted faces = edge cubes (not corners, not face-only).\n'
            'Edges of a cube: 12 edges x (4-2) interior cubes per edge = 12x2 = 24.\n'
            '(Subtract 2 from each edge for the corner cubes shared between 3 faces.)'
        ),
        'answer': '64 total; 24 with exactly 2 painted faces.',
    },
    # -- Probability & Combinatorics -------------------------------------------
    {
        'type': 'Probability',
        'problem': 'A bag has 3 red and 5 blue marbles. Two drawn without replacement. P(both red)?',
        'reasoning': (
            'P(first red) = 3/8.\n'
            'P(second red | first red) = 2/7 (one fewer red, one fewer total).\n'
            'P(both red) = (3/8) x (2/7) = 6/56 = 3/28 ~= 0.107.'
        ),
        'answer': '3/28 ~= 10.7%',
    },
    {
        'type': 'Combinatorics',
        'problem': 'How many ways to arrange the letters in MENSA?',
        'reasoning': (
            'MENSA has 5 distinct letters. Number of arrangements = 5! = 5x4x3x2x1 = 120.\n'
            'If letters repeated, divide by factorial of repeat count (none here).'
        ),
        'answer': '120',
    },
    {
        'type': 'Probability',
        'problem': 'Two fair dice rolled. P(sum = 7)?',
        'reasoning': (
            'Total outcomes: 6x6 = 36.\n'
            'Pairs summing to 7: (1,6),(2,5),(3,4),(4,3),(5,2),(6,1) -- 6 pairs.\n'
            'P(sum=7) = 6/36 = 1/6 ~= 16.7%.'
        ),
        'answer': '1/6',
    },
    {
        'type': 'Combinatorics',
        'problem': 'How many ways to choose 3 people from a group of 10?',
        'reasoning': (
            'Combination (order does not matter): C(10,3) = 10! / (3! x 7!) = 120.\n'
            'C(n,k) = n(n-1)(n-2).../ k! = (10x9x8)/(3x2x1) = 720/6 = 120.'
        ),
        'answer': '120',
    },
    # -- Lateral Thinking ------------------------------------------------------
    {
        'type': 'Lateral Thinking',
        'problem': 'A man walks into a restaurant and orders albatross soup. After one sip he goes home and kills himself. Why?',
        'reasoning': (
            'He was shipwrecked with his wife. She died. His companion told him he was eating '
            '"albatross soup" -- but it was actually his wife\'s flesh. When he tasted real '
            'albatross soup in the restaurant, he realized what he had really eaten before, '
            'overwhelming him with guilt.'
        ),
        'answer': (
            'He realized the "albatross soup" he ate during the shipwreck was not '
            'actually albatross -- it was his deceased wife. The guilt was unbearable.'
        ),
    },
    {
        'type': 'Lateral Thinking',
        'problem': 'A woman shoots her husband, then has dinner with him. How?',
        'reasoning': (
            'No rule says "shoot" must mean with a gun. She is a photographer: she takes '
            'his photograph (shoots him), then they go to dinner.'
        ),
        'answer': 'She is a photographer. She shoots his photograph.',
    },
    {
        'type': 'Lateral Thinking',
        'problem': 'How can you drop a raw egg onto a concrete floor without breaking it?',
        'reasoning': 'Concrete floors are very hard. A raw egg will not break the concrete floor.',
        'answer': 'Drop it -- concrete floors are very hard and will not break from an egg.',
    },
    # -- Mathematical Reasoning ------------------------------------------------
    {
        'type': 'Mathematical Reasoning',
        'problem': (
            'A train 200 m long travels at 60 km/h. How long to fully pass a 400 m platform?'
        ),
        'reasoning': (
            'Total distance to clear platform = train length + platform length = 200+400=600 m.\n'
            'Speed = 60 km/h = 60x1000/3600 = 16.67 m/s.\n'
            'Time = distance/speed = 600/16.67 ~= 36 seconds.'
        ),
        'answer': '36 seconds',
    },
    {
        'type': 'Mathematical Reasoning',
        'problem': 'A clock reads 3:00. What is the angle between the hour and minute hands?',
        'reasoning': (
            'The clock face is 360deg, divided into 12 hours = 30deg per hour.\n'
            'At 3:00: minute hand at 12 (0deg), hour hand at 3 (90deg).\n'
            'Angle between them = 90deg.'
        ),
        'answer': '90deg',
    },
    {
        'type': 'Mathematical Reasoning',
        'problem': 'If you have a 3-gallon jug and a 5-gallon jug, how do you measure exactly 4 gallons?',
        'reasoning': (
            'Step 1: Fill 5-gallon jug. Pour into 3-gallon until full (leaves 2 in 5-gal).\n'
            'Step 2: Empty 3-gallon. Pour 2 gallons from 5-gal into 3-gal jug.\n'
            'Step 3: Fill 5-gallon again. Pour into 3-gal jug (which has 2, needs 1 more).\n'
            'Step 4: 5-gallon now has 5-1=4 gallons. Done.'
        ),
        'answer': '4 gallons in the 5-gallon jug after 4 steps.',
    },
    {
        'type': 'Mathematical Reasoning',
        'problem': 'You have 12 identical-looking coins; one is heavier. Find it in 3 weighings.',
        'reasoning': (
            'Divide into groups of 4-4-4.\n'
            'Weigh group 1 vs group 2:\n'
            '  If balanced -> heavy coin in group 3. Weigh 2 from group 3 vs 2 others.\n'
            '  If unbalanced -> heavy coin in heavier group of 4.\n'
            'Subdivide the suspect group of 4 into 2-2, weigh them. '
            'The heavier side has 2 suspects.\n'
            'Weigh the 2 suspects against each other -> heavier one found.\n'
            '3 weighings total: log₃(12) ~= 2.26 -> 3 is sufficient.'
        ),
        'answer': '3 weighings using divide-by-3 strategy.',
    },
    # -- Critical Thinking / Argument Analysis ---------------------------------
    {
        'type': 'Critical Thinking',
        'problem': (
            'Argument: "Every time I carry an umbrella, it doesn\'t rain. '
            'My umbrella must prevent rain." Identify the flaw.'
        ),
        'reasoning': (
            'Flaw: Post hoc ergo propter hoc (correlation != causation).\n'
            'The person carries an umbrella BECAUSE rain is likely. When they carry it '
            'and it does not rain, that is coincidence or selection bias (they perhaps '
            'carry it less on clearly sunny days). The umbrella has no causal power over weather.'
        ),
        'answer': 'Post hoc / correlation-causation fallacy. Carrying an umbrella cannot control weather.',
    },
    {
        'type': 'Critical Thinking',
        'problem': (
            '"Our competitor\'s product failed in tests. Therefore our product is the best." '
            'What logical fallacy is this?'
        ),
        'reasoning': (
            'False dilemma / false dichotomy: assumes only two options exist (their product or ours). '
            'There may be many other competitors. Also a non sequitur: competitor failure does not '
            'establish that your product is the BEST -- only that theirs failed.'
        ),
        'answer': 'False dilemma (false dichotomy) and non sequitur.',
    },
]

# Key cognitive topics to train via Wikipedia
COGNITIVE_WIKI_TOPICS = [
    'Deductive_reasoning', 'Inductive_reasoning', 'Abductive_reasoning',
    'Logical_fallacy', 'Syllogism', 'Cognitive_bias', 'Heuristic',
    'Lateral_thinking', 'Critical_thinking', 'Abstract_reasoning',
    'Working_memory', 'Fluid_and_crystallized_intelligence',
    'IQ_classification', 'Mensa_International',
    'Dunning-Kruger_effect', 'Occam\'s_razor',
    'Game_theory', 'Decision_theory', 'Bayesian_inference',
    'Mathematical_puzzle', 'Logic_puzzle', 'Brain_teaser',
]


def _wiki_text(title: str, session: requests.Session) -> str:
    url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{title}'
    try:
        r = session.get(url, timeout=15)
        if r.ok:
            return r.json().get('extract', '')
    except Exception:
        pass
    return ''


def _oeis_sequences(session: requests.Session, max_seqs: int = 120) -> list:
    """Fetch core OEIS sequences with descriptions and examples."""
    results = []
    for start in range(0, max_seqs, 25):
        url = ('https://oeis.org/search?q=keyword%3Acore&fmt=json'
               f'&start={start}&count=25')
        try:
            r = session.get(url, timeout=20)
            if not r.ok:
                break
            data = r.json()
            for seq in data.get('results', []):
                name   = seq.get('name', '')
                values = ','.join(str(v) for v in (seq.get('data') or '').split(',')[:12])
                comment = ' '.join((seq.get('comment') or [])[:2])
                formula = ' '.join((seq.get('formula') or [])[:1])
                text = (
                    f'Integer sequence A{seq.get("number",""):06d}: {name}\n'
                    f'First terms: {values}\n'
                )
                if comment:
                    text += f'Notes: {comment[:400]}\n'
                if formula:
                    text += f'Formula: {formula[:200]}\n'
                results.append(text)
        except Exception as e:
            print(f'  [WARN] OEIS page {start}: {e}', flush=True)
        time.sleep(0.5)
    return results


def build_cognitive(out_dir: Path, node: str) -> list:
    """Stage 27: Cognitive / IQ / logical reasoning corpus."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()
    items = []
    ok = 0

    # 1. Hand-crafted reasoning problems
    print(f'  Cognitive: {len(COGNITIVE_PROBLEMS)} hand-crafted problems', flush=True)
    for p in COGNITIVE_PROBLEMS:
        text = (
            f'Cognitive reasoning -- {p["type"]}\n\n'
            f'Problem: {p["problem"]}\n\n'
            f'Step-by-step reasoning:\n{p["reasoning"]}\n\n'
            f'Answer: {p["answer"]}'
        )
        if _train(text, node, session):
            ok += 1
        time.sleep(0.08)

    # 2. Wikipedia cognitive/reasoning articles
    print(f'  Cognitive: fetching {len(COGNITIVE_WIKI_TOPICS)} Wikipedia articles', flush=True)
    for title in COGNITIVE_WIKI_TOPICS:
        extract = _wiki_text(title, session)
        if extract and len(extract) > 100:
            text = f'Cognitive science / reasoning -- {title.replace("_"," ")}\n\n{extract}'
            if _train(text, node, session):
                ok += 1
        time.sleep(0.15)

    # 3. OEIS integer sequences
    print('  Cognitive: fetching OEIS core sequences...', flush=True)
    seqs = _oeis_sequences(session, max_seqs=120)
    print(f'  Cognitive: {len(seqs)} OEIS sequences', flush=True)
    for seq_text in seqs:
        if _train(seq_text, node, session):
            ok += 1
        time.sleep(0.08)

    print(f'  Cognitive corpus: {ok} items trained', flush=True)
    items.append({'stage': 27, 'type': 'cognitive', 'trained': ok,
                  'modality': 'text', 'tags': ['iq', 'reasoning', 'logic']})
    return items


# ==============================================================================
# STAGE 28 -- SORTING & ALGORITHMS CORPUS
# ==============================================================================

SORTING_ALGORITHMS = [
    {
        'name': 'Bubble Sort',
        'category': 'Comparison / Exchange',
        'description': (
            'Repeatedly steps through the list, compares adjacent elements, and swaps them '
            'if they are in the wrong order. Each pass "bubbles" the largest unsorted element '
            'to its correct position at the end.'
        ),
        'complexity': {'best': 'O(n)', 'average': 'O(n^2)', 'worst': 'O(n^2)', 'space': 'O(1)'},
        'properties': 'Stable: YES. In-place: YES. Adaptive: YES (O(n) when nearly sorted with early-exit).',
        'use_when': 'Nearly sorted data; teaching/demonstration; n < 20.',
        'avoid_when': 'Large datasets. Never use in production for n > 1000.',
        'walkthrough': (
            'Array: [5, 3, 1, 4, 2]\n'
            'Pass 1: compare(5,3)->swap -> [3,5,1,4,2]; compare(5,1)->swap -> [3,1,5,4,2]; '
            'compare(5,4)->swap -> [3,1,4,5,2]; compare(5,2)->swap -> [3,1,4,2,5]. Max(5) placed.\n'
            'Pass 2: -> [1,3,2,4,5] (4 placed).\n'
            'Continue until sorted: [1,2,3,4,5].'
        ),
        'python': '''\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:   # already sorted -- O(n) best case
            break
    return arr''',
        'cpp': '''\
void bubble_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; ++i) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; ++j)
            if (a[j] > a[j+1]) { swap(a[j], a[j+1]); swapped = true; }
        if (!swapped) break;
    }
}''',
        'rust': '''\
fn bubble_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in 0..n {
        let mut swapped = false;
        for j in 0..n - i - 1 {
            if arr[j] > arr[j + 1] { arr.swap(j, j + 1); swapped = true; }
        }
        if !swapped { break; }
    }
}''',
        'interview': [
            ('Why is bubble sort O(n) best case?', 'With the swapped flag, if no swaps occur in a pass, the array is sorted -- we exit after one O(n) pass.'),
            ('Is bubble sort stable?', 'Yes -- it only swaps adjacent elements when a[j] > a[j+1], never when equal, preserving relative order.'),
        ],
    },
    {
        'name': 'Selection Sort',
        'category': 'Comparison / Selection',
        'description': (
            'Divides the array into a sorted and unsorted region. On each pass, finds the minimum '
            'element in the unsorted region and swaps it to the end of the sorted region.'
        ),
        'complexity': {'best': 'O(n^2)', 'average': 'O(n^2)', 'worst': 'O(n^2)', 'space': 'O(1)'},
        'properties': 'Stable: NO (standard version). In-place: YES. Adaptive: NO.',
        'use_when': 'When writes are expensive (minimizes swaps -- exactly n-1 swaps). Small arrays.',
        'avoid_when': 'Large data; when stability required (use insertion sort instead).',
        'walkthrough': (
            'Array: [5, 3, 1, 4, 2]\n'
            'Pass 1: min of [5,3,1,4,2]=1 at idx 2. Swap idx 0 and 2 -> [1,3,5,4,2].\n'
            'Pass 2: min of [3,5,4,2]=2 at idx 4. Swap idx 1 and 4 -> [1,2,5,4,3].\n'
            'Pass 3: min of [5,4,3]=3 at idx 4. Swap idx 2 and 4 -> [1,2,3,4,5].\n'
            'Done in n-1=4 swaps.'
        ),
        'python': '''\
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr''',
        'cpp': '''\
void selection_sort(vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n - 1; ++i) {
        int m = i;
        for (int j = i + 1; j < n; ++j) if (a[j] < a[m]) m = j;
        if (m != i) swap(a[i], a[m]);
    }
}''',
        'rust': '''\
fn selection_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in 0..n {
        let m = (i..n).min_by_key(|&j| arr[j]).unwrap();
        arr.swap(i, m);
    }
}''',
        'interview': [
            ('Why is selection sort always O(n^2)?', 'It always scans the entire remaining unsorted section to find the minimum, regardless of input order.'),
            ('When would you choose selection sort over insertion sort?', 'When writes/swaps are very expensive -- selection sort does at most n-1 swaps vs insertion sort\'s O(n^2) shifts.'),
        ],
    },
    {
        'name': 'Insertion Sort',
        'category': 'Comparison / Insertion',
        'description': (
            'Builds the sorted array one element at a time by taking each unsorted element '
            'and inserting it at the correct position in the already-sorted prefix. '
            'Works like sorting a hand of playing cards.'
        ),
        'complexity': {'best': 'O(n)', 'average': 'O(n^2)', 'worst': 'O(n^2)', 'space': 'O(1)'},
        'properties': 'Stable: YES. In-place: YES. Adaptive: YES (O(nk) for k inversions).',
        'use_when': 'Small arrays (n<50); nearly sorted data; online sorting (elements arrive one by one); as base case for hybrid sorts (Timsort).',
        'avoid_when': 'Large reverse-sorted arrays.',
        'walkthrough': (
            'Array: [5, 3, 1, 4, 2]\n'
            'i=1: key=3. 5>3 -> shift 5. Insert 3: [3,5,1,4,2].\n'
            'i=2: key=1. 5>1,3>1 -> shift both. Insert 1: [1,3,5,4,2].\n'
            'i=3: key=4. 5>4 -> shift. Insert 4: [1,3,4,5,2].\n'
            'i=4: key=2. 5>2,4>2,3>2 -> shift. Insert 2: [1,2,3,4,5].'
        ),
        'python': '''\
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr''',
        'cpp': '''\
void insertion_sort(vector<int>& a) {
    for (int i = 1; i < (int)a.size(); ++i) {
        int key = a[i], j = i - 1;
        while (j >= 0 && a[j] > key) { a[j+1] = a[j]; --j; }
        a[j+1] = key;
    }
}''',
        'rust': '''\
fn insertion_sort(arr: &mut Vec<i32>) {
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;
        while j > 0 && arr[j - 1] > key { arr[j] = arr[j - 1]; j -= 1; }
        arr[j] = key;
    }
}''',
        'interview': [
            ('What is adaptive sorting?', 'An adaptive sort exploits existing order. Insertion sort on a nearly sorted array runs in O(n + k) where k is the number of inversions.'),
            ('How does Timsort use insertion sort?', 'Timsort uses insertion sort for small runs (< 64 elements) before merging, getting the best of both algorithms.'),
        ],
    },
    {
        'name': 'Merge Sort',
        'category': 'Comparison / Divide-and-Conquer',
        'description': (
            'Recursively divides the array in half, sorts each half, then merges the two '
            'sorted halves. Guarantees O(n log n) in all cases. The merge step is the key: '
            'compare the front elements of both halves, take the smaller.'
        ),
        'complexity': {'best': 'O(n log n)', 'average': 'O(n log n)', 'worst': 'O(n log n)', 'space': 'O(n)'},
        'properties': 'Stable: YES. In-place: NO (O(n) auxiliary). Adaptive: NO.',
        'use_when': (
            'When stability required. Sorting linked lists (O(1) space for list merge). '
            'External sorting (large files). When worst-case guarantee matters.'
        ),
        'avoid_when': 'Memory-constrained environments. Cache-sensitive code (heapsort or introsort better).',
        'walkthrough': (
            'Array: [38, 27, 43, 3, 9, 82, 10]\n'
            'Divide: [38,27,43,3] and [9,82,10]\n'
            'Divide: [38,27],[43,3] | [9,82],[10]\n'
            'Sort singles: [27,38],[3,43] | [9,82],[10]\n'
            'Merge: [3,27,38,43] | [9,10,82]\n'
            'Merge final: [3,9,10,27,38,43,82]'
        ),
        'python': '''\
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    return result + left[i:] + right[j:]''',
        'cpp': '''\
void merge(vector<int>& a, int l, int m, int r) {
    vector<int> tmp(a.begin()+l, a.begin()+r+1);
    int i=0, j=m-l+1, k=l;
    while (i<=m-l && j<=r-l) a[k++] = (tmp[i]<=tmp[j]) ? tmp[i++] : tmp[j++];
    while (i<=m-l) a[k++]=tmp[i++];
    while (j<=r-l) a[k++]=tmp[j++];
}
void merge_sort(vector<int>& a, int l, int r) {
    if (l>=r) return;
    int m=(l+r)/2;
    merge_sort(a,l,m); merge_sort(a,m+1,r); merge(a,l,m,r);
}''',
        'rust': '''\
fn merge_sort(arr: &[i32]) -> Vec<i32> {
    if arr.len() <= 1 { return arr.to_vec(); }
    let mid = arr.len() / 2;
    let (l, r) = (merge_sort(&arr[..mid]), merge_sort(&arr[mid..]));
    let (mut i, mut j, mut res) = (0, 0, Vec::with_capacity(arr.len()));
    while i < l.len() && j < r.len() {
        if l[i] <= r[j] { res.push(l[i]); i += 1; }
        else { res.push(r[j]); j += 1; }
    }
    res.extend_from_slice(&l[i..]); res.extend_from_slice(&r[j..]); res
}''',
        'interview': [
            ('Why is merge sort preferred for linked lists?', 'Linked list merge sort runs in O(1) extra space -- pointer rewiring instead of copying. Array merge sort needs O(n).'),
            ('What makes merge sort stable?', 'The merge step takes from the left half when left[i] == right[j] (using <=), preserving relative order.'),
            ('How do you sort a large file that does not fit in RAM?', 'External merge sort: split into sorted chunks, merge k-way. Classic algorithm for external storage.'),
        ],
    },
    {
        'name': 'Quicksort',
        'category': 'Comparison / Divide-and-Conquer',
        'description': (
            'Picks a pivot element, partitions the array so all elements less than pivot come '
            'before it and all greater after it, then recursively sorts both partitions. '
            'In-place and cache-friendly. Fastest in practice for most inputs.'
        ),
        'complexity': {'best': 'O(n log n)', 'average': 'O(n log n)', 'worst': 'O(n^2)', 'space': 'O(log n) stack'},
        'properties': 'Stable: NO. In-place: YES. Adaptive: somewhat (degenerate on sorted input without randomization).',
        'use_when': 'General-purpose sorting. Best average performance. Cache-friendly. std::sort in most languages uses introsort (quicksort + heapsort + insertion sort).',
        'avoid_when': 'When worst-case O(n^2) is unacceptable (use introsort/heapsort). When stability required.',
        'walkthrough': (
            'Array: [3,6,8,10,1,2,1], pivot=last(1)\n'
            'Partition: elements < 1 go left, >= go right.\n'
            'After partition: [1,1,8,10,6,2,3] with pivot 1 at idx 1.\n'
            'Recurse on [1] (sorted) and [8,10,6,2,3].\n'
            'Key insight: pivot ends up in its final sorted position after partition.'
        ),
        'python': '''\
def quicksort(arr, lo=0, hi=None):
    if hi is None: hi = len(arr) - 1
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort(arr, lo, p - 1)
        quicksort(arr, p + 1, hi)
    return arr

def partition(arr, lo, hi):
    pivot, i = arr[hi], lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1; arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[hi] = arr[hi], arr[i+1]
    return i + 1''',
        'cpp': '''\
int partition(vector<int>& a, int lo, int hi) {
    int pivot = a[hi], i = lo - 1;
    for (int j = lo; j < hi; ++j)
        if (a[j] <= pivot) swap(a[++i], a[j]);
    swap(a[i+1], a[hi]); return i + 1;
}
void quicksort(vector<int>& a, int lo, int hi) {
    if (lo < hi) { int p=partition(a,lo,hi); quicksort(a,lo,p-1); quicksort(a,p+1,hi); }
}''',
        'rust': '''\
fn quicksort(arr: &mut [i32]) {
    if arr.len() <= 1 { return; }
    let p = partition(arr);
    quicksort(&mut arr[..p]);
    quicksort(&mut arr[p + 1..]);
}
fn partition(arr: &mut [i32]) -> usize {
    let hi = arr.len() - 1;
    let pivot = arr[hi];
    let mut i = 0;
    for j in 0..hi {
        if arr[j] <= pivot { arr.swap(i, j); i += 1; }
    }
    arr.swap(i, hi); i
}''',
        'interview': [
            ('How do you prevent quicksort O(n^2) worst case?', 'Use random pivot selection or median-of-three pivot. Introsort switches to heapsort after depth exceeds 2.log(n).'),
            ('Why is quicksort faster than merge sort in practice?', 'Better cache locality (in-place), smaller constant factors, and the average case is very close to O(n log n) with good pivot selection.'),
            ('What is Dutch National Flag partitioning (3-way quicksort)?', 'Handles many duplicate keys efficiently. Partitions into <pivot, ==pivot, >pivot regions. O(n) for all-equal arrays.'),
        ],
    },
    {
        'name': 'Heap Sort',
        'category': 'Comparison / Selection',
        'description': (
            'Builds a max-heap from the array, then repeatedly extracts the maximum element '
            'to the end. Uses the heap data structure (complete binary tree stored in-array). '
            'Guarantees O(n log n) and O(1) extra space.'
        ),
        'complexity': {'best': 'O(n log n)', 'average': 'O(n log n)', 'worst': 'O(n log n)', 'space': 'O(1)'},
        'properties': 'Stable: NO. In-place: YES. Adaptive: NO.',
        'use_when': 'When O(1) extra space AND O(n log n) worst case are both required. Embedded systems.',
        'avoid_when': 'Cache-sensitive code (poor locality vs quicksort). When stability needed.',
        'walkthrough': (
            'Array: [4, 10, 3, 5, 1]\n'
            'Build max-heap: [10, 5, 3, 4, 1]\n'
            'Extract max (10), swap with last: [1, 5, 3, 4, 10]. Heapify [1,5,3,4].\n'
            'Extract 5, swap: [4, 1, 3, 5, 10]. Heapify [4,1,3].\n'
            'Extract 4: [3, 1, 4, 5, 10]. Continue -> [1, 3, 4, 5, 10].'
        ),
        'python': '''\
def heapify(arr, n, i):
    largest, l, r = i, 2*i+1, 2*i+2
    if l < n and arr[l] > arr[largest]: largest = l
    if r < n and arr[r] > arr[largest]: largest = r
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1): heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    return arr''',
        'cpp': '''\
void heapify(vector<int>& a, int n, int i) {
    int lg=i, l=2*i+1, r=2*i+2;
    if (l<n && a[l]>a[lg]) lg=l;
    if (r<n && a[r]>a[lg]) lg=r;
    if (lg!=i) { swap(a[i],a[lg]); heapify(a,n,lg); }
}
void heap_sort(vector<int>& a) {
    int n=a.size();
    for (int i=n/2-1;i>=0;--i) heapify(a,n,i);
    for (int i=n-1;i>0;--i) { swap(a[0],a[i]); heapify(a,i,0); }
}''',
        'rust': '''\
fn heapify(arr: &mut Vec<i32>, n: usize, i: usize) {
    let (mut lg, l, r) = (i, 2*i+1, 2*i+2);
    if l < n && arr[l] > arr[lg] { lg = l; }
    if r < n && arr[r] > arr[lg] { lg = r; }
    if lg != i { arr.swap(i, lg); heapify(arr, n, lg); }
}
fn heap_sort(arr: &mut Vec<i32>) {
    let n = arr.len();
    for i in (0..n/2).rev() { heapify(arr, n, i); }
    for i in (1..n).rev() { arr.swap(0, i); heapify(arr, i, 0); }
}''',
        'interview': [
            ('Why is heap sort not cache-friendly?', 'Heap accesses jump between distant memory locations (parent/child relationships span the array), causing many cache misses vs quicksort\'s sequential access.'),
            ('What is the time to build a heap?', 'O(n) -- not O(n log n). Heapifying from n/2 down to 0 is linear because most nodes are near the bottom.'),
        ],
    },
    {
        'name': 'Timsort',
        'category': 'Hybrid (Merge + Insertion)',
        'description': (
            'Hybrid algorithm used by Python\'s sorted(), Java\'s Arrays.sort(), and Android. '
            'Finds natural "runs" (already sorted sub-sequences), extends short runs with '
            'insertion sort, then merges runs using a modified merge sort. '
            'Extremely fast on real-world data which usually has partial ordering.'
        ),
        'complexity': {'best': 'O(n)', 'average': 'O(n log n)', 'worst': 'O(n log n)', 'space': 'O(n)'},
        'properties': 'Stable: YES. In-place: NO. Adaptive: YES (exploits existing order).',
        'use_when': 'This IS the sort for general-purpose use. Python built-in uses it.',
        'avoid_when': 'Embedded/memory-constrained (use introsort or heapsort). Pure performance in C++ (std::sort uses introsort).',
        'walkthrough': (
            'minrun ~= 32-64 (chosen so n/minrun is a power of 2).\n'
            '1. Scan array for natural ascending/descending runs. Reverse descending runs.\n'
            '2. If run is shorter than minrun, extend with insertion sort.\n'
            '3. Push runs onto a stack. Merge adjacent runs when stack balance invariant violated.\n'
            '4. Merge all remaining runs at end.\n'
            'Key insight: real data has natural runs (timestamps, nearly-sorted lists). '
            'Timsort is O(n) on sorted/reverse-sorted input.'
        ),
        'python': '# Python\'s sorted() and list.sort() use Timsort natively.\n# CPython implementation: Objects/listobject.c timsort_merge()',
        'cpp': '# C++ std::sort uses introsort, not timsort. Use std::stable_sort for Timsort-like guarantees.',
        'rust': '# Rust\'s slice::sort() uses timsort (pdqsort variant). slice::sort_unstable() uses pdqsort.',
        'interview': [
            ('What is the minrun in Timsort and why?', 'minrun is 32-64, chosen so n/minrun is close to a power of 2 to optimize the merge phase. Uses insertion sort to fill short runs.'),
            ('What invariant does Timsort maintain on the run stack?', 'For runs A, B, C on stack: |C| > |B| + |A| and |B| > |A|. Ensures balanced merges and O(n log n) total.'),
        ],
    },
    {
        'name': 'Counting Sort',
        'category': 'Non-Comparison / Integer',
        'description': (
            'Counts occurrences of each distinct value, computes prefix sums of counts, '
            'then places each element at its correct output position. '
            'O(n + k) where k is the range of input values.'
        ),
        'complexity': {'best': 'O(n+k)', 'average': 'O(n+k)', 'worst': 'O(n+k)', 'space': 'O(n+k)'},
        'properties': 'Stable: YES. In-place: NO. Non-comparison-based.',
        'use_when': 'Integer data with small range k. Radix sort uses it as subroutine. k <= 10xn.',
        'avoid_when': 'Large range k (e.g. sorting 32-bit integers: k=4B). Non-integer data.',
        'walkthrough': (
            'Array: [1, 3, 1, 2, 0, 3], k=4 (values 0-3)\n'
            'Count: [1, 2, 1, 2] (index=value, value=count)\n'
            'Prefix sum: [1, 3, 4, 6]\n'
            'Place elements right-to-left: output[prefix[arr[i]]-1] = arr[i], decrement.\n'
            'Result: [0, 1, 1, 2, 3, 3]'
        ),
        'python': '''\
def counting_sort(arr, max_val=None):
    if not arr: return arr
    k = (max_val or max(arr)) + 1
    count = [0] * k
    for x in arr: count[x] += 1
    for i in range(1, k): count[i] += count[i-1]
    out = [0] * len(arr)
    for x in reversed(arr):
        count[x] -= 1; out[count[x]] = x
    return out''',
        'cpp': '''\
vector<int> counting_sort(vector<int>& a, int k) {
    vector<int> cnt(k+1,0), out(a.size());
    for (int x : a) ++cnt[x];
    for (int i=1;i<=k;++i) cnt[i]+=cnt[i-1];
    for (int i=a.size()-1;i>=0;--i) out[--cnt[a[i]]]=a[i];
    return out;
}''',
        'rust': '// Rust: use counting_sort for u8/u16 data; BTreeMap for generic integer sort',
        'interview': [
            ('Why iterate right-to-left when placing elements?', 'It preserves stability -- equal elements keep their original relative order from the input.'),
            ('What is the relationship between counting sort and radix sort?', 'Radix sort uses counting sort as a stable subroutine, applying it digit by digit from least to most significant.'),
        ],
    },
    {
        'name': 'Radix Sort',
        'category': 'Non-Comparison / Digit',
        'description': (
            'Sorts integers digit by digit from least significant digit (LSD) to most '
            'significant digit (MSD), using a stable sort (counting sort) at each digit level. '
            'O(d x n) where d is number of digits.'
        ),
        'complexity': {'best': 'O(d.n)', 'average': 'O(d.n)', 'worst': 'O(d.n)', 'space': 'O(n+b)'},
        'properties': 'Stable: YES. In-place: NO. Non-comparison-based.',
        'use_when': 'Fixed-width integers/strings. Sorting IP addresses, phone numbers, fixed-length codes. When d is small.',
        'avoid_when': 'Floating point (complex digit extraction). Variable-length strings (use MSD radix). When d x n > n log n.',
        'walkthrough': (
            'Array: [170, 45, 75, 90, 802, 24, 2, 66]\n'
            'Sort by 1s digit: [170,90,802,2,24,45,75,66]\n'
            'Sort by 10s digit: [802,2,24,45,66,170,75,90]\n'
            'Sort by 100s digit: [2,24,45,66,75,90,170,802]'
        ),
        'python': '''\
def radix_sort(arr):
    if not arr: return arr
    max_val = max(arr)
    exp = 1
    while max_val // exp > 0:
        arr = counting_sort_by_digit(arr, exp)
        exp *= 10
    return arr

def counting_sort_by_digit(arr, exp):
    n = len(arr)
    out, cnt = [0]*n, [0]*10
    for x in arr: cnt[(x // exp) % 10] += 1
    for i in range(1, 10): cnt[i] += cnt[i-1]
    for i in range(n-1, -1, -1):
        d = (arr[i] // exp) % 10; cnt[d] -= 1; out[cnt[d]] = arr[i]
    return out''',
        'cpp': '// Similar to Python; use d passes of counting_sort on each digit position.',
        'rust': '// Rust: radix sort available in rdst crate. Manual impl uses counting sort per digit.',
        'interview': [
            ('LSD vs MSD radix sort -- what is the difference?', 'LSD (least significant digit first) is iterative, stable, simpler. MSD (most significant first) is recursive, can short-circuit early, natural for variable-length strings.'),
        ],
    },
    {
        'name': 'Bucket Sort',
        'category': 'Distribution',
        'description': (
            'Distributes elements into buckets (usually equal-width ranges), sorts each bucket '
            '(typically with insertion sort), then concatenates. Works well for uniformly '
            'distributed floating-point data.'
        ),
        'complexity': {'best': 'O(n+k)', 'average': 'O(n + n^2/k + k)', 'worst': 'O(n^2)', 'space': 'O(n+k)'},
        'properties': 'Stable: depends on sub-sort. In-place: NO.',
        'use_when': 'Uniform distribution over known range. Floating-point in [0,1). Parallel sort.',
        'avoid_when': 'Skewed distributions (all elements in one bucket -> O(n^2)).',
        'walkthrough': (
            'Array: [0.78, 0.17, 0.39, 0.26, 0.72, 0.94, 0.21, 0.12, 0.23, 0.68]\n'
            'Create 10 buckets for [0,0.1), [0.1,0.2) ... [0.9,1.0)\n'
            'Distribute: bucket[1]=[0.17,0.12], bucket[2]=[0.26,0.21,0.23], ...\n'
            'Sort each bucket with insertion sort.\n'
            'Concatenate: [0.12,0.17,0.21,0.23,0.26,0.39,0.68,0.72,0.78,0.94]'
        ),
        'python': '''\
def bucket_sort(arr, k=10):
    if not arr: return arr
    lo, hi = min(arr), max(arr)
    rng = hi - lo or 1
    buckets = [[] for _ in range(k)]
    for x in arr:
        idx = min(int((x - lo) / rng * k), k - 1)
        buckets[idx].append(x)
    return [x for b in buckets for x in sorted(b)]''',
        'cpp': '// k buckets as vector<vector<double>>; sort each; concatenate.',
        'rust': '// Similar to Python; Vec<Vec<f64>> for buckets.',
        'interview': [
            ('What is the average-case analysis of bucket sort?', 'With n elements and n buckets uniformly distributed, expected bucket size is 1 -> insertion sort on each is O(1) -> total O(n).'),
        ],
    },
    {
        'name': 'Shell Sort',
        'category': 'Comparison / Insertion variant',
        'description': (
            'Generalizes insertion sort by sorting elements far apart first, then gradually '
            'reducing the gap. Uses a sequence of decreasing gap sizes (e.g. n/2, n/4, ..., 1). '
            'Last pass is insertion sort on a nearly sorted array.'
        ),
        'complexity': {'best': 'O(n log n)', 'average': 'O(n^1.5) or O(n log^2n)', 'worst': 'O(n^2)', 'space': 'O(1)'},
        'properties': 'Stable: NO. In-place: YES. Adaptive: YES.',
        'use_when': 'Medium-sized arrays (~1000-5000). Embedded without recursion. Simple implementation with decent performance.',
        'avoid_when': 'Large n (use introsort/timsort). When exact complexity guarantees needed.',
        'walkthrough': (
            'Array: [8,7,6,5,4,3,2,1], gap=4\n'
            'Sort pairs 4 apart: compare [8,4],[7,3],[6,2],[5,1] -> swap all -> [4,3,2,1,8,7,6,5]\n'
            'gap=2: -> [2,1,4,3,6,5,8,7] ... gap=1 (insertion sort): -> [1,2,3,4,5,6,7,8]'
        ),
        'python': '''\
def shell_sort(arr):
    gap = len(arr) // 2
    while gap > 0:
        for i in range(gap, len(arr)):
            temp, j = arr[i], i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]; j -= gap
            arr[j] = temp
        gap //= 2
    return arr''',
        'cpp': '''\
void shell_sort(vector<int>& a) {
    for (int gap=a.size()/2; gap>0; gap/=2)
        for (int i=gap;i<(int)a.size();++i) {
            int tmp=a[i], j=i;
            for (;j>=gap&&a[j-gap]>tmp;j-=gap) a[j]=a[j-gap];
            a[j]=tmp;
        }
}''',
        'rust': '''\
fn shell_sort(arr: &mut Vec<i32>) {
    let mut gap = arr.len() / 2;
    while gap > 0 {
        for i in gap..arr.len() {
            let tmp = arr[i]; let mut j = i;
            while j >= gap && arr[j - gap] > tmp { arr[j] = arr[j - gap]; j -= gap; }
            arr[j] = tmp;
        }
        gap /= 2;
    }
}''',
        'interview': [
            ('What gap sequence gives the best performance for shell sort?', 'Ciura\'s sequence (1,4,10,23,57,132,301,701) empirically gives best performance. Hibbard\'s (2^k-1) gives O(n^1.5) worst case.'),
        ],
    },
]

# Sorting comparison table and meta-knowledge
SORTING_META = """
SORTING ALGORITHM COMPARISON TABLE
===================================
Algorithm      | Best      | Average   | Worst     | Space    | Stable | In-Place
---------------|-----------|-----------|-----------|----------|--------|----------
Bubble Sort    | O(n)      | O(n^2)     | O(n^2)     | O(1)     | YES    | YES
Selection Sort | O(n^2)     | O(n^2)     | O(n^2)     | O(1)     | NO     | YES
Insertion Sort | O(n)      | O(n^2)     | O(n^2)     | O(1)     | YES    | YES
Shell Sort     | O(n log n)| O(n^1.5)  | O(n^2)     | O(1)     | NO     | YES
Merge Sort     | O(n log n)| O(n log n)| O(n log n)| O(n)     | YES    | NO
Quicksort      | O(n log n)| O(n log n)| O(n^2)     | O(log n) | NO     | YES
Heap Sort      | O(n log n)| O(n log n)| O(n log n)| O(1)     | NO     | YES
Timsort        | O(n)      | O(n log n)| O(n log n)| O(n)     | YES    | NO
Counting Sort  | O(n+k)    | O(n+k)    | O(n+k)    | O(n+k)   | YES    | NO
Radix Sort     | O(d.n)    | O(d.n)    | O(d.n)    | O(n+b)   | YES    | NO
Bucket Sort    | O(n+k)    | O(n+k)    | O(n^2)     | O(n+k)   | *      | NO

DECISION GUIDE:
  General purpose (unknown input):     -> Timsort (Python) / Introsort (C++)
  Need guaranteed O(n log n), O(1):    -> Heap Sort
  Need stable, guaranteed O(n log n):  -> Merge Sort
  Small array (n < 50):                -> Insertion Sort
  Integer data, small range:           -> Counting Sort -> Radix Sort
  Uniform floats [0,1):                -> Bucket Sort
  Nearly sorted data:                  -> Insertion Sort or Timsort

LOWER BOUND: Any comparison-based sort requires Ω(n log n) comparisons.
Proof: Decision tree has n! leaves (permutations); tree height >= log₂(n!) = Ω(n log n).
Non-comparison sorts (counting/radix/bucket) beat this by using numeric properties.
"""

SEARCHING_CORPUS = [
    {
        'name': 'Binary Search',
        'description': (
            'Finds a target in a sorted array in O(log n) by repeatedly halving the search space. '
            'Compare target to middle element: if equal found; if target < mid search left half; else right half.'
        ),
        'complexity': 'O(log n) time, O(1) space (iterative). Requires sorted input.',
        'python': '''\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1  # not found''',
        'interview': [
            ('Find first/last occurrence of a target in a sorted array with duplicates.',
             'Binary search with modified condition: for first occurrence, when arr[mid]==target set hi=mid-1; for last, set lo=mid+1.'),
            ('Find the rotation point in a rotated sorted array.',
             'Binary search: if arr[mid] > arr[hi], rotation is in right half; else in left half.'),
        ],
    },
]


def build_sorting(out_dir: Path, node: str) -> list:
    """Stage 28: Sorting and searching algorithms corpus."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = _make_session()
    items = []
    ok = 0

    # 1. Individual algorithm entries
    for algo in SORTING_ALGORITHMS:
        text_parts = [
            f'SORTING ALGORITHM: {algo["name"]}',
            f'Category: {algo["category"]}',
            '',
            f'DESCRIPTION:\n{algo["description"]}',
            '',
            'COMPLEXITY:',
            f'  Best case:    {algo["complexity"]["best"]}',
            f'  Average case: {algo["complexity"]["average"]}',
            f'  Worst case:   {algo["complexity"]["worst"]}',
            f'  Space:        {algo["complexity"]["space"]}',
            '',
            f'PROPERTIES: {algo["properties"]}',
            '',
            f'WHEN TO USE: {algo["use_when"]}',
            f'WHEN TO AVOID: {algo["avoid_when"]}',
            '',
            f'STEP-BY-STEP WALKTHROUGH:\n{algo["walkthrough"]}',
            '',
            f'PYTHON IMPLEMENTATION:\n{algo["python"]}',
            '',
            f'C++ IMPLEMENTATION:\n{algo["cpp"]}',
            '',
            f'RUST IMPLEMENTATION:\n{algo["rust"]}',
        ]
        if algo.get('interview'):
            text_parts += ['', 'INTERVIEW QUESTIONS:']
            for q, a in algo['interview']:
                text_parts.append(f'Q: {q}\nA: {a}')
        if _train('\n'.join(text_parts), node, session):
            ok += 1
        time.sleep(0.1)

    # 2. Comparison table and decision guide
    if _train(SORTING_META, node, session):
        ok += 1

    # 3. Searching algorithms
    for algo in SEARCHING_CORPUS:
        text = (
            f'SEARCHING ALGORITHM: {algo["name"]}\n\n'
            f'{algo["description"]}\n'
            f'Complexity: {algo["complexity"]}\n\n'
            f'PYTHON:\n{algo["python"]}\n'
        )
        for q, a in algo.get('interview', []):
            text += f'\nQ: {q}\nA: {a}\n'
        if _train(text, node, session):
            ok += 1
        time.sleep(0.1)

    # 4. Wikipedia articles on sorting and algorithms
    algo_wiki = [
        'Sorting_algorithm', 'Comparison_sort', 'Computational_complexity_theory',
        'Big_O_notation', 'In-place_algorithm', 'Stable_sort',
        'Timsort', 'Introsort', 'Pdqsort',
    ]
    for title in algo_wiki:
        r = session.get(f'https://en.wikipedia.org/api/rest_v1/page/summary/{title}', timeout=10)
        if r.ok:
            extract = r.json().get('extract', '')
            if extract:
                if _train(f'Algorithms -- {title.replace("_"," ")}\n\n{extract}', node, session):
                    ok += 1
        time.sleep(0.15)

    print(f'  Sorting/algorithms corpus: {ok} items trained', flush=True)
    items.append({'stage': 28, 'type': 'sorting_algorithms', 'trained': ok,
                  'modality': 'text', 'tags': ['sorting', 'algorithms', 'coding']})
    return items


# -- Entry point ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Stages 27-28: Cognitive + Sorting corpus')
    ap.add_argument('--stages',   default='27,28')
    ap.add_argument('--node',     default=DEFAULT_NODE)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    args = ap.parse_args()

    stages    = {int(s.strip()) for s in args.stages.split(',')}
    train_dir = Path(args.data_dir) / 'training'

    print('Cognitive + Sorting Corpus Builder -- Stages 27/28')
    print(f'  Node   : {args.node}')
    print(f'  Stages : {sorted(stages)}')

    session    = _make_session()
    all_items: dict = {}

    if 27 in stages:
        print('\n[Stage 27] Cognitive / IQ / logical reasoning')
        all_items[27] = build_cognitive(train_dir / 'stage27_cognitive', args.node)

    if 28 in stages:
        print('\n[Stage 28] Sorting & searching algorithms')
        all_items[28] = build_sorting(train_dir / 'stage28_sorting', args.node)

    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage27_28_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest -> {mpath}')


if __name__ == '__main__':
    main()
