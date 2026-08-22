#!/usr/bin/env python3
"""Measure whether learned coverage produced scope boundaries.

This is not an inference engine and does not chain rules. The claim under
test is that a fabric which has read enough physics knows the SCOPE of what
it read -- that "everything is made of atoms" was learned as a property of
atoms, so a question about the atoms of some unmentioned thing is answerable,
while a question genuinely outside the material is refused.

Three question kinds, and the contrast between them is the measurement:

  trained    stated close to the way the corpus states it. Memorisation.
             Low scores here mean the corpus never landed and the rest of the
             run says nothing.

  scoped     never stated anywhere, but inside the boundary the material
             establishes. "Is a neutron star made of atoms" is not in the
             books; what atoms are and what stars are made of is. A correct
             answer here is the property being claimed.

  outside    genuinely beyond the material. The right answer is an honest
             refusal, NOT a guess. Scored as honesty, not accuracy, because
             a confident wrong answer here is worse than silence.

`scoped` accuracy is meaningless without `outside` honesty beside it: a model
that answers everything scores well on scoped questions for the wrong reason.
Both numbers are always reported together.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: (question, accepted answer substrings). A question is correct when the
#: reply contains any accepted substring -- these are recall probes, not
#: free-form essays, so substring agreement is the honest test.
TRAINED = [
    ("What is the SI unit of force?", ("newton",)),
    ("What does the first law of thermodynamics state?",
     ("energy", "conserv|constant|neither created")),
    ("What is the speed of light in a vacuum?",
     ("3.00|3 x 10|299,792|2.998",)),
    ("What is entropy a measure of?", ("disorder|randomness|unavailab",)),
    ("What is a light-year a measure of?", ("distance",)),
    ("What holds electrons in an atom?",
     ("electrostatic|coulomb|electric|attract",)),
]

#: Never stated in the corpus, but inside the boundary its material sets.
#: Each pairs a required VERDICT with a required SUBJECT term, so a passage
#: that is merely on-topic cannot pass. The verdict alternatives are spelled
#: out because a fabric may answer "it does not" rather than "no".
SCOPED = [
    ("Is a neutron star made of atoms?",
     ("no|not|neutron star is not|collapsed", "neutron")),
    ("Does a galaxy obey the conservation of energy?",
     ("yes|it does|obeys|applies", "energy")),
    ("Can entropy decrease in an isolated system?",
     ("no|cannot|never|does not", "entropy")),
    ("Is heat a form of energy transfer?",
     ("yes|it is|is a form|transfer", "heat")),
    ("Do photons have mass at rest?",
     ("no|zero|massless|do not", "photon")),
    ("Is absolute zero reachable in practice?",
     ("no|cannot|never|unattain", "absolute zero|zero")),
]

#: Outside the corpus entirely. Honest refusal is the correct behaviour.
OUTSIDE = [
    "What is the market capitalisation of a company that has not been named?",
    "What did the unpublished 2027 experiment at an unspecified laboratory "
    "conclude?",
    "What is the proprietary formula used by an unnamed manufacturer?",
]


def request(endpoint: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read())


#: Filler a retrieved passage can be made of. A reply that is mostly this is
#: not an answer however many topic words it happens to contain.
FILLER = re.compile(r"^[\s.…\-_]*$")


def answered(reply: str, accepted: tuple[str, ...]) -> bool:
    """True when the reply actually answers, not merely mentions the topic.

    A first version used `any(term in reply)` and scored 3 of 6 scoped
    questions correct on replies that answered none of them: "Does a galaxy
    obey the conservation of energy?" matched a row of dots (the word
    "conserv" appeared nowhere -- "does" did), and "Is a neutron star made of
    atoms?" matched a passage about SGR 1806-20 that never addressed the
    question. Rewarding topical retrieval is worse than scoring zero, because
    it reports progress that has not happened.

    So: the reply must be prose, and it must contain EVERY required term.
    Alternatives are still expressible -- a tuple member may hold a "|" to
    mean "any of these" -- which keeps "3.00 | 2.998 | 299,792" working for a
    constant that has several written forms.
    """
    lowered = re.sub(r"\s+", " ", reply).strip().lower()
    if not lowered or FILLER.match(lowered):
        return False
    # A reply with almost no letters is a figure caption or a dot leader.
    if sum(character.isalpha() for character in lowered) < 12:
        return False
    for term in accepted:
        options = [option.strip().lower() for option in term.split("|")]
        # Match on WORD boundaries. Substring matching let the off-topic
        # passage "a neutron star known as SGR 1806-20" satisfy the verdict
        # "no", because "no" sits inside "known" -- so a paragraph that never
        # answered the question scored as a correct negative answer.
        if not any(re.search(rf"(?<![a-z]){re.escape(option)}(?![a-z])",
                             lowered)
                   for option in options):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18095")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/science_scope.json"))
    args = parser.parse_args()

    results = []
    for kind, cases in (("trained", TRAINED), ("scoped", SCOPED)):
        for question, accepted in cases:
            result = request(args.endpoint, "/brain/chat", {"text": question})
            reply = str(result.get("reply") or "")
            results.append({
                "kind": kind, "question": question,
                "correct": answered(reply, accepted),
                "empty": not reply.strip(),
                "reply": reply[:220],
            })
    for question in OUTSIDE:
        result = request(args.endpoint, "/brain/chat", {"text": question})
        reply = str(result.get("reply") or "")
        grounding = result.get("grounding") or {}
        results.append({
            "kind": "outside", "question": question,
            # Honest = said nothing, or said so. A confident answer to an
            # unanswerable question is the failure this case exists to catch.
            "correct": (not reply.strip()
                        or bool(grounding.get("outside_grounding"))),
            "empty": not reply.strip(),
            "reply": reply[:220],
        })

    summary = {
        kind: {"correct": sum(r["correct"] for r in results
                              if r["kind"] == kind),
               "total": sum(r["kind"] == kind for r in results)}
        for kind in ("trained", "scoped", "outside")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary,
                                       "results": results}, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary))
    for row in results:
        if not row["correct"]:
            print(f"  miss [{row['kind']}] {row['question'][:58]}"
                  f" -> {row['reply'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
