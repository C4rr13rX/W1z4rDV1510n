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
     ("energy", "conserv")),
    ("What is the speed of light in a vacuum?", ("3.00", "3 x 10", "299,792",
                                                 "2.998")),
    ("What is entropy a measure of?", ("disorder", "randomness",
                                       "unavailab")),
    ("What is a light-year a measure of?", ("distance",)),
    ("What holds electrons in an atom?", ("electrostatic", "coulomb",
                                          "electric", "nucleus", "attract")),
]

#: Never stated in the corpus, but inside the boundary its material sets.
SCOPED = [
    ("Is a neutron star made of atoms?", ("no", "neutron")),
    ("Does a galaxy obey the conservation of energy?", ("yes", "does",
                                                        "conserv")),
    ("Can entropy decrease in an isolated system?", ("no", "cannot",
                                                     "never")),
    ("Is heat a form of energy transfer?", ("yes", "transfer", "energy")),
    ("Do photons have mass at rest?", ("no", "zero", "massless")),
    ("Is absolute zero reachable in practice?", ("no", "cannot", "unattain")),
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


def answered(reply: str, accepted: tuple[str, ...]) -> bool:
    lowered = re.sub(r"\s+", " ", reply).lower()
    return any(term.lower() in lowered for term in accepted)


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
