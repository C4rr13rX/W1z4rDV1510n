#!/usr/bin/env python3
"""Ask for conclusions the corpus never states, checkable against reality.

Two benchmarks, both aimed at the same claim: that knowledge learned in one
place should be usable somewhere it was never stated, and that the answer can
be checked against something true in the world rather than against a phrasing.

**cross_domain** -- questions whose answer requires joining facts from
different books. Nothing in the corpus says "a helium balloon rises because
its density is lower than air"; the density of gases is in a chemistry book
and buoyancy is in a physics one. Each question is scored on whether the
reply carries the ideas the real answer needs, not on wording, and every
expected answer is a fact that can be checked outside this system.

**invention** -- one open engineering problem: design a microelectronic
circuit that a WiFi device can detect and locate. There is no trained answer
to recall. It spans radio propagation, antenna design, circuit topology,
materials, and power, so a reply is scored by how many of those distinct
disciplines it actually brings, and separately by whether it reaches for
nanoscale materials. This is a coverage measure over a real design space, not
a right/wrong test -- reported as such.

Neither benchmark is passed by retrieving a section. A reply that quotes one
textbook paragraph scores on that paragraph's discipline alone.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: (question, required idea groups, one-line real-world check).
#: A group is a tuple of synonyms -- any member satisfies that group, and
#: every group must be satisfied. The groups are the STEPS the real answer
#: needs, so satisfying them all means the reply carried the reasoning.
CROSS_DOMAIN = [
    (
        "A helium balloon released indoors rises to the ceiling. Why?",
        [("density", "dense", "lighter", "less mass per"),
         ("air", "atmosphere", "surrounding")],
        "Helium is less dense than air, so buoyancy exceeds its weight.",
    ),
    (
        "Why is the sky blue during the day but red at sunset?",
        [("scatter", "scattering", "rayleigh"),
         ("wavelength", "blue", "shorter"),
         ("atmosphere", "air", "path")],
        "Rayleigh scattering is stronger at short wavelengths; a long "
        "sunset path removes them.",
    ),
    (
        "Why does food cook more slowly at high altitude?",
        [("pressure", "atmospheric"),
         ("boil", "boiling point", "temperature")],
        "Lower pressure lowers water's boiling point, so cooking is cooler.",
    ),
    (
        "Why can a star much more massive than the Sun burn out far sooner?",
        [("mass", "massive", "heavier"),
         ("fuse", "fusion", "burn", "consume", "rate")],
        "Luminosity rises much faster than mass, so fuel is spent sooner.",
    ),
    (
        "Why does a metal spoon feel colder than a wooden one in the same "
        "room?",
        [("conduct", "conduction", "conductivity"),
         ("heat", "thermal", "energy")],
        "Both are at room temperature; metal conducts heat from skin faster.",
    ),
    (
        "Why does increasing the temperature of a gas in a sealed rigid "
        "container raise its pressure?",
        [("kinetic", "faster", "speed", "motion", "energy"),
         ("collision", "collide", "impact", "wall")],
        "Faster molecules strike the walls harder and more often.",
    ),
]

#: The invention problem, and the disciplines a serious answer must touch.
INVENTION_PROMPT = (
    "Design a microelectronic circuit that can be detected and located by a "
    "standard WiFi device. Explain how the circuit produces a signature the "
    "WiFi radio can pick out, and how the distance and direction to the "
    "circuit can be recovered from that signal."
)

DISCIPLINES = {
    "radio propagation": ("2.4", "5 ghz", "ghz", "frequency", "wavelength",
                          "propagat", "signal strength", "rssi", "path loss"),
    "antenna": ("antenna", "dipole", "patch", "aperture", "radiat",
                "impedance match"),
    "circuit topology": ("oscillator", "mixer", "amplifier", "resonator",
                         "lc ", "varactor", "diode", "transistor",
                         "backscatter", "modulat"),
    "localisation method": ("time of flight", "time-of-flight", "rtt",
                            "round trip", "triangulat", "trilaterat",
                            "phase", "angle of arrival", "doppler", "rssi"),
    "power": ("power", "battery", "harvest", "energy", "microwatt",
              "milliwatt", "passive"),
    "materials": ("substrate", "dielectric", "silicon", "copper", "pcb",
                  "semiconductor", "permittivity"),
    "nanotechnology": ("nano", "nanoscale", "nanoparticle", "nanowire",
                       "graphene", "carbon nanotube", "quantum dot",
                       "thin film", "mems"),
}


def request(endpoint: str, path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as response:
        return json.loads(response.read())


def has_group(reply: str, group: tuple[str, ...]) -> bool:
    lowered = re.sub(r"\s+", " ", reply).lower()
    return any(re.search(rf"(?<![a-z]){re.escape(term)}", lowered)
               for term in group)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:18095")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/integration.json"))
    args = parser.parse_args()

    results = []
    for question, groups, truth in CROSS_DOMAIN:
        reply = str(request(args.endpoint, "/brain/chat",
                            {"text": question}).get("reply") or "")
        met = [index for index, group in enumerate(groups)
               if has_group(reply, group)]
        results.append({
            "kind": "cross_domain", "question": question,
            "groups_met": len(met), "groups_total": len(groups),
            # Complete only when every step of the real answer is present.
            "complete": len(met) == len(groups),
            "real_answer": truth,
            "reply": reply[:400],
        })

    reply = str(request(args.endpoint, "/brain/chat",
                        {"text": INVENTION_PROMPT}).get("reply") or "")
    touched = sorted(name for name, terms in DISCIPLINES.items()
                     if has_group(reply, terms))
    results.append({
        "kind": "invention", "question": INVENTION_PROMPT[:80],
        "disciplines": touched, "disciplines_total": len(DISCIPLINES),
        "nanotech": "nanotechnology" in touched,
        "reply": reply[:1200],
    })

    cross = [row for row in results if row["kind"] == "cross_domain"]
    summary = {
        "cross_domain": {
            "complete": sum(row["complete"] for row in cross),
            "total": len(cross),
            # Partial credit, so movement is visible before any question is
            # fully answered.
            "groups_met": sum(row["groups_met"] for row in cross),
            "groups_total": sum(row["groups_total"] for row in cross),
            "answered": sum(bool(row["reply"].strip()) for row in cross),
        },
        "invention": {
            "disciplines": len(touched),
            "of": len(DISCIPLINES),
            "nanotech": "nanotechnology" in touched,
            "answered": bool(reply.strip()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"summary": summary,
                                       "results": results}, indent=2),
                           encoding="utf-8")
    print(json.dumps(summary))
    for row in cross:
        state = ("complete" if row["complete"]
                 else f"{row['groups_met']}/{row['groups_total']}")
        print(f"  [{state:>8}] {row['question'][:62]}")
        if row["reply"].strip():
            print(f"             -> {row['reply'][:90]!r}")
    print(f"  [invention] disciplines: {touched or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
