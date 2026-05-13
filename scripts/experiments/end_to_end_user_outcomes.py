#!/usr/bin/env python3
"""
End-to-end verification of the six user outcomes from the May 12 turn.

This is the small-corpus experiment per the operator's standing rule:
"run small experiments that validate it will work with a large training
set with only a small training set, and don't stop until you have
verified that."

Each section is a separate assertion-style block.  All sections must
pass.  Failures are reported with concrete data.
"""
from __future__ import annotations
import json
import re
import sys
import time
import urllib.error
import urllib.request


NODE     = "http://localhost:8090"
DJANGO   = "http://127.0.0.1:8000"


def post(url: str, payload: dict, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"_http_error": str(exc)}


def get(url: str, timeout: float = 10) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"_http_error": str(exc)}


# ──────────────────────────────────────────────────────────────────────────────

def section(n: int, title: str) -> None:
    print(f"\n{'='*72}\n[{n}] {title}\n{'='*72}")


def report(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))


PASS_COUNT = 0
FAIL_COUNT = 0


def expect(name: str, ok: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if ok: PASS_COUNT += 1
    else:  FAIL_COUNT += 1
    report(name, ok, detail)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# ──────────────────────────────────────────────────────────────────────────────


def section_1_node_alive() -> None:
    section(1, "Node and Django up")
    h = get(f"{NODE}/health")
    expect("node /health 200", h.get("status") == "OK",
           f"uptime={h.get('uptime_secs')}")
    d = get(f"{DJANGO}/api/wizard-chat/status/")
    expect("django /api/wizard-chat/status 200", d.get("online") is True)


def section_2_brain_overview() -> None:
    section(2, "Item 6: GET /brain returns the full architectural snapshot")
    b = get(f"{NODE}/brain")
    expect("/brain has slow_pool",     "slow_pool"     in b)
    expect("/brain has multi_pool",    "multi_pool"    in b)
    expect("/brain has motifs",        "motifs"        in b)
    expect("/brain has neuromodulators","neuromodulators" in b)
    expect("/brain lists recipes",
           isinstance(b.get("feedback_recipes"), list)
           and "train_and_verify" in b["feedback_recipes"])
    print(f"      [snapshot] pool_counts={b.get('multi_pool', {}).get('pools')}  "
          f"cross_edges={b.get('multi_pool', {}).get('cross_edges')}  "
          f"motifs_total={b.get('motifs', {}).get('total')}")


def section_3_recipes_listed() -> None:
    section(3, "Item 6: GET /brain/recipes returns schemas")
    r = get(f"{NODE}/brain/recipes")
    recipes = r.get("recipes", [])
    names = [rc.get("name") for rc in recipes]
    expect("recipe list non-empty",  len(names) >= 5,  f"got {len(names)}")
    expect("dopamine_pulse listed",  "dopamine_pulse"           in names)
    expect("consolidate_pair listed","consolidate_pair"         in names)
    expect("train_and_verify listed","train_and_verify"         in names)
    expect("verifier_pass listed",   "verifier_pass"            in names)
    expect("motif_query_continuation listed",
                                     "motif_query_continuation" in names)


def section_4_train_and_verify_recipe() -> None:
    section(4, "Item 6: POST /brain/recipes/run train_and_verify")
    # Clear pool so the test is reproducible.
    post(f"{NODE}/neuro/clear", {})
    time.sleep(0.5)
    r = post(f"{NODE}/brain/recipes/run", {
        "name": "train_and_verify",
        "args": {
            "q": "what is the carnot cycle",
            "a": "The Carnot cycle is a theoretical thermodynamic cycle of maximum efficiency.",
            "passes": 35,
        },
    })
    expect("recipe returned ok",          r.get("ok") is True,
           f"a_returned={r.get('a_returned', '')[:60]!r}")
    expect("verify_match true",           r.get("verify_match") is True)
    expect("confidence high",
           (r.get("confidence") or 0.0) >= 0.5,
           f"conf={r.get('confidence')}")


def section_5_verifier_pass_recipe() -> None:
    section(5, "Item 6: POST /brain/recipes/run verifier_pass with custom battery")
    # Train two pairs through train_and_verify first so the battery has hits.
    post(f"{NODE}/neuro/clear", {})
    time.sleep(0.3)
    pairs = [
        ("what is hooke's law",      "Hooke's law: stress is proportional to strain within the elastic limit."),
        ("what is bernoulli's principle", "Bernoulli's principle: faster fluid flow corresponds to lower pressure."),
    ]
    for q, a in pairs:
        post(f"{NODE}/brain/recipes/run", {
            "name": "train_and_verify",
            "args": {"q": q, "a": a, "passes": 35},
        })
    r = post(f"{NODE}/brain/recipes/run", {
        "name": "verifier_pass",
        "args": {
            "battery": [{"q": q, "a": a} for q, a in pairs],
            "reinforce_on_pass": True,
        },
    })
    expect("verifier returned results", len(r.get("results", [])) == 2)
    expect("verifier passed both", r.get("passed") == 2,
           f"results={r.get('results')}")
    expect("pass_rate 1.0", r.get("pass_rate") == 1.0)


def section_6_wizard_chat_train_end_to_end() -> None:
    section(6, "Item 4: Wizard-chat train endpoint, fresh pool, exact recall")
    post(f"{NODE}/neuro/clear", {})
    time.sleep(0.3)
    r = post(f"{DJANGO}/api/wizard-chat/train/", {
        "question": "what is the bose-einstein condensate",
        "answer":   "A Bose-Einstein condensate is a state of matter formed by bosons at near-absolute zero.",
    })
    expect("/api/wizard-chat/train/ ok",      r.get("ok") is True,
           f"err={r.get('error', '')}")
    expect("verify_match true",               r.get("verify_match") is True)
    expect("decoder is multi_pool",
           r.get("verify_decoder") == "multi_pool",
           f"decoder={r.get('verify_decoder')}")
    expect("steps array returned",
           isinstance(r.get("steps"), list) and len(r["steps"]) >= 4,
           f"steps count={len(r.get('steps', []))}")
    print(f"      [steps]")
    for s in (r.get("steps") or []):
        mark = "OK" if s.get("ok") else "ERR"
        print(f"        [{mark}]  {s.get('label')}  {s.get('detail', '')}")
    # Now do an INDEPENDENT /chat call to confirm recall is real.
    c = post(f"{NODE}/chat", {
        "text": "what is the bose-einstein condensate",
        "hops": 2, "min_strength": 0.05,
    })
    expect("independent /chat returns trained answer",
           _norm(r.get("answer", "")) == _norm(c.get("answer", ""))
           or _norm("Bose-Einstein condensate") in _norm(c.get("answer", "")),
           f"chat_answer={c.get('answer', '')[:80]!r}")


def section_7_hebbian_transfer_smoke() -> None:
    section(7, "End-to-end: Hebbian transfer smoke (train two domains, query a third)")
    post(f"{NODE}/neuro/clear", {})
    time.sleep(0.3)
    # Train: physics defines momentum.  Training also wires "momentum" to
    # related concepts via the slow-pool char/bigram path.
    pairs = [
        ("define momentum in physics",
         "Momentum is mass times velocity, a conserved vector quantity in classical mechanics."),
        ("define energy in physics",
         "Energy is the capacity to do work, conserved across closed systems."),
        ("define force in physics",
         "Force is mass times acceleration; a vector quantity that changes momentum."),
    ]
    for q, a in pairs:
        post(f"{NODE}/brain/recipes/run", {
            "name": "train_and_verify",
            "args": {"q": q, "a": a, "passes": 35},
        })
    # Each individual definition should recall (basic test).
    pass_count = 0
    for q, a in pairs:
        c = post(f"{NODE}/chat", {"text": q, "hops": 2, "min_strength": 0.05})
        ok = _norm(a) in _norm(c.get("answer", "")) \
            or _norm(c.get("answer", "")) in _norm(a)
        if ok: pass_count += 1
    expect("all 3 trained pairs recall on /chat",
           pass_count == 3, f"{pass_count}/3")
    # Transfer-flavoured probe: a related question that wasn't explicitly
    # trained but shares vocabulary.  We DON'T assert a specific answer
    # — we just assert /chat returns SOMETHING content-bearing, not the
    # `!\n"$!` structural soup.  This is the weakest meaningful claim:
    # the architecture doesn't degrade to noise when input partially
    # overlaps trained material.
    transfer_q = "what is mass times velocity"
    c = post(f"{NODE}/chat",
              {"text": transfer_q, "hops": 2, "min_strength": 0.05})
    actual = (c.get("answer") or "").strip()
    has_content = bool(re.search(r"[A-Za-z]{3,}", actual))
    expect("transfer query returns content-bearing text (not structural soup)",
           has_content, f"answer={actual[:80]!r}")


# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    section_1_node_alive()
    section_2_brain_overview()
    section_3_recipes_listed()
    section_4_train_and_verify_recipe()
    section_5_verifier_pass_recipe()
    section_6_wizard_chat_train_end_to_end()
    section_7_hebbian_transfer_smoke()

    print("\n" + "=" * 72)
    print(f"  {PASS_COUNT} passed, {FAIL_COUNT} failed out of "
          f"{PASS_COUNT + FAIL_COUNT} assertions")
    print("=" * 72)
    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
