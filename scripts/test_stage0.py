#!/usr/bin/env python3
"""
W1z4rD V1510n -- Stage 0 Validation Tests
=========================================
Small meaningful tests against the toddler-foundations training set.

Tests both exact trained questions and rephrased variants so we can tell
whether Hebbian activation is working or just memorising string keys.

Usage:
  python scripts/test_stage0.py [--node http://127.0.0.1:8090] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import httpx
except ImportError:
    sys.exit("Missing: pip install httpx")

# ---------------------------------------------------------------------------
# Test cases
# Each tuple: (label, question, expected_keyword_in_answer)
# "expected_keyword" is just a loose word-level check -- it does NOT need to
# match exactly.  It gives us a binary signal for scoring.
# ---------------------------------------------------------------------------

EXACT_QUESTIONS = [
    # Exact phrases from TODDLER_CONCEPTS -- should all score well.
    # Keywords are checked as substrings in both /chat answer and /qa/query top result.
    # Multiple keywords separated by | mean ANY one match passes.
    ("eyes (exact)",       "What are eyes?",                    "see"),
    ("brain (exact)",      "What is a brain?",                  "nervous|organ|cerebr|neuron|memory|skull"),
    ("dog (exact)",        "What is a dog?",                    "canis|animal|domesticat|mammal|wolf|pet"),
    ("apple (exact)",      "What is an apple?",                 "fruit"),
    ("water (exact)",      "What is water?",                    "liquid|hydrogen|drink"),
    ("circle (exact)",     "What is a circle?",                 "round"),
    ("gravity (exact)",    "What is gravity?",                  "force|pull|attract"),
    ("sun (exact)",        "What is the sun?",                  "star|solar|light"),
    ("color red (exact)",  "What is the color red?",            "color|colour|wavelength"),
    ("triangle (exact)",   "What is a triangle?",               "three|sides|angle"),
    ("teacher (exact)",    "What is a teacher?",                "teach|educat|learn"),
    ("energy (exact)",     "What is energy?",                   "work|power|capacit|convert|heat|kinetic|motion"),
    ("language (exact)",   "What is language?",                 "communicat|speech|symbol|linguistic"),
    ("sentence (exact)",   "What is a sentence?",               "word|clause|grammar|written|complete"),
    ("music (exact)",      "What is music?",                    "sound|rhythm|melody"),
]

REPHRASED_QUESTIONS = [
    # Paraphrasings -- tests whether Hebbian weight generalises across token variants
    ("heart (rephrase)",   "Tell me about the heart.",          "pump|blood|muscle|cardiac"),
    ("rain (rephrase)",    "How does rain happen?",             "water|cloud|droplet|precipitat"),
    ("skin (rephrase)",    "Describe what skin does.",          "body|protect|organ|cover"),
    ("math (rephrase)",    "Why do we study math?",             "number|calculat|quantit|problem"),
    ("sleep (rephrase)",   "Why do we need to sleep?",         "rest|brain|body|recover"),
    ("tree (rephrase)",    "What do trees do for us?",          "plant|wood|oxygen|grow|forest"),
    ("ocean (rephrase)",   "Describe the ocean.",               "water|sea|salt|deep"),
    ("learn (rephrase)",   "What does it mean to learn?",       "knowledge|skill|understand|acquire"),
]

EDGE_CASES = [
    # Genuinely unknown / nonsense -- expect low confidence.
    # Note: as Wikipedia training proceeds, calculus/blockchain become known --
    # only pure nonsense words remain reliably unknown.
    ("unknown: blockchain","What is a blockchain?",             None),
    ("nonsense",           "What is a flibbertigibbet?",        None),
    ("nonsense2",          "What is a snorkelblast?",           None),
]

ALL_TESTS = [
    ("=== EXACT QUESTIONS (trained verbatim) ===", EXACT_QUESTIONS),
    ("=== REPHRASED QUESTIONS (word-overlap only) ===", REPHRASED_QUESTIONS),
    ("=== EDGE CASES (not trained -- expect uncertainty) ===", EDGE_CASES),
]

# ---------------------------------------------------------------------------
# Thresholds (raw activation, not normalized confidence)
# ---------------------------------------------------------------------------
# The /chat endpoint returns normalized confidence (always 1.0 for the top
# result) which is meaningless for thresholding.  We use /qa/query to get
# the raw Hebbian activation score for all checks.
PASS_ACT    = 0.12   # trained question: raw activation must be >= this
EDGE_ACT    = 2.90   # edge case PASSES if top raw activation is < this
                     # Activation floor rises as training scales -- anchor bursts for
                     # each concept push trained concepts to 3-9; common-token noise
                     # for unknown words lands ~2.79.  Set just below the lowest
                     # trained concept (circle ~2.99) to keep a clean gap.


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def neuro_ask(client: httpx.Client, node_url: str, question: str) -> dict:
    """POST /chat -> full response dict (same handler as /neuro/ask)."""
    try:
        r = client.post(
            f"{node_url}/chat",
            json={"text": question, "hops": 2, "top_k": 5, "min_strength": 0.01},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def qa_query(client: httpx.Client, node_url: str, question: str) -> dict:
    """POST /qa/query -> QA report dict."""
    try:
        r = client.post(
            f"{node_url}/qa/query",
            json={"question": question},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def extract_best_activation(qa_resp: dict) -> float:
    """Pull the top raw activation score from a /qa/query response.

    The normalized 'confidence' field in /chat is always 1.0 for the top
    result and is meaningless for thresholding.  Raw activation is the actual
    Hebbian score before normalization and is what the min_activation gate
    in QaRuntime uses -- it's the right metric for pass/fail decisions.
    """
    results = qa_resp.get("report", {}).get("results", [])
    if results:
        return max(r.get("activation", 0.0) for r in results)
    return 0.0


def best_qa_answer(qa_resp: dict) -> str:
    """Return the answer text with the highest raw activation."""
    results = qa_resp.get("report", {}).get("results", [])
    if not results:
        return ""
    top = max(results, key=lambda r: r.get("activation", 0.0))
    return top.get("answer", "")


def keyword_hit(response_text: str, keyword: Optional[str]) -> bool:
    """Return True if any | -separated keyword variant appears in response_text."""
    if keyword is None:
        return True   # edge cases: keyword check irrelevant
    text = response_text.lower()
    return any(k.strip() in text for k in keyword.lower().split("|"))


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label: str
    question: str
    keyword: Optional[str]
    confidence: float
    response: str
    qa_answer: str
    passed: bool
    note: str


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_tests(node_url: str, verbose: bool) -> None:
    client = httpx.Client()

    # Quick health check
    try:
        h = client.get(f"{node_url}/health", timeout=5)
        info = h.json()
        print(f"Node: {info.get('node_id', '?')}  status={info.get('status')}  uptime={info.get('uptime_secs')}s")
    except Exception as e:
        sys.exit(f"Node not reachable at {node_url}: {e}")

    # Also peek at QA store stats via /qa/stats if available, else skip
    try:
        qs = client.get(f"{node_url}/qa/stats", timeout=5)
        if qs.status_code == 200:
            d = qs.json()
            print(f"QA store: {d.get('pairs_ingested','?')} pairs  "
                  f"{d.get('answer_entries','?')} answers  "
                  f"{d.get('question_neurons','?')} q-neurons")
    except Exception:
        pass

    print()

    all_results: list[Result] = []

    for section_label, cases in ALL_TESTS:
        is_edge = "EDGE" in section_label
        print(section_label)
        print("-" * len(section_label))

        for label, question, keyword in cases:
            qa_resp  = qa_query(client, node_url, question)
            chat_resp = neuro_ask(client, node_url, question)

            if "error" in qa_resp and "error" in chat_resp:
                result = Result(
                    label=label, question=question, keyword=keyword,
                    confidence=0.0, response="[REQUEST FAILED]",
                    qa_answer="", passed=False, note=qa_resp.get("error", "")
                )
            else:
                act  = extract_best_activation(qa_resp)
                top_qa = best_qa_answer(qa_resp)
                response_text = chat_resp.get("answer", "") if "error" not in chat_resp else ""

                if is_edge:
                    passed = act < EDGE_ACT
                    note = "correctly uncertain" if passed else f"false positive (act={act:.3f})"
                else:
                    kw_ok = keyword_hit(response_text + " " + top_qa, keyword)
                    passed = act >= PASS_ACT and kw_ok
                    if act < PASS_ACT:
                        note = f"low act={act:.3f}"
                    elif not kw_ok:
                        note = f"keyword '{keyword}' missing (act={act:.3f})"
                    else:
                        note = f"act={act:.3f}"

                result = Result(
                    label=label, question=question, keyword=keyword,
                    confidence=act, response=response_text,
                    qa_answer=top_qa, passed=passed, note=note,
                )

            status = "PASS" if result.passed else "FAIL"
            print(f"  [{status}] {result.label:<30} {result.note}")

            if verbose or not result.passed:
                print(f"         Q: {result.question}")
                if result.response:
                    print(f"         R: {result.response[:180]}")
                if result.qa_answer and result.qa_answer != result.response:
                    print(f"        QA: {result.qa_answer[:180]}")
                print()

            all_results.append(result)

        print()

    # Summary
    trained_tests = [r for r in all_results if "unknown" not in r.label and r.label != "nonsense"]
    edge_tests    = [r for r in all_results if "unknown" in r.label or r.label == "nonsense"]

    trained_pass = sum(1 for r in trained_tests if r.passed)
    edge_pass    = sum(1 for r in edge_tests    if r.passed)

    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Trained questions : {trained_pass}/{len(trained_tests)} passed")
    print(f"Edge cases        : {edge_pass}/{len(edge_tests)} correctly uncertain")
    print()

    failures = [r for r in trained_tests if not r.passed]
    if failures:
        print("FAILURES (trained questions that should have answers):")
        for r in failures:
            print(f"  - {r.label}: {r.note}")
        print()

    total = len(all_results)
    total_pass = sum(1 for r in all_results if r.passed)
    pct = 100 * total_pass / total if total else 0
    print(f"Overall: {total_pass}/{total} ({pct:.0f}%)")

    if pct >= 80:
        print("\nArchitecture looks solid -- ready for more training data.")
    elif pct >= 55:
        print("\nPartially working -- review failures above; may need tuning.")
    else:
        print("\nSignificant gaps -- investigate before scaling training.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0 validation tests")
    parser.add_argument("--node", default="http://127.0.0.1:8090", help="Node API URL")
    parser.add_argument("--verbose", action="store_true", help="Show all responses, not just failures")
    args = parser.parse_args()
    run_tests(args.node, args.verbose)
