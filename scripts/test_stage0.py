#!/usr/bin/env python3
"""
W1z4rD V1510n — Stage 0 Validation Tests
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
# "expected_keyword" is just a loose word-level check — it does NOT need to
# match exactly.  It gives us a binary signal for scoring.
# ---------------------------------------------------------------------------

EXACT_QUESTIONS = [
    # Exact phrases from TODDLER_CONCEPTS — should all score well
    ("eyes (exact)",       "What are eyes?",                    "see"),
    ("brain (exact)",      "What is a brain?",                  "organ"),
    ("dog (exact)",        "What is a dog?",                    "animal"),
    ("apple (exact)",      "What is an apple?",                 "fruit"),
    ("water (exact)",      "What is water?",                    "liquid"),
    ("circle (exact)",     "What is a circle?",                 "round"),
    ("gravity (exact)",    "What is gravity?",                  "force"),
    ("sun (exact)",        "What is the sun?",                  "star"),
    ("color red (exact)",  "What is the color red?",            "color"),
    ("triangle (exact)",   "What is a triangle?",               "three"),
    ("teacher (exact)",    "What is a teacher?",                "learn"),
    ("energy (exact)",     "What is energy?",                   "work"),
    ("language (exact)",   "What is language?",                 "communicate"),
    ("sentence (exact)",   "What is a sentence?",               "complete"),
    ("music (exact)",      "What is music?",                    "sound"),
]

REPHRASED_QUESTIONS = [
    # Paraphrasings — tests whether Hebbian weight generalises across token variants
    ("heart (rephrase)",   "Tell me about the heart.",          "pump"),
    ("rain (rephrase)",    "How does rain happen?",             "water"),
    ("skin (rephrase)",    "Describe what skin does.",          "body"),
    ("math (rephrase)",    "Why do we study math?",             "number"),
    ("sleep (rephrase)",   "Why do we need to sleep?",         "rest"),
    ("tree (rephrase)",    "What do trees do for us?",          "plant"),
    ("ocean (rephrase)",   "Describe the ocean.",               "water"),
    ("learn (rephrase)",   "What does it mean to learn?",       "knowledge"),
]

EDGE_CASES = [
    # Things we definitely haven't trained — expect low confidence / don't-know
    ("unknown: calculus",  "What is calculus?",                 None),
    ("unknown: blockchain","What is a blockchain?",             None),
    ("nonsense",           "What is a flibbertigibbet?",        None),
]

ALL_TESTS = [
    ("=== EXACT QUESTIONS (trained verbatim) ===", EXACT_QUESTIONS),
    ("=== REPHRASED QUESTIONS (word-overlap only) ===", REPHRASED_QUESTIONS),
    ("=== EDGE CASES (not trained — expect uncertainty) ===", EDGE_CASES),
]

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
PASS_CONF   = 0.30   # confidence ≥ this → PASS for trained questions
EDGE_CONF   = 0.25   # confidence < this for untrained → edge case PASS


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def neuro_ask(client: httpx.Client, node_url: str, question: str) -> dict:
    """POST /chat → full response dict (same handler as /neuro/ask)."""
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
    """POST /qa/query → QA report dict."""
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


def extract_confidence(resp: dict) -> float:
    """Pull best confidence from a /chat response."""
    qa = resp.get("qa_candidates") or []
    if qa:
        return max(a.get("confidence", 0.0) for a in qa)
    return 0.0


def keyword_hit(response_text: str, keyword: Optional[str]) -> bool:
    if keyword is None:
        return True   # edge cases: keyword check irrelevant
    return keyword.lower() in response_text.lower()


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
            resp = neuro_ask(client, node_url, question)

            if "error" in resp:
                result = Result(
                    label=label, question=question, keyword=keyword,
                    confidence=0.0, response="[REQUEST FAILED]",
                    qa_answer="", passed=False, note=resp["error"]
                )
            else:
                conf = extract_confidence(resp)
                response_text = resp.get("answer", "")
                qa_answers = resp.get("qa_candidates") or []
                best_qa = qa_answers[0].get("answer", "") if qa_answers else ""

                if is_edge:
                    passed = conf < EDGE_CONF
                    note = "correctly uncertain" if passed else f"false positive (conf={conf:.2f})"
                else:
                    kw_ok = keyword_hit(response_text + " " + best_qa, keyword)
                    passed = conf >= PASS_CONF and kw_ok
                    if conf < PASS_CONF:
                        note = f"low conf={conf:.2f}"
                    elif not kw_ok:
                        note = f"keyword '{keyword}' missing (conf={conf:.2f})"
                    else:
                        note = f"conf={conf:.2f}"

                result = Result(
                    label=label, question=question, keyword=keyword,
                    confidence=conf, response=response_text,
                    qa_answer=best_qa, passed=passed, note=note,
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
        print("\nArchitecture looks solid — ready for more training data.")
    elif pct >= 55:
        print("\nPartially working — review failures above; may need tuning.")
    else:
        print("\nSignificant gaps — investigate before scaling training.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0 validation tests")
    parser.add_argument("--node", default="http://127.0.0.1:8090", help="Node API URL")
    parser.add_argument("--verbose", action="store_true", help="Show all responses, not just failures")
    args = parser.parse_args()
    run_tests(args.node, args.verbose)
