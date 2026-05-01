#!/usr/bin/env python3
"""
train_concept_bindings.py — bind key Q->A pairs into the GA-tuned multi-pool fabric.

The slow-pool path (/media/train_sequence) trained by the rest of the
curriculum gives the system character-level recall.  The multi-pool path
(/multi_pool/train_pair) is what the GA-experimental search tuned for the
+0.024 orig-combined gain (0.840 -> 0.864): position-augmented + trigram
encoding, lr=0.825, passes=35, mp_confidence_threshold=0.345.

This script binds the conversational identity pairs and a handful of
foundational identity facts through that path AFTER the curriculum has
trained the slow pool.  /query/integrated routes through multi_pool first
when its confidence > 0.345, so these bindings give the chat fast,
high-confidence recall of greetings + identity without disturbing the
broad knowledge that lives in the slow pool.

Sends the runtime's defaults (passes / lr / hops / min_strength /
mp_confidence_threshold / use_eem_fallback all match the GA winner since
2026-05-01).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# Pull the full 36-pair greeting + identity list from the conversational
# training script so we don't drift between the two surfaces.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from train_conversations import CONVERSATIONS  # noqa: E402


# Foundational identity facts beyond greetings.  These deliberately
# overlap with the greeting answers so the multi-pool concept neurons
# accumulate evidence from multiple Q phrasings.
IDENTITY_BINDINGS: list[tuple[str, str]] = [
    ("what is w1z4rd",
     "W1z4rD is a distributed Hebbian neural AI node that learns from training data and experience."),
    ("what is w1z4rd vision",
     "W1z4rD Vision is the neural AI system: Hebbian learning, multi-pool fabric, slow-pool char-chain decoder, EEM equation matrix."),
    ("how do you work",
     "I learn through Hebbian co-occurrence: training pairs strengthen synapses between input and output atoms, then queries propagate activation through the resulting graph."),
    ("how do you learn",
     "Each training pair runs Hebbian updates between source and target atoms; concepts emerge through mini-column collapse; concept-to-concept edges form via the multi-pool fabric."),
    ("what is hebbian learning",
     "Hebbian learning is the rule that neurons which fire together wire together: co-active atoms strengthen their connection proportional to their joint activation."),
]


def _load_code_corpus() -> list[tuple[str, str]]:
    """Pull every (Q, A) from build_code_corpus.CORPUS.

    build_code_corpus imports httpx + neuro_client at module level for its
    own training loop; we don't need those — only the CORPUS data — so we
    stub the deps if they aren't installed.  Returns a flat list of
    (question, answer) tuples covering JS/TS/CSS/HTML/Angular/Ionic/Python/
    PHP/C/C++/C#/Rust/Perl/JSON/XML/applied math/Blender/KiCad/ngspice/
    OpenSCAD/Verilator/Linux+Windows terminal/Git/Docker/agent decisions.
    """
    import importlib.util
    import types
    for stub in ("httpx", "neuro_client"):
        if stub not in sys.modules:
            mod = types.ModuleType(stub)
            if stub == "neuro_client":
                mod.NeuroClient = object  # type: ignore[attr-defined]
            sys.modules[stub] = mod
    spec = importlib.util.spec_from_file_location(
        "build_code_corpus",
        str(__import__("pathlib").Path(__file__).resolve().parent / "build_code_corpus.py"),
    )
    if spec is None or spec.loader is None:
        return []
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as exc:
        print(f"WARN: could not import build_code_corpus.CORPUS: {exc}", file=sys.stderr)
        return []
    pairs: list[tuple[str, str]] = []
    for _title, _disc, qas in getattr(m, "CORPUS", []):
        for q, a in qas:
            pairs.append((q, a))
    return pairs


CODE_BINDINGS: list[tuple[str, str]] = _load_code_corpus()


def _post(url: str, payload: dict, timeout: float = 60) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default="http://localhost:8090")
    args = ap.parse_args()

    try:
        h = _get(f"{args.node}/health")
        print(f"Node online: {h.get('node_id', '?')}")
    except Exception as exc:
        print(f"ERROR: node offline: {exc}", file=sys.stderr)
        return 1

    pairs = list(CONVERSATIONS) + IDENTITY_BINDINGS + CODE_BINDINGS
    total = len(pairs)
    print(f"Pair budget: {len(CONVERSATIONS)} greetings + {len(IDENTITY_BINDINGS)} identity + "
          f"{len(CODE_BINDINGS)} code = {total} total")
    ok = 0
    t0 = time.time()
    for i, (q, a) in enumerate(pairs, 1):
        # Omit passes/lr so the runtime applies the GA-tuned defaults
        # (passes=35, lr=0.825) banked on 2026-05-01.
        try:
            resp = _post(f"{args.node}/multi_pool/train_pair", {
                "src_pool": "in",  "src": q,
                "tgt_pool": "out", "tgt": a,
            }, timeout=120)
            stats = (resp.get("stats") or {}).get("pools") or {}
            in_n  = (stats.get("in")  or {}).get("neurons", 0)
            out_n = (stats.get("out") or {}).get("neurons", 0)
            xedges = (resp.get("stats") or {}).get("cross_edges", 0)
            print(f"[{i:2d}/{total}] {q!r:38s} -> in={in_n:4d} out={out_n:4d} x={xedges}")
            sys.stdout.flush()
            ok += 1
        except Exception as exc:
            print(f"[{i:2d}/{total}] FAILED {q!r}: {exc}", file=sys.stderr)

    dt = time.time() - t0
    print(f"\nBound {ok}/{total} pairs into multi-pool fabric in {dt:.1f}s")
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())
