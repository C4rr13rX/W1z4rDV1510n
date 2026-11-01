#!/usr/bin/env python3
"""
train_deterministic_innovation.py — Bind the 30-Year Digital Innovation Continuum
into the W1z4rD multi-pool Hebbian neural fabric.
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request

INNOVATION_CURRICULUM: list[tuple[str, str]] = [
    (
        "what is the deterministic path of digital innovation",
        "The deterministic continuum traverses graph topology (PageRank), distributed content DAGs (Git/Merkle), cryptographic conservation invariants (AMMs), continuous latent manifold attention (Transformers), and neuro-symbolic self-healing autonomy.",
    ),
    (
        "what is epoch 1 of digital innovation",
        "Epoch 1 (1996-2004) solved the global discovery bottleneck through graph topology and stationary eigenvector centrality (PageRank), turning isolated hypermedia into a structured, searchable directed graph.",
    ),
    (
        "what is epoch 2 of digital innovation",
        "Epoch 2 (2004-2010) decoupled state from monolithic hardware through distributed consensus (Paxos/Raft), content-addressed Merkle DAGs (Git), MapReduce, and horizontal cloud virtualization (AWS).",
    ),
    (
        "what is epoch 3 of digital innovation",
        "Epoch 3 (2008-2018) eliminated counterparty risk and centralized gatekeepers through Nakamoto consensus, Turing-complete smart contracts, and automated market maker conservation invariants (x * y = k).",
    ),
    (
        "what is epoch 4 of digital innovation",
        "Epoch 4 (2017-2023) solved high-dimensional unstructured representation by projecting chaotic discrete language and multimodal signals into continuous geometric embedding spaces via scaled dot-product attention.",
    ),
    (
        "what is epoch 5 of digital innovation",
        "Epoch 5 (2024-2026) unifies neural intuition with formal mathematical theorem proving: neuro-symbolic multi-agent swarms, continuous bio-inspired Hebbian learning, and self-healing supervision (Guardian).",
    ),
    (
        "how does the deterministic innovation core protect capital",
        "It calculates the Shannon entropy of incoming sensory attention, validates the cryptographic Merkle audit trail, and simulates constant-product AMM slippage before permitting trade execution.",
    ),
    (
        "how does w1z4rd synthesize digital innovation",
        "W1z4rD implements Hebbian co-occurrence across multi-pool topologies, integrating continuous perception with deterministic theorem verification and automated Guardian recovery.",
    ),
]


def post_pair(node: str, q: str, a: str, timeout: float = 120.0) -> dict:
    payload = json.dumps({
        "src_pool": "in",
        "src": q,
        "tgt_pool": "out",
        "tgt": a,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{node}/multi_pool/train_pair",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=payload,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    node = "http://localhost:8090"
    print(f"Connecting to W1z4rD node at {node}...")
    try:
        with urllib.request.urlopen(f"{node}/health", timeout=5) as r:
            health = json.loads(r.read())
            print(f"Node Online: {health.get('node_id', 'w1z4rd_node')}")
    except Exception as exc:
        print(f"Node offline or unreachable: {exc}", file=sys.stderr)
        return 1

    total = len(INNOVATION_CURRICULUM)
    print(f"Training {total} foundational digital innovation concept pairs into multi-pool fabric...")
    t0 = time.time()
    success = 0

    for i, (q, a) in enumerate(INNOVATION_CURRICULUM, 1):
        try:
            resp = post_pair(node, q, a)
            totals = (resp.get("stats") or {}).get("totals") or {}
            atoms = totals.get("atoms", 0)
            concepts = totals.get("concepts", 0)
            synapses = totals.get("exc_synapses", 0)
            print(f"[{i}/{total}] {q[:42]:42s} -> Atoms: {atoms:4d} | Concepts: {concepts:3d} | Synapses: {synapses:6d}")
            success += 1
        except Exception as exc:
            print(f"[{i}/{total}] FAILED {q}: {exc}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Successfully trained {success}/{total} pairs into W1z4rD brain in {elapsed:.2f}s.")
    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
