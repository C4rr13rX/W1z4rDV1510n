#!/usr/bin/env python3
"""
train_deterministic_innovation.py — Bind the Pre-1980 Foundations and the 30-Year Digital Innovation Continuum
into the W1z4rD multi-pool Hebbian neural fabric.
"""
from __future__ import annotations
import json
import sys
import time
import urllib.request

INNOVATION_CURRICULUM: list[tuple[str, str]] = [
    (
        "what would embarrass your parents if discovered living in 1980 and missing from quantitative trading",
        "Missing the Kelly Criterion (1956 J.L. Kelly Jr.), Ed Thorp's quantitative market-neutral hedge fund and delta hedging (1966 Beat the Dealer, 1969 Princeton Newport Partners), Harry Markowitz's Mean-Variance Efficient Frontier (1952), Benoit Mandelbrot's Pareto fat-tail risk (1963), and Black-Scholes-Merton option pricing (1973). Without Kelly sizing f* = (bp - q)/b, trading with positive edge leads to ruin or sub-optimal geometric growth.",
    ),
    (
        "what is epoch 0 of digital innovation",
        "Epoch 0 (Pre-1980) established the mathematical bedrock of quantitative finance and information theory: Markowitz mean-variance optimization (1952), the Kelly Criterion for geometric capital growth (1956), Mandelbrot fat-tailed Pareto distributions (1963), Thorp convertible arbitrage (1969), and the Black-Scholes continuous delta hedge formula (1973).",
    ),
    (
        "what is the kelly criterion and half kelly",
        "Formulated by John Larry Kelly Jr. at Bell Labs in 1956 and applied to financial markets by Edward Thorp, the Kelly Criterion f* = (b*p - q)/b calculates the fraction of bankroll to allocate to maximize asymptotic log-wealth growth without risking gambler's ruin. Half-Kelly halves the fraction to safeguard against parameter uncertainty and Mandelbrot heavy tails.",
    ),
    (
        "how do pre 1980 foundations graduate strategies to live trading",
        "By applying Half-Kelly sizing and Mandelbrot fat-tail Expected Shortfall hurdles, strategies clear real DEX gas costs, achieve positive net expectancy with t-stat >= 2.0, and satisfy the live readiness gate.",
    ),
    (
        "what is the deterministic path of digital innovation",
        "The deterministic continuum traverses Epoch 0 quantitative foundations (Kelly/Thorp/Mandelbrot), graph topology (PageRank), distributed content DAGs (Git/Merkle), cryptographic conservation invariants (AMMs), continuous latent manifold attention (Transformers), and neuro-symbolic self-healing autonomy.",
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
        "It calculates the Shannon entropy of incoming sensory attention, validates the cryptographic Merkle audit trail, sizes positions via Half-Kelly, and simulates constant-product AMM slippage before permitting trade execution.",
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
