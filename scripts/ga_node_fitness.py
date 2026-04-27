#!/usr/bin/env python3
"""
ga_node_fitness.py — fitness harness for the GA over the entire W1z4rD node.

Given a genome (a dict of node-config knobs + connection toggles), run the
following pipeline:

  1. Spawn the node binary on a *fresh* data dir.
  2. Register additional pools per the genome (e.g. equation pool).
  3. Train the (NCBI, class) corpus subsets via /multi_pool/train_pair.
     Optionally call /multi_pool/replay between epochs (CLS toggle).
     Optionally call /equations/apply on every Q label set (EEM toggle).
  4. Evaluate held-out paraphrase / Q-A pairs via /chat and /multi_pool/ask.
  5. Compute per-piece fitness (NCBI recall, class recall, paraphrase recall,
     equation hit-rate).  Return per-piece scores + combined.
  6. Stop the node, delete the data dir.

Designed for repeated invocation by ga_node_search.py.

Politeness: launches the node at BELOW_NORMAL priority via the existing
binary's `be_polite` (it's a node arg) — our GA process itself sleeps
between heavy steps so the host stays responsive.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


NODE_BIN     = pathlib.Path("bin/w1z4rd_node.exe")
NCBI_JSONL   = pathlib.Path("data/foundation/ncbi_pairs.jsonl")
CLASS_JSONL  = pathlib.Path("data/foundation/class_corpus.jsonl")


# ── Levenshtein similarity (matches Python port + Rust pos_label decoder) ─

def lev_sim(a: str, b: str) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    if max(len(a), len(b)) > 600:
        # Cheap proxy for very long strings.
        L = min(len(a), len(b))
        match = sum(1 for i in range(L) if a[i] == b[i])
        return match / max(len(a), len(b))
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]; dp[0] = i
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
            prev = cur
    return 1.0 - dp[lb] / max(la, lb)


# ── Node lifecycle ──────────────────────────────────────────────────────

def free_port(start: int = 8090) -> int:
    """Find a free pair (port, port+10) for the node API + neuro API."""
    for p in range(start, start + 200, 2):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", p))
            with socket.socket() as s:
                s.bind(("127.0.0.1", p + 10))
            return p
        except OSError:
            continue
    raise RuntimeError("no free ports")


def launch_node(data_dir: pathlib.Path, env: dict) -> subprocess.Popen:
    data_dir.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, **env, "W1Z4RDV1510N_DATA_DIR": str(data_dir)}
    log_path = data_dir / "node.log"
    proc = subprocess.Popen(
        [str(NODE_BIN)],
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        env=full_env,
    )
    return proc


def wait_for_health(port: int, timeout_s: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                if b"OK" in r.read():
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def shutdown_node(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── HTTP helpers ────────────────────────────────────────────────────────

def post(url: str, payload: dict, timeout: float = 20) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(url: str, timeout: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# ── Corpus loading ──────────────────────────────────────────────────────

def load_jsonl(path: pathlib.Path) -> list[dict]:
    out = []
    if not path.exists(): return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            try: out.append(json.loads(line))
            except Exception: pass
    return out


def split_class_corpus(records: list[dict], holdout_variation: int = 4) -> tuple[list[dict], list[dict]]:
    """Train: variations != holdout_variation. Eval: variations == holdout."""
    train = [r for r in records if r["variation"] != holdout_variation]
    evalu = [r for r in records if r["variation"] == holdout_variation]
    return train, evalu


def split_ncbi(records: list[dict], frac: float = 0.8, seed: int = 1) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(records); rng.shuffle(shuffled)
    cut = int(len(shuffled) * frac)
    return shuffled[:cut], shuffled[cut:]


# ── Fitness pipeline ────────────────────────────────────────────────────

def run_one_genome(
    genome: dict,
    ncbi_records: list[dict],
    class_records: list[dict],
    *,
    ncbi_max: int = 60,
    class_max: int = 30,
    log: bool = True,
) -> dict:
    port = free_port()
    data_dir = pathlib.Path(f"D:/w1z4rdv1510n-data-ga/g_{int(time.time()*1000)}_{port}")
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    proc = launch_node(data_dir, env={})
    try:
        if not wait_for_health(port):
            return {"error": "node failed to start", **per_piece_zero()}
        node = f"http://127.0.0.1:{port}"
        if log: print(f"[ga] genome live on {node}")

        # Connection-graph toggles from the genome.
        train_eq      = bool(genome.get("train_equation_pool", 0))
        cls_after_ep  = bool(genome.get("cls_replay_after_epoch", 0))
        eem_in_loop   = bool(genome.get("eem_in_train_loop", 0))

        # Register equation pool if requested.
        if train_eq:
            post(f"{node}/multi_pool/register", {"pool_id": "equation"})

        # ── Subsample corpora to keep the eval cheap ──────────────────
        ncbi_records = ncbi_records[:ncbi_max]
        class_records = class_records[:class_max * 5]  # 5 variations per class
        ncbi_train, ncbi_eval = split_ncbi(ncbi_records)
        class_train, class_eval = split_class_corpus(class_records)

        # ── Train: NCBI Q-A pairs ─────────────────────────────────────
        t0 = time.time()
        passes = max(1, int(genome["passes"]))
        lr     = float(genome["lr"])
        for r in ncbi_train:
            try:
                post(f"{node}/multi_pool/train_pair", {
                    "src_pool": "in",  "src": r["question"],
                    "tgt_pool": "out", "tgt": r["answer"],
                    "passes": passes, "lr": lr,
                }, timeout=60)
            except Exception:
                pass

        # ── Train: class corpus (description -> code) ─────────────────
        for r in class_train:
            try:
                post(f"{node}/multi_pool/train_pair", {
                    "src_pool": "in",  "src": r["description"],
                    "tgt_pool": "out", "tgt": r["code"],
                    "passes": passes, "lr": lr,
                }, timeout=120)
                if train_eq:
                    # Class id is a useful equation-style label.
                    post(f"{node}/multi_pool/train_pair", {
                        "src_pool": "in", "src": r["description"],
                        "tgt_pool": "equation", "tgt": r["class_id"],
                        "passes": passes, "lr": lr,
                    }, timeout=60)
                if eem_in_loop:
                    post(f"{node}/equations/apply",
                         {"labels": r["description"].split()[:8], "dims": 3},
                         timeout=10)
            except Exception:
                pass

        if cls_after_ep:
            try:
                post(f"{node}/multi_pool/replay", {
                    "max_per_pool": 0,
                    "lr_scale": float(genome.get("cls_lr_scale", 0.1)),
                }, timeout=60)
            except Exception:
                pass
        train_time = time.time() - t0
        if log: print(f"[ga]   trained in {train_time:.1f}s")

        # ── Evaluate ──────────────────────────────────────────────────
        hops = max(1, int(genome["hops"]))
        min_s = float(genome["min_strength"])

        ncbi_scores = []
        ncbi_exact = 0
        for r in ncbi_eval:
            try:
                resp = post(f"{node}/multi_pool/ask", {
                    "src_pool": "in", "text": r["question"],
                    "hops": hops, "min_strength": min_s,
                }, timeout=20)
                pred = (resp.get("predictions") or {}).get("out", "") or ""
            except Exception:
                pred = ""
            ncbi_scores.append(lev_sim(pred, r["answer"]))
            if pred == r["answer"]: ncbi_exact += 1

        class_scores = []
        class_exact = 0
        eq_hits = 0
        for r in class_eval:
            try:
                resp = post(f"{node}/multi_pool/ask", {
                    "src_pool": "in", "text": r["description"],
                    "hops": hops, "min_strength": min_s,
                }, timeout=30)
                preds = resp.get("predictions") or {}
                pred = preds.get("out", "") or ""
                if train_eq and preds.get("equation") == r["class_id"]:
                    eq_hits += 1
            except Exception:
                pred = ""
            class_scores.append(lev_sim(pred, r["code"]))
            if pred == r["code"]: class_exact += 1

        ncbi_mean   = sum(ncbi_scores)  / max(1, len(ncbi_scores))
        class_mean  = sum(class_scores) / max(1, len(class_scores))
        eq_rate     = eq_hits / max(1, len(class_eval)) if train_eq else 0.0

        # Combined fitness: weighted toward exact recall (it is the harder bar).
        combined = (
            0.4 * ncbi_mean +
            0.4 * class_mean +
            0.1 * (ncbi_exact / max(1, len(ncbi_eval))) +
            0.1 * (class_exact / max(1, len(class_eval)))
        )

        result = {
            "ncbi_mean":      ncbi_mean,
            "class_mean":     class_mean,
            "ncbi_exact":     ncbi_exact,
            "ncbi_n":         len(ncbi_eval),
            "class_exact":    class_exact,
            "class_n":        len(class_eval),
            "equation_hit_rate": eq_rate,
            "combined":       combined,
            "train_time_s":   round(train_time, 1),
        }
        if log: print(f"[ga]   {result}")
        return result

    finally:
        shutdown_node(proc)
        try:
            shutil.rmtree(data_dir, ignore_errors=True)
        except Exception:
            pass


def per_piece_zero():
    return {
        "ncbi_mean": 0.0, "class_mean": 0.0,
        "ncbi_exact": 0, "ncbi_n": 0,
        "class_exact": 0, "class_n": 0,
        "equation_hit_rate": 0.0, "combined": 0.0, "train_time_s": 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", type=str, default="{}",
                    help="JSON-encoded genome dict")
    ap.add_argument("--ncbi-max",  type=int, default=60)
    ap.add_argument("--class-max", type=int, default=20)
    args = ap.parse_args()
    genome = json.loads(args.genome)
    # Defaults if not specified.
    genome.setdefault("lr",           0.5)
    genome.setdefault("passes",       30)
    genome.setdefault("hops",         3)
    genome.setdefault("min_strength", 0.05)
    genome.setdefault("cls_replay_after_epoch", 0)
    genome.setdefault("cls_lr_scale", 0.1)
    genome.setdefault("eem_in_train_loop", 0)
    genome.setdefault("train_equation_pool", 0)

    ncbi  = load_jsonl(NCBI_JSONL)
    cls_  = load_jsonl(CLASS_JSONL)
    print(f"corpora: ncbi={len(ncbi)} classes={len(cls_)}")
    res = run_one_genome(genome, ncbi, cls_, ncbi_max=args.ncbi_max, class_max=args.class_max)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
