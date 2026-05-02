#!/usr/bin/env python3
"""
verify_and_reinforce.py — verifier-driven consolidation for the W1z4rD node.

Runs a battery of (question, expected_answer) pairs against /chat.  For
each pair that passes the match check, fires the existing dopamine-gated
LTP capture path so the *specific synapses that produced the correct
answer* get hardened against future training.  Pairs that fail are
appended to a regression queue for human review — failed pairs are NEVER
reinforced (that would lock in a wrong answer).

Biology this maps to:
  • train_pair lays down Hebbian edges with a transient `trace` tag.
  • A correct response means those tagged synapses participated in the
    successful propagation.
  • record_episode with surprise>0 raises dopamine, and the runtime's
    flush_dopamine_potentiation captures tagged synapses into late-LTP.
  • Captured synapses cluster at max_weight=4.0, becoming the dominant
    basin for that input — competing paths can't easily displace them.

Run modes:
  --check-only       Run battery, print pass/fail, no reinforcement.
  --reinforce-only   Skip /chat verification, just fire reinforcement on
                     every pair (use when you trust the battery as-is).
  (default)          Verify, reinforce only the passes, log the failures.

Match modes (per battery entry):
  exact     — normalize whitespace+case, expect identical match
  contains  — every key phrase listed must appear in the answer
  any_of    — at least one key phrase listed must appear

Battery defaults to scripts/verification_battery.json; if that file is
missing it's built at startup from train_convos_atoms.CONVERSATIONS and
train_concept_bindings.IDENTITY_BINDINGS so the verifier stays in sync
with the training data without a separate maintenance burden.
"""
from __future__ import annotations
import argparse
import base64
import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

DEFAULT_NODE = "http://localhost:8090"
DEFAULT_BATTERY = Path(__file__).resolve().parent / "verification_battery.json"
REGRESSION_LOG = Path("D:/w1z4rdv1510n-data/training/regression_queue.jsonl")
RESULT_LOG     = Path("D:/w1z4rdv1510n-data/training/verifier.log")

# Reinforcement tuning — same shape as wizard_chat/_wizard_train but
# slightly more passes since we know the answer is correct (no risk of
# burning in a wrong correction).
REINFORCE_PASSES   = 5
REINFORCE_LR_SLOW  = 0.40   # /media/train_sequence base_lr
REINFORCE_SURPRISE = 0.5    # record_episode surprise → dopamine pulse
MULTI_POOL_PASSES  = 10     # extra /multi_pool/train_pair passes


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _post(node: str, path: str, payload: dict, timeout: float = 30) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(node.rstrip("/") + path, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as exc:
        return {"error": str(exc)}


def _get(node: str, path: str, timeout: float = 5) -> dict:
    try:
        with urllib.request.urlopen(node.rstrip("/") + path, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


# ── Match logic ─────────────────────────────────────────────────────────────

_NORM_RE = re.compile(r"\s+")

def _norm(s: str) -> str:
    return _NORM_RE.sub(" ", (s or "").strip().lower())


def matches(actual: str, expected: str | list[str], mode: str) -> bool:
    a = _norm(actual)
    if mode == "exact":
        return a == _norm(expected if isinstance(expected, str) else expected[0])
    if mode == "contains":
        phrases = expected if isinstance(expected, list) else [expected]
        return all(_norm(p) in a for p in phrases)
    if mode == "any_of":
        phrases = expected if isinstance(expected, list) else [expected]
        return any(_norm(p) in a for p in phrases)
    raise ValueError(f"unknown match mode: {mode}")


# ── Battery loading ─────────────────────────────────────────────────────────

def _build_default_battery() -> list[dict]:
    """Pull (Q, A) pairs from the training scripts so the battery doesn't
    drift from what the curriculum actually trained.  Stub the optional
    deps `httpx`/`neuro_client` the way train_concept_bindings.py does so
    we can import it without a full venv."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    import types
    for stub in ("httpx", "neuro_client"):
        if stub not in sys.modules:
            mod = types.ModuleType(stub)
            if stub == "neuro_client":
                mod.NeuroClient = object
            sys.modules[stub] = mod

    battery: list[dict] = []
    # Greetings + identity from the training scripts themselves.
    try:
        from train_convos_atoms import CONVERSATIONS as GREETINGS
        for q, a in GREETINGS:
            battery.append({"q": q, "a": a, "match": "any_of",
                             "phrases": _key_phrases(a)})
    except Exception as exc:
        print(f"WARN: could not import train_convos_atoms.CONVERSATIONS: {exc}",
              file=sys.stderr)

    try:
        from train_concept_bindings import IDENTITY_BINDINGS
        for q, a in IDENTITY_BINDINGS:
            battery.append({"q": q, "a": a, "match": "any_of",
                             "phrases": _key_phrases(a)})
    except Exception as exc:
        print(f"WARN: could not import train_concept_bindings.IDENTITY_BINDINGS: "
              f"{exc}", file=sys.stderr)

    return battery


_STOPWORDS = {"a","an","the","is","i","am","are","you","to","of","in","on","at","for",
              "and","or","with","my","your","me","yes","no","ok","ask","what","who","how"}


def _key_phrases(answer: str) -> list[str]:
    """Pick discriminative tokens from the expected answer to use as
    `any_of` phrases.  Falls back to the literal answer if nothing
    discriminative is found.  Strips short/common words so we don't pass
    on accidental stopword overlap with garbage output."""
    words = re.findall(r"[A-Za-z0-9]+", answer)
    keep = [w for w in words if len(w) >= 4 and w.lower() not in _STOPWORDS]
    if not keep:
        keep = [answer]
    # Keep up to 3 most distinctive (longest); avoids noisy any_of with too many
    # short tokens that match incidentally.
    keep.sort(key=len, reverse=True)
    return keep[:3]


def load_battery(path: Path) -> list[dict]:
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    print(f"[verifier] {path} not found — building default battery from "
          f"training scripts", file=sys.stderr)
    return _build_default_battery()


# ── Reinforcement primitives ────────────────────────────────────────────────

def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _spans(text: str, role: str, y: float, idx: int, total: int) -> list:
    return [{"text": text, "role": role, "bold": False, "italic": False,
             "size_ratio": 1.0, "x_frac": 0.5, "y_frac": y,
             "seq_index": idx, "seq_total": total}]


def reinforce_pair(node: str, q: str, a: str) -> dict:
    """Reinforce a verified-correct pair on BOTH pool paths:
      • slow pool — train_sequence + record_episode(surprise=0.5) to
        fire the dopamine-gated late-LTP capture on the path that
        produced the correct answer.
      • multi_pool — extra train_pair passes so the concept-bound
        bridge edges get pushed to (or held at) max_weight, hardening
        the routing decision.
    """
    out = {"slow_pool": [], "multi_pool": None}

    # 1) Slow-pool sequence training with neuromodulator-gated LTP.
    q_words = q.split()
    a_words = a.split()
    total = len(q_words) + len(a_words)
    frames = []
    for j, w in enumerate(q_words):
        frames.append({
            "modality": "text", "t_secs": j * 0.05, "lr_scale": 1.0,
            "data_b64": _b64(w), "text": w,
            "spans": _spans(w, "body", 0.5, j, total),
        })
    a_start = (len(q_words) - 1) * 0.05 + 0.4 if q_words else 0.4
    for j, w in enumerate(a_words):
        frames.append({
            "modality": "text", "t_secs": a_start + j * 0.10, "lr_scale": 1.0,
            "data_b64": _b64(w), "text": w,
            "spans": _spans(w, "body", 0.5, len(q_words) + j, total),
        })
    for _ in range(REINFORCE_PASSES):
        sid = str(uuid.uuid4())
        r = _post(node, "/media/train_sequence", {
            "session_id": sid, "base_lr": REINFORCE_LR_SLOW,
            "tau_secs": 2.0, "frames": frames,
        }, timeout=20)
        out["slow_pool"].append("err" if r.get("error") else "ok")

    # 2) Dopamine pulse — captures the tagged synapses into late-LTP.
    _post(node, "/neuro/record_episode", {
        "context": q, "outcome": a, "surprise": REINFORCE_SURPRISE,
    }, timeout=10)

    # 3) Multi-pool extra passes so the concept binding holds at saturation.
    r = _post(node, "/multi_pool/train_pair", {
        "src_pool": "in",  "src": q,
        "tgt_pool": "out", "tgt": a,
        "passes": MULTI_POOL_PASSES,  # GA-tuned LR is the runtime default
    }, timeout=60)
    out["multi_pool"] = "err" if r.get("error") else "ok"
    return out


# ── Verifier loop ───────────────────────────────────────────────────────────

def verify_one(node: str, entry: dict) -> dict:
    q = entry["q"]
    expected = entry.get("phrases") or entry.get("a") or ""
    mode = entry.get("match", "any_of")
    r = _post(node, "/chat", {
        "text": q, "hops": 2, "min_strength": 0.05,
    }, timeout=10)
    actual = r.get("answer") or ""
    decoder = r.get("decoder", "?")
    pass_ = matches(actual, expected, mode)
    return {
        "q": q, "expected": expected, "match_mode": mode,
        "actual": actual, "decoder": decoder, "pass": pass_,
        "ts": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }


def run(node: str, battery: list[dict], *, check_only: bool,
         reinforce_only: bool) -> dict:
    REGRESSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    summary = {"total": len(battery), "passed": 0, "failed": 0,
                "reinforced": 0, "by_decoder": {}}
    t0 = time.time()
    for i, entry in enumerate(battery, 1):
        if reinforce_only:
            r = reinforce_pair(node, entry["q"], entry["a"])
            summary["reinforced"] += 1
            print(f"[{i:3d}/{len(battery)}] FORCED  {entry['q']!r:40s} -> {r}")
            continue

        result = verify_one(node, entry)
        decoder = result["decoder"]
        summary["by_decoder"][decoder] = summary["by_decoder"].get(decoder, 0) + 1

        with RESULT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")

        if result["pass"]:
            summary["passed"] += 1
            mark = "PASS"
            if not check_only:
                reinforce_pair(node, entry["q"], entry["a"])
                summary["reinforced"] += 1
                mark = "PASS+REINFORCED"
            print(f"[{i:3d}/{len(battery)}] {mark:18s} ({decoder:10s}) "
                  f"{entry['q']!r}")
        else:
            summary["failed"] += 1
            with REGRESSION_LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result) + "\n")
            short_actual = (result["actual"] or "")[:60].replace("\n", "\\n")
            print(f"[{i:3d}/{len(battery)}] FAIL ({decoder:10s})        "
                  f"{entry['q']!r:30s} -> {short_actual!r}", file=sys.stderr)

    summary["elapsed_s"] = round(time.time() - t0, 1)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default=DEFAULT_NODE)
    ap.add_argument("--battery", type=Path, default=DEFAULT_BATTERY)
    ap.add_argument("--check-only", action="store_true",
                     help="Verify only — never reinforce.")
    ap.add_argument("--reinforce-only", action="store_true",
                     help="Skip /chat verification — reinforce every pair "
                     "(use only when battery is fully trusted).")
    args = ap.parse_args()

    if args.check_only and args.reinforce_only:
        print("--check-only and --reinforce-only are mutually exclusive",
              file=sys.stderr)
        return 2

    health = _get(args.node, "/health")
    if health.get("error"):
        print(f"ERROR: node offline: {health['error']}", file=sys.stderr)
        return 1
    print(f"Node online: {health.get('node_id', '?')}")

    battery = load_battery(args.battery)
    print(f"Battery: {len(battery)} pairs")

    s = run(args.node, battery,
             check_only=args.check_only,
             reinforce_only=args.reinforce_only)
    print(f"\n=== Verifier summary ===")
    print(f"  total:      {s['total']}")
    print(f"  passed:     {s['passed']}  ({100*s['passed']/max(s['total'],1):.1f}%)")
    print(f"  failed:     {s['failed']}")
    print(f"  reinforced: {s['reinforced']}")
    print(f"  decoders:   {s['by_decoder']}")
    print(f"  elapsed:    {s['elapsed_s']}s")
    print(f"  results:    {RESULT_LOG}")
    if s["failed"]:
        print(f"  regressions:{REGRESSION_LOG}")

    # Exit 0 if everything passed, 3 if any regressed (so curricula can
    # detect and decide whether to halt or continue).
    return 0 if s["failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
