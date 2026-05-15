"""tools/training_standard/drive_corpora.py — feed registry corpora to the node.

Walks every TOML in the registry, reads the JSONL corpus listed in
`inputs[kind="corpus"]`, and POSTs each (prompt, response) pair to the
wizard node's /sensor/observe endpoint as paired text — establishing
the cross-pool associations the benchmarks later verify.

After each script's corpus has been fed N times (REPEATS, default 6
for robust Hebbian consolidation), the runner is invoked via
`mark-trained <id>` so the benchmarks fire and the event log records
the result.

Usage:
  python -m tools.training_standard.drive_corpora                # all
  python -m tools.training_standard.drive_corpora --script <id>  # one
  python -m tools.training_standard.drive_corpora --repeats 10

The wizard node URL is read from W1Z4RD_NODE_URL (default
http://127.0.0.1:8090).  Failures during /sensor/observe are logged
but don't halt the run — partial training is still useful.

This is the "curriculum executor" — the runner inside training_standard
verifies BENCHMARKS, this script does the actual TEACHING via paired
observations.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Local import paths so the script works without installing the package.
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.schema import load_registry  # noqa: E402
from tools.training_standard import runner as runner_mod   # noqa: E402


NODE_URL = os.environ.get("W1Z4RD_NODE_URL", "http://127.0.0.1:8090")


def post_observe(text: str, paired_text: str, session_id: str | None,
                  lr: float = 1.5, timeout: float = 15.0) -> tuple[bool, str]:
    """Send one paired-text observation.  Returns (ok, error_msg)."""
    body = {
        "kind":        "text",
        "text":        text,
        "paired_text": paired_text,
        "lr":          lr,
    }
    if session_id:
        body["session_id"] = session_id
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{NODE_URL.rstrip('/')}/sensor/observe",
        data=raw, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            _ = r.read()
        return True, ""
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except Exception as e:
        return False, str(e)


def read_corpus_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def smoke_test_memorization(rows: list[dict], n: int = 3,
                              timeout: float = 30.0) -> tuple[bool, list[dict]]:
    """Query the first `n` prompts back through /chat and check that the
    multi_pool decoder fires (not char_chain).  Returns (ok, samples).
    The pipeline is silently failing if every query falls through to
    char_chain after a script's first repeat — drive_corpora aborts the
    run in that case rather than wasting hours.

    "Ok" means at least one of the n queries came back with
    decoder == "multi_pool".  Strict — if it's char_chain across the
    board, training is not landing in the queryable fabric.
    """
    samples: list[dict] = []
    multi_pool_hits = 0
    for row in rows[:n]:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        if not prompt:
            continue
        body = json.dumps({"text": prompt}).encode("utf-8")
        req = urllib.request.Request(
            f"{NODE_URL.rstrip('/')}/chat",
            data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read())
        except Exception as e:
            samples.append({"prompt": prompt, "error": str(e)})
            continue
        decoder = resp.get("decoder")
        if decoder == "multi_pool":
            multi_pool_hits += 1
        samples.append({
            "prompt":     prompt,
            "decoder":    decoder,
            "confidence": resp.get("confidence"),
            "answer":     (resp.get("answer") or "")[:80],
        })
    return (multi_pool_hits > 0, samples)


def drive_one(script, repeats: int, project_root: Path,
                verbose: bool = False, smoke: bool = True) -> dict:
    """Feed one TOML script's corpus to the node, then mark trained.

    After the FIRST repeat, runs an inline smoke test querying 3 trained
    prompts back through /chat.  If every query returns char_chain (not
    multi_pool), the training pipeline is silently failing — drive_corpora
    aborts that script and reports the failure.  Caller can pass
    `smoke=False` to skip (used for re-training scripts whose corpora
    have only one item, etc.).
    """
    summary = {
        "script_id":   script.id,
        "category":    script.category,
        "phase":       script.phase,
        "pairs":       0,
        "posted_ok":   0,
        "posted_fail": 0,
        "repeats":     repeats,
        "smoke_ok":    None,
    }
    # Each corpus input becomes one stream of paired observations.
    for inp in script.inputs:
        if inp.kind != "corpus":
            continue
        corpus_path = project_root / inp.path
        rows = read_corpus_jsonl(corpus_path)
        if not rows:
            print(f"  [{script.id}] corpus missing or empty: {corpus_path}",
                    flush=True)
            continue
        summary["pairs"] += len(rows)
        sess = f"train:{script.id}"
        smoke_ran = False
        for r in range(repeats):
            for row in rows:
                prompt = (row.get("prompt") or row.get("question") or "").strip()
                resp   = (row.get("response") or row.get("answer") or "").strip()
                if not prompt or not resp:
                    continue
                ok, err = post_observe(prompt, resp, session_id=sess, lr=1.5)
                if ok:
                    summary["posted_ok"] += 1
                else:
                    summary["posted_fail"] += 1
                    if verbose:
                        print(f"  [{script.id}] post fail: {err}", flush=True)
            # Inline smoke after first repeat: verify that /chat now
            # returns multi_pool (not char_chain) for prompts we just
            # trained.  Catches silent failures BEFORE we waste another
            # 5 repeats on a broken pipeline.
            if smoke and not smoke_ran and r == 0 and rows:
                smoke_ok, samples = smoke_test_memorization(rows, n=3)
                summary["smoke_ok"] = smoke_ok
                summary["smoke_samples"] = samples
                smoke_ran = True
                if smoke_ok:
                    if verbose:
                        print(f"  [{script.id}] smoke OK (multi_pool decoder fired)",
                                flush=True)
                else:
                    print(f"  [{script.id}] !! SMOKE FAILED — every query fell "
                            f"to char_chain after first repeat:",
                            flush=True)
                    for s in samples:
                        print(f"     prompt: {s.get('prompt','')[:60]!r}",
                                flush=True)
                        print(f"     decoder: {s.get('decoder')}  "
                                f"conf: {s.get('confidence')}  "
                                f"ans: {s.get('answer')!r}",
                                flush=True)
                    print(f"  [{script.id}] aborting remaining repeats; "
                            "fabric is not capturing observations.",
                            flush=True)
                    break  # bail on remaining repeats for this script
        if verbose:
            print(f"  [{script.id}] corpus done — {repeats} reps × "
                    f"{len(rows)} pairs", flush=True)

    # After corpus pass, mark trained so the runner's benchmarks fire.
    # We construct argparse-like args and call cmd_mark_trained directly
    # to avoid spawning a subprocess.
    class _Args: pass
    a = _Args()
    a.script_id = script.id
    try:
        runner_mod.cmd_mark_trained(a)
        summary["marked"] = True
    except Exception as e:
        summary["marked"] = False
        summary["mark_error"] = str(e)
    return summary


def main(argv: list[str] | None = None) -> int:
    global NODE_URL
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default="",
                     help="train only this script id (default: all in phase order)")
    p.add_argument("--repeats", type=int, default=6,
                     help="repeat the corpus this many times per script "
                          "(default 6 — enough for cross-edges to saturate)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--node", default=NODE_URL,
                     help=f"node URL (default {NODE_URL})")
    args = p.parse_args(argv)

    # Override module-level node URL.
    NODE_URL = args.node.rstrip("/")
    runner_mod.NODE_URL = NODE_URL

    registry = load_registry(runner_mod.REGISTRY_DIR)
    if args.script:
        if args.script not in registry:
            print(f"unknown script: {args.script!r}", file=sys.stderr)
            return 2
        plan = [registry[args.script]]
    else:
        plan = sorted(registry.values(), key=lambda s: (s.phase, s.id))

    print(f"[drive_corpora] curriculum: {len(plan)} script(s), "
            f"repeats={args.repeats}", flush=True)
    summaries: list[dict] = []
    t0 = time.time()
    for s in plan:
        print(f"\n=== {s.id} (phase {s.phase}, {s.category}) ===", flush=True)
        summ = drive_one(s, args.repeats, _PROJECT_ROOT, verbose=args.verbose)
        summaries.append(summ)
        print(f"  pairs={summ['pairs']}  ok={summ['posted_ok']}  "
                f"fail={summ['posted_fail']}  marked={summ.get('marked')}",
                flush=True)
    dt = time.time() - t0
    total_ok   = sum(s["posted_ok"]   for s in summaries)
    total_fail = sum(s["posted_fail"] for s in summaries)
    print(f"\n[drive_corpora] done in {dt:.1f}s — "
            f"{total_ok} observations posted, {total_fail} failed",
            flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
