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


def drive_one(script, repeats: int, project_root: Path,
                verbose: bool = False) -> dict:
    """Feed one TOML script's corpus to the node, then mark trained."""
    summary = {
        "script_id":   script.id,
        "category":    script.category,
        "phase":       script.phase,
        "pairs":       0,
        "posted_ok":   0,
        "posted_fail": 0,
        "repeats":     repeats,
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
        for r in range(repeats):
            for row in rows:
                prompt = (row.get("prompt") or row.get("question") or "").strip()
                resp   = (row.get("response") or row.get("answer") or "").strip()
                if not prompt or not resp:
                    continue
                # paired_text drives the cross-pool training: keyboard_text
                # source atoms = response chars, the modality pool (also
                # keyboard_text for text-only training, which falls back
                # to within-pool sequence + own-concept reinforcement).
                # session_id keeps the per-script temporal cache scoped
                # so two scripts' observations don't bleed into each
                # other's sequence edges.
                ok, err = post_observe(prompt, resp, session_id=sess, lr=1.5)
                if ok:
                    summary["posted_ok"] += 1
                else:
                    summary["posted_fail"] += 1
                    if verbose:
                        print(f"  [{script.id}] post fail: {err}", flush=True)
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
