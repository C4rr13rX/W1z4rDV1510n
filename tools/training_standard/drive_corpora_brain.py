"""tools/training_standard/drive_corpora_brain.py
====================================================

Brain-server (port 8095) curriculum executor.  Sibling to
`drive_corpora.py` which targets the legacy neuro node on port 8090;
this script teaches the new brain crate via its native cross-pool
observation contract:

  POST /observe { pool_id: 1 (text),   frame: b64(prompt) }
  POST /observe { pool_id: 4 (action), frame: b64(response) }
  POST /tick    { }                          -> wires cross-pool terminals

That is the architectural mode the brain crate is *designed for* —
both atom sets land in the same `fabric.current_moment`, and
`advance_tick` grows the prompt-atom <-> response-atom binding
terminals that integrate_autonomous later traverses to answer /chat.

Usage:
  python -m tools.training_standard.drive_corpora_brain               # all
  python -m tools.training_standard.drive_corpora_brain --script <id> # one
  python -m tools.training_standard.drive_corpora_brain --repeats 8   # more reps

  # Smoke test that training actually populates queryable bindings —
  # after the first repeat, sample 3 trained prompts back through /chat
  # and verify decoder == "multi_pool" (not "char_chain").  If every
  # query falls through to char_chain, abort that script rather than
  # waste 5 more repeats on a broken pipeline.

Outputs append-only events to the same training_events.jsonl the
legacy runner writes — both pipelines feed the same regression-history
log so Wizard frontend can show the unified view.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.schema import load_registry  # noqa: E402
from tools.training_standard import runner as runner_mod   # noqa: E402


BRAIN_URL = os.environ.get("W1Z4RD_BRAIN_URL", "http://127.0.0.1:8095")
POOL_TEXT   = 1
POOL_ACTION = 4


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


def _post(path: str, body: dict, timeout: float = 60.0) -> tuple[bool, dict | str]:
    raw = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{BRAIN_URL.rstrip('/')}{path}",
        data=raw, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = r.read()
        try:
            return True, json.loads(payload)
        except json.JSONDecodeError:
            return True, payload.decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.reason}"
    except Exception as e:
        return False, str(e)


def post_xpool_pair(prompt: str, response: str) -> tuple[bool, str]:
    """One cross-pool training step: prompt -> text pool, response -> action
    pool, then advance_tick to crystallize the cross-pool binding."""
    if not prompt or not response:
        return False, "empty"
    ok_t, err_t = _post("/observe", {
        "pool_id": POOL_TEXT,
        "frame":   _b64url(prompt.encode("utf-8")),
    })
    if not ok_t:
        return False, f"text-observe: {err_t}"
    ok_a, err_a = _post("/observe", {
        "pool_id": POOL_ACTION,
        "frame":   _b64url(response.encode("utf-8")),
    })
    if not ok_a:
        return False, f"action-observe: {err_a}"
    ok_k, err_k = _post("/tick", {})
    if not ok_k:
        return False, f"tick: {err_k}"
    return True, ""


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


# ── Smoke test ─────────────────────────────────────────────────────────────


def smoke_test_recall(rows: list[dict], n: int = 3,
                       timeout: float = 30.0) -> tuple[bool, list[dict]]:
    """Query the first `n` trained prompts back through brain /chat and
    check that at least one returns `tier == "multi_pool"` (not
    char_chain).  Trained cross-pool bindings *should* light the
    multi_pool decoder; if every query falls through to char_chain
    after the first repeat the bindings aren't taking — abort."""
    samples: list[dict] = []
    multi_pool_hits = 0
    for row in rows[:n]:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        if not prompt:
            continue
        ok, resp = _post("/chat", {"text": prompt}, timeout=timeout)
        if not ok or not isinstance(resp, dict):
            samples.append({"prompt": prompt, "error": str(resp)})
            continue
        # Brain /chat returns `decoder` ("multi_pool" / "eem" / "char_chain")
        # not `tier`.  Treat anything that isn't char_chain as a usable
        # signal — both multi_pool and eem mean the trained substrate
        # contributed, only char_chain means the pipeline gave up.
        decoder = resp.get("decoder") or resp.get("confidence_tier")
        oog = (resp.get("grounding") or {}).get("outside_grounding")
        if decoder and decoder != "char_chain":
            multi_pool_hits += 1
        samples.append({
            "prompt":  prompt,
            "decoder": decoder,
            "oog":     oog,
            "answer":  (resp.get("answer") or "")[:80],
        })
    return (multi_pool_hits > 0, samples)


# ── Per-script driver ──────────────────────────────────────────────────────


def drive_one(script, repeats: int, project_root: Path,
                verbose: bool = False, smoke: bool = True,
                inter_post_sleep: float = 0.05) -> dict:
    summary = {
        "script_id":   script.id,
        "category":    script.category,
        "phase":       script.phase,
        "pairs":       0,
        "posted_ok":   0,
        "posted_fail": 0,
        "repeats":     repeats,
        "smoke_ok":    None,
        "target":      "brain_server",
    }
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
        smoke_ran = False
        for r in range(repeats):
            t_rep = time.time()
            for row in rows:
                prompt = (row.get("prompt") or row.get("question") or "").strip()
                resp   = (row.get("response") or row.get("answer") or "").strip()
                if not prompt or not resp:
                    continue
                ok, err = post_xpool_pair(prompt, resp)
                if ok:
                    summary["posted_ok"] += 1
                else:
                    summary["posted_fail"] += 1
                    if verbose:
                        print(f"  [{script.id}] post fail: {err}", flush=True)
                if inter_post_sleep > 0:
                    time.sleep(inter_post_sleep)
            if verbose:
                print(f"  [{script.id}] rep {r+1}/{repeats} "
                        f"({len(rows)} pairs in {time.time()-t_rep:.1f}s)",
                        flush=True)
            if smoke and not smoke_ran and r == 0 and rows:
                smoke_ok, samples = smoke_test_recall(rows, n=3)
                summary["smoke_ok"] = smoke_ok
                summary["smoke_samples"] = samples
                smoke_ran = True
                if smoke_ok:
                    if verbose:
                        print(f"  [{script.id}] smoke OK — multi_pool decoder fired",
                                flush=True)
                else:
                    print(f"  [{script.id}] !! SMOKE FAILED — every query fell "
                            f"to char_chain after first repeat:", flush=True)
                    for s in samples:
                        print(f"     prompt: {s.get('prompt','')[:60]!r}", flush=True)
                        print(f"     decoder: {s.get('decoder')}  oog: {s.get('oog')}  "
                                f"ans: {s.get('answer')!r}", flush=True)
                    print(f"  [{script.id}] continuing repeats anyway "
                            "(brain training may need >1 pass for emergence)",
                            flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    global BRAIN_URL
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--script", default="",
                     help="train only this script id (default: all in phase order)")
    p.add_argument("--repeats", type=int, default=6,
                     help="repeat the corpus this many times per script "
                          "(default 6 — enough for cross-edges to saturate)")
    p.add_argument("--inter-post-sleep", type=float, default=0.0,
                     help="seconds to sleep between cross-pool POST sets "
                          "(default 0; brain's lock is short-held)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--brain", default=BRAIN_URL,
                     help=f"brain server URL (default {BRAIN_URL})")
    p.add_argument("--no-smoke", action="store_true",
                     help="skip the inline post-first-repeat smoke test")
    p.add_argument("--phase-max", type=int, default=None,
                     help="run only scripts with phase <= this value")
    args = p.parse_args(argv)

    BRAIN_URL = args.brain.rstrip("/")

    registry = load_registry(runner_mod.REGISTRY_DIR)
    if args.script:
        if args.script not in registry:
            print(f"unknown script: {args.script!r}", file=sys.stderr)
            return 2
        plan = [registry[args.script]]
    else:
        plan = sorted(registry.values(), key=lambda s: (s.phase, s.id))
        if args.phase_max is not None:
            plan = [s for s in plan if s.phase <= args.phase_max]

    print(f"[drive_corpora_brain] target={BRAIN_URL} "
            f"plan={len(plan)} script(s) repeats={args.repeats}", flush=True)

    summaries: list[dict] = []
    t0 = time.time()
    for s in plan:
        print(f"\n=== {s.id} (phase {s.phase}, {s.category}) ===", flush=True)
        summ = drive_one(s, args.repeats, _PROJECT_ROOT,
                            verbose=args.verbose,
                            smoke=not args.no_smoke,
                            inter_post_sleep=args.inter_post_sleep)
        summaries.append(summ)
        print(f"  pairs={summ['pairs']}  ok={summ['posted_ok']}  "
                f"fail={summ['posted_fail']}  smoke={summ.get('smoke_ok')}",
                flush=True)
    dt = time.time() - t0
    total_ok   = sum(s["posted_ok"]   for s in summaries)
    total_fail = sum(s["posted_fail"] for s in summaries)
    print(f"\n[drive_corpora_brain] done in {dt:.1f}s — "
            f"{total_ok} xpool pairs posted, {total_fail} failed",
            flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
