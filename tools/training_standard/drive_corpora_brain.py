"""tools/training_standard/drive_corpora_brain.py
====================================================

Brain-server (port 8095) curriculum executor.  Updated for Stage 16
(see STATE.md / README.md "Brain crate" section).  Sibling to
`drive_corpora.py` which targets the legacy neuro node on port 8090;
this script teaches the brain crate via its native cross-pool
observation contract:

  POST /observe { pool_id: 1 (text),   frame: b64(prompt) }
  POST /observe { pool_id: 4 (action), frame: b64(response) }
  POST /tick    { }                          -> wires cross-pool terminals

That is the architectural mode the brain crate is *designed for* —
both atom sets land in the same `fabric.current_moment`, and
`advance_tick` grows the prompt-atom <-> response-atom binding
terminals that decode_best_trained_binding later traverses to answer
both /chat and /integrate (Stage 16 unified them via the same
authoritative decoder).

Stage 16 defaults:
  --burst            ON   dense-burst is required for word-level
                          concept emergence under wide corpora.
                          Per ARCHITECTURE.md §4.D.1 and the Stage 13
                          empirical findings; toggle off with
                          --epoch-interleaved if you specifically want
                          the legacy schedule for comparison.
  --sleep-between    ON   between each script's phase, POST /sleep so
                          weak concepts get pruned and recent moment
                          fingerprints get re-fired (CLS consolidation).
                          McClelland/McNaughton/O'Reilly 1995.
  --canonical-env    OFF  if set, applies the Stage-16 canonical
                          BRAIN_SPARSITY_ACTION / BRAIN_MIN_ATOM_SCORE
                          via /sensor/observe?... no, we can't change
                          the running brain's env mid-flight — those
                          must be set BEFORE the brain server starts.
                          See scripts/run_full_training.py for the
                          end-to-end orchestrator that handles env.

Usage:
  python -m tools.training_standard.drive_corpora_brain               # all
  python -m tools.training_standard.drive_corpora_brain --script <id> # one
  python -m tools.training_standard.drive_corpora_brain --repeats 8   # more reps

  # Smoke test that training actually populates queryable bindings —
  # after the first repeat, sample 3 trained prompts back through /chat.
  # Under Stage 16 we accept ANY non-char_chain decoder OR a reply that
  # matches one of the prompt's trained responses (categorical entries
  # can be multi-valued, so substring-match against the canonical
  # benchmark would miss legitimate trained hits).

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
import http.client
from urllib.parse import urlparse
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.training_standard.schema import load_registry  # noqa: E402
from tools.training_standard import runner as runner_mod   # noqa: E402


BRAIN_URL = os.environ.get("W1Z4RD_BRAIN_URL", "http://127.0.0.1:8090")
# When True (default), routes /observe/tick/chat/sleep/checkpoint
# through the /brain/* prefix on the merged main node binary.  Set
# WIZARD_USE_BRAIN_PREFIX=0 to fall back to the top-level paths the
# standalone w1z4rd_brain_server binary exposes.  Same convention as
# tools/wizard_session.py and web/wizard_chat/views.py.
_BRAIN_PREFIX = "" if os.environ.get(
    "WIZARD_USE_BRAIN_PREFIX", "1").strip().lower() in {"0","false","no"} else "/brain"

def _path(suffix: str) -> str:
    return f"{_BRAIN_PREFIX}{suffix}"

POOL_TEXT   = 1
POOL_ACTION = 4


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii").rstrip("=")


# Persistent HTTP connection — Windows pays a heavy per-connection
# tax (TIME_WAIT + new socket setup); on a 1.5M-neuron brain that
# inflated per-tick wall time from ~0.5 ms to ~2 s.  Keep one
# connection open and reuse it for every observe / tick / chat call.
_HTTP_CONN: http.client.HTTPConnection | None = None
_HTTP_CONN_URL: str = ""

def _get_conn() -> http.client.HTTPConnection:
    global _HTTP_CONN, _HTTP_CONN_URL
    if _HTTP_CONN is None or _HTTP_CONN_URL != BRAIN_URL:
        if _HTTP_CONN is not None:
            try: _HTTP_CONN.close()
            except Exception: pass
        u = urlparse(BRAIN_URL)
        host = u.hostname or "localhost"
        port = u.port or (443 if u.scheme == "https" else 80)
        if u.scheme == "https":
            _HTTP_CONN = http.client.HTTPSConnection(host, port, timeout=60)
        else:
            _HTTP_CONN = http.client.HTTPConnection(host, port, timeout=60)
        _HTTP_CONN_URL = BRAIN_URL
    return _HTTP_CONN

def _post(path: str, body: dict, timeout: float = 60.0) -> tuple[bool, dict | str]:
    """POST via persistent connection.  On any error closes + retries once
    so a stale keep-alive doesn't kill a whole training run."""
    raw = json.dumps(body).encode("utf-8")
    for attempt in range(2):
        conn = _get_conn()
        conn.timeout = timeout
        try:
            conn.request("POST", path, body=raw,
                         headers={"Content-Type": "application/json"})
            r = conn.getresponse()
            payload = r.read()
            if r.status >= 400:
                return False, f"HTTP {r.status}: {r.reason}"
            try:
                return True, json.loads(payload)
            except json.JSONDecodeError:
                return True, payload.decode("utf-8", "replace")
        except (http.client.HTTPException, ConnectionError, OSError) as exc:
            # Drop the dead conn and let the next iteration reopen it.
            global _HTTP_CONN
            try: conn.close()
            except Exception: pass
            _HTTP_CONN = None
            if attempt == 1:
                return False, str(exc)
    return False, "unreachable"


def post_xpool_pair(prompt: str, response: str) -> tuple[bool, str]:
    """One cross-pool training step: prompt -> text pool, response -> action
    pool, then advance_tick to crystallize the cross-pool binding."""
    if not prompt or not response:
        return False, "empty"
    ok_t, err_t = _post(_path("/observe"), {
        "pool_id": POOL_TEXT,
        "frame":   _b64url(prompt.encode("utf-8")),
    })
    if not ok_t:
        return False, f"text-observe: {err_t}"
    ok_a, err_a = _post(_path("/observe"), {
        "pool_id": POOL_ACTION,
        "frame":   _b64url(response.encode("utf-8")),
    })
    if not ok_a:
        return False, f"action-observe: {err_a}"
    ok_k, err_k = _post(_path("/tick"), {})
    if not ok_k:
        return False, f"tick: {err_k}"
    return True, ""


def read_corpus_jsonl(path: Path, *, skip_rows: int = 0,
                      limit_rows: int | None = None) -> list[dict]:
    """Read a bounded logical-row window from a JSONL corpus.

    Blank or malformed physical lines do not count toward ``skip_rows``.
    Keeping only the requested window in memory makes multi-gigabyte corpora
    safe to train as independently benchmarked, resumable chunks.
    """
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        valid_index = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if valid_index < skip_rows:
                valid_index += 1
                continue
            out.append(row)
            valid_index += 1
            if limit_rows is not None and len(out) >= limit_rows:
                break
    return out


# ── Smoke test ─────────────────────────────────────────────────────────────


def smoke_test_recall(rows: list[dict], n: int = 3,
                       timeout: float = 30.0) -> tuple[bool, list[dict]]:
    """Query the first `n` trained prompts back through brain /chat
    and check that at least one returns either:
      - a non-char_chain decoder (multi_pool / eem), OR
      - a `reply` string that's a non-empty substring of any trained
        response for this prompt (Stage 16 accept-any-trained-categorical
        semantics — categorical entries can be multi-valued).

    If every query falls through to char_chain AND no reply matches
    a trained response, the bindings aren't taking — log warning but
    keep going (later repeats may still consolidate).
    """
    # Build accepted-set for each prompt from this corpus batch:
    # multiple rows can share a prompt with different responses.
    accepted: dict[str, set[str]] = {}
    for row in rows:
        p = (row.get("prompt") or row.get("question") or "").strip()
        r = (row.get("response") or row.get("answer") or "").strip()
        if p and r:
            accepted.setdefault(p, set()).add(r)

    samples: list[dict] = []
    hits = 0
    # Sample evenly across the corpus, not just the first n (which
    # may all share the same response under one-hot corpora).
    if rows and n > 0:
        step = max(1, len(rows) // max(n, 1))
        probe_rows = [rows[i] for i in range(0, len(rows), step)][:n]
    else:
        probe_rows = []
    for row in probe_rows:
        prompt = (row.get("prompt") or row.get("question") or "").strip()
        if not prompt:
            continue
        ok, resp = _post(_path("/chat"), {"text": prompt}, timeout=timeout)
        if not ok or not isinstance(resp, dict):
            samples.append({"prompt": prompt, "error": str(resp)})
            continue
        decoder = resp.get("decoder") or resp.get("confidence_tier")
        oog     = (resp.get("grounding") or {}).get("outside_grounding")
        # Stage 16 brain /chat returns BOTH `reply` (the canonical
        # decoded answer) and `answer` (same string, kept for the
        # legacy Wizard-chat client).  Prefer reply.
        reply   = (resp.get("reply") or resp.get("answer") or "").strip()
        # Hit if decoder is substantive OR reply is a trained answer
        # for this prompt (substring tolerant: 'animal' in 'animala'
        # decoder-residual case, etc.).
        decoder_ok = bool(decoder) and decoder != "char_chain"
        trained_set = accepted.get(prompt, set())
        reply_ok = bool(reply) and any(
            v and (v in reply or reply in v) for v in trained_set
        )
        if decoder_ok or reply_ok:
            hits += 1
        samples.append({
            "prompt":     prompt,
            "decoder":    decoder,
            "oog":        oog,
            "reply":      reply[:80],
            "trained":    sorted(trained_set)[:4],
            "decoder_ok": decoder_ok,
            "reply_ok":   reply_ok,
        })
    return (hits > 0, samples)


# ── Sleep / replay (Stage 16 CLS consolidation) ───────────────────────────


def post_sleep_cycle(min_use_count: int = 2,
                     stale_ticks: int = 1000,
                     replay_count: int = 24,
                     replay_strength: float = 0.5,
                     timeout: float = 30.0) -> tuple[bool, dict | str]:
    """Trigger Stage-16 sleep cycle: prune weak concepts then replay
    recent moment fingerprints to consolidate surviving patterns.
    Called between scripts in the curriculum so each phase's bindings
    settle before the next phase introduces new patterns that could
    interfere via partial-atom overlap."""
    return _post(_path("/sleep"), {
        "min_use_count":   min_use_count,
        "stale_ticks":     stale_ticks,
        "replay_count":    replay_count,
        "replay_strength": replay_strength,
    }, timeout=timeout)


def post_checkpoint(timeout: float = 60.0) -> tuple[bool, dict | str]:
    """Persist brain.bin to the configured data dir."""
    return _post(_path("/checkpoint"), {}, timeout=timeout)


# ── Per-script driver ──────────────────────────────────────────────────────


def _midcheck(script, posted_so_far: int, project_root: Path,
              verbose: bool = False) -> dict | None:
    """Run the script's benchmarks + regression-protected benchmarks
    mid-training and append a mid_train_benchmark event so the live
    panel can chart progress without waiting for end-of-script.

    Returns the summary dict (also logged), or None on import failure.
    """
    try:
        from tools.training_standard import runner as runner_mod
        from tools.training_standard.schema import load_registry
    except Exception as exc:  # pragma: no cover — runner is in-repo
        print(f"  [{script.id}] midcheck import failed: {exc}", flush=True)
        return None

    registry = load_registry(runner_mod.REGISTRY_DIR)
    own = runner_mod.run_benchmarks(script)
    own_pass = sum(1 for r in own if r.passed)
    prot_summary: dict[str, tuple[int, int]] = {}
    for rp in script.regression_protects:
        if rp.script_id not in registry:
            continue
        prot = registry[rp.script_id]
        prr = runner_mod.run_benchmarks(prot)
        prot_summary[prot.id] = (sum(1 for r in prr if r.passed), len(prr))
        for r in prr:
            runner_mod._emit_benchmark_event(
                prot.id, r, prot.category, prot.phase, "mid_train_regression",
            )
    for r in own:
        runner_mod._emit_benchmark_event(
            script.id, r, script.category, script.phase, "mid_train_self",
        )
    runner_mod.append_event({
        "kind":          "mid_train_benchmark",
        "script_id":     script.id,
        "category":      script.category,
        "phase":         script.phase,
        "posted_ok":     posted_so_far,
        "self_pass":     own_pass,
        "self_total":    len(own),
        "protected":     {pid: {"pass": p, "total": t}
                          for pid, (p, t) in prot_summary.items()},
    })
    if verbose:
        prot_str = ", ".join(f"{pid}={p}/{t}" for pid, (p, t) in prot_summary.items()) or "-"
        print(f"  [{script.id}] midcheck @ {posted_so_far} rows: "
              f"self={own_pass}/{len(own)}  protected={prot_str}",
              flush=True)
    return {"self_pass": own_pass, "self_total": len(own),
            "protected": prot_summary}


def drive_one(script, repeats: int, project_root: Path,
                verbose: bool = False, smoke: bool = True,
                inter_post_sleep: float = 0.05,
                burst: bool = False,
                midcheck_rows: int = 50_000,
                limit_rows: int | None = None,
                start_row: int = 0) -> dict:
    """Drive one registry script's corpus through the brain.

    `burst=False` (default): epoch-interleaved schedule.  Each rep is
    one full pass through all pairs.  After `repeats` passes, every
    pair has been observed `repeats` times spread across the whole
    corpus.

    `burst=True`: dense-burst schedule.  Each pair is observed
    `repeats` times back-to-back BEFORE moving to the next pair.
    Within recent_atoms_window of any pool, the same prompt sequence
    repeats `repeats` consecutive times.  This is required for
    word-level (rather than morpheme-level) concept emergence under
    `concept_emergence_threshold=3` — see Pool::collapse_tail_to_concept
    and ARCHITECTURE.md §4.D.1.

    Empirical: dense-burst lifted toddler categorical recall from
    71.9% to 90.6%, and food/body went 0% → 90+% on the canonical
    32-pair test (scripts/brain_dense_burst_toddler_categorical.py).
    """
    summary = {
        "script_id":   script.id,
        "category":    script.category,
        "phase":       script.phase,
        "pairs":       0,
        "posted_ok":   0,
        "posted_fail": 0,
        "repeats":     repeats,
        "burst":       burst,
        "smoke_ok":    None,
        "target":      "brain_server",
        "start_row":   start_row,
    }
    for inp in script.inputs:
        if inp.kind != "corpus":
            continue
        corpus_path = project_root / inp.path
        rows = read_corpus_jsonl(
            corpus_path, skip_rows=start_row, limit_rows=limit_rows
        )
        if not rows:
            print(f"  [{script.id}] corpus missing or empty: {corpus_path}",
                    flush=True)
            continue
        if verbose and (start_row or limit_rows is not None):
            end_row = start_row + len(rows)
            print(f"  [{script.id}] logical rows [{start_row}, {end_row})",
                  flush=True)
        summary["pairs"] += len(rows)
        smoke_ran = False
        t0 = time.time()

        next_midcheck = midcheck_rows if midcheck_rows > 0 else None

        if burst:
            # Dense-burst: pair outer, reps inner.
            for row in rows:
                prompt = (row.get("prompt") or row.get("question") or "").strip()
                resp   = (row.get("response") or row.get("answer") or "").strip()
                if not prompt or not resp:
                    continue
                for _ in range(repeats):
                    ok, err = post_xpool_pair(prompt, resp)
                    if ok:
                        summary["posted_ok"] += 1
                    else:
                        summary["posted_fail"] += 1
                        if verbose:
                            print(f"  [{script.id}] post fail: {err}", flush=True)
                    if inter_post_sleep > 0:
                        time.sleep(inter_post_sleep)
                    if next_midcheck is not None and summary["posted_ok"] >= next_midcheck:
                        _midcheck(script, summary["posted_ok"], project_root, verbose)
                        next_midcheck += midcheck_rows
            if verbose:
                print(f"  [{script.id}] burst done ({len(rows)} pairs x "
                        f"{repeats} reps in {time.time()-t0:.1f}s)", flush=True)
        else:
            # Epoch-interleaved schedule (default).
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
                    if next_midcheck is not None and summary["posted_ok"] >= next_midcheck:
                        _midcheck(script, summary["posted_ok"], project_root, verbose)
                        next_midcheck += midcheck_rows
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
    p.add_argument("--repeats", type=int, default=4,
                     help="repeat the corpus this many times per script "
                          "(default 4 — Stage 16 dense-burst needs fewer reps "
                          "than epoch-interleaved; toddler-style corpora can "
                          "use --repeats 8 explicitly)")
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
    p.add_argument("--midcheck-rows", type=int, default=50_000,
                     help="re-run the script's own + protected benchmarks "
                          "every N successful POSTs (default 50,000; set to 0 "
                          "to disable mid-training checks).  Events logged "
                          "as kind=mid_train_benchmark in training_events.jsonl.")
    p.add_argument("--limit-rows", type=int, default=None,
                     help="cap rows ingested per script per input to N "
                          "(useful for smoke tests on large corpora).  "
                          "Applies BEFORE repeats — N rows are observed "
                          "`repeats` times.")
    p.add_argument("--start-row", type=int, default=0,
                     help="skip this many valid JSONL rows before reading; "
                          "combine with --limit-rows for resumable chunks")
    # Stage 16: --burst is now the DEFAULT.  Use --epoch-interleaved to
    # force the legacy schedule (mainly for comparison / regression).
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--burst", dest="burst", action="store_true",
                     help="dense-burst training schedule (DEFAULT for Stage 16): "
                          "each pair observed `repeats` times back-to-back "
                          "before moving to the next pair.  Required for "
                          "word-level concept emergence on wide corpora; "
                          "see ARCHITECTURE.md §4.D.1.")
    grp.add_argument("--epoch-interleaved", dest="burst", action="store_false",
                     help="legacy epoch-interleaved schedule (one full pass "
                          "per rep).  Mostly useful for comparison.")
    p.set_defaults(burst=True)
    # Stage 16 CLS consolidation between scripts.
    sleep_grp = p.add_mutually_exclusive_group()
    sleep_grp.add_argument("--sleep-between", dest="sleep_between",
                            action="store_true",
                            help="POST /sleep between scripts to consolidate "
                                 "(DEFAULT for Stage 16).  Prunes weak "
                                 "concepts + replays recent moment fingerprints.")
    sleep_grp.add_argument("--no-sleep-between", dest="sleep_between",
                            action="store_false",
                            help="skip the inter-script sleep cycle")
    p.set_defaults(sleep_between=True)
    p.add_argument("--checkpoint", action="store_true",
                     help="POST /checkpoint at the end so brain.bin persists "
                          "the trained state to disk")
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

    schedule = "dense-burst" if args.burst else "epoch-interleaved"
    print(f"[drive_corpora_brain] target={BRAIN_URL}  schedule={schedule}  "
            f"plan={len(plan)} script(s)  repeats={args.repeats}  "
            f"sleep_between={args.sleep_between}", flush=True)

    summaries: list[dict] = []
    t0 = time.time()
    for idx, s in enumerate(plan):
        print(f"\n=== [{idx+1}/{len(plan)}] {s.id} "
                f"(phase {s.phase}, {s.category}) ===", flush=True)
        summ = drive_one(s, args.repeats, _PROJECT_ROOT,
                            verbose=args.verbose,
                            smoke=not args.no_smoke,
                            inter_post_sleep=args.inter_post_sleep,
                            burst=args.burst,
                            midcheck_rows=args.midcheck_rows,
                            limit_rows=args.limit_rows,
                            start_row=args.start_row)
        summaries.append(summ)
        print(f"  pairs={summ['pairs']}  ok={summ['posted_ok']}  "
                f"fail={summ['posted_fail']}  smoke={summ.get('smoke_ok')}",
                flush=True)
        # CLS consolidation between scripts.  Not on the LAST one
        # (final checkpoint is enough).
        if args.sleep_between and idx + 1 < len(plan):
            ok_s, resp_s = post_sleep_cycle()
            if ok_s and isinstance(resp_s, dict):
                print(f"  [sleep] pruned={resp_s.get('pruned',0)} "
                        f"replayed={resp_s.get('replayed',0)} "
                        f"tick_now={resp_s.get('tick_now','?')}", flush=True)
            else:
                print(f"  [sleep] failed: {resp_s}", flush=True)

    dt = time.time() - t0
    total_ok   = sum(s["posted_ok"]   for s in summaries)
    total_fail = sum(s["posted_fail"] for s in summaries)
    print(f"\n[drive_corpora_brain] done in {dt:.1f}s — "
            f"{total_ok} xpool pairs posted, {total_fail} failed",
            flush=True)

    if args.checkpoint:
        ok_c, resp_c = post_checkpoint()
        if ok_c and isinstance(resp_c, dict):
            print(f"[checkpoint] wrote {resp_c.get('written_bytes','?')} bytes "
                    f"to {resp_c.get('path','?')}", flush=True)
        else:
            print(f"[checkpoint] failed: {resp_c}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
