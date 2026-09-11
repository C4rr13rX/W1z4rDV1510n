python3 - <<'PY'
"""Why do THIS interval's 14 failures happen, in the record's own words?

The size hypothesis is dead: `go-systems:0:131072` and
`go-systems:131072:262144` -- 131,072 rows each -- admitted 16.5 h and 13.7 h
ago, while `jupyter-scientific-full:201344:262144` at 60,800 rows has failed 14
times over the same span. A work unit twice the size converts and this one does
not, so resizing it would have been another inert fix.

So dump the failures verbatim -- every key, not a summarised `reason`, since
the last probe found that field absent and reported a vacuous "?" for all 14 --
and read WHERE each died. Before the gate means the pass never finished; at the
gate means a real verdict. That split decides resize-versus-repair.

Corpus size is read with `stat`, never by counting lines: the previous revision
of this probe called `sum(1 for _ in fh)` on multi-gigabyte corpora and never
returned inside its transport bound.
"""
import collections
import json
import os
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}


def load(path):
    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception as exc:
        out.setdefault("errors", []).append(f"{path}: {exc}")
    return rows


health = load(f"{R}/curriculum-health.jsonl")
current = "jupyter-scientific-full:201344:262144"
try:
    with open(f"{R}/deferred-replay-active.json", "r", encoding="utf-8") as fh:
        current = str(json.load(fh).get("interval_id") or current)
except Exception:
    pass
out["current_interval"] = current

failures = [r for r in health
            if str(r.get("interval_id") or "") == current
            and str(r.get("kind") or "") == "deferred_replay_failed"]
out["failure_count"] = len(failures)
out["failure_keys_seen"] = sorted({k for r in failures for k in r})
out["failures_verbatim"] = [
    {k: (str(v)[:700] if isinstance(v, str) else v) for k, v in r.items()}
    for r in failures[-5:]
]
out["failure_ages_hours"] = [
    round((out["now"] - float(r.get("updated_unix") or r.get("unix") or 0)) / 3600.0, 2)
    for r in failures
]

for good in ("go-systems:0:131072", "go-systems:131072:262144"):
    hits = [r for r in health if str(r.get("interval_id") or "") == good]
    out[f"kinds__{good}"] = dict(collections.Counter(
        str(r.get("kind") or "?") for r in hits))

# Per-phase cost, from stat only.
progress = {}
for name in os.listdir(R):
    if not name.endswith(".progress.json"):
        continue
    path = os.path.join(R, name)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        continue
    corpus = str(payload.get("corpus") or "")
    entry = {
        "age_h": round((out["now"] - os.path.getmtime(path)) / 3600.0, 2),
        "durable_next_row": payload.get("durable_next_row"),
        "accepted_episodes": payload.get("accepted_episodes"),
        "corpus": os.path.basename(corpus),
    }
    try:
        entry["corpus_gb"] = round(os.path.getsize(corpus) / 2**30, 3)
    except Exception:
        pass
    progress[name] = entry
out["progress_files"] = dict(sorted(
    progress.items(), key=lambda kv: kv[1]["age_h"])[:12])

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
