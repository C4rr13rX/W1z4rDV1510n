python3 - <<'PY'
"""Which suite is the 12th, and is the forward block actually advancing?

Two questions the previous probe could not answer, for two different reasons.

The gate is reporting `first_passed_suites=11, confirm_passed_suites=11,
total_suites=12, passed=false` over and over: eleven of twelve suites pass on
both the first run and the confirmation, so the block is being rejected by ONE
suite, deterministically. The suite's identity is in the record but not in the
fields the last probe projected, so this one prints raw records instead of a
chosen shape.

The second question is whether row 49152 is frozen or merely sampled at rest.
`continuous_canary` freezes the row by design, so a single reading of 0 rows/s
proves nothing; this samples the freshest writer three times across a minute
and reports the actual row deltas. It also dumps one raw ledger line so the
real timestamp key is visible rather than guessed -- the last probe's
"no events in 6 h" was an artefact of guessing `unix`/`timestamp`, and a
filter that can never match reports 0 forever.
"""
import collections
import json
import os
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
LEDGER = R / "curriculum-health.jsonl"
out = {"now": time.time()}

# 1. Raw tail: schema first, interpretation second.
lines = []
try:
    with LEDGER.open("rb") as fh:
        fh.seek(max(0, LEDGER.stat().st_size - 400000))
        lines = fh.read().decode("utf-8", "replace").splitlines()[1:]
except Exception as exc:  # noqa: BLE001
    out["tail_err"] = str(exc)

out["raw_last_3"] = [ln[:2000] for ln in lines[-3:]]
out["ledger_keys"] = sorted({k for ln in lines[-400:] if ln.strip()
                             for k in (json.loads(ln).keys()
                                       if ln.strip().startswith("{") else [])})

# 2. Every gate/canary record in the tail, verbatim but bounded, so the suite
#    names survive whatever nesting they live in.
gate_raw, canary_raw = [], []
for ln in lines:
    ln = ln.strip()
    if not ln.startswith("{"):
        continue
    try:
        ev = json.loads(ln)
    except Exception:  # noqa: BLE001
        continue
    kind = str(ev.get("kind") or "")
    if "gate" in kind or "admit" in kind or "harvest" in kind:
        gate_raw.append(ev)
    if "canary" in kind:
        canary_raw.append(ev)

def shrink(ev, cap=1400):
    s = json.dumps(ev, default=str)
    return s if len(s) <= cap else s[:cap] + "...<trunc>"

out["gate_raw_tail"] = [shrink(e) for e in gate_raw[-6:]]
out["canary_raw_tail"] = [shrink(e) for e in canary_raw[-6:]]

# 3. Suite-name harvesting: look for any list/dict of suite results anywhere
#    in the record, and count how often each suite is reported failing.
fail_counter = collections.Counter()
pass_counter = collections.Counter()


def walk(node, depth=0):
    if depth > 6:
        return
    if isinstance(node, dict):
        name = node.get("suite") or node.get("name") or node.get("suite_name")
        if name is not None and ("passed" in node or "ok" in node or "status" in node):
            ok = node.get("passed", node.get("ok", node.get("status")))
            ok = ok if isinstance(ok, bool) else str(ok).lower() in ("true", "pass", "passed", "ok")
            (pass_counter if ok else fail_counter)[str(name)[:80]] += 1
        for v in node.values():
            walk(v, depth + 1)
    elif isinstance(node, list):
        for v in node[:64]:
            walk(v, depth + 1)


for ev in gate_raw[-60:] + canary_raw[-60:]:
    walk(ev)
out["suite_fail_counts"] = fail_counter.most_common(20)
out["suite_pass_counts"] = pass_counter.most_common(20)

# 4. Gate artifacts on disk carry the per-suite verdict when the ledger does not.
art_dirs = []
for pat in ("gate-*", "*gate*", "evidence"):
    for p in R.glob(pat):
        try:
            art_dirs.append({"path": str(p), "is_dir": p.is_dir(),
                             "age_s": round(out["now"] - p.stat().st_mtime, 1)})
        except OSError:
            continue
art_dirs.sort(key=lambda d: d["age_s"])
out["gate_artifact_paths"] = art_dirs[:10]

newest = next((d for d in art_dirs if d["is_dir"]), None)
if newest:
    p = pathlib.Path(newest["path"])
    kids = sorted(p.rglob("*"), key=lambda q: -q.stat().st_mtime if q.exists() else 0)[:14]
    out["gate_artifact_children"] = [
        {"f": str(q.relative_to(p)), "age_s": round(out["now"] - q.stat().st_mtime, 1),
         "size": q.stat().st_size if q.is_file() else None}
        for q in kids if q.exists()
    ]
    for q in kids:
        if q.is_file() and q.suffix in (".json", ".jsonl") and q.stat().st_size < 200000:
            out.setdefault("gate_artifact_sample", {})[str(q.relative_to(p))] = \
                q.read_text(encoding="utf-8", errors="replace")[:3000]
            break

# 5. Is the block advancing? Three samples, real deltas, not one reading.
prog = R / "go-systems.progress.json"
samples = []
for _ in range(3):
    snap = {"t": round(time.time(), 1)}
    try:
        snap["progress"] = json.loads(prog.read_text())
        snap["progress_age_s"] = round(time.time() - prog.stat().st_mtime, 1)
    except Exception as exc:  # noqa: BLE001
        snap["err"] = str(exc)
    try:
        st = json.loads((R / "curriculum-supervisor.status.json").read_text())
        snap["status_row"] = st.get("durable_next_row")
        snap["status_state"] = st.get("state")
        snap["status_age_s"] = round(
            time.time() - (R / "curriculum-supervisor.status.json").stat().st_mtime, 1)
    except Exception:  # noqa: BLE001
        pass
    samples.append(snap)
    time.sleep(30)
out["samples"] = samples

print("PROBE_JSON " + json.dumps(out, default=str))
PY
