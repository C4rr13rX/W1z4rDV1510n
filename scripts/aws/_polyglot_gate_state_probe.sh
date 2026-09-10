#!/bin/bash
# `polyglot` is the one suite still failing the 12-suite enterprise gate, and
# the go-systems corpus now training forward is the repair aimed at it. Two
# questions this answers, neither of which the counts in the health ledger can:
#
#   1. WHICH polyglot case fails, and on what -- `results[].name` carries only
#      a boolean, so the per-case detail has to come from the suite artifact.
#   2. Is the block still converging, measured over a window long enough to
#      contain a batch commit rather than a fixed six seconds.
set -uo pipefail

python3 - <<'PY'
import glob
import json
import pathlib
import subprocess
import time

runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        return (r.stdout or "").strip()
    except Exception as exc:
        return f"ERR {exc}"


def load(path):
    try:
        return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


# --- 1. the polyglot verdict, per case ------------------------------------
gates = sorted(glob.glob(str(runtime / "*.enterprise-gate.json")),
               key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)
detail = []
for path in gates[:2]:
    body = load(path)
    for result in (body.get("results") or []):
        if result.get("name") != "polyglot":
            continue
        detail.append({
            "file": pathlib.Path(path).name,
            "age_hours": round(
                (time.time() - pathlib.Path(path).stat().st_mtime) / 3600, 1),
            "passed": result.get("passed"),
            # Whatever the suite actually recorded, verbatim -- the key names
            # have moved before and guessing one publishes an empty string.
            "keys": sorted(result),
            "result": {k: str(v)[:400] for k, v in result.items()},
        })
out["polyglot_results"] = detail

# The suite's own artifact, if it writes one beside the gate summary.
suite = sorted(glob.glob(str(runtime / "*polyglot*")),
               key=lambda p: pathlib.Path(p).stat().st_mtime, reverse=True)
out["polyglot_artifacts"] = [
    {"name": pathlib.Path(p).name,
     "age_hours": round((time.time() - pathlib.Path(p).stat().st_mtime) / 3600, 1),
     "bytes": pathlib.Path(p).stat().st_size}
    for p in suite[:8]]
for path in suite[:2]:
    if pathlib.Path(path).suffix == ".json":
        out.setdefault("polyglot_artifact_bodies", {})[
            pathlib.Path(path).name] = str(load(path))[:2500]

# --- 2. is the forward block converging -----------------------------------
def freshest():
    best, chosen = None, None
    for path in glob.glob(str(runtime / "*.progress.json")) + [
            str(runtime / "curriculum-supervisor.status.json")]:
        p = pathlib.Path(path)
        try:
            age = time.time() - p.stat().st_mtime
        except Exception:
            continue
        if best is None or age < best:
            best, chosen = age, p
    return chosen, best


path, age = freshest()
first = load(path) if path else {}
t0 = time.time()
row0 = first.get("durable_next_row")
acc0 = first.get("accepted_episodes")
row1, acc1 = row0, acc0
while time.time() - t0 < 150:
    time.sleep(3)
    body = load(path) if path else {}
    row1, acc1 = body.get("durable_next_row"), body.get("accepted_episodes")
    if row1 is not None and row1 != row0:
        break
elapsed = max(1e-6, time.time() - t0)
out["heartbeat"] = {
    "file": path.name if path else None,
    "age_seconds": round(age or -1, 1),
    "row_from": row0, "row_to": row1,
    "accepted_from": acc0, "accepted_to": acc1,
    "sample_seconds": round(elapsed, 1),
    "rows_per_second": (round((row1 - row0) / elapsed, 3)
                        if isinstance(row0, int) and isinstance(row1, int)
                        else None),
}
out["status"] = load(runtime / "curriculum-supervisor.status.json")
out["status_age_seconds"] = round(
    time.time() - (runtime / "curriculum-supervisor.status.json").stat().st_mtime, 1)
out["stage"] = sh("cat", str(runtime / "curriculum-service-supervisor.stage"))
out["memory"] = sh("free", "-g")

print("PROBE_JSON " + json.dumps(out))
PY
