python3 - <<'PY'
"""Is the polyglot 6/6 stable, or did one sample land on a lucky tick?

The defect it replaces was reproducible: twelve consecutive gate runs over
57.6 h all emitted ledger.go for the canonical row. A single 6/6 against a
brain that is training underneath is weaker evidence than that, and CLAUDE.md
requires repeated sampling for exactly this reason -- the enterprise gate
itself runs the suite twice (first + confirm) and rejects on either.

Three runs spaced across ~9 minutes, with the trained row sampled alongside so
the brain is provably still ingesting between samples rather than idle.
"""
import json
import os
import subprocess
import time

PROJ = "/srv/wizard/project"
R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time(), "runs": []}

env = dict(os.environ)
env["PYTHONPATH"] = f"{PROJ}/scripts:" + env.get("PYTHONPATH", "")


def row_now():
    for name in ("go-systems.progress.json", "curriculum-supervisor.status.json"):
        try:
            with open(os.path.join(R, name), encoding="utf-8") as fh:
                data = json.load(fh)
            return {"file": name, "row": data.get("durable_next_row"),
                    "state": data.get("state"),
                    "age_s": round(time.time() - os.path.getmtime(os.path.join(R, name)), 1)}
        except Exception:  # noqa: BLE001
            continue
    return None


for attempt in range(3):
    if attempt:
        time.sleep(200)
    started = time.time()
    entry = {"attempt": attempt, "row_before": row_now()}
    try:
        proc = subprocess.run(
            ["python3", "scripts/programming_polyglot_composition.py",
             "--endpoint", "http://127.0.0.1:18095",
             "--output", f"/tmp/polyglot_stability_{attempt}.json"],
            cwd=PROJ, capture_output=True, text=True, timeout=600, env=env)
        entry["exit_code"] = proc.returncode
        entry["summary"] = (proc.stdout or "").strip()[:400]
        entry["stderr_tail"] = (proc.stderr or "").strip()[-400:]
    except Exception as exc:  # noqa: BLE001
        entry["error"] = str(exc)
    entry["elapsed_s"] = round(time.time() - started, 1)
    try:
        with open(f"/tmp/polyglot_stability_{attempt}.json", encoding="utf-8") as fh:
            rep = json.load(fh)
        target = next(
            (r for r in rep.get("results", [])
             if r.get("name") == "javascript_go_order_workers"
             and r.get("kind") == "canonical"), None)
        entry["canonical_files"] = (target or {}).get("files")
        entry["canonical_executes"] = (target or {}).get("executes")
        entry["failing_rows"] = [
            {"name": r.get("name"), "kind": r.get("kind"), "files": r.get("files")}
            for r in rep.get("results", []) if not r.get("executes")
        ]
    except Exception as exc:  # noqa: BLE001
        entry["report_error"] = str(exc)
    entry["row_after"] = row_now()
    out["runs"].append(entry)

out["all_clean"] = all(r.get("exit_code") == 0 for r in out["runs"])
out["canonical_always_dedup"] = all(
    "dedup.go" in (r.get("canonical_files") or []) for r in out["runs"])
print("PROBE_JSON " + json.dumps(out, default=str))
PY
