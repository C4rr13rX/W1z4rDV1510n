python3 - <<'PY'
"""Which polyglot row fails on the CURRENT brain, and how long has it blocked?

The enterprise gate rejects on `polyglot` alone: 5/6 projects, 11/12 component
executions, 11 of 12 suites otherwise green. One row, one component.

The suite writes a full per-row report next to the gate summary, so the
identity is already on disk and does not need the suite re-run to recover it.
This finds every copy under the host tree and reports the newest, plus the
whole enterprise-gate history so "how long" is measured rather than inferred
from the six records that happened to fit in the last probe's tail.

An earlier local run of this same suite failed with
`file not found: VersionedStore.java` while emitting an unrelated
`migrations.py`, so the per-row `files` list is reported verbatim: a component
that "does not execute" because it was never emitted is a composition fault,
and one that executes wrongly is a content fault. Those need different repairs.
"""
import json
import pathlib
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}


def find(pattern):
    try:
        r = subprocess.run(
            ["find", "/srv/wizard", "-name", pattern, "-type", "f",
             "-not", "-path", "*/target/*", "-printf", "%T@ %p\n"],
            capture_output=True, text=True, timeout=120)
        rows = []
        for line in (r.stdout or "").splitlines():
            ts, _, path = line.partition(" ")
            try:
                rows.append((float(ts), path))
            except ValueError:
                continue
        return sorted(rows, reverse=True)
    except Exception as exc:  # noqa: BLE001
        return [(0.0, f"ERR {exc}")]


hits = find("polyglot_composition.json") + find("polyglot*.json")
seen, uniq = set(), []
for ts, path in sorted(hits, reverse=True):
    if path in seen:
        continue
    seen.add(path)
    uniq.append((ts, path))
out["polyglot_artifacts"] = [
    {"path": p, "age_h": round((out["now"] - t) / 3600.0, 2)} for t, p in uniq[:12]
]

# The newest report, decoded down to the failing row.
for ts, path in uniq[:6]:
    try:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:  # noqa: BLE001
        continue
    if not isinstance(data, dict) or "results" not in data:
        continue
    rows = []
    for row in data.get("results", []):
        rows.append({
            "name": row.get("name"), "kind": row.get("kind"),
            "executes": row.get("executes"),
            "files": row.get("files"),
            "components": [
                {"component": c.get("component"), "executes": c.get("executes"),
                 "exact": c.get("exact"), "detail": str(c.get("detail") or "")[:400]}
                for c in row.get("components", [])
            ],
            "intent": json.dumps(row.get("intent_diagnostics"), default=str)[:600],
        })
    out.setdefault("reports", {})[path] = {
        "age_h": round((out["now"] - ts) / 3600.0, 2),
        "summary": data.get("summary"),
        "rows": rows,
        "oov": data.get("oov"),
    }
    if len(out.get("reports", {})) >= 3:
        break

# Full enterprise-gate history: when did polyglot start blocking?
hist = []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{") or "enterprise_gate" not in line:
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if str(ev.get("kind") or "") != "enterprise_gate_confirmation":
                continue
            hist.append({
                "h_ago": round((out["now"] - float(ev.get("updated_unix") or 0)) / 3600.0, 2),
                "phase": ev.get("phase"), "passed": ev.get("passed"),
                "first": ev.get("first_passed_suites"),
                "confirm": ev.get("confirm_passed_suites"),
                "total": ev.get("total_suites"),
            })
except Exception as exc:  # noqa: BLE001
    out["hist_err"] = str(exc)

out["enterprise_gate_total"] = len(hist)
out["enterprise_gate_ever_passed"] = [h for h in hist if h.get("passed")][:5]
out["enterprise_gate_recent"] = hist[-14:]
out["enterprise_gate_first"] = hist[:3]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
