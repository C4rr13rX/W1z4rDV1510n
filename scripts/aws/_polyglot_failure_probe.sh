python3 - <<'PY'
"""Name the polyglot rows that fail, and separate capability from harness.

`polyglot` is now the sole enterprise-gate blocker: 11 of 12 suites pass, ten
consecutive confirmations agree, and it owns 18 of its 19 lifetime failures in
the last three days. The suite exits 1 rather than 75, so its own fixture-
infrastructure branch has already ruled itself out -- either a component did
not execute AND was not byte-exact, or an OOV row answered when it should have
abstained.

Which of those two it is decides the repair and they point opposite ways: a
component that stops executing is a recall/composition regression, while an
OOV row that answers is an over-eager gate. Read the report rather than infer,
because the suite prints only the summary and the gate keeps only the exit
code (`named_failure_vs_population`).
"""
import glob, json, os, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

reports = glob.glob(os.path.join(R, "**", "polyglot.json"), recursive=True)
reports.sort(key=os.path.getmtime)
out["report_paths"] = [p.replace(R, "") for p in reports[-6:]]
if reports:
    path = reports[-1]
    out["report_age_h"] = round((time.time() - os.path.getmtime(path)) / 3600, 2)
    report = json.load(open(path, encoding="utf-8"))
    out["summary"] = report.get("summary")
    rows = []
    for row in report.get("results") or []:
        diag = row.get("intent_diagnostics") or {}
        rows.append({
            "name": row.get("name"),
            "kind": row.get("kind"),
            "executes": row.get("executes"),
            "files": row.get("files"),
            "answer_branch": diag.get("answer_branch"),
            "components": [
                {"c": c.get("component"), "exec": c.get("executes"),
                 "exact": c.get("exact"), "detail": (c.get("detail") or "")[:300]}
                for c in row.get("components") or []
            ],
        })
    out["rows"] = rows
    out["oov"] = [
        {"prompt": o.get("prompt")[:70], "honest": o.get("honest"),
         "reply": (str(o.get("reply") or ""))[:200]}
        for o in report.get("oov") or []
    ]

# Which of the two exit-1 branches fired, computed the same way the suite does.
if out.get("rows") is not None:
    failed = [c for r in out["rows"] for c in r["components"] if not c["exec"]]
    out["verdict"] = {
        "failed_components": len(failed),
        "all_failed_are_exact": bool(failed) and all(c["exact"] for c in failed),
        "oov_all_honest": all(o["honest"] for o in out.get("oov") or []),
        "rows_not_executing": [f"{r['name']}/{r['kind']}"
                               for r in out["rows"] if not r["executes"]],
    }

print("PROBEJSON " + json.dumps(out))
PY
