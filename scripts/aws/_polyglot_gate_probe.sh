python3 - <<'PY'
"""Read why `polyglot` -- and only `polyglot` -- rejects every interval.

Measured 2026-09-09: the enterprise gate scores 11/12 with `tick_delta` 0 and
`structure_unchanged` true, ten consecutive confirmations all 11-then-11, so
this is not the transient dip `enterprise_gate_confirmed` was written to
absorb. It is one deterministic suite, and it has re-deferred every replayed
interval for three days -- exactly the 96 h admission drought.

The four suites that used to fail consistently (platform, cross_project,
composition, semantic_stress) have not failed since 2026-09-06, so the
question this probe answers is whether polyglot fails on capability or was
traded away by whatever repaired those four. `capstone_subject_guard_tradeoff`
is the precedent: four tightenings each traded one suite for another.

`programming_polyglot_composition` writes per-component detail and
`intent_diagnostics` that the gate's own summary line throws away, and the
answer branch must be READ rather than inferred (RECALL_PATH_FIELD_GUIDE).
"""
import json, os, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

path = os.path.join(R, "polyglot.json")
out["report_age_h"] = (round((time.time() - os.path.getmtime(path)) / 3600, 2)
                       if os.path.exists(path) else None)
try:
    report = json.load(open(path, encoding="utf-8"))
except Exception as error:
    out["read_error"] = str(error)[:200]
    report = {}

out["summary"] = report.get("summary")

rows = []
for row in report.get("results") or []:
    diagnostics = row.get("intent_diagnostics") or {}
    rows.append({
        "name": row.get("name"),
        "kind": row.get("kind"),
        "executes": row.get("executes"),
        "files": row.get("files"),
        # Which arm of the if/else chain actually ran. Inferring this from the
        # reply has been wrong before.
        "answer_branch": diagnostics.get("answer_branch"),
        "route": diagnostics.get("route"),
        "score": diagnostics.get("score") or diagnostics.get("best_score"),
        "components": [{
            "component": item.get("component"),
            "executes": item.get("executes"),
            "exact": item.get("exact"),
            "detail": (item.get("detail") or "")[:300],
        } for item in row.get("components") or []],
    })
out["results"] = rows
out["oov"] = [{"honest": item.get("honest"),
               "reply_len": len(str(item.get("reply") or "")),
               "reply_head": str(item.get("reply") or "")[:160]}
              for item in report.get("oov") or []]

# Did something land right before polyglot started failing on 2026-09-07?
stamps = {}
for name in ("polyglot.json", "composition.json", "semantic-stress.json",
             "cross-project.json", "platform.json"):
    p = os.path.join(R, name)
    if os.path.exists(p):
        stamps[name] = round((time.time() - os.path.getmtime(p)) / 3600, 2)
out["report_ages_h"] = stamps

for label, p in (
        ("brain_binary", "/srv/wizard/project/target/release/w1z4rd_brain_server"),
        ("supervisor_py", "/srv/wizard/project/scripts/programming_curriculum_supervisor.py"),
        ("polyglot_py", "/srv/wizard/project/scripts/programming_polyglot_composition.py"),
        ("native_eval_py", "/srv/wizard/project/scripts/programming_native_enterprise_eval.py"),
        ("project_eval_py", "/srv/wizard/project/scripts/programming_project_eval.py"),
):
    try:
        out[f"{label}_age_h"] = round(
            (time.time() - os.path.getmtime(p)) / 3600, 2)
    except OSError as error:
        out[f"{label}_age_h"] = str(error)[:80]

print("PROBEJSON " + json.dumps(out))
PY
