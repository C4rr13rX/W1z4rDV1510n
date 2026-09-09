python3 - <<'PY'
"""Was `dedup.go` ever a candidate for the canonical prompt, or never recalled?

The repair depends entirely on this. `merge_grounded_file_manifests` selects
from `feature_candidates`; if the Go deduplication component is in that pool
and lost the selection, the defect is in selection and a behaviour-coverage
check fixes it. If it was never recalled, selection is innocent and the
repair belongs in recall or curriculum -- an entirely different fix.

`intent_diagnostics.component_recall` records, per LANGUAGE+BEHAVIOUR subset,
the manifest that subset actually retrieved, and `component_routes` records
the ranked alternatives. Read them rather than inferring from the reply: the
answer branch is an if/else chain and reasoning from the output has been
wrong here twice (RECALL_PATH_FIELD_GUIDE).
"""
import json, os, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

report = json.load(open(os.path.join(R, "polyglot.json"), encoding="utf-8"))
rows = []
for row in report.get("results") or []:
    if row.get("name") != "javascript_go_order_workers":
        continue
    diagnostics = row.get("intent_diagnostics") or {}
    entry = {
        "kind": row.get("kind"),
        "executes": row.get("executes"),
        "files": row.get("files"),
        "answer_branch": diagnostics.get("answer_branch"),
        "diagnostic_keys": sorted(diagnostics.keys()),
        "intent_labels": diagnostics.get("intent_labels")
                         or diagnostics.get("labels"),
        "component_recall": diagnostics.get("component_recall"),
        "manifest_composition_ready": diagnostics.get(
            "manifest_composition_ready"),
        "feature_candidate_count": diagnostics.get("feature_candidates")
                                   or diagnostics.get("feature_candidate_count"),
    }
    routes = diagnostics.get("component_routes") or []
    entry["component_routes"] = [{
        "labels": item.get("labels"),
        "artifacts": [a.get("files") or a.get("file")
                      for a in (item.get("artifacts") or [])],
    } for item in routes]
    rows.append(entry)

out["rows"] = rows
print("PROBEJSON " + json.dumps(out))
PY
