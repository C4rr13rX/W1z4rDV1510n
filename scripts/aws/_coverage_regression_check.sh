python3 - <<'PY'
"""Would the new coverage rule have rejected the cases that already pass?

The fix can only choose BETWEEN selections that pass every prior check, and
it falls back to the old rule when nothing satisfies coverage -- so it cannot
make a composable request uncomposable. What it CAN do is pick a different
selection. This replays the rule against the labels and routes the live brain
actually recorded for all six polyglot cells, so the claim rests on the
recorded data rather than on reasoning about it.
"""
import json, os

R = "/srv/wizard/runtime/programming-integrated-20260713"
report = json.load(open(os.path.join(R, "polyglot.json"), encoding="utf-8"))


def behaviour_group(label):
    if label.endswith(":SECURITY:AUTHORIZATION"):
        return "authorization"
    if label.endswith(":API:IDEMPOTENT_COMMAND"):
        return "idempotency"
    if label.endswith(":PERSISTENCE:ATOMIC_TRANSACTION") or \
       label.endswith(":DOMAIN:ATOMIC_LEDGER_TRANSFER"):
        return "transaction"
    if label.endswith(":OBSERVABILITY:CORRELATED_LOGGING") or \
       label.endswith(":ENTERPRISE:SECRET_REDACTION"):
        return "observability"
    if label.endswith(":RESILIENCE:CIRCUIT_BREAKER"):
        return "circuit_breaker"
    if label.endswith(":ENTERPRISE:BOUNDED_RETRY") or \
       label.endswith(":RESILIENCE:ASYNC_RETRY"):
        return "retry"
    if label.endswith(":CONCURRENCY:DEDUPLICATION"):
        return "deduplication"
    if label.endswith(":INTEGRATION:TRANSACTIONAL_OUTBOX"):
        return "outbox"
    if label.endswith(":STATE:OPTIMISTIC_CONCURRENCY"):
        return "optimistic_concurrency"
    if label.endswith(":ENTERPRISE:BATCHING"):
        return "batching"
    return None


out = []
for row in report.get("results") or []:
    diagnostics = row.get("intent_diagnostics") or {}
    labels = diagnostics.get("labels") or []
    requested = sorted({g for g in map(behaviour_group, labels) if g})
    languages = sum(1 for label in labels if ":LANGUAGE:" in label)
    ceiling = min(max(len(requested), languages), 4)
    ceiling = max(ceiling, 2)

    # Provenance exactly as the fix builds it: the RANKED route only.
    served = {}
    for item in diagnostics.get("component_routes") or []:
        groups = [g for g in map(behaviour_group, item.get("labels") or []) if g]
        files = []
        for artifact in item.get("artifacts") or []:
            files.extend(artifact.get("files") or [])
        for group in groups:
            served.setdefault(group, set()).update(files)

    servable = sorted(set(requested) & set(served))
    delivered = set(row.get("files") or [])
    covered = sorted(g for g in servable
                     if served.get(g, set()) & delivered)
    out.append({
        "case": row.get("name"),
        "kind": row.get("kind"),
        "passes_today": row.get("executes"),
        "requested_groups": requested,
        "component_ceiling": ceiling,
        "servable": servable,
        "covered_by_delivered_files": covered,
        "delivered": sorted(delivered),
        "would_still_be_chosen": covered == servable,
    })

print("PROBEJSON " + json.dumps({"cells": out}))
PY
