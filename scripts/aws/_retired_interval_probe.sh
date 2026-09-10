python3 - <<'PY'
"""Do the 70 retired quarantine intervals come back, or are they lost work?

`--auto-quarantine-recovery` is on and the ledger carries 157
`automatic_quarantine_recovery` events, so ordinary quarantine is retested
without help. `unrestorable_quarantine_retired` fired 70 times, and "retired"
does not read like something a retest revisits.

That distinction decides whether 2.78M deferred rows are waiting or gone, so
this reads the retirement records themselves rather than inferring from the
name. It does NOT guess where interval state lives: an earlier pass invented
three metadata filenames, found none of them in 135 directories, and reported
a state histogram that was entirely an artefact of the guess. Here the ledger
is the source, and the runtime directory is listed rather than probed by name.
"""
import collections
import json
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

# What files actually exist at the top level, so the next probe need not guess.
try:
    out["runtime_listing"] = sorted(
        p.name for p in R.iterdir()
        if p.is_file() and not p.name.endswith((".log", ".wbrain")))[:60]
except Exception as exc:  # noqa: BLE001
    out["runtime_listing"] = [f"ERR {exc}"]

kinds = collections.Counter()
retired, recovered, invalidated = [], [], []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            kind = str(ev.get("kind") or "")
            kinds[kind] += 1
            if kind not in ("unrestorable_quarantine_retired",
                            "automatic_quarantine_recovery",
                            "false_semantic_quarantine_invalidated",
                            "quarantine_retest_admitted"):
                continue
            record = {
                "h_ago": round((out["now"] - float(ev.get("updated_unix") or 0)) / 3600.0, 2)
                if ev.get("updated_unix") else None,
                "phase": ev.get("phase"),
                "interval_id": ev.get("interval_id"),
                "reason": str(ev.get("reason") or ev.get("note") or "")[:220],
                "keys": sorted(k for k in ev.keys() if k != "kind")[:14],
            }
            if kind == "unrestorable_quarantine_retired":
                retired.append(record)
            elif kind == "automatic_quarantine_recovery":
                recovered.append(record)
            else:
                invalidated.append(record)
except Exception as exc:  # noqa: BLE001
    out["ledger_err"] = str(exc)

out["retired_count"] = len(retired)
out["retired_first"] = retired[:2]
out["retired_last"] = retired[-4:]
out["retired_reasons"] = collections.Counter(
    r["reason"] for r in retired).most_common(8)
out["retired_phases"] = collections.Counter(
    str(r["phase"]) for r in retired).most_common(8)
out["recovery_count"] = len(recovered)
out["recovery_last"] = recovered[-3:]
out["invalidated"] = invalidated[-4:]
out["quarantine_kind_counts"] = [
    (k, n) for k, n in kinds.most_common() if "quarantine" in k or "defer" in k]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
