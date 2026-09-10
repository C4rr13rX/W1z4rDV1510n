python3 - <<'PY'
"""Bucket every gate rejection in the health ledger, then name the binding stage.

The watchdog reported one `last_failure` ("deferred replay worker exited 1").
The named-failure lesson is that acting on that row alone repairs the tail: a
previous `last_failure` turned out to be 6.6% of 288. So this counts the whole
population and splits it by the stage that actually raised.

It also reads the REAL ledger. An earlier probe in this same session read
`curriculum-admissions.jsonl`, which does not exist, and reported zero failures
of every kind -- the vacuous-zero signature: a pattern that can never match
reports 0 forever. `append_health_event` writes `curriculum-health.jsonl`
(supervisor line 1498), so that is the only file with authority here.

Read-only. Nothing is restarted and no eval is invoked: a restart during
`state: training` discards the interval.
"""
import collections
import json
import os
import re
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
LEDGER = os.path.join(R, "curriculum-health.jsonl")
now = time.time()
out = {"now": now, "ledger": LEDGER, "ledger_exists": os.path.exists(LEDGER)}

rows = []
if out["ledger_exists"]:
    out["ledger_bytes"] = os.path.getsize(LEDGER)
    for line in open(LEDGER, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            continue
out["ledger_rows"] = len(rows)

# Prove the pattern CAN be non-zero before trusting any absence below.
kinds = collections.Counter(str(r.get("kind") or r.get("event") or "?") for r in rows)
out["kind_counts"] = dict(kinds.most_common(40))


def when(row):
    for key in ("unix", "updated_unix", "ts", "time"):
        value = row.get(key)
        if isinstance(value, (int, float)) and value > 1e9:
            return float(value)
    return 0.0


# ---- Which stage raised, across every replay failure ----------------------
fails = [r for r in rows if str(r.get("kind") or "") == "deferred_replay_failed"]
out["deferred_replay_failed_total"] = len(fails)

stage_counts = collections.Counter()
reason_counts = collections.Counter()
for row in fails:
    blob = json.dumps(row)
    stage = str(row.get("stage") or row.get("gate_stage") or "")
    reason = str(row.get("reason") or row.get("error") or row.get("detail") or "")
    if not stage:
        found = re.search(r'"stage":\s*"([^"]+)"', blob)
        stage = found.group(1) if found else "unrecorded"
    stage_counts[stage] += 1
    low = reason.lower()
    if "exited" in low or "signal" in low or "sigterm" in low:
        bucket = "worker_exit_or_signal"
    elif "yield" in low or "memory" in low or "free" in low:
        bucket = "resource_yield"
    elif "regression" in low:
        bucket = re.sub(r"[^a-z_ ]", "", low)[:48].strip()
    elif "timeout" in low or "timed out" in low:
        bucket = "timeout"
    elif reason:
        bucket = re.sub(r"[^a-z_ ]", "", low)[:48].strip()
    else:
        bucket = "no_reason_recorded"
    reason_counts[bucket] += 1
out["failure_stage_counts"] = dict(stage_counts.most_common(20))
out["failure_reason_buckets"] = dict(reason_counts.most_common(20))

# ---- Enterprise suite detail: which of the 12 actually fail, and how often
suite_fail = collections.Counter()
suite_seen = collections.Counter()
ent_scores = []
for row in rows:
    kind = str(row.get("kind") or "")
    if kind not in ("enterprise_gate_unconfirmed", "enterprise_gate_confirmation"):
        continue
    ent_scores.append({
        "age_h": round((now - when(row)) / 3600.0, 1),
        "kind": kind,
        "first": row.get("first_passed_suites", row.get("passed_suites")),
        "confirm": row.get("confirm_passed_suites"),
        "total": row.get("total_suites"),
        "passed": row.get("passed"),
    })
out["enterprise_recent"] = ent_scores[-14:]
out["enterprise_events"] = len(ent_scores)

# The gate report itself names the failing suites; read the newest per phase.
reports = []
for name in sorted(os.listdir(R)):
    if not name.endswith(".enterprise-gate.json"):
        continue
    path = os.path.join(R, name)
    try:
        data = json.load(open(path, encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        reports.append({"file": name, "err": str(error)[:80]})
        continue
    suites = data.get("suites") or data.get("results") or []
    failing, passing = [], []
    if isinstance(suites, dict):
        items = suites.items()
    else:
        items = [(s.get("name") or s.get("suite") or "?", s) for s in suites]
    for suite_name, detail in items:
        ok = detail.get("passed") if isinstance(detail, dict) else bool(detail)
        (passing if ok else failing).append(suite_name)
        suite_seen[suite_name] += 1
        if not ok:
            suite_fail[suite_name] += 1
    reports.append({
        "file": name,
        "age_h": round((now - os.path.getmtime(path)) / 3600.0, 1),
        "passed_suites": data.get("passed_suites"),
        "total_suites": data.get("total_suites"),
        "tick_delta": data.get("tick_delta"),
        "structure_unchanged": data.get("structure_unchanged"),
        "failing": failing,
    })
out["enterprise_reports"] = reports[-10:]
out["suite_failure_counts"] = dict(suite_fail.most_common(20))
out["suite_seen_counts"] = dict(suite_seen.most_common(20))

# ---- When did anything last actually resolve? ----------------------------
admit_kinds = ("deferred_replay_admitted", "quarantine_retest_admitted",
               "phase_forward_harvested", "fully_deferred_block_advanced",
               "protected_hebbian_cadence_combined_admission")
admits = [(when(r), str(r.get("kind"))) for r in rows
          if str(r.get("kind") or "") in admit_kinds]
admits.sort()
out["recent_admissions"] = [
    {"age_h": round((now - t) / 3600.0, 1), "kind": k} for t, k in admits[-12:]
]

# ---- The named stderr log, timestamped against the process generation -----
paths = re.findall(r"stderr=(\S+?\.log)", json.dumps(rows[-400:]))
out["stderr_paths_recent"] = list(dict.fromkeys(paths))[-4:]
for path in out["stderr_paths_recent"][-2:]:
    entry = {"path": path, "exists": os.path.exists(path)}
    if entry["exists"]:
        entry["age_h"] = round((now - os.path.getmtime(path)) / 3600.0, 1)
        entry["size"] = os.path.getsize(path)
        with open(path, encoding="utf-8", errors="replace") as handle:
            entry["tail"] = handle.read()[-1800:].splitlines()[-18:]
    out.setdefault("stderr_detail", []).append(entry)

print("PROBE_JSON " + json.dumps(out, default=str))
PY
