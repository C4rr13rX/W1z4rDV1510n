python3 - <<'PY'
"""Name the suites that are rejecting every replayed interval.

`enterprise_gate_clean` demands 12 of 12, and the supervisor's own docstring
records that four suites fail consistently while four others flicker. The
confirmation re-run only neutralises the flicker, so a consistently failing
suite re-defers every interval forever -- measured here as 312
`deferred_replay_failed` against 20 admitted.

Which suites, and whether they are the SAME suites now as when that docstring
was written, decides the repair: a suite failing on real capability needs
curriculum or an architecture fix, while a suite failing on infrastructure
(exit 75, a timeout, an empty answer from an unhydrated brain) is a harness
fault that must never be scored as a capability gap. `named_failure_vs_
population` -- bucket them all rather than reading the single last failure.
"""
import collections, glob, json, os, re, time

R = "/srv/wizard/runtime/programming-integrated-20260713"
out = {"now": time.time()}

# ---------------------------------------------------------------- current
# The newest gate report carries per-suite `results`, which the truncated
# `last_failure` string in the health ledger cannot.
reports = sorted(glob.glob(os.path.join(R, "*enterprise-gate*.json")),
                 key=os.path.getmtime)
out["gate_reports"] = [os.path.basename(p) for p in reports[-6:]]
if reports:
    newest = reports[-1]
    try:
        report = json.load(open(newest, encoding="utf-8"))
    except Exception as error:
        report = {"read_error": str(error)[:200]}
    out["newest_report"] = os.path.basename(newest)
    out["newest_age_h"] = round((time.time() - os.path.getmtime(newest)) / 3600, 2)
    out["newest"] = {
        key: report.get(key) for key in
        ("passed", "passed_suites", "total_suites", "tick_delta",
         "structure_unchanged", "infrastructure_only_failure",
         "residency_before", "residency_after", "updated_unix")
    }
    suites = []
    for row in report.get("results") or []:
        entry = {
            "name": row.get("name"),
            "passed": row.get("passed"),
            "exit_code": row.get("exit_code"),
            "infra": row.get("infrastructure_failure"),
            "timed_out": row.get("timed_out"),
            "secs": row.get("elapsed_seconds"),
        }
        summary = row.get("summary")
        if isinstance(summary, dict):
            # Keep only the scalar score-shaped keys; the full summaries run to
            # megabytes and SSM truncates stdout.
            entry["summary"] = {
                k: v for k, v in summary.items()
                if isinstance(v, (int, float, bool, str)) and len(str(v)) <= 60
            }
        if not row.get("passed"):
            entry["stderr_tail"] = (row.get("stderr_tail") or "")[-400:]
        suites.append(entry)
    out["newest_suites"] = suites

# ------------------------------------------------------------ history
# Every replay failure embeds the whole enterprise dict, so the per-suite
# verdicts are recoverable from the ledger even for reports since rotated.
suite_fail = collections.Counter()
suite_seen = collections.Counter()
per_day = collections.defaultdict(collections.Counter)
confirmations = []
failures = 0
first_fail_unix = {}
last_fail_unix = {}

try:
    lines = open(R + "/curriculum-health.jsonl", encoding="utf-8").read().splitlines()
except Exception as error:
    lines = []
    out["ledger_error"] = str(error)[:200]

for line in lines:
    if "enterprise" not in line:
        continue
    try:
        event = json.loads(line)
    except Exception:
        continue
    kind = event.get("kind")
    when = event.get("updated_unix") or 0
    if kind == "enterprise_gate_confirmation":
        confirmations.append({
            "ago_h": round((out["now"] - when) / 3600, 2),
            "first": event.get("first_passed_suites"),
            "confirm": event.get("confirm_passed_suites"),
            "total": event.get("total_suites"),
            "passed": event.get("passed"),
        })
        continue
    if kind != "deferred_replay_failed":
        continue
    error = str(event.get("error") or "")
    if "enterprise regression" not in error:
        continue
    failures += 1
    day = time.strftime("%Y-%m-%d", time.gmtime(when))
    # `results` is repr'd into the error, so each suite appears as
    # {'name': 'platform', 'passed': False, ...} in source order.
    for name, verdict in re.findall(
            r"'name':\s*'([a-z_]+)',\s*'passed':\s*(True|False)", error):
        suite_seen[name] += 1
        if verdict == "False":
            suite_fail[name] += 1
            per_day[day][name] += 1
            last_fail_unix[name] = max(last_fail_unix.get(name, 0), when)
            first_fail_unix.setdefault(name, when)

out["enterprise_regression_failures"] = failures
out["suite_failure_rate"] = {
    name: {
        "failed": suite_fail[name],
        "of": suite_seen[name],
        "rate": round(suite_fail[name] / suite_seen[name], 3) if suite_seen[name] else None,
        "last_fail_ago_h": (round((out["now"] - last_fail_unix[name]) / 3600, 1)
                            if name in last_fail_unix else None),
    }
    for name in sorted(suite_seen, key=lambda n: -suite_fail[n])
}
out["per_day_recent"] = {
    day: dict(counter) for day, counter in sorted(per_day.items())[-8:]
}
out["confirmations_recent"] = confirmations[-10:]

print("PROBEJSON " + json.dumps(out))
PY
