python3 - <<'PY'
"""Bucket EVERY deferred_replay_failed since the last admission, on `error`.

Schema confirmed empirically: the record type key is `kind`, and 1180 records
carry `error`. CLAUDE.md: an arm matching a generic substring swallows every
cause (317/324 once), and `last_failure` was 6.6% of its population -- so
bucket on a normalised whole-error signature, and print the arm sizes.

The go-systems enterprise gate passed 12/12 two hours ago, so the blocker is
NOT the enterprise suite. Find which stage says no.

Read-only.
"""
import json, os, time, collections, re

R = "/srv/wizard/runtime/programming-integrated-20260713"
NOW = time.time()
out = {"now": NOW}

recs = []
with open(f"{R}/curriculum-health.jsonl", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    recs.append(r)
            except Exception:
                pass

admits = [(i, r) for i, r in enumerate(recs)
          if "admitted" in str(r.get("kind"))]
out["admit_count"] = len(admits)
out["admits_tail"] = [
    {"i": i, "kind": r.get("kind"), "phase": r.get("phase"),
     "interval_id": r.get("interval_id"),
     "age_h": round((NOW - float(r.get("updated_unix") or NOW)) / 3600.0, 2)}
    for i, r in admits[-5:]]

cut = admits[-1][0] if admits else 0
since = recs[cut + 1:]
out["records_since_last_admit"] = len(since)
out["kind_mix_since"] = collections.Counter(
    str(r.get("kind")) for r in since).most_common(25)


def sig(text):
    t = re.sub(r"0x[0-9a-f]+|\b\d{3,}\b", "N", str(text))
    t = re.sub(r"/[\w./-]+\.(log|json|py|txt)", "<path>", t)
    return " ".join(t.split())[:220]


for kind in ("deferred_replay_failed", "midphase_gate_failed",
             "continuous_canary_failed", "enterprise_gate_unconfirmed",
             "completion_gate_failed", "fully_deferred_failure_absorbed"):
    pop = [r for r in since if str(r.get("kind")) == kind]
    if not pop:
        pop_all = [r for r in recs if str(r.get("kind")) == kind]
        out[kind] = {"since_last_admit": 0, "lifetime": len(pop_all)}
        continue
    b = collections.Counter()
    for r in pop:
        txt = r.get("error")
        if txt in (None, ""):
            txt = r.get("note") or r.get("reason") or r.get("detail") or ""
        b[sig(txt) or "<empty>"] += 1
    out[kind] = {
        "since_last_admit": len(pop),
        "buckets": b.most_common(8),
        "newest": {
            "age_h": round((NOW - float(pop[-1].get("updated_unix") or NOW)) / 3600.0, 2),
            "interval_id": pop[-1].get("interval_id"),
            "phase": pop[-1].get("phase"),
            "keys": sorted(pop[-1].keys()),
            "error_head": str(pop[-1].get("error") or "")[:600],
            "error_tail": str(pop[-1].get("error") or "")[-600:],
        },
    }

print("PROBE_JSON " + json.dumps(out, default=str)[:14000])
PY
