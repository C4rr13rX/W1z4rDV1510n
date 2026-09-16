"""Build a read-only dry run of the CANDIDATE census against the live host.

Every classifier in this repository that was verified only against a
hand-written payload has behaved differently on the host: `block_target_row`
passed 47 tests and is a key no replay ever emits; the `quarantine_ready` arm
passed 67 tests against a `service_stage` the test helper never set. The 35
tests added beside this change are worth exactly as much as their fixtures
resemble the ledger, so this asks the real ledger.

The question: with the barren-stall channel and the decisive-miss rule, what
does the census decide about the 22 unresolved intervals, and does the capacity
block say what the halt is actually asking the operator for?

Writes the candidate to /tmp, imports it from there, and asks it. Nothing under
/srv/wizard/project is touched, no unit is started or stopped, and no ledger is
written -- `replay_window_census` is a pure read.

Kept as a file rather than an inline heredoc because the payload contains
heredoc terminators, and nesting them silently truncates the script.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "programming_curriculum_supervisor.py"
TARGET = ROOT / "scripts" / "aws" / "_capacity_census_dry_run.sh"

raw = SOURCE.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")

wrapped = "\n".join(payload[index:index + 76]
                    for index in range(0, len(payload), 76))

script = f"""set -uo pipefail
WANT={digest}
mkdir -p /tmp/wizdry
cat >/tmp/wizdry/cap.b64 <<'B64_PAYLOAD_END'
{wrapped}
B64_PAYLOAD_END

python3 -c "
import base64, gzip, hashlib, pathlib
raw = gzip.decompress(base64.b64decode(
    pathlib.Path('/tmp/wizdry/cap.b64').read_text()))
pathlib.Path('/tmp/wizdry/capacity_supervisor.py').write_bytes(raw)
print('WROTE', hashlib.sha256(raw).hexdigest(), len(raw))
"
GOT=$(python3 -c "
import hashlib, pathlib
print(hashlib.sha256(
    pathlib.Path('/tmp/wizdry/capacity_supervisor.py').read_bytes()).hexdigest())
")
if [ "$WANT" != "$GOT" ]; then
  echo "PROBE_JSON {{\\"error\\": \\"payload digest mismatch\\", \\"want\\": \\"$WANT\\", \\"got\\": \\"$GOT\\"}}"
  exit 0
fi

python3 /tmp/wizdry/capacity_body.py
"""

body = r'''
"""What would the candidate census decide, against this host's own ledger?"""
import json
import pathlib
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

sys.path.insert(0, "/srv/wizard/project")
sys.path.insert(0, "/srv/wizard/project/scripts")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "capacity_supervisor", "/tmp/wizdry/capacity_supervisor.py")
sup = importlib.util.module_from_spec(spec)
sys.modules["capacity_supervisor"] = sup
try:
    spec.loader.exec_module(sup)
    out["import"] = "ok"
except Exception as exc:  # noqa: BLE001
    out["import"] = "%s: %s" % (type(exc).__name__, exc)
    print("PROBE_JSON " + json.dumps(out))
    raise SystemExit(0)

out["barren_stalls"] = sup.replay_barren_stalls(R)
out["phase_rates"] = sup.measure_phase_rows_per_hour(R)

pending = sup.unresolved_deferred_intervals(R)
ordered = sup.order_replay_candidates(pending, sup.replay_stall_counts(R))
census = sup.replay_window_census(R, ordered, 150.0)

out["pending"] = len(pending)
out["window_hours"] = census["window_hours"]
out["burn_gb_per_hour"] = (census.get("burn") or {}).get("gb_per_hour")
out["free_gb"] = round(census["free_bytes"] / 2 ** 30, 2)
out["capacity"] = census["capacity"]
out["fits"] = census["fits"]
out["unknown_count"] = len(census["unknown"])
out["exceeds_count"] = len(census["exceeds"])
out["hopeless"] = sup.replay_queue_is_hopeless(census)

# Per-phase summary: the queue is 22 intervals over a handful of phases, and
# the phase is what carries the cost per row.
by_phase = {}
for row in census["intervals"]:
    entry = by_phase.setdefault(row["phase"], {
        "count": 0, "verdicts": {}, "eta_hours": row["eta_hours"],
        "rows_per_hour": row["rows_per_hour"],
        "rate_source": row["rate_source"], "rate_samples": row["rate_samples"],
    })
    entry["count"] += 1
    entry["verdicts"][row["verdict"]] = entry["verdicts"].get(
        row["verdict"], 0) + 1
out["by_phase"] = by_phase

# The interval that is training RIGHT NOW, and what the candidate says of it.
try:
    marker = json.loads((R / "deferred-replay-active.json").read_text())
except Exception:  # noqa: BLE001
    marker = {}
running = str(marker.get("interval_id") or "")
out["running_interval"] = running
out["running_marker_has_resume_row"] = "resume_row" in marker
for row in census["intervals"]:
    if row["interval_id"] == running:
        out["running_verdict"] = row
        break

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
'''

body_payload = base64.b64encode(body.encode("utf-8")).decode("ascii")
body_wrapped = "\n".join(body_payload[index:index + 76]
                         for index in range(0, len(body_payload), 76))
script = script.replace(
    "python3 /tmp/wizdry/capacity_body.py",
    "cat >/tmp/wizdry/capbody.b64 <<'BODY_PAYLOAD_END'\n"
    + body_wrapped
    + "\nBODY_PAYLOAD_END\n"
    "python3 -c \"\n"
    "import base64, pathlib\n"
    "pathlib.Path('/tmp/wizdry/capacity_body.py').write_bytes("
    "base64.b64decode(\n"
    "    pathlib.Path('/tmp/wizdry/capbody.b64').read_text()))\n"
    "\"\n"
    "python3 /tmp/wizdry/capacity_body.py",
)

TARGET.write_text(script, encoding="utf-8", newline="\n")
print("wrote", TARGET, len(script), "bytes; supervisor sha256", digest)
