"""Build a read-only dry run of the candidate supervisor against the live host.

Every previous version of a classifier in this repo was verified against a
hand-built payload and then behaved differently on the host: `block_target_row`
passed 47 tests and is a key no replay ever emits; the `quarantine_ready` arm
passed 67 tests against a `service_stage` the test helper never set. The only
way to know what a decision function does here is to run it against this host's
own ledger, so this writes the candidate to /tmp, imports it from there, and
asks it what it would decide. Nothing under /srv/wizard/project is touched and
no unit is started or stopped.

Kept as a file rather than an inline heredoc because the payload itself
contains heredoc terminators, and nesting them silently truncates the script --
which produced a probe that reported `FileNotFoundError` for a file it had
never finished writing.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "programming_curriculum_supervisor.py"
TARGET = ROOT / "scripts" / "aws" / "_census_dry_run.sh"

raw = SOURCE.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
payload = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")

# Wrapped so the remote heredoc body has no line long enough to trip anything
# in the SSM transport, which has already rewritten a payload once.
wrapped = "\n".join(payload[index:index + 76]
                    for index in range(0, len(payload), 76))

script = f"""set -uo pipefail
WANT={digest}
mkdir -p /tmp/wizdry
cat >/tmp/wizdry/sup.b64 <<'B64_PAYLOAD_END'
{wrapped}
B64_PAYLOAD_END

python3 -c "
import base64, gzip, hashlib, pathlib
raw = gzip.decompress(base64.b64decode(
    pathlib.Path('/tmp/wizdry/sup.b64').read_text()))
pathlib.Path('/tmp/wizdry/candidate_supervisor.py').write_bytes(raw)
print('WROTE', hashlib.sha256(raw).hexdigest(), len(raw))
"
GOT=$(python3 -c "
import hashlib, pathlib
print(hashlib.sha256(
    pathlib.Path('/tmp/wizdry/candidate_supervisor.py').read_bytes()).hexdigest())
")
if [ "$WANT" != "$GOT" ]; then
  echo "PROBE_JSON {{\\"error\\": \\"payload digest mismatch\\", \\"want\\": \\"$WANT\\", \\"got\\": \\"$GOT\\"}}"
  exit 0
fi

python3 /tmp/wizdry/_census_dry_run_body.py
"""

body = r'''
"""Ask the candidate supervisor what it would decide, against live state."""
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

# The supervisor imports siblings (`programming_integrated_retention`, ...), so
# loading it from /tmp by path still needs the project on `sys.path`. Those
# siblings are unchanged by this candidate, so resolving them from the deployed
# tree is correct -- only the file under test comes from /tmp, and it is loaded
# under its own module name so nothing shadows it.
sys.path.insert(0, "/srv/wizard/project")
sys.path.insert(0, "/srv/wizard/project/scripts")

spec = importlib.util.spec_from_file_location(
    "candidate_supervisor", "/tmp/wizdry/candidate_supervisor.py")
sup = importlib.util.module_from_spec(spec)
sys.modules["candidate_supervisor"] = sup
try:
    spec.loader.exec_module(sup)
    out["import"] = "ok"
except Exception as exc:  # noqa: BLE001
    out["import"] = "%s: %s" % (type(exc).__name__, exc)
    print("PROBE_JSON " + json.dumps(out))
    raise SystemExit(0)

out["burn"] = sup.measure_disk_burn_gb_per_hour(R)
out["rollback_reclaim_bytes"] = sup.rollback_reclaim_bytes(R)
out["phase_rates"] = sup.measure_phase_rows_per_hour(R)
out["stall_observations"] = sup.replay_stall_observations(R)

pending = sup.unresolved_deferred_intervals(R)
stalls = sup.replay_stall_counts(R)
ordered = sup.order_replay_candidates(pending, stalls)
census = sup.replay_window_census(R, ordered, 150.0)
out["pending"] = len(pending)
out["stall_counts"] = stalls
out["census"] = {
    "window_hours": census["window_hours"],
    "free_gb": round(census["free_bytes"] / 2 ** 30, 2),
    "window_gb": round(census["window_bytes"] / 2 ** 30, 2),
    "fits": census["fits"],
    "unknown": census["unknown"],
    "exceeds": census["exceeds"],
    "hopeless": sup.replay_queue_is_hopeless(census),
    "head": census["intervals"][:6],
}

# What the FIRST selection will see is not what the census sees now: a restart
# runs `recover_interrupted_deferred_replay` first, which records the halted
# generation's own rate into the ledger. Model that here.
try:
    marker = json.loads((R / "deferred-replay-active.json").read_text())
except Exception as exc:  # noqa: BLE001
    marker = {}
    out["marker_error"] = str(exc)
event = marker.get("interval") or {}
if event:
    interval_id = str(marker.get("interval_id") or "")
    digest = hashlib.sha256(interval_id.encode("utf-8")).hexdigest()[:16]
    start_row = int(event["start_row"])
    durable = sup.replay_pass_durable_row(
        R / ("deferred-replay-%s.progress.json" % digest),
        start_row, int(event["end_row"]))
    created = marker.get("created_unix")
    rows = max(0, durable - start_row)
    progress = R / ("deferred-replay-%s.progress.json" % digest)
    # Mirror the production clock exactly: training ended at the progress
    # file's last write, not at this probe's `time.time()`. Measuring to now
    # would fold the 107.7 h outage into the rate and understate it ~40x.
    try:
        ended = progress.stat().st_mtime
    except OSError:
        ended = time.time()
    hours = ((ended - float(created)) / 3600.0) if created else None
    out["recovery_would_record"] = {
        "interval_id": interval_id,
        "marker_state": marker.get("state"),
        "rows_trained": rows,
        "hours": round(hours, 2) if hours else None,
        "rows_per_hour": round(rows / hours, 1) if hours and hours > 0 else None,
        "clock_ends_at": "progress_mtime",
    }
    out["naive_now_clock_would_have_said"] = {
        "hours": round((time.time() - float(created)) / 3600.0, 2),
        "rows_per_hour": round(
            rows / ((time.time() - float(created)) / 3600.0), 1),
    }
    # Model the census the FIRST selection will actually see: the recovery
    # rollback has run by then, so free space is post-rollback and this
    # interval carries one own-stall observation.
    if hours and hours > 0:
        reclaim = out.get("rollback_reclaim_bytes")
        out["projected_first_selection"] = {
            "own_stall_rows_per_hour": round(rows / hours, 1),
            "span_from_scratch": int(event["end_row"]) - start_row,
            "eta_hours": round(
                (int(event["end_row"]) - start_row) / (rows / hours), 2),
            "note": ("window after the recovery rollback is "
                     "(free + reclaim - floor) / burn; reclaim is unmeasured "
                     "until that rollback publishes it"),
        }

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
'''

body_payload = base64.b64encode(body.encode("utf-8")).decode("ascii")
body_wrapped = "\n".join(body_payload[index:index + 76]
                         for index in range(0, len(body_payload), 76))
script = script.replace(
    "python3 /tmp/wizdry/_census_dry_run_body.py",
    "cat >/tmp/wizdry/body.b64 <<'BODY_PAYLOAD_END'\n"
    + body_wrapped
    + "\nBODY_PAYLOAD_END\n"
    "python3 -c \"\n"
    "import base64, pathlib\n"
    "pathlib.Path('/tmp/wizdry/body.py').write_bytes(base64.b64decode(\n"
    "    pathlib.Path('/tmp/wizdry/body.b64').read_text()))\n"
    "\"\n"
    "python3 /tmp/wizdry/body.py",
)

TARGET.write_text(script, encoding="utf-8", newline="\n")
print("wrote", TARGET, len(script), "bytes; supervisor sha256", digest)
