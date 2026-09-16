"""Build the deploy for the self-reclaiming disk halt and the window census.

Deploys two files and restarts, in this order:

  1. `scripts/programming_curriculum_supervisor.py` -- the halt now rolls back
     (the only reclaim that returns bytes on this volume: 414.02 GB measured)
     before it may call itself unrecoverable, and the replay queue refuses to
     spend a window on an interval measured not to fit inside one.
  2. the systemd unit -- exit 91 joins `RestartPreventExitStatus`, because
     `Restart=on-failure` with `RestartSec=10` would respawn a refusal every
     ten seconds forever.

Then it starts the unit, which is currently `failed` with `ExecMainStatus=90`
and has been since 2026-09-11T11:44:01Z. Starting it runs
`recover_interrupted_deferred_replay` against a marker still in
`state: "training"`, which rolls `brain.wbrain` back onto its guard and returns
the volume from 151.16 GB free to an expected ~565 GB.

Verification is by `df` either side, by the unit's own state, and by the stall
record the recovery must write -- never by "the command returned 0". A deploy
that copies a file it never loads has already cost this project 850 s for a
Python fix and 96 h for a Rust one, so the digest is compared on the host
before anything is started.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "scripts" / "aws" / "_deploy_halt_recovery.sh"

SUPERVISOR = ROOT / "scripts" / "programming_curriculum_supervisor.py"
UNIT = ROOT / "scripts" / "aws" / "wizard-curriculum-supervisor.service"


def payload(path: pathlib.Path) -> tuple[str, str]:
    raw = path.read_bytes()
    blob = base64.b64encode(gzip.compress(raw, 9)).decode("ascii")
    wrapped = "\n".join(blob[i:i + 76] for i in range(0, len(blob), 76))
    return hashlib.sha256(raw).hexdigest(), wrapped


sup_digest, sup_blob = payload(SUPERVISOR)
unit_digest, unit_blob = payload(UNIT)

script = f"""set -uo pipefail
PROJ=/srv/wizard/project
DST="$PROJ/scripts/programming_curriculum_supervisor.py"
RUNTIME=/srv/wizard/runtime/programming-integrated-20260713
UNIT_NAME=wizard-curriculum-supervisor.service
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WANT_SUP={sup_digest}
WANT_UNIT={unit_digest}

mkdir -p /tmp/wizdeploy
cat >/tmp/wizdeploy/sup.b64 <<'SUP_PAYLOAD_END'
{sup_blob}
SUP_PAYLOAD_END
cat >/tmp/wizdeploy/unit.b64 <<'UNIT_PAYLOAD_END'
{unit_blob}
UNIT_PAYLOAD_END

python3 -c "
import base64, gzip, pathlib
for name in ('sup', 'unit'):
    raw = gzip.decompress(base64.b64decode(
        pathlib.Path('/tmp/wizdeploy/%s.b64' % name).read_text()))
    pathlib.Path('/tmp/wizdeploy/%s.out' % name).write_bytes(raw)
    print('decoded', name, len(raw))
"

GOT_SUP=$(sha256sum /tmp/wizdeploy/sup.out | cut -d' ' -f1)
GOT_UNIT=$(sha256sum /tmp/wizdeploy/unit.out | cut -d' ' -f1)
echo "DIGEST sup want=$WANT_SUP got=$GOT_SUP"
echo "DIGEST unit want=$WANT_UNIT got=$GOT_UNIT"
if [ "$WANT_SUP" != "$GOT_SUP" ] || [ "$WANT_UNIT" != "$GOT_UNIT" ]; then
  echo "RESULT_JSON {{\\"error\\": \\"payload digest mismatch in transport\\"}}"
  exit 0
fi

# The volume BEFORE anything runs. The reclaim is measured as a df delta and
# never as a sum of file sizes: on this reflink volume those differ by orders
# of magnitude, and a summed reclaim reports a success it did not achieve.
FREE_BEFORE=$(python3 -c "import shutil;print(shutil.disk_usage('$RUNTIME').free)")

cp -a "$DST" "$DST.bak-$STAMP"
install -o ec2-user -g ec2-user -m 644 /tmp/wizdeploy/sup.out "$DST"

FRAGMENT=$(systemctl show "$UNIT_NAME" -p FragmentPath --value)
if [ -n "$FRAGMENT" ] && [ -f "$FRAGMENT" ]; then
  sudo cp -a "$FRAGMENT" "$FRAGMENT.bak-$STAMP"
  sudo install -o root -g root -m 644 /tmp/wizdeploy/unit.out "$FRAGMENT"
  sudo systemctl daemon-reload
fi

# Prove the file that will RUN is the file that was sent. A deploy verified by
# "the copy returned 0" is how a fix sat on disk unloaded for 96 h.
ON_DISK=$(sha256sum "$DST" | cut -d' ' -f1)
OWNER=$(stat -c '%U:%G' "$DST")
PREVENT=$(systemctl show "$UNIT_NAME" -p RestartPreventExitStatus --value)

# `failed` units refuse to start until reset.
sudo systemctl reset-failed "$UNIT_NAME" 2>/dev/null || true
sudo systemctl start "$UNIT_NAME"

# The recovery rollback stops and restarts the brain node, so give it room --
# but report what is true at each check rather than waiting for a fixed time
# and assuming.
sleep 90
STATE_1=$(systemctl show "$UNIT_NAME" -p ActiveState -p SubState -p ExecMainStatus --no-pager | tr '\\n' ' ')
FREE_1=$(python3 -c "import shutil;print(shutil.disk_usage('$RUNTIME').free)")
sleep 180
STATE_2=$(systemctl show "$UNIT_NAME" -p ActiveState -p SubState -p ExecMainStatus --no-pager | tr '\\n' ' ')
FREE_2=$(python3 -c "import shutil;print(shutil.disk_usage('$RUNTIME').free)")

python3 - <<'REPORT_END'
import json, pathlib, shutil, subprocess, time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {{"now": time.time()}}


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "").strip()[-3000:]
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)


out["unit"] = sh("systemctl show wizard-curriculum-supervisor.service "
                 "-p ActiveState -p SubState -p Result -p ExecMainStatus "
                 "-p NRestarts -p RestartPreventExitStatus --no-pager")
out["journal"] = sh("journalctl -u wizard-curriculum-supervisor.service "
                    "-n 25 --no-pager -o short-iso")
out["free_gb"] = round(shutil.disk_usage(R).free / 2 ** 30, 2)
try:
    out["status"] = json.loads(
        (R / "curriculum-supervisor.status.json").read_text())
except Exception as exc:  # noqa: BLE001
    out["status"] = {{"error": str(exc)}}
try:
    out["active_marker"] = json.loads(
        (R / "deferred-replay-active.json").read_text()).get("state")
except Exception:  # noqa: BLE001
    out["active_marker"] = None

# The events that prove the new code ran, rather than that a process started.
recent = []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            recent.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)
interesting = {{
    "deferred_replay_interrupted_before_gate",
    "deferred_replay_rollback_reclaim",
    "disk_floor_rollback_reclaim",
    "no_interval_fits_disk_window",
    "deferred_replay_admitted",
}}
out["new_events"] = [
    {{k: v for k, v in event.items()
      if k in ("kind", "interval_id", "phase", "rows_trained", "hours",
               "rows_per_hour", "reclaimed_bytes", "cleared_floor",
               "window_hours", "updated_unix")}}
    for event in recent[-40:] if event.get("kind") in interesting
]
out["census"] = sh("ps -o pid,etimes,rss,args -C python3 --no-headers | "
                   "grep -c curriculum_supervisor || true")
print("RESULT_JSON " + json.dumps(out, sort_keys=True, default=str))
REPORT_END

echo "DEPLOY_JSON {{\\"on_disk_sha\\": \\"$ON_DISK\\", \\"owner\\": \\"$OWNER\\", \\"prevent\\": \\"$PREVENT\\", \\"free_before\\": $FREE_BEFORE, \\"free_1\\": $FREE_1, \\"free_2\\": $FREE_2, \\"state_1\\": \\"$STATE_1\\", \\"state_2\\": \\"$STATE_2\\"}}"
"""

TARGET.write_text(script, encoding="utf-8", newline="\n")
print("wrote", TARGET, len(script), "bytes")
print("supervisor sha256", sup_digest)
print("unit sha256      ", unit_digest)
