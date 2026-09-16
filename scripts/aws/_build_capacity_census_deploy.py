"""Deploy the barren-stall channel, the decisive-miss rule, verdict-aware
selection, and the capacity block -- in chunks, because the payload no longer
fits one SSM document.

CHUNKING IS NOT AN OPTIMISATION HERE, IT IS THE ONLY WAY THROUGH.
`SendCommand` rejects a total parameter+document size over 97 KB with
`MaxDocumentSizeExceeded`, and `bootstrap_training_host.aws` runs the CLI with
`check=True` beside `capture_output=True` -- the evidence-deleting pair this
repository has already been bitten by -- so the failure surfaced only as
"returned non-zero exit status 254" with the reason discarded. The supervisor
is 280 KB of source; gzip+base64 is ~93 KB, and it crossed the limit the moment
this change added a few hundred lines. It will keep growing, so the transport
splits the payload rather than trimming the fix to fit.

Each part appends its slice of the base64 blob to /tmp; the installer decodes,
compares the sha256 against the digest computed here, and only then installs.
A deploy verified by "the copy returned 0" is how a fix sat on disk unloaded
for 96 h, so the digest is checked on the host before anything is restarted,
and the main PID is compared either side to prove the new file was LOADED.

What the restart does, in order:

  1. `recover_interrupted_deferred_replay` records the interrupted interval's
     stall -- with a rate, now that a barren generation records its hours.
  2. It rolls `brain.wbrain` back onto its guard, returning the blocks the
     interval appended. Measured by `df` either side, never by summing file
     sizes: on this reflink volume those differ by orders of magnitude.
  3. The census measures every pending interval, and selection now ORDERS ON
     THAT VERDICT. Before this, `replay_queue_is_hopeless` collapsed the
     verdicts to one boolean over the whole queue and selection took
     `pending[0]` from a sort that had never heard of them -- so one `unknown`
     anywhere kept the queue eligible while the head was measured to fail.
     Verified live: 21 of 22 `exceeds`, training a measured 18x miss.

If every interval exceeds, `run_deferred_replays` returns 91 and the unit must
not respawn it; `RestartPreventExitStatus=42 90 91` is deployed with it.

A refusal is not a retirement: every interval stays `deferred` and eligible,
and the same census passes the moment the burn falls or the volume grows.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUPERVISOR = ROOT / "scripts" / "programming_curriculum_supervisor.py"
UNIT = ROOT / "scripts" / "aws" / "wizard-curriculum-supervisor.service"
OUT = ROOT / "scripts" / "aws"

#: Base64 characters per SSM document. The hard limit is 97 KB for parameters
#: AND document combined; 40,000 leaves generous room for the wrapper, which
#: matters because exceeding it fails with the reason discarded by `check=True`.
CHUNK = 40_000


def blob(path: pathlib.Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return (hashlib.sha256(raw).hexdigest(),
            base64.b64encode(gzip.compress(raw, 9)).decode("ascii"))


def wrap(payload: str) -> str:
    return "\n".join(payload[i:i + 76] for i in range(0, len(payload), 76))


sup_digest, sup_blob = blob(SUPERVISOR)
unit_digest, unit_blob = blob(UNIT)

parts: list[pathlib.Path] = []
chunks = [sup_blob[i:i + CHUNK] for i in range(0, len(sup_blob), CHUNK)]
for index, chunk in enumerate(chunks):
    # `>` on the first part, `>>` after: a re-run must not append to a leftover
    # from the previous attempt, which would corrupt the payload in a way the
    # digest catches but only after a full round trip.
    redirect = ">" if index == 0 else ">>"
    body = f"""set -uo pipefail
mkdir -p /tmp/wizcap
cat {redirect}/tmp/wizcap/sup.b64 <<'CHUNK_END'
{wrap(chunk)}
CHUNK_END
echo "PART {index + 1}/{len(chunks)} bytes=$(wc -c </tmp/wizcap/sup.b64)"
"""
    path = OUT / f"_deploy_capacity_census.part{index + 1}.sh"
    path.write_text(body, encoding="utf-8", newline="\n")
    parts.append(path)

installer = f"""set -uo pipefail
PROJ=/srv/wizard/project
DST="$PROJ/scripts/programming_curriculum_supervisor.py"
RUNTIME=/srv/wizard/runtime/programming-integrated-20260713
UNIT_NAME=wizard-curriculum-supervisor.service
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
WANT_SUP={sup_digest}
WANT_UNIT={unit_digest}

cat >/tmp/wizcap/unit.b64 <<'UNIT_PAYLOAD_END'
{wrap(unit_blob)}
UNIT_PAYLOAD_END

python3 -c "
import base64, gzip, pathlib
for name in ('sup', 'unit'):
    raw = gzip.decompress(base64.b64decode(
        pathlib.Path('/tmp/wizcap/%s.b64' % name).read_text()))
    pathlib.Path('/tmp/wizcap/%s.out' % name).write_bytes(raw)
    print('decoded', name, len(raw))
"

GOT_SUP=$(sha256sum /tmp/wizcap/sup.out | cut -d' ' -f1)
GOT_UNIT=$(sha256sum /tmp/wizcap/unit.out | cut -d' ' -f1)
echo "DIGEST sup want=$WANT_SUP got=$GOT_SUP"
echo "DIGEST unit want=$WANT_UNIT got=$GOT_UNIT"
if [ "$WANT_SUP" != "$GOT_SUP" ] || [ "$WANT_UNIT" != "$GOT_UNIT" ]; then
  echo "RESULT_JSON {{\\"error\\": \\"payload digest mismatch in transport\\"}}"
  exit 0
fi

PID_BEFORE=$(systemctl show "$UNIT_NAME" -p MainPID --value)
FREE_BEFORE=$(python3 -c "import shutil;print(shutil.disk_usage('$RUNTIME').free)")

# Files written over SSM land root:root and the supervisor runs as ec2-user,
# which is its own long-standing failure mode here.
cp -a "$DST" "$DST.bak-$STAMP"
install -o ec2-user -g ec2-user -m 644 /tmp/wizcap/sup.out "$DST"

FRAGMENT=$(systemctl show "$UNIT_NAME" -p FragmentPath --value)
if [ -n "$FRAGMENT" ] && [ -f "$FRAGMENT" ]; then
  sudo cp -a "$FRAGMENT" "$FRAGMENT.bak-$STAMP"
  sudo install -o root -g root -m 644 /tmp/wizcap/unit.out "$FRAGMENT"
  sudo systemctl daemon-reload
fi

sudo systemctl reset-failed "$UNIT_NAME" 2>/dev/null || true
sudo systemctl restart "$UNIT_NAME"

# The recovery rollback stops and restarts the brain node and rewrites a
# multi-hundred-GB reflink clone, so give it room -- but report what is true at
# each check rather than waiting a fixed time and assuming it finished.
sleep 120
FREE_1=$(python3 -c "import shutil;print(shutil.disk_usage('$RUNTIME').free)")
sleep 180

python3 - <<REPORT_END
import json, pathlib, shutil, subprocess, time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {{"now": time.time()}}
out["free_before"] = int("$FREE_BEFORE")
out["free_mid"] = int("$FREE_1")
out["main_pid_before"] = "$PID_BEFORE"


def sh(cmd, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return (p.stdout or "").strip()[-2000:]
    except Exception as exc:  # noqa: BLE001
        return "%s: %s" % (type(exc).__name__, exc)


out["unit"] = sh("systemctl show wizard-curriculum-supervisor.service "
                 "-p ActiveState -p SubState -p Result -p ExecMainStatus "
                 "-p MainPID -p NRestarts -p RestartPreventExitStatus "
                 "--no-pager")
out["on_disk_sha256"] = sh("sha256sum '$DST' | cut -d' ' -f1")
out["owner"] = sh("stat -c '%U:%G' '$DST'")
out["free_after"] = shutil.disk_usage(R).free
out["reclaimed_gb"] = round(
    (out["free_after"] - out["free_before"]) / 2 ** 30, 2)
out["free_after_gb"] = round(out["free_after"] / 2 ** 30, 2)

try:
    out["status"] = json.loads(
        (R / "curriculum-supervisor.status.json").read_text())
except Exception as exc:  # noqa: BLE001
    out["status"] = str(exc)

stalls, refusals = [], []
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("kind")
            if kind == "deferred_replay_interrupted_before_gate":
                stalls.append(event)
            elif kind == "no_interval_fits_disk_window":
                refusals.append(event)
except OSError as exc:
    out["ledger_error"] = str(exc)
out["stalls_tail"] = stalls[-3:]
out["refusal_count"] = len(refusals)
if refusals:
    last = refusals[-1]
    out["refusal"] = {{
        "pending": last.get("pending"),
        "window_hours": last.get("window_hours"),
        "capacity": last.get("capacity"),
        "fits": last.get("fits"),
        "exceeds_count": len(last.get("exceeds") or []),
        "unknown_count": len(last.get("unknown") or []),
    }}

try:
    marker = json.loads((R / "deferred-replay-active.json").read_text())
    out["marker"] = {{k: v for k, v in marker.items() if k != "interval"}}
except Exception as exc:  # noqa: BLE001
    out["marker"] = str(exc)

print("RESULT_JSON " + json.dumps(out, default=str))
REPORT_END
"""

install_path = OUT / "_deploy_capacity_census.install.sh"
install_path.write_text(installer, encoding="utf-8", newline="\n")

print(f"supervisor sha256 {sup_digest}")
print(f"unit sha256 {unit_digest}")
print(f"base64 {len(sup_blob)} chars in {len(chunks)} parts")
for path in parts:
    print(" ", path.name, path.stat().st_size, "bytes")
print(" ", install_path.name, install_path.stat().st_size, "bytes")
