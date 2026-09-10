#!/bin/bash
# Ship commit c343443's two changed evaluator files to /srv/wizard/project and
# PROVE the deployed copy classifies a crash as infrastructure.
#
# Deliberately does NOT restart the supervisor. The midphase gate spawns
# `python scripts/programming_integrated_retention.py` as a fresh subprocess
# per invocation, so the next gate run reads the new file on its own -- and a
# restart right now would discard the in-flight go-systems replay interval,
# which `deferred-replay-active.json` reports as state:training.
#
# "Deploy is not load": the copy is verified by inode and sha here, and then
# the fixed code path is executed against the live brain.
set -uo pipefail

COMMIT=c34344398fd706002f72168caa274c8986fc98bc
BUCKET=wizard-vision-private-321572159829-us-east-1
KEY="wizard-vision/source/${COMMIT}/wizard-vision-source.tar.gz"
WANT_SHA=e4f34d2ba1756871da3481e7b449c1ca2676e2435032473691a9c0bdaf8916ad

PROJ=/srv/wizard/project
STAGE=$(mktemp -d /tmp/wvgate.XXXXXX)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
trap 'rm -rf "$STAGE"' EXIT

TARBALL="$STAGE/src.tar.gz"
if ! aws s3 cp "s3://${BUCKET}/${KEY}" "$TARBALL" --region us-east-1 --only-show-errors; then
    echo "PROBEJSON {\"error\":\"s3 download failed\",\"key\":\"${KEY}\"}"; exit 0
fi
GOT_SHA=$(sha256sum "$TARBALL" | awk '{print $1}')
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
    echo "PROBEJSON {\"error\":\"sha mismatch\",\"got\":\"${GOT_SHA}\"}"; exit 0
fi
mkdir -p "$STAGE/tree"
tar -xzf "$TARBALL" -C "$STAGE/tree" || { echo 'PROBEJSON {"error":"extract failed"}'; exit 0; }

FILES="
scripts/programming_integrated_retention.py
scripts/programming_debug_benchmark.py
"
BACKUP="$PROJ/.deploy-backup-$STAMP"
mkdir -p "$BACKUP"
for rel in $FILES; do
    src="$STAGE/tree/$rel"; dst="$PROJ/$rel"
    [ -f "$src" ] || { echo "PROBEJSON {\"error\":\"missing in bundle: $rel\"}"; exit 0; }
    [ -f "$dst" ] && { mkdir -p "$BACKUP/$(dirname "$rel")"; cp -p "$dst" "$BACKUP/$rel"; }
    install -m 0644 "$src" "$dst"
    # SSM writes land root:root; the supervisor is ec2-user and dies on write.
    chown 1000:1000 "$dst"
done
chown -R 1000:1000 "$BACKUP"

python3 - <<PY
import hashlib, json, os, pathlib, subprocess, sys, time

proj = pathlib.Path("$PROJ")
out = {"commit": "$COMMIT", "backup": "$BACKUP", "files": {}}
for rel in """$FILES""".split():
    p = proj / rel
    st = p.stat()
    out["files"][rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                         "uid": st.st_uid, "mode": oct(st.st_mode)[-4:],
                         "inode": st.st_ino, "size": st.st_size}

# Deploy is not load. Execute the DEPLOYED code path and check both directions.
sys.path.insert(0, str(proj / "scripts"))
sys.path.insert(0, str(proj))
try:
    import programming_integrated_retention as ret
    out["has_run_evaluator"] = hasattr(ret, "run_evaluator")
    out["has_EvaluatorUnavailable"] = hasattr(ret, "EvaluatorUnavailable")
    out["predict_timeout"] = __import__("programming_debug_benchmark").PREDICT_TIMEOUT_SECONDS

    # A crashed child must raise a message the supervisor calls transient.
    stale = pathlib.Path("/tmp/_gate_stale_report.json")
    stale.write_text(json.dumps({"exact": {"passed": 6, "total": 6}}))
    real_run = ret.run_evaluator
    ret.run_evaluator = lambda cmd: subprocess.CompletedProcess(
        cmd, 1, "", "socket.timeout: timed out\n")
    try:
        ret.debug_eval("http://127.0.0.1:18095", stale)
        out["crash_raises"] = False
    except ret.EvaluatorUnavailable as exc:
        out["crash_raises"] = True
        out["crash_message_head"] = str(exc)[:160]
        out["stale_report_removed"] = not stale.exists()
    finally:
        ret.run_evaluator = real_run

    from programming_curriculum_supervisor import GateCommandFailure, transient_gate_failure
    out["classified_transient"] = transient_gate_failure(GateCommandFailure(
        ["python", "retention.py"], 1, "",
        "Traceback (most recent call last):\n" + out.get("crash_message_head", "")))

    # And the real thing: run the deployed benchmark against the live brain.
    target = "/tmp/_deployed_integrated_debug.json"
    if os.path.exists(target):
        os.unlink(target)
    started = time.time()
    run = subprocess.run(["/usr/bin/python3", "scripts/programming_debug_benchmark.py",
                          "--endpoint", "http://127.0.0.1:18095", "--output", target],
                         cwd=str(proj), capture_output=True, text=True, timeout=3000)
    out["live_run"] = {"rc": run.returncode, "seconds": round(time.time() - started, 1),
                       "stdout": run.stdout[-400:], "stderr_tail": run.stderr[-500:],
                       "wrote_report": os.path.exists(target)}
except Exception as error:
    out["error"] = f"{type(error).__name__}: {error}"[:400]
print("PROBEJSON " + json.dumps(out))
PY
