python3 - <<'PY'
"""Restart the curriculum after a DELIBERATE halt, but only if it can train.

The wake-up read wrapper 0 / supervisor 0 / worker 0 and called it "no owner of
terminal state deferred_replay_training". The census was right and every cause
CLAUDE.md records for it was wrong: `df` shows 502 GB free (not the ENOSPC
crash loop), the row has been frozen 49 minutes across two samples 90 s apart
(not a cooperative-yield trough), and the unit reads:

    ActiveState=inactive  SubState=dead  Result=success
    ExecMainStatus=143    NRestarts=0

143 is SIGTERM, `Result=success`, and the journal shows an explicit
`Stopping ...` at 23:45:11 UTC followed by `systemctl reset-failed` at 00:17:32.
That is `_halt_burn_and_verify.sh` plus `_reset_unit_state.sh` -- the previous
session stopped training on purpose to beat the disk burn to ENOSPC, shipped
the 150 GB floor, and never started it again. `Restart=on-failure` cannot undo
a clean stop, so training stays down forever with no crash to alarm on. This is
a FOURTH cause for the same 0/0/0 census, and the only repair is to start it.

Starting is gated, not blind:

  * `load_registry` must load the real directory. A malformed `.toml` is
    all-or-nothing and retry-loops the WHOLE curriculum; the stderr log named
    in `last_failure` is exactly that SchemaError, 26.8 h old, so its repair is
    the thing to re-prove before handing the supervisor the wheel.
  * the deployed floor must be the 150 GB one, or the restart re-runs the race
    that caused the halt.
  * free disk must exceed the floor with room to train.

`deferred-replay-active.json` is `state: training` at row 211,688, so the
restart discards ~10,344 rows and resumes the interval from 201,344. That is
the known and accepted cost of a supervisor restart mid-interval; it preserves
the accept/quarantine invariant, which re-entering a half-trained interval
would not.

The brain (PID from the last yield) is left alive deliberately: the wrapper
adopts a node whose endpoint owner and runtime owner agree, so adoption avoids
a multi-minute re-hydration of a 363 GB container.
"""
import json
import os
import pathlib
import shutil
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
P = "/srv/wizard/project"
UNIT = "wizard-curriculum-supervisor"
out = {"now": time.time(), "gates": {}}


def sh(cmd, timeout=300):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "rc": proc.returncode,
            "out": (proc.stdout or "")[-4000:],
            "err": (proc.stderr or "")[-2000:],
        }
    except Exception as exc:
        return {"rc": None, "error": f"{type(exc).__name__}: {exc}"}


def rows():
    path = os.path.join(R, "deferred-replay-909de5e9d4936130.progress.json")
    best = None
    for name in os.listdir(R):
        if name.startswith("deferred-replay-") and name.endswith(".progress.json"):
            full = os.path.join(R, name)
            age = time.time() - os.path.getmtime(full)
            if best is None or age < best[1]:
                best = (full, age)
    picked = best[0] if best else path
    try:
        with open(picked, "r", encoding="utf-8") as handle:
            blob = json.load(handle)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "file": os.path.basename(picked),
        "durable_next_row": blob.get("durable_next_row"),
        "accepted_episodes": blob.get("accepted_episodes"),
        "age_s": round(time.time() - os.path.getmtime(picked), 1),
    }


# --- Gate 1: the registry must load, or the supervisor retry-loops. ---------
out["gates"]["registry"] = sh(
    f"cd {P} && python3 -c \""
    "from tools.training_standard.schema import load_registry;"
    "import tools.training_standard.runner as r;"
    "reg = load_registry(r.REGISTRY_DIR);"
    "print('REGISTRY_OK', len(reg));"
    "print(sorted(reg)[:40])\" 2>&1 | tail -20"
)
out["gates"]["go_toml"] = sh(
    f"grep -n 'category' {P}/tools/training_standard/registry/go_systems_001.toml 2>&1"
)
out["gates"]["registry_test"] = sh(
    f"cd {P} && python3 -m pytest tests/test_training_registry_schema.py -q 2>&1 | tail -8"
)

# --- Gate 2: the deployed floor must be the measured one. -------------------
out["gates"]["floor"] = sh(
    f"grep -n 'min-free-disk-gb' {P}/scripts/aws/run_programming_curriculum_service.sh"
)

# --- Gate 3: real free space, from df -- never from summing file sizes. -----
usage = shutil.disk_usage("/srv/wizard")
out["gates"]["disk"] = {
    "free_gb": round(usage.free / 2**30, 2),
    "total_gb": round(usage.total / 2**30, 2),
}
out["gates"]["memory"] = sh("free -g | head -3")

registry_ok = "REGISTRY_OK" in (out["gates"]["registry"].get("out") or "")
floor_ok = "--min-free-disk-gb 150" in (out["gates"]["floor"].get("out") or "")
disk_ok = out["gates"]["disk"]["free_gb"] > 200.0
out["gates"]["verdict"] = {
    "registry_ok": registry_ok,
    "floor_ok": floor_ok,
    "disk_ok": disk_ok,
}

out["before"] = {
    "rows": rows(),
    "unit": sh(
        f"systemctl show -p ActiveState -p SubState -p Result -p NRestarts "
        f"-p ExecMainStatus {UNIT} --no-pager"
    ),
    "active_json_state": sh(
        f"python3 -c \"import json;d=json.load(open('{R}/deferred-replay-active.json'));"
        "print(json.dumps({k: d.get(k) for k in "
        "('interval_id','phase','state','start_row','resume_row','end_row','created_unix')}))\" 2>&1"
    ),
}

if not (registry_ok and floor_ok and disk_ok):
    out["action"] = "REFUSED: a start gate failed; not handing the supervisor the wheel"
    print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
    raise SystemExit(0)

# --- Start, then prove it TRAINS. A started unit is not a training brain. ---
out["action"] = "start"
out["start"] = sh(f"systemctl start {UNIT}", timeout=240)
time.sleep(45)
out["after_start_unit"] = sh(
    f"systemctl show -p ActiveState -p SubState -p Result -p NRestarts {UNIT} --no-pager"
)

samples = []
for index in range(7):
    census = {"wrapper": 0, "supervisor": 0, "worker": 0}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                cmd = handle.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "run_programming_curriculum_service.sh" in cmd:
            census["wrapper"] += 1
        if "programming_curriculum_supervisor.py" in cmd:
            census["supervisor"] += 1
        if "drive_corpora_brain" in cmd:
            census["worker"] += 1
    samples.append(
        {
            "t": round(time.time() - out["now"], 1),
            "census": census,
            "rows": rows(),
            "free_gb": round(shutil.disk_usage("/srv/wizard").free / 2**30, 2),
        }
    )
    if index < 6:
        time.sleep(60)
out["samples"] = samples

out["tail_stderr"] = sh(f"tail -c 3000 {R}/curriculum-service.stderr.log 2>&1")
out["tail_status"] = sh(f"tail -c 1500 {R}/curriculum-supervisor.status.json 2>&1")
out["journal"] = sh(f"journalctl -u {UNIT} -n 40 --no-pager 2>&1 | tail -c 3000")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
