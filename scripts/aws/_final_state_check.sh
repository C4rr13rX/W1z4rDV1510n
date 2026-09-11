python3 - <<'PY'
import json, os, shutil, subprocess, time
R="/srv/wizard/runtime/programming-integrated-20260713"
def sh(c):
    p=subprocess.run(c,shell=True,capture_output=True,text=True,timeout=90)
    return (p.stdout+p.stderr).strip()[-1200:]
out={"now":time.time()}
out["unit"]=sh("systemctl show wizard-curriculum-supervisor.service -p ActiveState -p SubState -p NRestarts -p RestartPreventExitStatus --no-pager")
out["free_gb"]=round(shutil.disk_usage(R).free/2**30,2)
out["status"]=sh(f"tail -c 400 {R}/curriculum-supervisor.status.json")
# The running supervisor must be the generation that carries the replay guard.
out["guard_in_running_source"]=sh("grep -c replay_disk_floor_breached /srv/wizard/project/scripts/programming_curriculum_supervisor.py")
out["supervisor_start"]=sh("ps -o lstart= -p $(pgrep -f 'programming_curriculum_supervisor.py' | head -1)")
census={}
for n,pat in {"wrapper":"run_programming_curriculum_service.sh","supervisor":"programming_curriculum_supervisor.py","worker":"drive_corpora_brain"}.items():
    c=0
    for pid in os.listdir("/proc"):
        if not pid.isdigit(): continue
        try:
            cmd=open(f"/proc/{pid}/cmdline","rb").read().replace(b"\0",b" ").decode("utf-8","replace")
        except OSError: continue
        if pat in cmd: c+=1
    census[n]=c
out["census"]=census
print("PROBE_JSON "+json.dumps(out,sort_keys=True,default=str))
PY
