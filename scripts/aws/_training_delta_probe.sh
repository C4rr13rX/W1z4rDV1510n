#!/bin/bash
# Two samples separated by a real interval: is the restarted supervisor
# CONVERTING, or merely alive?
#
# CLAUDE.md's standing rule -- the measurement is the resolved count rising,
# not that a process is alive. A supervisor that hydrates forever and a
# supervisor that trains look identical in `systemctl is-active`.
set -uo pipefail

SLEEP="${1:-180}"

python3 - "$SLEEP" <<'PY'
import glob
import json
import os
import pathlib
import subprocess
import sys
import time

nap = int(sys.argv[1])
runtime = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")


def sh(*cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def sample():
    s = {"unix": time.time()}
    # Every replay progress file, newest first -- a NEW one appearing is itself
    # the signal that this pass started an interval.
    files = {}
    for path in glob.glob(str(runtime / "deferred-replay-*.progress.json")):
        p = pathlib.Path(path)
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
            files[p.name] = {
                "mtime": p.stat().st_mtime,
                "durable_next_row": body.get("durable_next_row"),
                "accepted_episodes": body.get("accepted_episodes"),
                "timed_batches": body.get("timed_batches"),
                "corpus": os.path.basename(str(body.get("corpus") or "")),
            }
        except Exception:
            continue
    s["progress"] = files

    # Brain hydration + tick: a brain still faulting its store in has not
    # started learning yet, and RSS is how that shows.
    rss = 0.0
    pid = ""
    for cand in sh("pgrep", "-f", "w1z4rdv1510n-node|brain_server|wizard-brain").split():
        try:
            with open(f"/proc/{cand}/status") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        val = float(line.split()[1]) / (1024 * 1024)
                        if val > rss:
                            rss, pid = val, cand
        except Exception:
            continue
    s["brain_rss_gb"] = round(rss, 3)
    s["brain_pid"] = pid

    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:18095/stats", timeout=25) as h:
            stats = json.loads(h.read().decode("utf-8"))
        s["tick"] = stats.get("tick")
        s["total_neurons"] = stats.get("total_neurons")
        s["resident_terminals"] = stats.get("resident_terminals")
    except Exception as exc:
        s["stats_error"] = f"{type(exc).__name__}: {exc}"

    log = runtime / "curriculum-service.stderr.log"
    s["log_size"] = log.stat().st_size if log.is_file() else 0
    s["supervisor"] = len(sh("pgrep", "-f", "programming_curriculum_supervisor").split())
    s["driver"] = len(sh("pgrep", "-f", "drive_corpora_brain").split())
    return s


first = sample()
time.sleep(nap)
second = sample()

out = {"seconds": round(second["unix"] - first["unix"], 1)}
for key in ("brain_rss_gb", "tick", "log_size", "supervisor", "driver",
            "total_neurons", "resident_terminals"):
    out[key] = [first.get(key), second.get(key)]

moved = {}
for name, now in second["progress"].items():
    was = first["progress"].get(name)
    if was is None:
        moved[name] = {"NEW": now}
    elif (was.get("durable_next_row") != now.get("durable_next_row")
          or was.get("accepted_episodes") != now.get("accepted_episodes")
          or was.get("timed_batches") != now.get("timed_batches")):
        moved[name] = {
            "corpus": now.get("corpus"),
            "durable_next_row": [was.get("durable_next_row"), now.get("durable_next_row")],
            "accepted_episodes": [was.get("accepted_episodes"), now.get("accepted_episodes")],
            "timed_batches": [was.get("timed_batches"), now.get("timed_batches")],
        }
out["progress_moved"] = moved
out["progress_file_count"] = [len(first["progress"]), len(second["progress"])]
out["stats_error"] = second.get("stats_error")

log = runtime / "curriculum-service.stderr.log"
if log.is_file() and second["log_size"] > first["log_size"]:
    with log.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(first["log_size"])
        out["new_log"] = fh.read()[:2500]

print("PROBE_JSON " + json.dumps(out, default=str)[:6000])
PY
