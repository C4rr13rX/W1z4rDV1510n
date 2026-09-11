python3 - <<'PY'
"""Attribute the burn from the brain's own counters instead of a stopwatch.

Every previous attribution on this volume was a `df` delta over a wall-clock
window, which measures the SUM and cannot separate its terms. `page_outs` and
`clean_skips` split it: bodies actually written versus evictions that wrote
nothing because the durable body was already identical.

What the numbers decide:

  * high `clean_skips` and a fallen burn -- the redundant-append term was
    dominant and the fix addressed it;
  * near-zero `clean_skips` -- bodies genuinely differ between evictions, so
    the growth is real learning writes and the next fix is delta-encoded
    terminal updates rather than suppression;
  * high `clean_skips` and an unchanged burn -- something else on this path is
    writing, and the eviction attribution was wrong.

`bytes_per_body` is published because it is the number the delta-encoding case
turns on: a ~71 KB mean body rewritten in full to record a handful of changed
terminals is a very different problem from one that is mostly new structure.

The node recycles once per cooperative memory yield and respawns from
`--node-bin`, so the rebuilt binary arrives on its own -- but only a stats
payload that CONTAINS the counters proves it did. Deploy is not load.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

R = "/srv/wizard/runtime/programming-integrated-20260713"
ENDPOINT = "http://127.0.0.1:18095/brain/stats"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (proc.stdout + proc.stderr)[-2000:]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def stats():
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


# Wait for the rebuilt node, which arrives on the next memory yield.
waited = 0.0
first = stats()
while "clean_skips" not in first and waited < 600:
    time.sleep(30)
    waited += 30
    first = stats()
out["waited_for_counters_seconds"] = waited
out["counters_present"] = "clean_skips" in first
if "clean_skips" not in first:
    out["stats_without_counters"] = first
    print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
    raise SystemExit(0)

free_a = shutil.disk_usage(R).free
t0 = time.time()
time.sleep(300)
second = stats()
free_b = shutil.disk_usage(R).free
elapsed = time.time() - t0

out["elapsed_seconds"] = round(elapsed, 1)
out["free_gb_first"] = round(free_a / 2**30, 2)
out["free_gb_second"] = round(free_b / 2**30, 2)
out["burn_gb_per_hour"] = round(
    (free_a - free_b) / 2**30 / (elapsed / 3600.0), 2
)

page_outs = int(second.get("page_outs", 0)) - int(first.get("page_outs", 0))
skips = int(second.get("clean_skips", 0)) - int(first.get("clean_skips", 0))
ticks = int(second.get("tick", 0)) - int(first.get("tick", 0))
out["page_outs_delta"] = page_outs
out["clean_skips_delta"] = skips
out["tick_delta"] = ticks
attempts = page_outs + skips
out["eviction_attempts"] = attempts
out["clean_skip_fraction"] = (
    round(skips / attempts, 4) if attempts > 0 else None
)
grown = free_a - free_b
out["bytes_per_body_written"] = (
    round(grown / page_outs) if page_outs > 0 else None
)
out["bytes_per_tick"] = round(grown / ticks) if ticks > 0 else None
out["bytes_suppressed_estimate_gb"] = (
    round(skips * (grown / page_outs) / 2**30, 2)
    if page_outs > 0 and skips > 0 else 0.0
)
out["totals"] = {
    key: second.get(key)
    for key in ("page_outs", "clean_skips", "tick", "total_neurons",
                "evicted_neurons", "resident_terminals", "total_terminals")
}

out["status_tail"] = sh(f"tail -c 500 {R}/curriculum-supervisor.status.json 2>&1")
out["unit"] = sh(
    "systemctl show wizard-curriculum-supervisor.service -p ActiveState "
    "-p SubState -p NRestarts --no-pager 2>&1"
).strip()
out["df"] = sh("df -h /srv/wizard | tail -1")
out["free_mem"] = sh("free -g | head -2")
out["node_inode"] = sh(
    "ls -i /srv/wizard/project/target/release/w1z4rd_brain_server 2>&1"
).strip()

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
