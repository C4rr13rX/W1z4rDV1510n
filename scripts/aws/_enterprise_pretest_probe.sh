python3 - <<'PY'
"""Pre-test all 12 enterprise suites while the block still has ~1.4 h to run.

polyglot -- the only suite failing the last seven gates -- now passes on the
partially-trained Go corpus. That answers the question the gate was failing on,
but not the question it will ask next: whether 131k rows of Go regressed one of
the other eleven. Corpus interference is the documented signature (the gate is
measured immediately after a large interval and generalisation dips while
trained recall stays perfect), so the eleven are the live risk now.

Two bounds are deliberate:

  * `--suite-timeout 240`, not the gate's 900. Worst case is 12*240 = 48 min
    against a ~1.4 h margin before the gate fires. Two enterprise runs
    overlapping would contend for memory against a `--min-free-memory-gb 3`
    floor at ~3.6 GB available, and tripping it mid-interval yields a resource
    yield that gets scored as a semantic failure. A suite that hits 240 s here
    is reported `timed_out`, NOT failed -- it is unmeasured, and the gate's own
    900 s may still pass it.
  * Memory is sampled between suites and the run aborts early if it falls
    below the margin, because protecting the in-flight interval outranks
    completing this measurement.

`tick_delta` will be non-zero because training is concurrently advancing the
tick. That is an artifact of measuring during a live block, not a mutation by
this probe: the runner passes `--no-train` to the six suites that would
otherwise write, and the other six are read-only by construction.
"""
import json
import os
import subprocess
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
PROJECT = "/srv/wizard/project"
ENDPOINT = "http://127.0.0.1:18095"
FLOOR_GB = 3.2
out = {"now": time.time()}


def avail_gb():
    for line in open("/proc/meminfo", encoding="utf-8"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024.0 * 1024.0)
    return 0.0


def status():
    try:
        data = json.load(open(
            os.path.join(R, "curriculum-supervisor.status.json"), encoding="utf-8"))
        return data.get("durable_next_row"), data.get("state")
    except Exception:  # noqa: BLE001
        return None, None


endpoint_args = ["--endpoint", ENDPOINT]
OUT = os.path.join(R, "_pretest_suites")
os.makedirs(OUT, exist_ok=True)
# Mirrors scripts/programming_enterprise_retention.py exactly, including which
# suites receive --no-train. Diverging here would measure a different gate.
SUITES = [
    ("python_enterprise", ["scripts/programming_enterprise_eval.py", *endpoint_args, "--no-train"]),
    ("multilanguage", ["scripts/programming_multilanguage_eval.py", *endpoint_args, "--no-train"]),
    ("native_enterprise", ["scripts/programming_native_enterprise_eval.py", *endpoint_args, "--no-train"]),
    ("platform", ["scripts/programming_platform_eval.py", *endpoint_args, "--no-train"]),
    ("project", ["scripts/programming_project_eval.py", *endpoint_args, "--no-train"]),
    ("typescript", ["scripts/programming_typescript_enterprise.py", *endpoint_args, "--no-train"]),
    ("cross_language", ["scripts/programming_cross_language_transfer.py", *endpoint_args, "--no-train"]),
    ("cross_project", ["scripts/programming_cross_project_composition.py", *endpoint_args]),
    ("composition", ["scripts/programming_composition_eval.py", *endpoint_args]),
    ("semantic_stress", ["scripts/programming_semantic_stress.py", *endpoint_args]),
    ("capstone_safety", ["scripts/programming_capstone_readiness.py", *endpoint_args]),
]

out["row_before"], out["state_before"] = status()
out["avail_before"] = round(avail_gb(), 2)
results = []
for name, command in SUITES:
    free = avail_gb()
    if free < FLOOR_GB:
        results.append({"suite": name, "skipped": "avail %.2f GB" % free})
        out["aborted_for_memory"] = True
        break
    started = time.time()
    try:
        proc = subprocess.run(
            ["/usr/bin/python3", *command, "--output",
             os.path.join(OUT, name + ".json")],
            cwd=PROJECT, capture_output=True, text=True, timeout=240)
        results.append({
            "suite": name,
            "rc": proc.returncode,
            "passed": proc.returncode == 0,
            "elapsed_s": round(time.time() - started, 1),
            "avail_gb": round(free, 2),
            "stdout_tail": proc.stdout.strip()[-600:],
            "stderr_tail": proc.stderr.strip()[-400:],
        })
    except subprocess.TimeoutExpired:
        results.append({
            "suite": name, "timed_out": True, "passed": None,
            "elapsed_s": round(time.time() - started, 1),
            "note": "unmeasured at 240s; the gate allows 900s",
        })
    except Exception as error:  # noqa: BLE001
        results.append({"suite": name, "error": str(error)[:200], "passed": None})

out["suites"] = results
out["passed_count"] = sum(1 for r in results if r.get("passed") is True)
out["failed"] = [r["suite"] for r in results if r.get("passed") is False]
out["unmeasured"] = [r["suite"] for r in results
                     if r.get("passed") is None and "skipped" not in r]
out["avail_after"] = round(avail_gb(), 2)
out["row_after"], out["state_after"] = status()
if out["row_before"] and out["row_after"]:
    out["forward_still_advancing"] = out["row_after"] > out["row_before"]
    out["rows_gained_during_pretest"] = out["row_after"] - out["row_before"]

print("PROBE_JSON " + json.dumps(out, default=str))
PY
