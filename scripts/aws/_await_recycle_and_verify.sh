python3 - <<'PY'
"""Wait for the natural memory recycle, then ask whether the fix took.

The brain server is NOT killed to hurry this along. A SIGTERM to the brain
mid-pass is scored as a replay failure -- measured previously as 19 of 19
memory yields being charged as semantic failures and rolling back everything
trained. The supervisor recycles it roughly every 40 minutes on its own, and
waiting costs nothing but wall-clock.

Nor is the supervisor restarted: `deferred-replay-active.json` is
state:training, so that would discard the whole interval back to start_row.

The check that matters is the process start time against the binary mtime.
A brain that predates its own binary is running the old code however clean
the deploy looked.
"""
import json, os, subprocess, time, urllib.request

BIN = "/srv/wizard/project/target/release/w1z4rd_brain_server"
BUILT = os.path.getmtime(BIN)
DEADLINE = time.time() + 2100

boot = 0.0
for line in open("/proc/stat"):
    if line.startswith("btime"):
        boot = float(line.split()[1])
HZ = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def brains():
    """Running brain servers, with start times. Anchor the pattern: an
    unanchored `w1z4rd_brain_server` matches the supervisor first, whose
    command line carries the binary path and whose ~23 MB looks exactly like
    a brain that never hydrated."""
    pids = subprocess.run(["pgrep", "-f", "release/w1z4rd_brain_server$"],
                          capture_output=True, text=True, timeout=30).stdout.split()
    found = []
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as handle:
                fields = handle.read().split()
            with open(f"/proc/{pid}/statm") as handle:
                rss = int(handle.read().split()[1]) * PAGE
            found.append({"pid": pid,
                          "started": boot + int(fields[21]) / HZ,
                          "rss_gb": round(rss / 2**30, 2)})
        except OSError:
            pass
    return found


out = {"binary_built_unix": BUILT, "waited_s": 0, "polls": []}
started = time.time()
current = None
while time.time() < DEADLINE:
    running = brains()
    fresh = [b for b in running if b["started"] > BUILT]
    out["polls"].append({
        "t": round(time.time() - started),
        "pids": [b["pid"] for b in running],
        "ages": [round(time.time() - b["started"]) for b in running],
        "rss": [b["rss_gb"] for b in running],
        "on_new_binary": bool(fresh),
    })
    if fresh and fresh[0]["rss_gb"] > 1.0:
        # Restarted AND hydrated; a brain still loading answers nothing useful.
        current = fresh[0]
        break
    time.sleep(60)

out["waited_s"] = round(time.time() - started)
out["running_new_binary"] = current is not None
if current is None:
    out["note"] = "no recycle onto the new binary within the wait window"
    print("PROBEJSON " + json.dumps(out))
    raise SystemExit(0)

out["brain"] = current
PROMPT = ("Build a polyglot project with a JavaScript transactional-outbox "
          "order service and a Go concurrency-safe work deduplicator.")
PARAPHRASE = ("Create idempotent Node.js outbox-event ordering code plus "
              "Golang synchronization that suppresses duplicate work in one "
              "repository.")


def ask(text):
    body = json.dumps({"text": text}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:18095/brain/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=240) as handle:
        return json.loads(handle.read().decode("utf-8"))


# Sample more than once: a live curriculum trains underneath the brain, and a
# single passing probe has been contradicted by a ten-sample repeat before.
for label, text in (("canonical", PROMPT), ("paraphrase", PARAPHRASE)):
    samples = []
    for _ in range(3):
        try:
            response = ask(text)
            reply = str(response.get("reply") or "")
            try:
                files = sorted(json.loads(reply).get("files", {}))
            except Exception:
                files = None
            diagnostics = response.get("intent_diagnostics") or {}
            samples.append({
                "files": files,
                "branch": diagnostics.get("answer_branch"),
                "routes": [
                    {"labels": [l.rsplit(":", 1)[-1] for l in (item.get("labels") or [])],
                     "artifacts": [a.get("files") for a in (item.get("artifacts") or [])]}
                    for item in (diagnostics.get("component_routes") or [])
                ],
            })
        except Exception as error:
            samples.append({"error": f"{type(error).__name__}: {error}"[:200]})
    out[label] = samples

canonical = out.get("canonical") or []
out["dedup_go_every_sample"] = bool(canonical) and all(
    sample.get("files") and "dedup.go" in sample["files"] for sample in canonical
)
print("PROBEJSON " + json.dumps(out))
PY
