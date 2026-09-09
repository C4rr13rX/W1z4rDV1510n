python3 - <<'PY'
"""Did the running brain load the new binary, and does it now compose dedup.go?

Two questions, in this order, because the first invalidates the second.
"Deploying a fix is not applying it": a binary on disk that no process has
executed ran the OLD code for 850 s while a grep of the file said "fixed". So
the brain's start time is compared against the binary's mtime before any
behavioural claim is made.

Then the actual question. /brain/chat is read-only on a settled brain, so
asking it costs nothing and answers in seconds what the enterprise gate would
otherwise take another interval -- about two hours -- to report.
"""
import json, os, subprocess, time, urllib.request

BIN = "/srv/wizard/project/target/release/w1z4rd_brain_server"
out = {"now": time.time()}

out["binary_mtime"] = os.path.getmtime(BIN)
out["binary_age_h"] = round((out["now"] - out["binary_mtime"]) / 3600, 3)

# `pgrep -f w1z4rd_brain_server` matches the SUPERVISOR first -- its command
# line carries `--node-bin .../w1z4rd_brain_server` and reports ~23 MB, the
# exact signature of a brain that never hydrated. Anchor the pattern.
pids = subprocess.run(["pgrep", "-f", "release/w1z4rd_brain_server$"],
                      capture_output=True, text=True, timeout=30).stdout.split()
boot = 0.0
for line in open("/proc/stat"):
    if line.startswith("btime"):
        boot = float(line.split()[1])
hz = os.sysconf("SC_CLK_TCK")
procs = []
for pid in pids:
    try:
        with open(f"/proc/{pid}/stat") as handle:
            fields = handle.read().split()
        started = boot + int(fields[21]) / hz
        with open(f"/proc/{pid}/statm") as handle:
            rss_pages = int(handle.read().split()[1])
        procs.append({
            "pid": pid,
            "started_unix": started,
            "age_s": round(out["now"] - started, 1),
            "rss_gb": round(rss_pages * os.sysconf("SC_PAGE_SIZE") / 2**30, 2),
            "started_after_build": started > out["binary_mtime"],
        })
    except OSError:
        pass
out["brain_processes"] = procs
out["running_new_binary"] = any(p["started_after_build"] for p in procs)

PROMPT = ("Build a polyglot project with a JavaScript transactional-outbox "
          "order service and a Go concurrency-safe work deduplicator.")

def ask(text):
    body = json.dumps({"text": text}).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:18095/brain/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as handle:
        return json.loads(handle.read().decode("utf-8"))

if out["running_new_binary"]:
    try:
        response = ask(PROMPT)
        reply = str(response.get("reply") or "")
        try:
            files = sorted(json.loads(reply).get("files", {}))
        except Exception:
            files = None
        diagnostics = response.get("intent_diagnostics") or {}
        out["canonical"] = {
            "files": files,
            "reply_len": len(reply),
            "answer_branch": diagnostics.get("answer_branch"),
            "component_recall": diagnostics.get("component_recall"),
            "component_routes": [
                {"labels": item.get("labels"),
                 "artifacts": [a.get("files") for a in (item.get("artifacts") or [])]}
                for item in (diagnostics.get("component_routes") or [])
            ],
        }
        out["dedup_go_composed"] = bool(files and "dedup.go" in files)
    except Exception as error:
        out["chat_error"] = f"{type(error).__name__}: {error}"[:300]
else:
    out["note"] = ("brain has not restarted onto the new binary yet; the "
                   "supervisor relaunches it at the next memory recycle")

print("PROBEJSON " + json.dumps(out))
PY
