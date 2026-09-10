set -u
R=/srv/wizard/runtime/programming-integrated-20260713
show() {
  python3 - "$R" <<'PY'
import glob, json, os, sys, time
root = sys.argv[1]
out = {"t": time.time()}
st = os.path.join(root, "deferred-replay-active.json")
if os.path.exists(st):
    out["status"] = json.load(open(st))
prog = sorted(glob.glob(os.path.join(root, "deferred-replay-*.progress.json")),
              key=os.path.getmtime)
if prog:
    d = json.load(open(prog[-1]))
    out["progress"] = {k: d.get(k) for k in
                       ("durable_next_row", "accepted_episodes",
                        "current_batch_size", "batch_seconds_ema")}
    out["progress_file"] = os.path.basename(prog[-1])
print(json.dumps(out))
PY
}
show
sleep 60
show
echo "--- procs ---"
ps -eo pid,etimes,rss,comm --sort=-rss | head -6
echo "--- supervisor ---"
systemctl is-active wizard-curriculum-supervisor 2>&1 || true
