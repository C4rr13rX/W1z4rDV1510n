#!/usr/bin/env bash
set -euo pipefail

project_root="${WIZARD_PROJECT_ROOT:-/srv/wizard/project}"
runtime="${WIZARD_RUNTIME:-/srv/wizard/runtime/programming-integrated-20260713}"
corpus_root="${WIZARD_CORPUS_ROOT:-/srv/wizard/corpora}"
endpoint="${WIZARD_ENDPOINT:-http://127.0.0.1:18095}"
node_bin="${WIZARD_NODE_BIN:-${project_root}/target/release/w1z4rd_brain_server}"

cd "${project_root}"

ready=0
for _ in $(seq 1 300); do
  if curl -fsS "${endpoint}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  echo "Wizard brain endpoint did not become healthy: ${endpoint}" >&2
  exit 1
fi

status_state() {
  python3 - "${runtime}/curriculum-supervisor.status.json" <<'PY'
import json
import pathlib
import sys

try:
    payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
except (OSError, ValueError):
    payload = {}
print(payload.get("state", ""))
PY
}

common=(
  python3 scripts/programming_curriculum_supervisor.py
  --runtime "${runtime}"
  --endpoint "${endpoint}"
  --node-bin "${node_bin}"
  --corpus-root "${corpus_root}"
  --batch-size 256
  --lock-chunk-size 32
  --poll-seconds 2
  --checkpoint-rows 131072
  --gate-rows 131072
  --canary-rows 16384
  --max-live-lock-seconds 8
  --min-free-memory-gb 6
  --min-free-disk-gb 8
  --max-restarts 10
)

# The first pass accepts every safe forward span and records any failing span
# for later isolation. The second pass cannot declare the curriculum complete
# until every deferred interval passes the same comprehensive admission gate.
forward_rc=0
"${common[@]}" --auto-quarantine-recovery --forward-harvest || forward_rc=$?
if (( forward_rc != 0 )); then
  forward_state="$(status_state)"
  if [[ "${forward_state}" != "deferred_intervals_pending" ]]; then
    echo "Forward curriculum failed in state '${forward_state}'" >&2
    exit "${forward_rc}"
  fi
fi

replay_rc=0
"${common[@]}" --replay-deferred || replay_rc=$?
if (( replay_rc != 0 )); then
  replay_state="$(status_state)"
  if [[ "${replay_state}" == "deferred_replay_failed" ]]; then
    echo "Deferred replay needs a semantic repair; evidence is stable." >&2
    exit 42
  fi
  echo "Deferred replay failed in state '${replay_state}'" >&2
  exit "${replay_rc}"
fi
