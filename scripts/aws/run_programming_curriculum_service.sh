#!/usr/bin/env bash
set -euo pipefail

project_root="${WIZARD_PROJECT_ROOT:-/srv/wizard/project}"
runtime="${WIZARD_RUNTIME:-/srv/wizard/runtime/programming-integrated-20260713}"
corpus_root="${WIZARD_CORPUS_ROOT:-/srv/wizard/corpora}"
endpoint="${WIZARD_ENDPOINT:-http://127.0.0.1:18095}"
node_bin="${WIZARD_NODE_BIN:-${project_root}/target/release/w1z4rd_brain_server}"
supervisor_pid_file="${runtime}/curriculum-service-supervisor.pid"
supervisor_stage_file="${runtime}/curriculum-service-supervisor.stage"

cd "${project_root}"

# Own initial-node creation here instead of unconditionally pulling in a
# second systemd unit.  A recovered node intentionally survives curriculum
# wrapper restarts, and its runtime environment is visible before its socket
# binds.  Adopt that process; launch only when neither identity exists.
python3 - "${runtime}" "${node_bin}" "${endpoint}" <<'PY'
import pathlib
import sys

from scripts.programming_curriculum_supervisor import (
    endpoint_listener_pid,
    start_runtime_node,
    unique_runtime_node_pid,
)

runtime = pathlib.Path(sys.argv[1])
node_bin = pathlib.Path(sys.argv[2])
endpoint = sys.argv[3]
listener_pid = endpoint_listener_pid(endpoint)
runtime_pid = unique_runtime_node_pid(runtime)
if listener_pid and runtime_pid and listener_pid != runtime_pid:
    raise SystemExit(
        f"brain endpoint/runtime conflict: endpoint_pid={listener_pid} "
        f"runtime_pid={runtime_pid}"
    )
if listener_pid and not runtime_pid:
    raise SystemExit(
        f"brain endpoint belongs to unrecognized PID {listener_pid}"
    )
if not listener_pid and not runtime_pid:
    start_runtime_node(runtime, node_bin, endpoint)
PY

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

# The initial-node unit can spend minutes opening a production `.wbrain`
# before it owns the HTTP socket.  Its process is deliberately independent of
# this service, so a PID file left by an older node is not authoritative.  Once
# health is proven, require the socket owner and runtime-environment owner to
# agree and atomically publish that identity before the supervisor may mutate
# corpus state.
python3 - "${runtime}" "${endpoint}" <<'PY'
import os
import pathlib
import sys

from scripts.programming_curriculum_supervisor import (
    endpoint_listener_pid,
    unique_runtime_node_pid,
)

runtime = pathlib.Path(sys.argv[1])
endpoint = sys.argv[2]
listener_pid = endpoint_listener_pid(endpoint)
runtime_pid = unique_runtime_node_pid(runtime)
if not listener_pid or not runtime_pid or listener_pid != runtime_pid:
    raise SystemExit(
        "brain identity mismatch after health check: "
        f"endpoint_pid={listener_pid} runtime_pid={runtime_pid}"
    )
pid_path = runtime / "node.pid"
temporary = pid_path.with_suffix(pid_path.suffix + f".{os.getpid()}.tmp")
temporary.write_text(f"{runtime_pid}\n", encoding="ascii")
os.replace(temporary, pid_path)
PY

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

supervisor_pid_is_live() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local command
  command="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${command}" == *"programming_curriculum_supervisor.py"* ]] || return 1
  [[ "${command}" == *"--runtime ${runtime}"* ]] || return 1
}

clear_supervisor_identity() {
  local expected_pid="${1:-}"
  if [[ -n "${expected_pid}" && -s "${supervisor_pid_file}" ]]; then
    [[ "$(<"${supervisor_pid_file}")" == "${expected_pid}" ]] || return 0
  fi
  rm -f "${supervisor_pid_file}" "${supervisor_stage_file}"
}

run_supervisor_stage() {
  local stage="$1"
  shift
  printf '%s\n' "${stage}" >"${supervisor_stage_file}.tmp"
  mv -f "${supervisor_stage_file}.tmp" "${supervisor_stage_file}"
  "$@" &
  local pid=$!
  printf '%s\n' "${pid}" >"${supervisor_pid_file}.tmp"
  mv -f "${supervisor_pid_file}.tmp" "${supervisor_pid_file}"
  local rc=0
  wait "${pid}" || rc=$?
  clear_supervisor_identity "${pid}"
  return "${rc}"
}

common=(
  python3 scripts/programming_curriculum_supervisor.py
  --runtime "${runtime}"
  --endpoint "${endpoint}"
  --node-bin "${node_bin}"
  --corpus-root "${corpus_root}"
  # At million-neuron scale a 32-episode request can page in enough of the
  # fabric to cross the host's remaining headroom before the supervisor gets
  # another durable control boundary. Eight keeps acknowledgements frequent
  # enough for memory settlement and uncontended canary pauses.
  --batch-size 8
  --lock-chunk-size 8
  --poll-seconds 2
  --checkpoint-rows 131072
  --gate-rows 131072
  --canary-rows 16384
  --max-live-lock-seconds 8
  # 3 GB, not 8. The floor must sit BELOW the brain's working peak or the
  # guard trips before any interval can finish.
  #
  # Measured 2026-09-04 on this 15.26 GB host: the brain reaches an 8.6 GB
  # baseline within 47 s of launch, so an 8 GB free-memory floor left it only
  # 7.26 GB -- less than it needs to exist. Deferred replay yielded four
  # times in a row at 6.7-6.8 GB available, each recycle correctly returning
  # ~14.6 GB, and every interval died with "worker exited -15" before
  # completing. The guard was working perfectly and training still made no
  # progress.
  #
  # 3 GB leaves the brain ~12.2 GB (3.6 GB of working headroom above its
  # baseline) while staying 3 GB clear of the OOM point -- the kernel killed
  # the brain at anon-rss 15,450,400 kB.
  --min-free-memory-gb 3
  --min-free-disk-gb 8
  --max-restarts 10
)

# The first pass accepts every safe forward span and records any failing span
# for later isolation. The second pass cannot declare the curriculum complete
# until every deferred interval passes the same comprehensive admission gate.
adopted_stage=""
if [[ -s "${supervisor_pid_file}" ]]; then
  adopted_pid="$(<"${supervisor_pid_file}")"
  if supervisor_pid_is_live "${adopted_pid}"; then
    adopted_stage="$(
      cat "${supervisor_stage_file}" 2>/dev/null || printf 'forward\n'
    )"
    echo "Adopting ${adopted_stage} supervisor PID ${adopted_pid}."
    while supervisor_pid_is_live "${adopted_pid}"; do
      sleep 10
    done
  fi
  clear_supervisor_identity "${adopted_pid}"
fi

forward_rc=0
if [[ -z "${adopted_stage}" ]]; then
  run_supervisor_stage forward \
    "${common[@]}" --auto-quarantine-recovery --forward-harvest \
    || forward_rc=$?
elif [[ "${adopted_stage}" == "forward" ]]; then
  forward_state="$(status_state)"
  if [[ "${forward_state}" == "deferred_intervals_pending" ]]; then
    forward_rc=1
  elif [[ "${forward_state}" != "all_complete" ]]; then
    echo "Adopted forward supervisor ended in state '${forward_state}'" >&2
    exit 1
  fi
elif [[ "${adopted_stage}" == "replay" ]]; then
  replay_state="$(status_state)"
  if [[ "${replay_state}" == "deferred_replay_failed" ]]; then
    exit 42
  fi
  if [[ "${replay_state}" != "deferred_replay_complete" \
        && "${replay_state}" != "all_complete" ]]; then
    echo "Adopted replay supervisor ended in state '${replay_state}'" >&2
    exit 1
  fi
  exit 0
else
  echo "Unknown adopted supervisor stage '${adopted_stage}'" >&2
  exit 1
fi

if (( forward_rc != 0 )); then
  forward_state="$(status_state)"
  if [[ "${forward_state}" != "deferred_intervals_pending" ]]; then
    echo "Forward curriculum failed in state '${forward_state}'" >&2
    exit "${forward_rc}"
  fi
fi

replay_rc=0
run_supervisor_stage replay "${common[@]}" --replay-deferred || replay_rc=$?
if (( replay_rc != 0 )); then
  replay_state="$(status_state)"
  if [[ "${replay_state}" == "deferred_replay_failed" ]]; then
    echo "Deferred replay needs a semantic repair; evidence is stable." >&2
    exit 42
  fi
  echo "Deferred replay failed in state '${replay_state}'" >&2
  exit "${replay_rc}"
fi
