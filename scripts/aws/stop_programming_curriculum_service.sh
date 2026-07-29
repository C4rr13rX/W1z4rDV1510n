#!/usr/bin/env bash
set -u

runtime="${WIZARD_RUNTIME:-/srv/wizard/runtime/programming-integrated-20260713}"
supervisor_marker="scripts/programming_curriculum_supervisor.py --runtime ${runtime} "
worker_marker="tools.training_standard.drive_corpora_brain"
progress_marker="--progress-path ${runtime}/"

matching_pids() {
  local first="$1"
  local second="${2:-}"
  ps -eo pid=,args= | awk -v first="${first}" -v second="${second}" '
    index($0, first) && (second == "" || index($0, second)) { print $1 }
  '
}

mapfile -t workers < <(matching_pids "${worker_marker}" "${progress_marker}")
mapfile -t supervisors < <(matching_pids "${supervisor_marker}")
targets=("${workers[@]}" "${supervisors[@]}")

if (( ${#targets[@]} == 0 )); then
  rm -f "${runtime}/curriculum-supervisor.pid"
  exit 0
fi

# Stop the producer and its owner together. The node command matches neither
# marker and remains alive for the next supervisor to attach without hydrating
# the complete persisted brain.
kill -TERM "${targets[@]}" 2>/dev/null || true
for _ in $(seq 1 30); do
  alive=()
  for pid in "${targets[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive+=("${pid}")
    fi
  done
  if (( ${#alive[@]} == 0 )); then
    rm -f "${runtime}/curriculum-supervisor.pid"
    exit 0
  fi
  sleep 1
done

kill -KILL "${targets[@]}" 2>/dev/null || true
rm -f "${runtime}/curriculum-supervisor.pid"
