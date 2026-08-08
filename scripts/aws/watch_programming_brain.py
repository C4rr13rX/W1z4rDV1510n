#!/usr/bin/env python3
"""Wake the current Codex session only when programming-brain work needs it.

The curriculum supervisor, not Codex, owns healthy progress.  This watcher
polls a compact AWS status probe and resumes one explicit Codex session only
after the same actionable state is observed repeatedly.  Identical handled
events are deduplicated across watcher restarts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.aws.bootstrap_training_host import aws, send_and_wait
except ModuleNotFoundError:
    from bootstrap_training_host import aws, send_and_wait


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE = "i-0d7a6deeb0ead2dfc"
DEFAULT_RUNTIME = "/srv/wizard/runtime/programming-integrated-20260713"
COMPLETE_STATES = {"all_complete", "deferred_replay_complete"}


@dataclass(frozen=True)
class Decision:
    kind: str
    reason: str
    fingerprint: str = ""


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def completion_marker_valid(payload: dict) -> bool:
    brain = payload.get("production_brain") or {}
    course = payload.get("obstacle_course") or {}
    selectors = payload.get("brain_selectors") or {}
    capstone = payload.get("capstone") or {}
    reports = [
        brain.get("report"), course.get("report"),
        selectors.get("report"), capstone.get("report"),
    ]
    return bool(
        payload.get("passed") is True
        and brain.get("passed") is True
        and course.get("passed") == 1000
        and course.get("total") == 1000
        and selectors.get("passed") is True
        and capstone.get("passed") is True
        and capstone.get("independently_verified") is True
        and all(isinstance(path, str) and path.strip() for path in reports)
    )


def event_fingerprint(kind: str, probe: dict) -> str:
    status = probe.get("status") or {}
    identity = {
        "kind": kind,
        "host_state": probe.get("host_state"),
        "state": status.get("state"),
        "phase": status.get("phase"),
        "interval_id": status.get("interval_id"),
        "error": status.get("error"),
        "block_target_row": status.get("block_target_row"),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    return f"{kind}:{digest}"


def classify_probe(probe: dict, *, stall_seconds: float) -> Decision:
    """Classify only deterministic lifecycle evidence, never model quality."""
    host_state = str(probe.get("host_state") or "unknown")
    if host_state != "running":
        return Decision(
            "fix_required", f"AWS host is {host_state}",
            event_fingerprint("fix_required", probe),
        )

    status = probe.get("status") or {}
    state = str(status.get("state") or "missing")
    supervisor_count = int(probe.get("supervisor_count") or 0)
    wrapper_count = int(probe.get("wrapper_count") or 0)
    status_age = float(probe.get("status_age_seconds") or 0.0)
    service_stage = str(probe.get("service_stage") or "")

    if state == "deferred_intervals_pending" or (
        service_stage == "replay"
        and state.startswith("deferred_replay_")
        and state not in COMPLETE_STATES
    ):
        identity = {
            "host_state": host_state,
            "runtime": probe.get("runtime"),
            "status": {"state": "quarantine_ready"},
        }
        return Decision(
            "quarantine_ready",
            "forward harvesting is complete and quarantine replay is ready or active",
            event_fingerprint("quarantine_ready", identity),
        )

    if state in COMPLETE_STATES and supervisor_count == 0:
        return Decision(
            "milestone", f"automated stage reached {state}",
            event_fingerprint("milestone", probe),
        )
    if supervisor_count > 0 or wrapper_count > 0:
        if status_age > stall_seconds:
            return Decision(
                "fix_required",
                f"live curriculum control is stale for {status_age:.0f}s",
                event_fingerprint("fix_required", probe),
            )
        return Decision("healthy", f"automation owns {state}")
    return Decision(
        "fix_required",
        f"no curriculum supervisor or wrapper owns terminal state {state}",
        event_fingerprint("fix_required", probe),
    )


def remote_probe(profile: str, instance_id: str, runtime: str) -> dict:
    instance = aws(
        profile, "ec2", "describe-instances", "--instance-ids", instance_id,
        "--query", "Reservations[0].Instances[0].State.Name", "--output", "text",
    ).stdout.strip()
    if instance != "running":
        return {"host_state": instance, "observed_unix": time.time()}

    remote = f"""python3 - <<'PY'
import json, pathlib, time
runtime = pathlib.Path({runtime!r})
try:
    status = json.loads((runtime / 'curriculum-supervisor.status.json').read_text())
except Exception as exc:
    status = {{'state': 'missing', 'error': str(exc)}}
supervisors = wrappers = workers = 0
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command = path.read_bytes().replace(b'\\0', b' ').decode(errors='replace')
    except OSError:
        continue
    if 'programming_curriculum_supervisor.py' in command and {runtime!r} in command:
        supervisors += 1
    if 'run_programming_curriculum_service.sh' in command:
        wrappers += 1
    if 'tools.training_standard.drive_corpora_brain' in command and {runtime!r} in command:
        workers += 1
updated = float(status.get('updated_unix') or 0.0)
try:
    service_stage = (runtime / 'curriculum-service-supervisor.stage').read_text().strip()
except OSError:
    service_stage = ''
print(json.dumps({{
    'host_state': 'running', 'runtime': str(runtime), 'status': status,
    'supervisor_count': supervisors, 'wrapper_count': wrappers,
    'worker_count': workers, 'service_stage': service_stage,
    'status_age_seconds': max(0.0, time.time() - updated) if updated else 1e99,
    'observed_unix': time.time(),
}}, separators=(',', ':')))
PY"""
    invocation = send_and_wait(
        profile, instance_id, [remote], 300,
        comment="Probe Wizard programming brain watchdog state",
    )
    output = str(invocation.get("StandardOutputContent") or "").strip().splitlines()
    if not output:
        raise RuntimeError("AWS programming-brain probe returned no output")
    return json.loads(output[-1])


def codex_prompt(decision: Decision, probe: dict) -> str:
    return f"""A deterministic Wizard Vision programming-brain watchdog resumed this
same session because automation reached an actionable state. This is not a
routine progress poll.

Event: {decision.kind}: {decision.reason}
Evidence:
{json.dumps(probe, indent=2, sort_keys=True)}

Continue the full senior-software-engineer brain objective from authoritative
repository and AWS state. Diagnose and take concrete action until autonomous
training is healthy again, the next required stage is running, or genuinely
new user authority is required. Preserve neuron-scoped serialization and every
accept/quarantine/replay invariant. Run proportionate tests, update the brain
configuration/reproduction documentation with durable lessons, and commit and
push all non-generated work. Do not merely report status and do not reinterpret
an expected quarantine as overall completion.

Completion still requires a deterministic obstacle course of 1,000 distinct,
representative enterprise-software tasks. Group failures by capability, repair
them with appropriately licensed training material or a causal architecture
fix, and rerun affected tasks plus full retention until all 1,000 pass. Then
integrate the independent Wizard brain selectors in CoolCryptoUtilities,
configure C0D3R V2 as Brand Dozer's agent using this brain, and complete the
Multi-Scale Robot World project. Judge that capstone independently and
critically: it is not complete until it is a world-class 3D robot-design system
with credible real-world physics and fabrication-ready designs suitable for 3D
printing. Never accept the brain's own declaration of completion as evidence.
Follow `docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md`. Publish its generated
completion marker only after every referenced authoritative report exists and
passes a fresh requirement-by-requirement audit.
"""


def invoke_codex(session_id: str, decision: Decision, probe: dict,
                 log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stdout_path = log_dir / f"codex-{stamp}.jsonl"
    stderr_path = log_dir / f"codex-{stamp}.stderr.log"
    command = [
        "codex", "exec", "resume",
        "-c", 'approval_policy="never"',
        "-c", 'sandbox_mode="danger-full-access"',
        "--json", session_id, "-",
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, \
            stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            command, cwd=ROOT, input=codex_prompt(decision, probe),
            text=True, stdout=stdout, stderr=stderr, check=False,
        )
    return result.returncode


def observe(state: dict, decision: Decision, stability_polls: int) -> tuple[dict, bool]:
    if decision.kind == "healthy":
        state.update({
            "pending_fingerprint": "", "pending_count": 0,
            "last_invoked_fingerprint": "", "last_invoked_unix": 0.0,
        })
        return state, False
    if state.get("pending_fingerprint") == decision.fingerprint:
        count = int(state.get("pending_count") or 0) + 1
    else:
        count = 1
    state.update({
        "pending_fingerprint": decision.fingerprint,
        "pending_count": count,
    })
    return state, count >= stability_polls


def cooldown_elapsed(state: dict, decision: Decision, *, now: float,
                     retry_cooldown: float) -> bool:
    if state.get("last_invoked_fingerprint") != decision.fingerprint:
        return True
    return now - float(state.get("last_invoked_unix") or 0.0) >= retry_cooldown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--runtime", default=DEFAULT_RUNTIME)
    parser.add_argument("--session-id", default=os.environ.get("CODEX_THREAD_ID", ""))
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stability-polls", type=int, default=2)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument("--retry-cooldown", type=float, default=1800.0)
    parser.add_argument(
        "--state-path", type=Path,
        default=ROOT / "runtime/programming-brain-codex-watch/state.json",
    )
    parser.add_argument(
        "--completion-marker", type=Path,
        default=(
            ROOT / "runtime/programming-brain-codex-watch/"
            "objective-complete.json"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.session_id and not args.dry_run:
        parser.error("--session-id or CODEX_THREAD_ID is required")
    if args.stability_polls < 1 or args.poll_seconds < 1:
        parser.error("poll and stability values must be positive")

    state = read_json(args.state_path)
    while True:
        if completion_marker_valid(read_json(args.completion_marker)):
            state.update({"state": "objective_complete", "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            print(json.dumps({"decision": {"kind": "objective_complete"}}))
            return 0
        try:
            probe = remote_probe(args.profile, args.instance_id, args.runtime)
            decision = classify_probe(probe, stall_seconds=args.stall_seconds)
            required_polls = (
                1 if decision.kind in {"quarantine_ready", "milestone"}
                else args.stability_polls
            )
            state, stable = observe(state, decision, required_polls)
            trigger = stable and cooldown_elapsed(
                state, decision, now=time.time(),
                retry_cooldown=args.retry_cooldown,
            )
            state.update({
                "last_probe": probe,
                "last_decision": decision.__dict__,
                "updated_unix": time.time(),
            })
            atomic_json(args.state_path, state)
            print(json.dumps({"decision": decision.__dict__, "trigger": trigger}))
            if trigger and not args.dry_run:
                returncode = invoke_codex(
                    args.session_id, decision, probe, args.state_path.parent / "logs"
                )
                state["last_codex_returncode"] = returncode
                state["last_codex_unix"] = time.time()
                if returncode == 0:
                    state["last_invoked_fingerprint"] = decision.fingerprint
                    state["last_invoked_unix"] = time.time()
                atomic_json(args.state_path, state)
        except Exception as exc:
            state.update({"probe_error": str(exc), "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            print(f"watch probe failed: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
