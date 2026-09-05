#!/usr/bin/env python3
"""Wake a Claude Code session only when programming-brain work needs it.

The curriculum supervisor, not the agent, owns healthy progress.  This watcher
polls a compact AWS status probe and resumes one explicit Claude Code session
only after the same actionable state is observed repeatedly.  Identical handled
events are deduplicated across watcher restarts.

Runs Claude headless on Opus 5 at xhigh effort with permission checks bypassed,
so an alarm at 03:00 is repaired rather than queued behind a prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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


def append_activity(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{stamp}] {message.rstrip()}\n")
        stream.flush()


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


def classify_probe(probe: dict, *, stall_seconds: float,
                   admission_stall_hours: float = 6.0,
                   memory_floor_gb: float = 1.5) -> Decision:
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
        # Motion is not progress. Measured 2026-09-05: eight clean
        # yield/recycle cycles, seven intervals advanced, 18,568 accepted
        # episodes and "0 failed" on every pass -- with zero admissions for
        # two weeks, because the gate never executed once. A watcher that
        # only checks liveness reports that as healthy and burns the budget.
        admissions = probe.get("admissions") or {}
        gate_artifacts = int(admissions.get("gate_artifacts") or 0)
        since = admissions.get("hours_since_admission")
        counts = admissions.get("event_counts") or {}
        yields = int(counts.get("deferred_replay_resource_yield") or 0)

        if gate_artifacts == 0 and yields >= 2:
            return Decision(
                "fix_required",
                f"admission gate has never produced an artifact across "
                f"{yields} resource cycles: the gate is not running, so no "
                f"interval can ever admit",
                event_fingerprint("gate_never_ran", probe),
            )
        if since is not None and float(since) > admission_stall_hours:
            return Decision(
                "fix_required",
                f"no interval admitted for {float(since):.1f}h while the "
                f"curriculum reports itself active",
                event_fingerprint("no_admission", probe),
            )

        memory = probe.get("memory") or {}
        available = float(memory.get("available_gb") or 0.0)
        if available and available < memory_floor_gb:
            return Decision(
                "fix_required",
                f"host memory available is {available:.2f} GB, below the "
                f"{memory_floor_gb:.1f} GB alarm floor",
                event_fingerprint("memory_low", probe),
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
import json, os, pathlib, sys, time
runtime = pathlib.Path({runtime!r})
sys.path.insert(0, '/srv/wizard/project')
from scripts.programming_curriculum_supervisor import (
    curriculum_phases, read_json, unresolved_deferred_intervals,
)
try:
    status = json.loads((runtime / 'curriculum-supervisor.status.json').read_text())
except Exception as exc:
    status = {{'state': 'missing', 'error': str(exc)}}
supervisors = wrappers = workers = 0
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command = path.read_bytes().replace(b'\\0', b' ').decode(errors='replace')
        process_name = (path.parent / 'comm').read_text().strip().lower()
    except OSError:
        continue
    if (process_name.startswith('python')
            and 'programming_curriculum_supervisor.py' in command
            and {runtime!r} in command):
        supervisors += 1
    if process_name == 'bash' and 'run_programming_curriculum_service.sh' in command:
        wrappers += 1
    if (process_name.startswith('python')
            and 'tools.training_standard.drive_corpora_brain' in command
            and {runtime!r} in command):
        workers += 1
updated = float(status.get('updated_unix') or 0.0)
try:
    service_stage = (runtime / 'curriculum-service-supervisor.stage').read_text().strip()
except OSError:
    service_stage = ''
include_seed = any(
    (runtime / f'{{name}}.progress.json').is_file()
    for name in ('canonical-algorithms', 'gsm8k-domain-safe')
)
phases = curriculum_phases(pathlib.Path('/srv/wizard/corpora'), include_seed)
processed = 0
phase_rows = {{}}
for phase in phases:
    progress = read_json(runtime / f'{{phase.name}}.progress.json')
    durable = min(phase.rows, max(0, int(progress.get('durable_next_row') or 0)))
    phase_rows[phase.name] = durable
    processed += durable
total = sum(phase.rows for phase in phases)
intervals = {{}}
for event in unresolved_deferred_intervals(runtime):
    phase = str(event['phase'])
    start = max(0, int(event['start_row']))
    end = min(phase_rows.get(phase, 0), int(event['end_row']))
    if end > start:
        intervals.setdefault(phase, []).append((start, end))
deferred = 0
for spans in intervals.values():
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    deferred += sum(end - start for start, end in merged)
# --- admission + memory metrics -------------------------------------------
# Every one of these was green for 16 h while nothing admitted, so the probe
# must report the ones that actually distinguish progress from motion.
health = runtime / 'curriculum-health.jsonl'
kinds = {{}}
admitted_unix = 0.0
last_yield = {{}}
recent_fail = ''
try:
    for line in health.read_text(errors='replace').splitlines()[-4000:]:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        kind = str(ev.get('kind') or '')
        kinds[kind] = kinds.get(kind, 0) + 1
        when = float(ev.get('updated_unix') or 0.0)
        if kind == 'deferred_replay_admitted':
            admitted_unix = max(admitted_unix, when)
        elif kind == 'deferred_replay_resource_yield':
            last_yield = {{
                'unix': when,
                'before_gb': round(float(ev.get('available_bytes_before') or 0) / 2**30, 2),
                'after_gb': round(float(ev.get('available_bytes_after') or 0) / 2**30, 2),
            }}
        elif kind == 'deferred_replay_failed':
            recent_fail = str(ev.get('error') or '')[:180]
except OSError:
    pass

# A gate that never RUNS logs neither pass nor failure. Counting its
# artifacts is the only way to tell "failing" from "never executed".
gate_artifacts = len(list(runtime.glob('*interval_recall*')))

meminfo = {{}}
try:
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        key, _, rest = line.partition(':')
        meminfo[key] = int(rest.split()[0])
except Exception:
    pass
brain_rss_kb = 0
brain_age_s = 0.0
for path in pathlib.Path('/proc').glob('[0-9]*/comm'):
    try:
        if path.read_text().strip() != 'w1z4rd_brain_se':
            continue
        pid = path.parent.name
        for line in (path.parent / 'status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                brain_rss_kb = int(line.split()[1])
                break
        stat = (path.parent / 'stat').read_text().rsplit(')', 1)[1].split()
        clk = os.sysconf('SC_CLK_TCK')
        boot = 0.0
        for ln in pathlib.Path('/proc/stat').read_text().splitlines():
            if ln.startswith('btime'):
                boot = float(ln.split()[1]); break
        brain_age_s = max(0.0, time.time() - (boot + float(stat[19]) / clk))
    except Exception:
        continue

throughput = {{}}
try:
    newest = max(runtime.glob('deferred-replay-*.progress.json'),
                 key=lambda f: f.stat().st_mtime, default=None)
    if newest is not None:
        prog = json.loads(newest.read_text())
        throughput = {{
            'progress_file': newest.name,
            'accepted_episodes': prog.get('accepted_episodes'),
            'durable_next_row': prog.get('durable_next_row'),
            'batch_seconds_ema': prog.get('batch_seconds_ema'),
            'current_batch_size': prog.get('current_batch_size'),
            'age_seconds': round(time.time() - newest.stat().st_mtime, 1),
        }}
except Exception:
    pass
curriculum = {{
    'total_rows': total,
    'durable_processed_rows': processed,
    'accepted_rows': max(0, processed - deferred),
    'deferred_rows': deferred,
    'forward_remaining_rows': max(0, total - processed),
    'minimum_outstanding_rows': max(0, total - processed) + deferred,
    'include_seed_corpora': include_seed,
}}
print(json.dumps({{
    'host_state': 'running', 'runtime': str(runtime), 'status': status,
    'supervisor_count': supervisors, 'wrapper_count': wrappers,
    'worker_count': workers, 'service_stage': service_stage,
    'curriculum': curriculum,
    'admissions': {{
        'event_counts': kinds,
        'last_admitted_unix': admitted_unix,
        'hours_since_admission': (
            round((time.time() - admitted_unix) / 3600.0, 1)
            if admitted_unix else None),
        'gate_artifacts': gate_artifacts,
        'last_resource_yield': last_yield,
        'last_failure': recent_fail,
    }},
    'memory': {{
        'available_gb': round(meminfo.get('MemAvailable', 0) / 2**20, 2),
        'total_gb': round(meminfo.get('MemTotal', 0) / 2**20, 2),
        'swap_total_gb': round(meminfo.get('SwapTotal', 0) / 2**20, 2),
        'brain_rss_gb': round(brain_rss_kb / 2**20, 2),
        'brain_age_seconds': round(brain_age_s, 1),
    }},
    'throughput': throughput,
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


def agent_prompt(decision: Decision, probe: dict) -> str:
    return f"""A deterministic Wizard Vision programming-brain watchdog woke this
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



def resolve_claude() -> str | None:
    """Absolute path to the Claude Code launcher, or None if not installed.

    npm installs claude on Windows as claude.cmd/claude.ps1 shims rather than
    a real binary, and subprocess without shell=True only runs actual
    binaries, so a bare "claude" raises WinError 2. Resolve the shim
    explicitly, mirroring how the Codex launcher was located before it.
    """
    direct = shutil.which("claude")
    if direct:
        return direct
    for ext in (".cmd", ".exe", ".ps1", ".bat"):
        found = shutil.which("claude" + ext)
        if found:
            return found
    return None


def format_claude_event(payload: dict) -> str:
    """Render one stream-json line as a single activity log entry.

    Claude Code's stream-json is shaped differently from Codex's event feed:
    assistant/user turns carry a `message` with a `content` list of typed
    blocks, and the run ends with a `result` envelope.
    """
    kind = str(payload.get("type") or "")
    if kind == "system":
        sub = str(payload.get("subtype") or "")
        model = payload.get("model") or ""
        return f"CLAUDE system {sub} {model}".rstrip()
    if kind == "result":
        status = "ERROR" if payload.get("is_error") else "ok"
        turns = payload.get("num_turns")
        cost = payload.get("total_cost_usd")
        parts = [f"CLAUDE result [{status}]"]
        if turns is not None:
            parts.append(f"turns={turns}")
        if isinstance(cost, (int, float)):
            parts.append(f"cost=${cost:.2f}")
        return " ".join(parts)
    if kind not in {"assistant", "user"}:
        return ""
    content = (payload.get("message") or {}).get("content")
    if not isinstance(content, list):
        return ""
    lines = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = str(block.get("type") or "")
        if btype == "text":
            text = str(block.get("text") or "").strip().replace("\r", "")
            if text:
                lines.append(f"CLAUDE MESSAGE {text}")
        elif btype == "tool_use":
            name = str(block.get("name") or "tool")
            args = block.get("input") or {}
            detail = ""
            if isinstance(args, dict):
                # Surface the part a human would actually want in a log.
                for key in ("command", "file_path", "pattern", "description"):
                    if args.get(key):
                        detail = str(args[key]).replace("\n", " ")[:200]
                        break
            lines.append(f"CLAUDE TOOL {name} {detail}".rstrip())
        elif btype == "tool_result":
            if block.get("is_error"):
                text = str(block.get("content") or "")[:200].replace("\n", " ")
                lines.append(f"CLAUDE TOOL-ERROR {text}".rstrip())
    return "\n".join(lines)


def invoke_claude(session_id: str, decision: Decision, probe: dict,
                  log_dir: Path, activity_path: Path, *,
                  model: str = "opus", effort: str = "xhigh") -> int:
    """Wake one Claude Code session to repair an actionable training fault.

    Runs headless (`-p`) with stream-json so every tool call and message lands
    in the activity log. `--resume` keeps one continuous session so the agent
    retains what it already learned about this brain across alarms; if that
    session is gone, fall back to a fresh run rather than losing the alarm.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stdout_path = log_dir / f"claude-{stamp}.jsonl"
    stderr_path = log_dir / f"claude-{stamp}.stderr.log"
    executable = resolve_claude()
    if executable is None:
        append_activity(
            activity_path,
            "CLAUDE UNAVAILABLE: no claude launcher on PATH; install with "
            "`npm i -g @anthropic-ai/claude-code`. Supervision continues; "
            "alarms are logged but cannot wake Claude.",
        )
        return 127

    base = [
        executable, "-p",
        "--model", model,
        "--effort", effort,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]
    command = base + (["--resume", session_id] if session_id else [])
    append_activity(
        activity_path,
        f"ALARM waking Claude ({model}/{effort}): {decision.reason}",
    )
    returncode = _run_claude(command, decision, probe, stdout_path,
                             stderr_path, activity_path)
    if returncode != 0 and session_id:
        # A stale/absent session id must not swallow the alarm: retry once
        # without --resume so the fault still gets worked.
        stale = stderr_path.read_text(encoding="utf-8", errors="replace")[-400:]
        append_activity(
            activity_path,
            f"CLAUDE resume failed (rc={returncode}); retrying as a fresh "
            f"session. stderr tail: {stale.strip()[-200:]}",
        )
        returncode = _run_claude(base, decision, probe, stdout_path,
                                 stderr_path, activity_path)
    append_activity(activity_path, f"CLAUDE EXIT returncode={returncode}")
    return returncode


def _run_claude(command: list[str], decision: Decision, probe: dict,
                stdout_path: Path, stderr_path: Path,
                activity_path: Path) -> int:
    with stdout_path.open("a", encoding="utf-8") as stdout, \
            stderr_path.open("a", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, cwd=ROOT, text=True, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=stderr,
        )
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(agent_prompt(decision, probe))
        process.stdin.close()
        for line in process.stdout:
            stdout.write(line)
            stdout.flush()
            try:
                activity = format_claude_event(json.loads(line))
            except (ValueError, TypeError):
                activity = ""
            if activity:
                append_activity(activity_path, activity)
        process.stdout.close()
        return process.wait()

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
    parser.add_argument(
        "--session-id",
        default=(os.environ.get("CLAUDE_SESSION_ID")
                 or os.environ.get("CODEX_THREAD_ID", "")),
        help="Claude Code session to resume so context carries across alarms.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("WIZARD_WATCH_MODEL", "opus"),
        help="Model alias passed to `claude --model`.",
    )
    parser.add_argument(
        "--effort", default=os.environ.get("WIZARD_WATCH_EFFORT", "xhigh"),
        choices=("low", "medium", "high", "xhigh", "max"),
        help="Reasoning effort passed to `claude --effort`.",
    )
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--stability-polls", type=int, default=2)
    parser.add_argument("--stall-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--admission-stall-hours", type=float, default=6.0,
        help=(
            "Alarm when no interval has admitted for this long while the "
            "curriculum reports itself active. Motion is not progress: "
            "measured 2026-09-05, eight clean resource cycles and seven "
            "intervals advanced with zero admissions for two weeks."
        ),
    )
    parser.add_argument(
        "--memory-floor-gb", type=float, default=1.5,
        help="Alarm when host available memory falls below this.",
    )
    parser.add_argument("--retry-cooldown", type=float, default=1800.0)
    parser.add_argument(
        "--state-path", type=Path,
        default=ROOT / "runtime/programming-brain-watch/state.json",
    )
    parser.add_argument(
        "--completion-marker", type=Path,
        default=(
            ROOT / "runtime/programming-brain-watch/"
            "objective-complete.json"
        ),
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    # A missing session id is no longer fatal: Claude Code can start a fresh
    # session, and refusing to run would mean no supervision at all.
    if not args.session_id and not args.dry_run:
        print("no --session-id/CLAUDE_SESSION_ID; alarms start fresh sessions",
              file=sys.stderr)
    if args.stability_polls < 1 or args.poll_seconds < 1:
        parser.error("poll and stability values must be positive")

    state = read_json(args.state_path)
    activity_path = args.state_path.parent / "activity.log"
    append_activity(
        activity_path,
        f"WATCHER START pid={os.getpid()} session={args.session_id or 'dry-run'}",
    )
    while True:
        if completion_marker_valid(read_json(args.completion_marker)):
            state.update({"state": "objective_complete", "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            print(json.dumps({"decision": {"kind": "objective_complete"}}))
            return 0
        try:
            probe = remote_probe(args.profile, args.instance_id, args.runtime)
            decision = classify_probe(
                probe, stall_seconds=args.stall_seconds,
                admission_stall_hours=args.admission_stall_hours,
                memory_floor_gb=args.memory_floor_gb,
            )
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
            status = probe.get("status") or {}
            append_activity(
                activity_path,
                f"{decision.kind.upper()} {decision.reason}; "
                f"phase={status.get('phase', '-')} "
                f"state={status.get('state', probe.get('host_state', '-'))} "
                f"row={status.get('durable_next_row', '-')} "
                f"target={status.get('block_target_row', '-')}; "
                f"curriculum={probe.get('curriculum', {}).get('durable_processed_rows', '-')}"
                f"/{probe.get('curriculum', {}).get('total_rows', '-')} "
                f"accepted={probe.get('curriculum', {}).get('accepted_rows', '-')} "
                f"quarantined={probe.get('curriculum', {}).get('deferred_rows', '-')} "
                f"forward_remaining={probe.get('curriculum', {}).get('forward_remaining_rows', '-')} "
                f"minimum_outstanding={probe.get('curriculum', {}).get('minimum_outstanding_rows', '-')}",
            )
            if trigger and not args.dry_run:
                # Waking the agent is a REMEDIATION, not part of the probe.
                # Keep its failures out of the probe's except block: a missing
                # CLI used to surface as "PROBE ERROR ... cannot find
                # the file specified" on every poll, which read as though the
                # AWS probe itself had broken and hid the numbers it had just
                # fetched successfully.
                try:
                    returncode = invoke_claude(
                        args.session_id, decision, probe,
                        args.state_path.parent / "logs", activity_path,
                        model=args.model, effort=args.effort,
                    )
                except Exception as exc:  # noqa: BLE001 - remediation only
                    returncode = 127
                    append_activity(activity_path, f"AGENT INVOKE FAILED {exc}")
                state["last_agent_returncode"] = returncode
                state["last_agent_unix"] = time.time()
                if returncode == 0:
                    state["last_invoked_fingerprint"] = decision.fingerprint
                    state["last_invoked_unix"] = time.time()
                atomic_json(args.state_path, state)
            # The probe succeeded; make sure a stale error from an earlier
            # cycle does not keep showing up in the header.
            state.pop("probe_error", None)
            atomic_json(args.state_path, state)
        except Exception as exc:
            state.update({"probe_error": str(exc), "updated_unix": time.time()})
            atomic_json(args.state_path, state)
            append_activity(activity_path, f"PROBE ERROR {exc}")
            print(f"watch probe failed: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
