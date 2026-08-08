"""Cross-layer invariants for the persistent programming-brain runtime."""

from __future__ import annotations

import json
import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.programming_integrated_retention import foundation_eval, mutation_enabled
from scripts.independent_snapshot import publish_independent_copy
from scripts.programming_brain_eval import (
    BrainClient as FoundationBrainClient,
    K12,
    OOV,
    TODDLER,
)
from scripts.programming_curriculum_supervisor import (
    AdmissionInfrastructureError,
    CanaryInfrastructureError,
    GateCommandFailure,
    Phase,
    accept_last_good_guard,
    append_deferred_event,
    assert_training_not_quarantined,
    claim_curriculum_supervisor,
    ensure_last_good_guard,
    ensure_live_last_good_guard,
    finalize_canary_restore,
    guarded_admission_due,
    guarded_block_target,
    guard_state_identity,
    deferred_interval_id,
    deferred_handoff_exit_code,
    deferred_replay_command,
    deferred_replay_marker_path,
    latest_passing_canary_row,
    disk_floor_breached,
    memory_floor_breached,
    matching_runtime_node_pids,
    recycle_settled_runtime_node,
    mark_phase_forward_harvested,
    next_suspect_start,
    phase_offsets,
    pause_admission_for_infrastructure,
    preserve_admission_evidence,
    preserve_deferred_base,
    prune_resolved_deferred_bases,
    publish,
    quarantine_interval_ids,
    read_json,
    record_deferred_failure,
    advance_guard_across_deferred_block,
    deferred_coverage_ids,
    recover_interrupted_deferred_replay,
    replay_interval_recall_command,
    recall_command,
    restore_canary_quarantine,
    responsive_batch_size,
    run_admission_json_command,
    run_admission_operation,
    run_canary_json_command,
    run_phase,
    release_supervisor_claim,
    require_snapshot_copy_headroom,
    runtime_responsive_batch_size,
    settle_brain_for_admission,
    start_runtime_node,
    stop_runtime_node,
    suspect_intervals,
    topology_delta,
    transient_gate_failure,
    unresolved_deferred_intervals,
    valid_deferred_interval,
    verify_restored_topology,
)
from scripts.programming_enterprise_retention import run_suite, stable_structure
from scripts.programming_capstone_readiness import safe_manifest, structural_checks
from scripts.programming_experiential_generalization import (
    EXPERIENCE,
    HELDOUT,
    begin_experience_transaction,
    commit_experience_transaction,
    execute as execute_experience,
    retention_passed,
)
from scripts.programming_multidomain_synthesis import (
    ALTERNATIVE_PREMISES,
    DISCIPLINES,
    HEADER as MULTIDOMAIN_HEADER,
    PREMISES as MULTIDOMAIN_PREMISES,
    PRIMARY_FEATURE,
    execute as execute_multidomain,
    execute_no_retry_contradiction,
    retain_failed_gate_report,
    training_rows as multidomain_training_rows,
)
from scripts.programming_multidomain_holdout import (
    CLASS_NAME as HOLDOUT_CLASS_NAME,
    DOMAIN_REQUIREMENTS as HOLDOUT_REQUIREMENTS,
    holdout_prompt,
    execute as execute_multidomain_holdout,
)
from tools.training_standard.drive_corpora_brain import adapt_lock_chunk_size
from scripts.programming_parameterized_fulfillment import (
    FRAGMENTS as PARAMETERIZED_FULFILLMENT_FRAGMENTS,
    render_fulfillment_fixture,
    training_rows as parameterized_fulfillment_training_rows,
)
from scripts.programming_domain_transfer_holdout import (
    CLASS_NAME as TRANSFER_CLASS_NAME,
    REQUIREMENTS as TRANSFER_REQUIREMENTS,
    transfer_prompt,
)
from scripts.train_programming_brain import (
    SEED_STAGES,
    curriculum_commands,
    finalize_production_brain,
    guard_seed_stage,
    parameterized_admission_command,
    qualification_commands,
    qualification_state_signature,
    resolve_seed_guard,
)
from scripts.programming_exec_env import (
    benchmark_tool_env,
    isolated_tool_env,
    prepare_tool_command,
)
from scripts.programming_slow_batch_microbrains import read_events
from scripts.programming_corpus_recall import accepted_responses, sample_window
from scripts.programming_mobile_runtime_eval import summarize_trials
from tools.training_standard.drive_corpora_brain import (
    append_slow_batch_event,
    checkpoint_due,
    drive_one,
    honor_pause_control,
    pause_ack_path,
    post_pretrain_batch,
    read_corpus_jsonl,
    row_is_skipped,
)


class ProgrammingRuntimeContractTests(unittest.TestCase):
    def test_runtime_node_identity_uses_brain_directory_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            matching = Mock(pid=44, info={
                "pid": 44, "name": "w1z4rd_brain_server",
            })
            matching.environ.return_value = {
                "W1Z4RD_NODE_BRAIN_DIR": str(runtime / "brain"),
            }
            other = Mock(pid=45, info={
                "pid": 45, "name": "w1z4rd_brain_server",
            })
            other.environ.return_value = {
                "W1Z4RD_NODE_BRAIN_DIR": str(runtime / "other-brain"),
            }
            with patch(
                "scripts.programming_curriculum_supervisor.psutil.process_iter",
                return_value=[other, matching],
            ):
                self.assertEqual(matching_runtime_node_pids(runtime), [44])

    def test_loading_runtime_node_blocks_duplicate_before_socket_bind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            executable = runtime / "w1z4rd_brain_server"
            executable.write_bytes(b"binary")
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.endpoint_listener_pid",
                    return_value=0,
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.matching_runtime_node_pids",
                    return_value=[321],
                ),
                patch("scripts.programming_curriculum_supervisor.subprocess.Popen") as launch,
            ):
                with self.assertRaisesRegex(RuntimeError, "already owned by PID 321"):
                    start_runtime_node(runtime, executable, "http://127.0.0.1:18095")
            launch.assert_not_called()
            self.assertEqual((runtime / "node.pid").read_text().strip(), "321")

    def test_service_wrapper_reconciles_initial_node_identity(self) -> None:
        wrapper = (
            ROOT / "scripts" / "aws" / "run_programming_curriculum_service.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("listener_pid = endpoint_listener_pid(endpoint)", wrapper)
        self.assertIn("runtime_pid = unique_runtime_node_pid(runtime)", wrapper)
        self.assertIn("listener_pid != runtime_pid", wrapper)
        self.assertIn("os.replace(temporary, pid_path)", wrapper)
        self.assertIn(
            "if not listener_pid and not runtime_pid:\n"
            "    start_runtime_node(runtime, node_bin, endpoint)",
            wrapper,
        )
        unit = (
            ROOT / "scripts" / "aws" / "wizard-curriculum-supervisor.service"
        ).read_text(encoding="utf-8")
        self.assertNotIn("wizard-brain-initial.service", unit)

    def test_stop_adopts_loading_runtime_node_when_pid_file_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "node.pid").write_text("111\n", encoding="ascii")
            process = Mock()
            process.exe.return_value = "/srv/wizard/w1z4rd_brain_server"
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.endpoint_listener_pid",
                    side_effect=[0, 0],
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.matching_runtime_node_pids",
                    return_value=[222],
                ),
                patch("scripts.programming_curriculum_supervisor.process_alive", return_value=True),
                patch("scripts.programming_curriculum_supervisor.psutil.Process", return_value=process),
            ):
                stopped = stop_runtime_node(runtime, "http://127.0.0.1:18095")
            self.assertEqual(stopped, 222)
            process.terminate.assert_called_once_with()
            process.wait.assert_called_once_with(timeout=60.0)

    def test_foundation_report_preserves_exact_oov_failure_evidence(self) -> None:
        accepted = {prompt: {"accepted"} for prompt in K12}

        def reply(_endpoint: str, prompt: str) -> dict:
            toddler = dict(TODDLER)
            if prompt in toddler:
                return {"reply": toddler[prompt]}
            if prompt in accepted:
                return {"reply": "accepted"}
            if prompt == OOV[1]:
                return {
                    "reply": "incorrect learned answer",
                    "grounding": {"outside_grounding": False},
                    "route": {"decoder": "trained_binding"},
                }
            return {
                "reply": None,
                "grounding": {"outside_grounding": True},
            }

        with patch(
            "scripts.programming_integrated_retention.chat",
            side_effect=reply,
        ):
            report = foundation_eval("http://brain", accepted)

        self.assertEqual(report["toddler"], report["toddler_total"])
        self.assertEqual(report["k12"], report["k12_total"])
        self.assertEqual(report["oov"], 2)
        failed = [row for row in report["oov_rows"] if not row["passed"]]
        self.assertEqual(failed[0]["prompt"], OOV[1])
        self.assertEqual(failed[0]["reply"], "incorrect learned answer")
        self.assertEqual(failed[0]["route"]["decoder"], "trained_binding")

    def test_memory_floor_requires_forward_durable_progress(self) -> None:
        gib = 1024 * 1024 * 1024
        self.assertFalse(memory_floor_breached(6.0, 100, 100, 100, 1 * gib))
        self.assertFalse(memory_floor_breached(6.0, 100, 101, 100, 1 * gib))
        self.assertFalse(memory_floor_breached(0.0, 100, 101, 101, 1 * gib))
        self.assertFalse(memory_floor_breached(6.0, 100, 101, 101, 7 * gib))
        self.assertTrue(memory_floor_breached(6.0, 100, 101, 101, 5 * gib))
        runtime = ROOT / "runtime"
        self.assertFalse(disk_floor_breached(8.0, runtime, 100, 100, 100, 1 * gib))
        self.assertFalse(disk_floor_breached(8.0, runtime, 100, 101, 100, 1 * gib))
        self.assertFalse(disk_floor_breached(0.0, runtime, 100, 101, 101, 1 * gib))
        self.assertFalse(disk_floor_breached(8.0, runtime, 100, 101, 101, 9 * gib))
        self.assertTrue(disk_floor_breached(8.0, runtime, 100, 101, 101, 7 * gib))

    def test_attached_worker_has_the_same_memory_settlement_contract(self) -> None:
        source = (
            ROOT / "scripts" / "programming_curriculum_supervisor.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"state": "attached_resource_settling"', source)
        self.assertIn('"kind": "attached_resource_bounded_settlement"', source)
        self.assertIn(
            "not attach_recovered and not attached_resource_settled",
            source,
        )

    def test_settled_node_recycle_requires_zero_residency_and_stable_topology(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            status = runtime / "status.json"
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 10)
            args = SimpleNamespace(
                endpoint="http://127.0.0.1:18095",
                node_bin=runtime / "brain-server",
            )
            topology = {
                "tick": 42,
                "pool_count": 3,
                "total_neurons": 100,
                "total_concepts": 50,
                "total_binding": 40,
                "total_terminals": 500,
                "resident_terminals": 0,
            }
            replacement = SimpleNamespace(pid=222)
            memory = [
                SimpleNamespace(available=4 * 1024**3),
                SimpleNamespace(available=12 * 1024**3),
            ]
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.endpoint_json",
                    side_effect=[dict(topology), dict(topology)],
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.stop_runtime_node",
                    return_value=111,
                ) as stop,
                patch(
                    "scripts.programming_curriculum_supervisor.start_runtime_node",
                    return_value=replacement,
                ) as start,
                patch(
                    "scripts.programming_curriculum_supervisor.psutil.virtual_memory",
                    side_effect=memory,
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.append_health_event"
                ) as health,
            ):
                report = recycle_settled_runtime_node(
                    args, runtime, phase, 8, status
                )

            self.assertEqual(report["old_pid"], 111)
            self.assertEqual(report["replacement_pid"], 222)
            self.assertEqual(report["available_bytes_after"], 12 * 1024**3)
            stop.assert_called_once_with(runtime, args.endpoint)
            start.assert_called_once_with(runtime, args.node_bin, args.endpoint)
            health.assert_called_once()
            self.assertEqual(
                json.loads(status.read_text(encoding="utf-8"))["state"],
                "resource_node_recycled",
            )

    def test_settled_node_recycle_refuses_resident_neurons(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 10)
            args = SimpleNamespace(
                endpoint="http://127.0.0.1:18095",
                node_bin=runtime / "brain-server",
            )
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.endpoint_json",
                    return_value={"resident_terminals": 1},
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.stop_runtime_node"
                ) as stop,
            ):
                with self.assertRaisesRegex(RuntimeError, "every neuron"):
                    recycle_settled_runtime_node(
                        args, runtime, phase, 8, runtime / "status.json"
                    )
            stop.assert_not_called()

    def test_foundation_gate_allows_saturated_host_response_window(self) -> None:
        client = FoundationBrainClient(
            "http://127.0.0.1:18095", timeout=180.0
        )
        self.assertEqual(client.conn.timeout, 180.0)
        client.conn.close()

    def test_exact_block_boundary_never_launches_one_extra_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            phase = Phase(
                "corpus", "script", runtime / "corpus.jsonl", 100
            )
            publish(runtime / "corpus.progress.json", {
                "ram_next_row": 10,
                "durable_next_row": 10,
            })
            with patch(
                "scripts.programming_curriculum_supervisor.subprocess.Popen"
            ) as worker:
                result = run_phase(
                    SimpleNamespace(),
                    phase,
                    runtime,
                    runtime / "status.json",
                    10,
                )
            self.assertEqual(result, 0)
            worker.assert_not_called()

    def test_zero_row_corpus_window_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(
                '{"prompt":"one","response":"two"}\n',
                encoding="utf-8",
            )
            self.assertEqual(
                read_corpus_jsonl(corpus, limit_rows=0), []
            )

    def test_mobile_runtime_gate_requires_latency_scope_and_idle_identity(
            self) -> None:
        base = {
            "tick": 7,
            "pool_count": 3,
            "total_neurons": 100,
            "total_concepts": 50,
            "total_binding": 40,
            "binding_pool_id": 0,
            "resident_terminals": 0,
            "total_terminals": 1000,
        }
        trial = {
            "cold_correct": True,
            "warm_correct": True,
            "deterministic": True,
            "cold_seconds": 0.8,
            "warm_seconds": 0.2,
            "before_cold": base,
            "after_cold": {**base, "resident_terminals": 80},
            "after_warm": {**base, "resident_terminals": 90},
            "after_resleep": base,
        }
        report = summarize_trials(
            [trial], base, base, 1.0, 0.5, 0.10
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["observed"]["peak_resident_fraction"], 0.09
        )
        too_broad = {
            **trial,
            "after_warm": {**base, "resident_terminals": 101},
        }
        report = summarize_trials(
            [too_broad], base, base, 1.0, 0.5, 0.10
        )
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["bounded_residency"])

    def test_canary_transport_timeout_is_not_semantic_regression(self) -> None:
        timeout = GateCommandFailure(
            ["python", "eval.py"], 1, "",
            "TimeoutError: timed out while reading response",
        )
        regression = GateCommandFailure(
            ["python", "eval.py"], 1, "",
            "novel_paraphrase executes 4/5",
        )
        missing_fixture = GateCommandFailure(
            ["python", "eval.py"], 1, "",
            "FileNotFoundError: [Errno 2] No such file or directory: corpus.jsonl",
        )
        self.assertTrue(transient_gate_failure(timeout))
        self.assertTrue(transient_gate_failure(missing_fixture))
        self.assertFalse(transient_gate_failure(regression))
        self.assertTrue(
            transient_gate_failure(
                json.JSONDecodeError("partial output", "x", 0)
            )
        )
        self.assertTrue(
            transient_gate_failure(
                RuntimeError("gate command produced no JSON: evaluator")
            )
        )

    def test_canary_transport_retry_passes_without_deferred_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            failure = subprocess.CompletedProcess(
                ["python", "eval.py"], 1, "",
                "TimeoutError: timed out",
            )
            success = subprocess.CompletedProcess(
                ["python", "eval.py"], 0, '{"passed": true}\n', "",
            )
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.subprocess.run",
                    side_effect=[failure, success],
                ) as run,
                patch(
                    "scripts.programming_curriculum_supervisor.time.sleep"
                ),
            ):
                result = run_canary_json_command(
                    runtime,
                    Phase("corpus", "script", Path("corpus.jsonl"), 100),
                    64,
                    "foundation",
                    ["python", "eval.py"],
                )
            self.assertTrue(result["passed"])
            self.assertEqual(run.call_count, 2)
            events = [
                json.loads(line)
                for line in (runtime / "curriculum-health.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                events[0]["kind"],
                "continuous_canary_infrastructure_retry",
            )
            self.assertIsNone(events[0]["passed"])
            self.assertFalse(
                (runtime / "curriculum-deferred-intervals.jsonl").exists()
            )

    def test_exhausted_canary_transport_does_not_become_runtime_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            failure = subprocess.CompletedProcess(
                ["python", "eval.py"], 1, "",
                "ConnectionResetError: peer reset the connection",
            )
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.subprocess.run",
                    return_value=failure,
                ),
                patch(
                    "scripts.programming_curriculum_supervisor.time.sleep"
                ),
            ):
                with self.assertRaises(CanaryInfrastructureError):
                    run_canary_json_command(
                        runtime,
                        Phase(
                            "corpus", "script", Path("corpus.jsonl"), 100
                        ),
                        64,
                        "foundation",
                        ["python", "eval.py"],
                        attempts=2,
                    )
            self.assertFalse(
                (runtime / "curriculum-deferred-intervals.jsonl").exists()
            )

    def test_midphase_transport_retry_uses_shared_admission_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            failure = subprocess.CompletedProcess(
                ["python", "eval.py"], 1, "",
                "ConnectionRefusedError: endpoint unavailable",
            )
            success = subprocess.CompletedProcess(
                ["python", "eval.py"], 0, '{"passed": true}\n', "",
            )
            with (
                patch(
                    "scripts.programming_curriculum_supervisor.subprocess.run",
                    side_effect=[failure, success],
                ) as run,
                patch(
                    "scripts.programming_curriculum_supervisor.time.sleep"
                ),
            ):
                result = run_admission_json_command(
                    runtime,
                    Phase("corpus", "script", Path("corpus.jsonl"), 100),
                    96,
                    "midphase_gate",
                    "enterprise",
                    ["python", "eval.py"],
                )
            self.assertTrue(result["passed"])
            self.assertEqual(run.call_count, 2)
            event = json.loads(
                (runtime / "curriculum-health.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                event["kind"], "midphase_gate_infrastructure_retry"
            )
            self.assertFalse(
                (runtime / "curriculum-deferred-intervals.jsonl").exists()
            )

    def test_idle_settlement_operation_uses_shared_admission_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            operation = Mock(
                side_effect=[TimeoutError("timed out"), {"ok": True}]
            )
            with patch(
                "scripts.programming_curriculum_supervisor.time.sleep"
            ):
                result = run_admission_operation(
                    runtime,
                    Phase("corpus", "script", Path("corpus.jsonl"), 100),
                    96,
                    "idle_settlement",
                    "checkpoint",
                    operation,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(operation.call_count, 2)
            event = json.loads(
                (runtime / "curriculum-health.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(
                event["kind"], "idle_settlement_infrastructure_retry"
            )

    def test_exhausted_admission_pause_preserves_semantic_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            phase = Phase(
                "corpus", "script", Path("corpus.jsonl"), 100
            )
            status_path = runtime / "status.json"
            pause_admission_for_infrastructure(
                runtime,
                status_path,
                phase,
                100,
                "completion_gate",
                AdmissionInfrastructureError(
                    "completion evaluator unavailable"
                ),
            )
            status = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(
                status["state"],
                "completion_gate_infrastructure_paused",
            )
            self.assertFalse(
                (runtime / "curriculum-deferred-intervals.jsonl").exists()
            )

    def test_curriculum_supervisor_claim_rejects_live_runtime_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.programming_curriculum_supervisor."
            "matching_live_supervisor_pid",
            return_value=777,
        ):
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                claim_curriculum_supervisor(Path(directory))

    def test_curriculum_supervisor_claim_replaces_stale_pid_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.programming_curriculum_supervisor."
            "matching_live_supervisor_pid",
            return_value=0,
        ), patch(
            "scripts.programming_curriculum_supervisor.process_alive",
            return_value=False,
        ):
            runtime = Path(directory)
            claim = runtime / "curriculum-supervisor.pid"
            claim.write_text("999999\n", encoding="ascii")
            self.assertEqual(claim_curriculum_supervisor(runtime), claim)
            self.assertEqual(int(claim.read_text(encoding="ascii")), os.getpid())
            release_supervisor_claim(claim, os.getpid())
            self.assertFalse(claim.exists())

    def test_slow_microbrain_replay_deduplicates_ranges_and_keeps_latest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = Path(raw) / "slow.jsonl"
            ledger.write_text("\n".join([
                json.dumps({"logical_start_row": 0, "logical_end_row": 2,
                            "max_lock_seconds": 9}),
                json.dumps({"logical_start_row": 2, "logical_end_row": 4,
                            "max_lock_seconds": 10}),
                json.dumps({"logical_start_row": 0, "logical_end_row": 2,
                            "max_lock_seconds": 11}),
            ]) + "\n", encoding="utf-8")
            events = read_events(ledger, 2)
            self.assertEqual([event["logical_start_row"] for event in events], [2, 0])
            self.assertEqual(events[1]["max_lock_seconds"], 11)

    def test_pretrain_batch_reports_exact_slowest_lock_chunk(self) -> None:
        response = {
            "ok": True,
            "max_lock_millis": 9500,
            "max_lock_chunk_index": 17,
            "max_lock_chunk_len": 1,
            "max_lock_profile_ns": {"frame_concept_lookup": 9_000_000_000},
        }
        with patch(
            "tools.training_standard.drive_corpora_brain._post",
            return_value=(True, response),
        ):
            self.assertEqual(
                post_pretrain_batch([{"frames": []}], 1),
                (True, "", 9.5, 17, 1,
                 {"frame_concept_lookup": 9_000_000_000}),
            )

    def test_block_admission_settles_and_serializes_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime = Path(raw)
            args = argparse.Namespace(endpoint="http://brain")
            phase = Phase("corpus", "script", Path("corpus.jsonl"), 1000)
            with patch(
                "scripts.programming_curriculum_supervisor.endpoint_json",
                side_effect=[
                    {"tick": 10, "resident_terminals": 1234},
                    {"tick": 10, "resident_terminals": 0},
                ],
            ) as get_json, patch(
                "scripts.programming_curriculum_supervisor.endpoint_post_json",
                side_effect=[
                    {"neurons_serialized": 42, "promotions_drained": 3},
                    {"ok": True, "path": "brain.wbrain"},
                ],
            ) as post_json:
                report = settle_brain_for_admission(
                    args, phase, runtime, 500
                )
            self.assertEqual(report["after"]["resident_terminals"], 0)
            self.assertEqual(post_json.call_args_list[0].args[1], "/brain/sleep")
            self.assertEqual(
                post_json.call_args_list[1].args[1], "/brain/checkpoint"
            )
            self.assertEqual(get_json.call_count, 2)
            artifact = runtime / "corpus.row-500.idle-settlement.json"
            self.assertTrue(artifact.is_file())

    def test_block_admission_rejects_resident_terminals_after_sleep(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            args = argparse.Namespace(endpoint="http://brain")
            phase = Phase("corpus", "script", Path("corpus.jsonl"), 1000)
            with patch(
                "scripts.programming_curriculum_supervisor.endpoint_json",
                side_effect=[
                    {"resident_terminals": 1234},
                    {"resident_terminals": 1},
                ],
            ), patch(
                "scripts.programming_curriculum_supervisor.endpoint_post_json",
                side_effect=[{"neurons_serialized": 42}, {"ok": True}],
            ):
                with self.assertRaisesRegex(RuntimeError, "retained terminals"):
                    settle_brain_for_admission(args, phase, Path(raw), 500)

    def test_checkpoint_reports_authoritative_wbrain_path(self) -> None:
        source = (ROOT / "crates/node/src/brain_api.rs").read_text(
            encoding="utf-8"
        )
        self.assertIn('dir.join("brain.wbrain")', source)
        self.assertIn('"storage": if uses_wbrain { "wbrain" } else { "bin" }', source)

    def test_deferred_interval_preserves_exact_causal_base_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.bin"
            guard.write_bytes(b"causal-base")
            base = preserve_deferred_base(runtime, "phase:100:120")
            self.assertTrue(base.samefile(guard))
            guard.unlink()
            guard.write_bytes(b"later-guard")
            self.assertEqual(base.read_bytes(), b"causal-base")

    def test_deferred_wbrain_base_links_immutable_guard_not_live_brain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            live = brain / "brain.wbrain"
            guard = brain / "brain.last-good.wbrain"
            live.write_bytes(b"mutable-candidate")
            guard.write_bytes(b"accepted-base")
            (brain / "brain.last-good.json").write_text(
                json.dumps({"guard": str(guard)}), encoding="utf-8"
            )

            base = preserve_deferred_base(runtime, "phase:120:140")
            self.assertTrue(base.samefile(guard))
            self.assertFalse(base.samefile(live))
            guard.unlink()
            live.write_bytes(b"later-candidate")
            self.assertEqual(base.read_bytes(), b"accepted-base")

    def test_deferred_interval_preserves_gate_details_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "cross-project.json").write_text(
                '{"passed": false, "reply": "candidate answer"}',
                encoding="utf-8",
            )
            (runtime / "semantic-stress.json").write_text(
                '{"passed": false, "failed_case": "integration-4"}',
                encoding="utf-8",
            )
            (runtime / "corpus.row-20.enterprise.json").write_text(
                '{"passed": false, "passed_suites": 10}',
                encoding="utf-8",
            )
            evidence = preserve_admission_evidence(
                runtime, "corpus", 20, "corpus:10:20",
                "cross-project and semantic-stress failed",
                attempt="candidate-row-20",
            )

            (runtime / "cross-project.json").write_text(
                '{"passed": true, "reply": "restored answer"}',
                encoding="utf-8",
            )
            self.assertEqual(
                json.loads((evidence / "cross-project.json").read_text(
                    encoding="utf-8"
                ))["reply"],
                "candidate answer",
            )
            self.assertEqual(
                json.loads((evidence / "semantic-stress.json").read_text(
                    encoding="utf-8"
                ))["failed_case"],
                "integration-4",
            )
            failure = read_json(evidence / "failure.json")
            self.assertEqual(failure["interval_id"], "corpus:10:20")
            self.assertEqual(failure["trained_rows"], 20)
            self.assertEqual(
                sorted(failure["captured_files"]),
                [
                    "corpus.row-20.enterprise.json",
                    "cross-project.json",
                    "semantic-stress.json",
                ],
            )

    def test_deferred_ranges_skip_only_the_half_open_suspect_rows(self) -> None:
        ranges = ((10, 20), (30, 31))
        self.assertFalse(row_is_skipped(9, ranges))
        self.assertTrue(row_is_skipped(10, ranges))
        self.assertTrue(row_is_skipped(19, ranges))
        self.assertFalse(row_is_skipped(20, ranges))
        self.assertTrue(row_is_skipped(30, ranges))
        self.assertFalse(row_is_skipped(31, ranges))

    def test_deferred_interval_ledger_must_resolve_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            interval_id = deferred_interval_id("corpus", 100, 120)
            append_deferred_event(runtime, {
                "interval_id": interval_id,
                "status": "deferred",
                "phase": "corpus",
                "start_row": 100,
                "end_row": 120,
            })
            append_deferred_event(runtime, {
                "interval_id": "other:1:2",
                "status": "deferred",
                "phase": "other",
                "start_row": 1,
                "end_row": 2,
            })
            self.assertEqual(
                [row["interval_id"] for row in unresolved_deferred_intervals(runtime, "corpus")],
                [interval_id],
            )
            append_deferred_event(runtime, {
                "interval_id": interval_id,
                "status": "resolved",
                "phase": "corpus",
            })
            self.assertEqual(unresolved_deferred_intervals(runtime, "corpus"), [])
            source = (ROOT / "scripts/programming_curriculum_supervisor.py").read_text(
                encoding="utf-8"
            )
            self.assertIn('"state": "deferred_intervals_pending"', source)
            self.assertIn("stop_runtime_node(runtime, args.endpoint)", source)
            self.assertIn("restored = restore_canary_quarantine(runtime)", source)
            self.assertIn("start_runtime_node(runtime, args.node_bin, args.endpoint)", source)
            self.assertIn('"W1Z4RD_TICK_HOUSEKEEPING": "lazy"', source)
            self.assertIn('"W1Z4RD_DEFER_PROMOTION": "1"', source)
            self.assertIn("args.restart_node_after_attach", source)
            self.assertIn('"--skip-range"', source)
            self.assertIn("and ram < block_target_row", source)
            self.assertIn("TimeoutError,", source)
            self.assertIn("urllib.error.URLError", source)

    def test_quarantine_starts_after_latest_passing_canary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "curriculum-health.jsonl").write_text(
                "\n".join((
                    json.dumps({
                        "kind": "continuous_canary", "phase": "corpus",
                        "trained_rows": 120, "passed": True,
                    }),
                    "not-json",
                    json.dumps({
                        "kind": "continuous_canary", "phase": "other",
                        "trained_rows": 180, "passed": True,
                    }),
                    json.dumps({
                        "kind": "continuous_canary", "phase": "corpus",
                        "trained_rows": 160, "passed": False,
                    }),
                    json.dumps({
                        "kind": "continuous_canary", "phase": "corpus",
                        "trained_rows": 140, "passed": True,
                    }),
                )) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(latest_passing_canary_row(runtime, "corpus", 100), 140)
            self.assertEqual(latest_passing_canary_row(runtime, "missing", 100), 100)
            self.assertEqual(
                latest_passing_canary_row(
                    runtime, "corpus", 100, before_row=140
                ),
                120,
            )
            self.assertEqual(
                latest_passing_canary_row(
                    runtime, "corpus", 100, after_unix=1.0
                ),
                100,
            )

    def test_quarantine_start_advances_past_existing_deferred_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "curriculum-health.jsonl").write_text(
                json.dumps({
                    "kind": "continuous_canary", "phase": "corpus",
                    "trained_rows": 16896, "passed": True,
                }) + "\n",
                encoding="utf-8",
            )
            for start, end in ((0, 16640), (16896, 32768)):
                append_deferred_event(runtime, {
                    "interval_id": deferred_interval_id("corpus", start, end),
                    "status": "deferred", "phase": "corpus",
                    "start_row": start, "end_row": end,
                })
            self.assertEqual(
                next_suspect_start(runtime, "corpus", 33536, 0), 32768
            )

    def test_comprehensive_failure_ignores_narrow_passing_canaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "curriculum-health.jsonl").write_text(
                json.dumps({
                    "kind": "continuous_canary", "phase": "corpus",
                    "trained_rows": 140, "passed": True,
                    "updated_unix": 20.0,
                }) + "\n",
                encoding="utf-8",
            )
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 110, 120),
                "status": "deferred", "phase": "corpus",
                "start_row": 110, "end_row": 120,
            })
            self.assertEqual(
                suspect_intervals(
                    runtime, "corpus", 160, 100,
                    canary_after_unix=10.0,
                    use_passing_canary=False,
                ),
                [(100, 110), (120, 160)],
            )

    def test_quarantine_epoch_does_not_reuse_pre_restore_canary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "curriculum-health.jsonl").write_text(
                json.dumps({
                    "kind": "continuous_canary", "phase": "corpus",
                    "trained_rows": 16896, "passed": True,
                    "updated_unix": 10.0,
                }) + "\n",
                encoding="utf-8",
            )
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 0, 16640),
                "status": "deferred", "phase": "corpus",
                "start_row": 0, "end_row": 16640,
            })
            self.assertEqual(
                next_suspect_start(
                    runtime, "corpus", 16896, 0, canary_after_unix=20.0
                ),
                16640,
            )

    def test_quarantine_does_not_jump_across_unexamined_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            for start, end in ((0, 100), (120, 200)):
                append_deferred_event(runtime, {
                    "interval_id": deferred_interval_id("corpus", start, end),
                    "status": "deferred", "phase": "corpus",
                    "start_row": start, "end_row": end,
                })
            self.assertEqual(
                next_suspect_start(runtime, "corpus", 220, 0), 100
            )
            self.assertEqual(
                suspect_intervals(runtime, "corpus", 220, 0),
                [(100, 120), (200, 220)],
            )

    def test_zero_width_deferred_interval_is_never_replayed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 20, 20),
                "status": "deferred", "phase": "corpus",
                "start_row": 20, "end_row": 20,
            })
            self.assertFalse(valid_deferred_interval({
                "phase": "corpus", "start_row": 20, "end_row": 20,
            }))
            self.assertEqual(unresolved_deferred_intervals(runtime), [])

    def test_forward_harvest_requires_complete_deferred_tail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 100)
            publish(runtime / "corpus.progress.json", {
                "ram_next_row": 20,
                "durable_next_row": 20,
            })
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 20, 60),
                "status": "deferred", "phase": "corpus",
                "start_row": 20, "end_row": 60,
            })
            with self.assertRaisesRegex(RuntimeError, "coverage stops"):
                mark_phase_forward_harvested(runtime, phase, {"row": 20})

            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 60, 100),
                "status": "deferred", "phase": "corpus",
                "start_row": 60, "end_row": 100,
            })
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted")
            publish(brain / "brain.last-good.json", {
                "phase": "corpus", "row": 20, "guard": str(guard),
            })
            report = mark_phase_forward_harvested(
                runtime, phase, {"row": 20}
            )
            self.assertTrue(report["forward_harvest_only"])
            self.assertEqual(phase_offsets(
                runtime / "corpus.progress.json"
            ), (100, 100))
            self.assertEqual(
                read_json(runtime / "corpus.progress.json")[
                    "forward_harvested_from_guard_row"
                ],
                20,
            )

    def test_same_phase_restore_can_reuse_verified_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted")
            publish(brain / "brain.last-good.json", {
                "phase": "corpus", "row": 20, "guard": str(guard),
            })
            publish(runtime / "corpus.progress.json", {
                "ram_next_row": 40, "durable_next_row": 40,
            })
            publish(runtime / "curriculum-canary-quarantine.json", {
                "state": "failed",
            })

            finalize_canary_restore(
                runtime, {"phase": "corpus", "row": 20},
                retain_guard=True,
            )

            self.assertTrue(guard.exists())
            self.assertTrue((brain / "brain.last-good.json").exists())
            self.assertFalse(
                (runtime / "curriculum-canary-quarantine.json").exists()
            )
            self.assertEqual(
                phase_offsets(runtime / "corpus.progress.json"), (20, 20)
            )

    def test_deferred_replay_command_targets_only_the_exact_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            corpus = runtime / "corpus.jsonl"
            phase = Phase("corpus", "script", corpus, 1000)
            event = {
                "interval_id": "corpus:100:120",
                "phase": "corpus",
                "start_row": 100,
                "end_row": 120,
            }
            args = SimpleNamespace(
                endpoint="http://brain",
                batch_size=256,
                lock_chunk_size=32,
                max_live_lock_seconds=8.0,
                inter_batch_yield_seconds=0.0,
                checkpoint_rows=131072,
            )
            command = deferred_replay_command(
                args, phase, runtime, event
            )
            self.assertEqual(
                command[command.index("--start-row") + 1], "100"
            )
            self.assertEqual(
                command[command.index("--limit-rows") + 1], "20"
            )
            self.assertNotIn("--skip-range", command)
            self.assertIn("deferred-replay-", " ".join(command))

            recall = replay_interval_recall_command(
                args, phase, runtime, event
            )
            self.assertEqual(
                recall[recall.index("--start-row") + 1], "100"
            )
            self.assertEqual(
                recall[recall.index("--window-rows") + 1], "20"
            )

    def test_deferred_replay_is_guarded_and_comprehensively_admitted(self) -> None:
        source = (
            ROOT / "scripts" / "programming_curriculum_supervisor.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"state": "deferred_replay_training"', source)
        self.assertIn("restore_rejected_deferred_replay(", source)
        self.assertIn(
            "run_completion_gate(\n"
            "                args, phase, runtime, frozenset({interval_id})",
            source,
        )
        self.assertIn(
            '"final-brain deferred replay passed comprehensive admission"',
            source,
        )

    def test_committed_deferred_replay_recovers_without_duplicate_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.bin"
            guard.write_bytes(b"accepted-before-replay")
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus",
                "row": 1000,
                "guard": str(guard),
            }), encoding="utf-8")
            event = {
                "interval_id": "corpus:100:120",
                "phase": "corpus",
                "start_row": 100,
                "end_row": 120,
                "status": "deferred",
            }
            append_deferred_event(runtime, event)
            publish(deferred_replay_marker_path(runtime), {
                "state": "admitted",
                "phase": "corpus",
                "interval_id": event["interval_id"],
                "interval": event,
            })
            recover_interrupted_deferred_replay(
                SimpleNamespace(), runtime,
                {"corpus": Phase(
                    "corpus", "script", runtime / "corpus.jsonl", 1000
                )},
            )
            self.assertEqual(unresolved_deferred_intervals(runtime), [])
            self.assertFalse(guard.exists())
            self.assertFalse(deferred_replay_marker_path(runtime).exists())

    def test_deferred_failure_attributes_only_new_epoch_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.bin"
            guard.write_bytes(b"accepted")
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus",
                "row": 0,
                "guard": str(guard),
                "created_unix": 20.0,
            }), encoding="utf-8")
            (runtime / "curriculum-health.jsonl").write_text(json.dumps({
                "kind": "continuous_canary",
                "phase": "corpus",
                "trained_rows": 16896,
                "passed": True,
                "updated_unix": 10.0,
            }) + "\n", encoding="utf-8")
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("corpus", 0, 16640),
                "status": "deferred", "phase": "corpus",
                "start_row": 0, "end_row": 16640,
            })
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 100000)
            event = record_deferred_failure(
                runtime, phase, 16896, 16896,
                "novel_paraphrase executes 4/5",
                "continuous_canary_failed",
            )
            self.assertEqual(event["start_row"], 16640)
            self.assertEqual(event["end_row"], 16896)
            self.assertNotEqual(event["start_row"], event["end_row"])

    def test_fully_deferred_block_advances_only_logical_guard_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted-neural-state")
            metadata = brain / "brain.last-good.json"
            proof = {"topology": {"tick": 42, "total_neurons": 7}}
            metadata.write_text(json.dumps({
                "phase": "corpus", "row": 0, "guard": str(guard),
                "storage": "wbrain", "checkpoint_proof": proof,
            }), encoding="utf-8")
            for start, end in ((0, 40), (40, 75), (75, 100)):
                append_deferred_event(runtime, {
                    "interval_id": deferred_interval_id("corpus", start, end),
                    "status": "deferred", "phase": "corpus",
                    "start_row": start, "end_row": end,
                })
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 200)
            before = guard.read_bytes()
            report = advance_guard_across_deferred_block(
                runtime, phase, 100
            )
            self.assertIsNotNone(report)
            self.assertFalse(report["neural_state_changed"])
            self.assertEqual(guard.read_bytes(), before)
            advanced = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(advanced["row"], 100)
            self.assertEqual(advanced["checkpoint_proof"], proof)
            self.assertEqual(len(report["deferred_interval_ids"]), 3)
            self.assertEqual(
                deferred_coverage_ids(runtime, "corpus", 0, 100),
                report["deferred_interval_ids"],
            )

    def test_deferred_block_refuses_to_advance_across_a_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted")
            metadata = brain / "brain.last-good.json"
            metadata.write_text(json.dumps({
                "phase": "corpus", "row": 0, "guard": str(guard),
            }), encoding="utf-8")
            for start, end in ((0, 40), (50, 100)):
                append_deferred_event(runtime, {
                    "interval_id": deferred_interval_id("corpus", start, end),
                    "status": "deferred", "phase": "corpus",
                    "start_row": start, "end_row": end,
                })
            phase = Phase("corpus", "script", runtime / "corpus.jsonl", 200)
            self.assertIsNone(deferred_coverage_ids(runtime, "corpus", 0, 100))
            self.assertIsNone(advance_guard_across_deferred_block(
                runtime, phase, 100
            ))
            self.assertEqual(
                json.loads(metadata.read_text(encoding="utf-8"))["row"], 0
            )

    def test_deferred_failure_records_disjoint_uncovered_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.bin"
            guard.write_bytes(b"accepted")
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 0,
                "guard": str(guard), "created_unix": 20.0,
            }), encoding="utf-8")
            for start, end in ((0, 100), (120, 200)):
                append_deferred_event(runtime, {
                    "interval_id": deferred_interval_id("corpus", start, end),
                    "status": "deferred", "phase": "corpus",
                    "start_row": start, "end_row": end,
                })
            event = record_deferred_failure(
                runtime, Phase("corpus", "script", Path("x"), 1000),
                220, 220, "regression", "continuous_canary_failed",
            )
            self.assertEqual(event["suspect_intervals"], [
                {"start_row": 100, "end_row": 120},
                {"start_row": 200, "end_row": 220},
            ])
            recorded = unresolved_deferred_intervals(runtime, "corpus")
            self.assertEqual(
                [(row["start_row"], row["end_row"]) for row in recorded],
                [(0, 100), (100, 120), (120, 200), (200, 220)],
            )

    def test_quarantine_retest_owns_exact_disjoint_interval_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            corpus = runtime / "corpus.jsonl"
            corpus.write_text("{}\n", encoding="utf-8")
            quarantine = {
                "deferred_events": [
                    {"interval_id": "phase:100:120"},
                    {"interval_id": "phase:200:220"},
                    {"interval_id": "phase:100:120"},
                ],
                "interval_id": "legacy:1:2",
            }
            self.assertEqual(quarantine_interval_ids(quarantine), [
                "phase:100:120", "phase:200:220", "legacy:1:2",
            ])
            for interval_id, start, end in (
                ("phase:0:100", 0, 100),
                ("phase:100:120", 100, 120),
                ("phase:120:200", 120, 200),
                ("phase:200:220", 200, 220),
            ):
                append_deferred_event(runtime, {
                    "interval_id": interval_id,
                    "status": "deferred",
                    "phase": "phase",
                    "start_row": start,
                    "end_row": end,
                })
            args = argparse.Namespace(endpoint="http://127.0.0.1:1")
            command = recall_command(
                args,
                Phase("phase", "script", corpus, 220),
                runtime,
                220,
                32,
                frozenset({"phase:100:120", "phase:200:220"}),
            )
            skip_ranges = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--skip-range"
            ]
            self.assertEqual(skip_ranges, ["0:100", "120:200"])

    def test_continuous_canary_attributes_concurrent_topology_growth(self) -> None:
        self.assertEqual(
            topology_delta(
                {"tick": 10, "total_neurons": 20, "total_binding": 3},
                {"tick": 14, "total_neurons": 25, "total_binding": 5},
            ),
            {
                "tick": 4,
                "pool_count": 0,
                "total_neurons": 5,
                "total_concepts": 0,
                "total_binding": 2,
                "total_terminals": 0,
            },
        )

    def test_standalone_server_honors_shared_brain_directory_precedence(self) -> None:
        source = (ROOT / "crates/node/src/bin/brain_server.rs").read_text(
            encoding="utf-8"
        )
        main = source[source.index("async fn main()") :]
        self.assertIn("brain_api::default_node_brain_dir()", main)
        self.assertNotIn("let data = data_dir();", main)

    def test_wal_durable_training_still_bounds_snapshot_tail(self) -> None:
        self.assertFalse(checkpoint_due(4096, 4095))
        self.assertTrue(checkpoint_due(4096, 4096))
        self.assertFalse(checkpoint_due(0, 100_000))

    def test_read_only_retention_cannot_enter_training_branch(self) -> None:
        self.assertFalse(mutation_enabled(read_only=True))
        self.assertTrue(mutation_enabled(read_only=False))

    def test_supervisor_status_survives_transient_windows_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "status.json"
            import scripts.programming_curriculum_supervisor as supervisor
            real_replace = supervisor.os.replace
            attempts = 0

            def transient_lock(source: Path, destination: Path) -> None:
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise PermissionError("simulated reader lock")
                real_replace(source, destination)

            with patch.object(supervisor.os, "replace",
                              side_effect=transient_lock), \
                    patch.object(supervisor.time, "sleep"):
                publish(target, {"ram_next_row": 12, "durable_next_row": 10})
            self.assertEqual(phase_offsets(target), (12, 10))
            self.assertEqual(attempts, 3)

    def test_enterprise_suite_failure_is_preserved(self) -> None:
        result = run_suite("failure", ["-c", "import sys; sys.exit(7)"], 5)
        self.assertFalse(result["passed"])
        self.assertEqual(result["exit_code"], 7)

    def test_enterprise_suite_timeout_is_preserved(self) -> None:
        result = run_suite(
            "timeout", ["-c", "import time; time.sleep(1)"], 0.01,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["timed_out"])

    def test_enterprise_structure_guard_ignores_cache_residency(self) -> None:
        baseline = {
            "pool_count": 13, "total_neurons": 100,
            "total_concepts": 25, "total_binding": 7,
            "binding_pool_id": 0, "resident_terminals": 500,
            "evicted_neurons": 40,
        }
        paged_in = dict(
            baseline, resident_terminals=800, evicted_neurons=35,
        )
        self.assertEqual(stable_structure(baseline), stable_structure(paged_in))
        rewired = dict(baseline, total_binding=8)
        self.assertNotEqual(stable_structure(baseline), stable_structure(rewired))

    def test_compiler_caches_are_confined_to_benchmark_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = isolated_tool_env(root)
            for key in ("GOCACHE", "GOMODCACHE", "DOTNET_CLI_HOME", "NUGET_PACKAGES"):
                self.assertTrue(Path(environment[key]).is_relative_to(root))
                self.assertTrue(Path(environment[key]).is_dir())
            self.assertEqual(
                environment["DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE"], "1"
            )
            self.assertEqual(environment["NUGET_CERT_REVOCATION_MODE"], "offline")
            self.assertEqual(environment["RestoreIgnoreFailedSources"], "true")

    def test_shared_compiler_cache_stays_inside_repository_runtime(self) -> None:
        environment = benchmark_tool_env()
        runtime = ROOT / "runtime" / "benchmark-tool-cache"
        for key in ("GOCACHE", "GOMODCACHE", "DOTNET_CLI_HOME", "NUGET_PACKAGES"):
                self.assertTrue(Path(environment[key]).is_relative_to(runtime))

    def test_csharp_evaluator_forces_offline_package_sources(self) -> None:
        source = (
            ROOT / "scripts" / "programming_multilanguage_eval.py"
        ).read_text(encoding="utf-8")
        self.assertIn('(work / "NuGet.Config").write_text(', source)
        self.assertIn("<packageSources><clear />", source)

    def test_shared_dotnet_runner_forces_offline_restore_and_no_servers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = prepare_tool_command(
                ("dotnet", "run", "--project", "Eval.csproj"), root
            )
            self.assertIn("--disable-build-servers", command)
            config = root / "NuGet.Config"
            self.assertTrue(config.is_file())
            self.assertIn(
                "<packageSources><clear />",
                config.read_text(encoding="utf-8"),
            )

    def test_semantic_stress_fails_on_any_missing_recall(self) -> None:
        source = (ROOT / "scripts" / "programming_semantic_stress.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('all(row["recalled"] for row in rows)', source)

    def test_code_intent_inhibits_cross_domain_raw_fallback(self) -> None:
        source = (ROOT / "crates/node/src/brain_api.rs").read_text(encoding="utf-8")
        self.assertIn("has_programming_language_intent", source)
        self.assertIn("programming_response_compatible", source)
        self.assertIn("&& !raw_programming_compatible", source)
        self.assertIn("directly_underspecified", source)
        self.assertIn('"raw_fallback_inhibited"', source)

    def test_capstone_safety_rejects_prose_and_requires_kernel_boundaries(self) -> None:
        self.assertEqual(safe_manifest("an unrelated math answer"), {})
        manifest = safe_manifest(json.dumps({"files": {
            "tsconfig.json": '{"compilerOptions":{"strict":true}}',
            "src/physics/kernel.ts": "SI units CODATA gravity electrostatic symplectic collision conservation validity error budget refine coarsen deterministic inverse",
            "src/render/three.ts": "Three instancing LOD origin worker Float64Array budget",
            "tests/kernel.test.ts": "deterministic inverse test",
            "README.md": "roadmap limitations",
        }}))
        checks = structural_checks(manifest)
        self.assertTrue(checks["renderer_separated"])
        self.assertTrue(all(checks.values()))

    def test_enterprise_retention_includes_capstone_safety(self) -> None:
        source = (ROOT / "scripts/programming_enterprise_retention.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('("capstone_safety"', source)

    def test_experiential_fixture_requires_repair_and_transfers_relation(self) -> None:
        self.assertFalse(execute_experience(EXPERIENCE, EXPERIENCE.broken)[0])
        self.assertTrue(execute_experience(EXPERIENCE, EXPERIENCE.corrected)[0])
        self.assertFalse(execute_experience(HELDOUT, HELDOUT.broken)[0])
        self.assertTrue(execute_experience(HELDOUT, HELDOUT.corrected)[0])
        self.assertNotEqual(EXPERIENCE.function, HELDOUT.function)
        self.assertNotEqual(EXPERIENCE.factor, HELDOUT.factor)
        self.assertNotEqual(EXPERIENCE.offset, HELDOUT.offset)

    def test_experiential_admission_requires_all_protected_retention(self) -> None:
        report = {"after_debug": {
            "foundation": {"toddler": 32, "toddler_total": 32,
                           "k12": 16, "k12_total": 16,
                           "oov": 3, "oov_total": 3},
            "python": {"summary": {
                "trained": {"executes": 5, "syntax_valid": 5, "count": 5},
                "novel": {"executes": 5, "syntax_valid": 5, "count": 5},
            }},
            "debug": {"transfer": {"passed": 4, "total": 4}},
        }}
        self.assertTrue(retention_passed(report))
        report["after_debug"]["debug"]["transfer"]["passed"] = 3
        self.assertFalse(retention_passed(report))

    def test_experiential_training_keeps_guard_until_admitted_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            snapshot = runtime / "brain" / "brain.bin"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"accepted-before-experience")
            with patch(
                "scripts.programming_experiential_generalization.request",
                return_value={"ok": True, "path": str(snapshot), "tick": 41},
            ):
                guard, metadata = begin_experience_transaction("http://brain", runtime)
            self.assertEqual(guard.read_bytes(), snapshot.read_bytes())
            self.assertTrue(metadata.is_file())
            self.assertGreaterEqual(snapshot.stat().st_nlink, 2)
            with patch(
                "scripts.programming_experiential_generalization.request",
                return_value={"tick": 41},
            ):
                resumed = begin_experience_transaction("http://brain", runtime)
            self.assertEqual(resumed, (guard, metadata))
            with patch(
                "scripts.programming_experiential_generalization.request",
                return_value={"ok": True, "path": str(snapshot), "tick": 47},
            ):
                committed = commit_experience_transaction(
                    "http://brain", guard, metadata
                )
            self.assertEqual(committed["tick"], 47)
            self.assertFalse(guard.exists())
            self.assertFalse(metadata.exists())

    def test_experiential_wbrain_guard_is_an_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            snapshot = runtime / "brain" / "brain.wbrain"
            snapshot.parent.mkdir()
            snapshot.write_bytes(b"accepted-container")
            with patch(
                "scripts.programming_experiential_generalization.request",
                return_value={"ok": True, "path": str(snapshot), "tick": 41},
            ):
                guard, metadata = begin_experience_transaction(
                    "http://brain", runtime
                )
            self.assertEqual(guard.name, "brain.experience-last-good.wbrain")
            self.assertFalse(guard.samefile(snapshot))
            snapshot.write_bytes(b"mutated-container")
            self.assertEqual(guard.read_bytes(), b"accepted-container")
            recorded = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(recorded["storage"], "wbrain")
            self.assertIn(recorded["guard_mode"], {"copy", "reflink"})

    def test_experiential_batch_uses_deployed_bulk_route(self) -> None:
        source = (ROOT / "scripts/programming_experiential_generalization.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"/brain/pretrain_bindings"', source)
        self.assertNotIn('"/brain/pretrain/batch"', source)

    def test_multidomain_fixture_requires_twelve_independent_premises(self) -> None:
        self.assertEqual(len(DISCIPLINES), 12)
        self.assertEqual(len({premise.name for premise in DISCIPLINES}), 12)
        self.assertEqual(set(PRIMARY_FEATURE), {premise.name for premise in DISCIPLINES})
        complete = "".join(
            premise.source for premise in MULTIDOMAIN_HEADER + MULTIDOMAIN_PREMISES
        )
        self.assertTrue(execute_multidomain(complete)[0])

        self.assertFalse(execute_no_retry_contradiction(complete)[0])
        no_retry = "".join(
            premise.source for premise in MULTIDOMAIN_HEADER
        ) + "".join(
            (ALTERNATIVE_PREMISES[0].source
             if premise.name == "async_retry" else premise.source)
            for premise in MULTIDOMAIN_PREMISES
        )
        self.assertTrue(execute_no_retry_contradiction(no_retry)[0])
        responses = [response for _, response in multidomain_training_rows()]
        self.assertTrue(all(complete not in response for response in responses))
        source = (ROOT / "scripts/programming_multidomain_synthesis.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("active_training_pids", source)
        self.assertIn('"concurrent_mutation_detected"', source)

    def test_multidomain_holdout_changes_domain_and_is_causally_ablatable(self) -> None:
        self.assertEqual(HOLDOUT_CLASS_NAME, "ResilientFulfillmentService")
        self.assertEqual(set(HOLDOUT_REQUIREMENTS), {
            premise.name for premise in DISCIPLINES
        })
        full = holdout_prompt()
        self.assertIn(HOLDOUT_CLASS_NAME, full)
        self.assertNotIn("AdaptiveCoordinator", full)
        for name, requirement in HOLDOUT_REQUIREMENTS.items():
            self.assertIn(requirement, full)
            self.assertNotIn(requirement, holdout_prompt(name))

    def test_parameterized_fulfillment_motif_executes_unseen_symbols(self) -> None:
        for class_name, method_name in [
            ("ResilientFulfillmentService", "fulfill"),
            ("DurableWarehouseEngine", "allocate_order"),
        ]:
            source = render_fulfillment_fixture(class_name, method_name)
            self.assertTrue(
                execute_multidomain_holdout(source, class_name, method_name)[0]
            )
        motif_rows = parameterized_fulfillment_training_rows()
        self.assertEqual(len(motif_rows), len(PARAMETERIZED_FULFILLMENT_FRAGMENTS) * 2)
        self.assertTrue(all("inventory fulfillment domain" in prompt
                            for prompt, _ in motif_rows[:len(PARAMETERIZED_FULFILLMENT_FRAGMENTS)]))
        self.assertTrue(all("inventory fulfillment domain" not in prompt
                            for prompt, _ in motif_rows[len(PARAMETERIZED_FULFILLMENT_FRAGMENTS):]))
        responses = [response for _, response in motif_rows]
        self.assertTrue(all("class ResilientFulfillmentService" not in response
                            for response in responses))
        self.assertTrue(all("class DurableWarehouseEngine" not in response
                            for response in responses))
        supervisor = (ROOT / "scripts/programming_parameterized_fulfillment.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"concurrent_mutation_detected"', supervisor)

    def test_domain_transfer_holdout_changes_state_contract(self) -> None:
        self.assertEqual(TRANSFER_CLASS_NAME, "ResilientJobScheduler")
        self.assertEqual(set(TRANSFER_REQUIREMENTS), {
            premise.name for premise in DISCIPLINES
        })
        full = transfer_prompt()
        self.assertIn("capacity 10", full)
        self.assertIn("method named schedule", full)
        self.assertNotIn("Fulfillment", full)
        self.assertNotIn("inventory initialized", full)
        for name, requirement in TRANSFER_REQUIREMENTS.items():
            self.assertIn(requirement, full)
            self.assertNotIn(requirement, transfer_prompt(name))

    def test_multidomain_failed_gate_keeps_authoritative_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "gate.json"
            output.write_text('{"passed":false,"passed_suites":11}', encoding="utf-8")

            def failed() -> dict:
                raise RuntimeError("gate failed")

            report = retain_failed_gate_report(failed, output)
            self.assertFalse(report["passed"])
            self.assertEqual(report["passed_suites"], 11)

    def test_phase_completion_gate_includes_strict_enterprise_retention(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("programming_enterprise_retention.py", source)
        self.assertIn('enterprise.get("tick_delta") != 0', source)
        self.assertIn('enterprise.get("structure_unchanged") is not True', source)

    def test_corpus_sampler_covers_both_ends_of_trained_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "corpus.jsonl"
            corpus.write_text(
                "".join(
                    f'{{"prompt":"p{index}","response":"r{index}"}}\n'
                    for index in range(10)
                ),
                encoding="utf-8",
            )
            probes, rows = sample_window(corpus, 2, 6, 3)
            self.assertEqual(rows, 6)
            self.assertEqual([row["prompt"] for row in probes], ["p2", "p4", "p7"])
            probes, rows = sample_window(corpus, 2, 6, 3, ((3, 6),))
            self.assertEqual(rows, 6)
            self.assertEqual([row["prompt"] for row in probes], ["p2", "p6", "p7"])

    def test_corpus_recall_accepts_prior_durable_supervision(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime = Path(raw)
            current = runtime / "current.jsonl"
            prior = runtime / "prior.jsonl"
            current.write_text(
                '{"prompt":"same","response":"new wording"}\n', encoding="utf-8"
            )
            prior.write_text(
                '{"prompt":"same","response":"retained wording"}\n', encoding="utf-8"
            )
            accepted = accepted_responses([current, prior], {"same"})
            self.assertEqual(
                accepted["same"], {"new wording", "retained wording"}
            )
            (runtime / "prior.progress.json").write_text(json.dumps({
                "corpus": str(prior), "durable_next_row": 1,
            }), encoding="utf-8")
            args = argparse.Namespace(endpoint="http://brain")
            phase = Phase("current", "reasoning", current, 1)
            command = recall_command(args, phase, runtime, 1, 1)
            self.assertIn("--accepted-corpus", command)
            self.assertIn(str(prior.resolve()), command)
            append_deferred_event(runtime, {
                "interval_id": deferred_interval_id("current", 0, 1),
                "status": "deferred", "phase": "current",
                "start_row": 0, "end_row": 1,
            })
            command = recall_command(args, phase, runtime, 1, 1)
            self.assertIn("--skip-range", command)
            self.assertIn("0:1", command)

    def test_direct_pretrain_is_chunked_between_retention_gates(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--limit-rows", str(max(0, block_target_row - ram))', source)
        self.assertIn("guarded_block_target", source)
        self.assertIn("admit_midphase_candidate(", source)
        self.assertIn("guarded_admission_due(", source)
        self.assertIn('"--no-checkpoint"', source)
        self.assertIn('"--gate-rows", type=int, default=131072', source)
        self.assertIn('"--checkpoint-rows", type=int, default=131072', source)

    def test_dedicated_corpus_supervisor_preserves_live_inference_windows(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--batch-size", type=int, default=32', source)
        self.assertIn('"--inter-batch-yield-seconds", type=float, default=0.0', source)
        self.assertIn('"--max-batch-seconds", str(args.max_live_lock_seconds)', source)
        driver = (ROOT / "tools/training_standard/drive_corpora_brain.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("adaptive_batch_reductions", driver)
        self.assertIn("adapt_lock_chunk_size(", driver)
        self.assertIn("adaptive_lock_increases", driver)
        self.assertIn("--initial-lock-chunk-size", driver)
        self.assertIn('"lock_chunk_size": lock_chunk_size', driver)
        self.assertIn('previous_progress.get("max_batch_seconds"', driver)

    def test_continuous_canaries_quarantine_before_more_training_is_admitted(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("run_continuous_canary", source)
        self.assertIn('"curriculum-health.jsonl"', source)
        self.assertIn('"state": "continuous_canary_failed"', source)
        self.assertIn('"suspect_start_row": suspect_start', source)
        self.assertIn('"suspect_end_row": suspect_end', source)
        self.assertIn('"suspect_intervals": [', source)
        self.assertIn("failed_row = max(", source)
        self.assertIn("worker.terminate()", source)
        self.assertIn("if code == 86:", source)

    def test_persisted_canary_quarantine_blocks_supervisor_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            (runtime / "curriculum-canary-quarantine.json").write_text(
                json.dumps({
                    "state": "continuous_canary_failed",
                    "phase": "corpus",
                    "candidate_row": 200,
                    "last_good": {"row": 100},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "restore the guarded snapshot"):
                assert_training_not_quarantined(runtime)

    def test_canary_quarantine_restore_rewinds_snapshot_wal_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.bin"
            snapshot.write_bytes(b"accepted")
            guard = brain / "brain.last-good.bin"
            guard.hardlink_to(snapshot)
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 100,
            }), encoding="utf-8")
            snapshot.unlink()
            snapshot.write_bytes(b"rejected-candidate")
            (brain / "brain.wal").write_bytes(b"rejected-wal")
            progress = runtime / "corpus.progress.json"
            progress.write_text(json.dumps({
                "ram_next_row": 200, "durable_next_row": 200,
                "max_batch_seconds": 7.0,
            }), encoding="utf-8")
            (runtime / "curriculum-canary-quarantine.json").write_text(
                json.dumps({
                    "phase": "corpus", "candidate_row": 200,
                    "last_good": {"phase": "corpus", "row": 100},
                }),
                encoding="utf-8",
            )
            health = runtime / "curriculum-health.jsonl"
            health.write_text(json.dumps({
                "passed": False,
                "suspect_start_row": 120,
                "suspect_end_row": 200,
            }) + "\n", encoding="utf-8")
            restored = restore_canary_quarantine(runtime)
            self.assertEqual(restored["row"], 100)
            self.assertEqual(snapshot.read_bytes(), b"accepted")
            self.assertFalse((brain / "brain.wal").exists())
            self.assertFalse(guard.exists())
            self.assertFalse((runtime / "curriculum-canary-quarantine.json").exists())
            self.assertIn('"suspect_start_row": 120', health.read_text(encoding="utf-8"))
            rewound = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(rewound["ram_next_row"], 100)
            self.assertEqual(rewound["durable_next_row"], 100)
            self.assertEqual(rewound["max_batch_seconds"], 7.0)

    def test_chunk_snapshot_guard_survives_until_explicit_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.bin"
            snapshot.write_bytes(b"accepted-state")
            phase = Phase("phase-a", "script-a", runtime / "corpus.jsonl", 10)
            guard = ensure_last_good_guard(runtime, phase, 4)
            self.assertTrue(guard.exists())
            self.assertTrue(snapshot.samefile(guard))
            snapshot.unlink()
            snapshot.write_bytes(b"candidate-state")
            self.assertEqual(guard.read_bytes(), b"accepted-state")
            self.assertEqual(ensure_last_good_guard(runtime, phase, 6), guard)
            accept_last_good_guard(runtime)
            self.assertFalse(guard.exists())

    def test_wbrain_guard_is_an_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.wbrain"
            snapshot.write_bytes(b"accepted-container")
            phase = Phase("phase-a", "script-a", runtime / "corpus.jsonl", 10)
            guard = ensure_last_good_guard(runtime, phase, 4)
            self.assertEqual(guard.name, "brain.last-good.wbrain")
            self.assertFalse(snapshot.samefile(guard))
            snapshot.write_bytes(b"rejected-container")
            self.assertEqual(guard.read_bytes(), b"accepted-container")
            metadata = json.loads(
                (brain / "brain.last-good.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["storage"], "wbrain")
            self.assertIn(metadata["guard_mode"], {"copy", "reflink"})
            accept_last_good_guard(runtime)
            self.assertFalse(guard.exists())

    def test_wbrain_guard_reserves_one_independent_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.wbrain"
            snapshot.write_bytes(b"accepted-container")
            phase = Phase("phase-a", "script-a", runtime / "corpus.jsonl", 10)
            with patch(
                "scripts.programming_curriculum_supervisor."
                "require_snapshot_copy_headroom"
            ) as headroom:
                ensure_last_good_guard(runtime, phase, 4)
            headroom.assert_called_once_with(
                snapshot,
                copies=1,
                operation="independent .wbrain guard",
            )

    def test_reflink_guard_is_independent_without_full_copy_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "brain.wbrain"
            guard = root / "brain.last-good.wbrain"
            source.write_bytes(b"accepted-container")

            def emulate_reflink(origin: Path, destination: Path) -> bool:
                destination.write_bytes(origin.read_bytes())
                return True

            with patch(
                "scripts.independent_snapshot._clone_reflink",
                side_effect=emulate_reflink,
            ):
                headroom = Mock(side_effect=AssertionError("must not run"))
                mode = publish_independent_copy(
                    source,
                    guard,
                    operation="test reflink guard",
                    require_full_copy_headroom=headroom,
                )

            self.assertEqual(mode, "reflink")
            self.assertFalse(source.samefile(guard))
            source.write_bytes(b"candidate-container")
            self.assertEqual(guard.read_bytes(), b"accepted-container")
            headroom.assert_not_called()

    def test_unsupported_reflink_uses_headroom_checked_copy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "brain.wbrain"
            guard = root / "brain.last-good.wbrain"
            source.write_bytes(b"accepted-container")

            with patch(
                "scripts.independent_snapshot._clone_reflink",
                return_value=False,
            ):
                headroom = Mock()
                mode = publish_independent_copy(
                    source,
                    guard,
                    operation="test copy guard",
                    require_full_copy_headroom=headroom,
                )

            self.assertEqual(mode, "copy")
            headroom.assert_called_once_with(
                source, 1, "test copy guard"
            )
            self.assertEqual(guard.read_bytes(), source.read_bytes())

    def test_snapshot_publication_preserves_posix_service_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "brain.wbrain"
            guard = root / "brain.last-good.wbrain"
            source.write_bytes(b"accepted-container")

            with (
                patch(
                    "scripts.independent_snapshot._clone_reflink",
                    return_value=False,
                ),
                patch("scripts.independent_snapshot.os.name", "posix"),
                patch(
                    "scripts.independent_snapshot.os.chown", create=True
                ) as chown,
            ):
                publish_independent_copy(
                    source,
                    guard,
                    operation="test owner preservation",
                    require_full_copy_headroom=Mock(),
                )

            source_stat = source.stat()
            chown.assert_called_once_with(
                guard.with_suffix(guard.suffix + ".tmp"),
                source_stat.st_uid,
                source_stat.st_gid,
            )

    def test_snapshot_copy_refuses_to_consume_rollback_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "brain.wbrain"
            source.write_bytes(b"accepted-container")
            with patch(
                "scripts.programming_curriculum_supervisor.shutil.disk_usage",
                return_value=SimpleNamespace(free=1),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "insufficient disk headroom"
                ):
                    require_snapshot_copy_headroom(
                        source, copies=2, operation="test guard"
                    )

    def test_resolved_deferred_bases_are_pruned_without_touching_active_or_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted-container")
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 0, "guard": str(guard),
            }), encoding="utf-8")

            first_id = deferred_interval_id("corpus", 0, 10)
            second_id = deferred_interval_id("corpus", 10, 20)
            first = preserve_deferred_base(runtime, first_id)
            second = preserve_deferred_base(runtime, second_id)
            append_deferred_event(runtime, {
                "interval_id": first_id, "phase": "corpus",
                "start_row": 0, "end_row": 10, "status": "deferred",
                "base_snapshot": str(first),
            })
            append_deferred_event(runtime, {
                "interval_id": second_id, "phase": "corpus",
                "start_row": 10, "end_row": 20, "status": "deferred",
                "base_snapshot": str(second),
            })
            replacement_id = deferred_interval_id("corpus", 2, 8)
            append_deferred_event(runtime, {
                "interval_id": replacement_id, "phase": "corpus",
                "start_row": 2, "end_row": 8, "status": "deferred",
                "base_snapshot": str(first),
            })
            unknown = runtime / "deferred" / "manual-experiment"
            unknown.mkdir()
            (unknown / "brain.base.wbrain").write_bytes(b"keep")

            append_deferred_event(runtime, {
                "interval_id": first_id, "phase": "corpus",
                "status": "resolved",
            })
            self.assertEqual(prune_resolved_deferred_bases(runtime), [])
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())
            self.assertTrue(unknown.exists())

            append_deferred_event(runtime, {
                "interval_id": replacement_id, "phase": "corpus",
                "status": "resolved",
            })
            self.assertEqual(
                prune_resolved_deferred_bases(runtime),
                [first.parent.resolve()],
            )
            self.assertFalse(first.parent.exists())

            append_deferred_event(runtime, {
                "interval_id": second_id, "phase": "corpus",
                "status": "resolved",
            })
            self.assertEqual(
                prune_resolved_deferred_bases(runtime),
                [second.parent.resolve()],
            )
            self.assertFalse(second.parent.exists())
            self.assertTrue(unknown.exists())

    def test_regenerated_guard_for_same_proven_state_reuses_causal_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"first-accepted-container")
            metadata = {
                "phase": "corpus",
                "row": 100,
                "storage": "wbrain",
                "guard": str(guard),
                "checkpoint_proof": {
                    "topology": {
                        "tick": 7,
                        "pool_count": 2,
                        "total_neurons": 11,
                        "binding_posting_generations": 3,
                    },
                },
            }
            (brain / "brain.last-good.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            first_id = deferred_interval_id("corpus", 100, 110)
            first = preserve_deferred_base(runtime, first_id)
            identity = guard_state_identity(metadata)
            append_deferred_event(runtime, {
                "interval_id": first_id,
                "phase": "corpus",
                "start_row": 100,
                "end_row": 110,
                "status": "deferred",
                "base_snapshot": str(first),
                "base_state_identity": identity,
            })

            guard.unlink()
            guard.write_bytes(b"regenerated-equivalent-container")
            second = preserve_deferred_base(
                runtime, deferred_interval_id("corpus", 110, 120)
            )
            self.assertTrue(first.samefile(second))
            self.assertEqual(second.read_bytes(), b"first-accepted-container")
            self.assertFalse(second.samefile(guard))

    def test_live_guard_owns_checkpoint_barrier_and_topology_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            (brain / "brain.wbrain").write_bytes(b"accepted-container")
            phase = Phase("phase-a", "script-a", runtime / "corpus.jsonl", 10)
            args = SimpleNamespace(endpoint="http://127.0.0.1:18600")
            topology = {
                "tick": 4, "pool_count": 2, "total_neurons": 9,
                "total_concepts": 5, "total_binding": 3,
                "total_terminals": 12,
            }
            with patch(
                "scripts.programming_curriculum_supervisor.endpoint_post_json",
                return_value={"ok": True},
            ) as checkpoint, patch(
                "scripts.programming_curriculum_supervisor.endpoint_json",
                return_value=topology,
            ):
                ensure_live_last_good_guard(args, runtime, phase, 4)
            checkpoint.assert_called_once()
            metadata = json.loads(
                (brain / "brain.last-good.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["checkpoint_proof"]["row"], 4)
            self.assertEqual(metadata["checkpoint_proof"]["topology"], topology)

    def test_restore_topology_proof_rejects_stale_container(self) -> None:
        restored = {"checkpoint_proof": {"topology": {
            "tick": 10, "pool_count": 2, "total_neurons": 20,
            "total_concepts": 12, "total_binding": 7,
            "total_terminals": 40,
        }}}
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            verify_restored_topology(restored, {
                "tick": 9, "pool_count": 2, "total_neurons": 19,
                "total_concepts": 12, "total_binding": 7,
                "total_terminals": 40,
            })

    def test_guard_acceptance_requires_matching_phase_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted-container")
            metadata = brain / "brain.last-good.json"
            metadata.write_text(json.dumps({
                "phase": "metamath", "row": 100,
                "guard": str(guard),
            }), encoding="utf-8")

            self.assertFalse(accept_last_good_guard(runtime, "mathinstruct"))
            self.assertTrue(guard.exists())
            self.assertTrue(metadata.exists())

            self.assertTrue(accept_last_good_guard(runtime, "metamath"))
            self.assertFalse(guard.exists())
            self.assertFalse(metadata.exists())

    def test_canary_restore_replaces_authoritative_wbrain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.wbrain"
            snapshot.write_bytes(b"rejected-container")
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted-container")
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 100,
                "snapshot": str(snapshot), "guard": str(guard),
                "storage": "wbrain", "guard_mode": "copy",
            }), encoding="utf-8")
            (brain / "brain.wal").write_bytes(b"rejected-wal")
            (runtime / "corpus.progress.json").write_text(json.dumps({
                "ram_next_row": 200, "durable_next_row": 200,
            }), encoding="utf-8")
            (runtime / "curriculum-canary-quarantine.json").write_text(
                json.dumps({
                    "phase": "corpus", "candidate_row": 200,
                    "last_good": {
                        "phase": "corpus", "row": 100,
                        "snapshot": str(snapshot), "guard": str(guard),
                    },
                }), encoding="utf-8",
            )
            restored = restore_canary_quarantine(runtime, finalize=False)
            self.assertEqual(restored["snapshot"], str(snapshot))
            self.assertEqual(snapshot.read_bytes(), b"accepted-container")
            self.assertTrue(guard.exists())
            self.assertFalse((brain / "brain.wal").exists())
            self.assertEqual(
                json.loads((runtime / "corpus.progress.json").read_text())[
                    "durable_next_row"
                ],
                200,
            )
            self.assertTrue(
                (runtime / "curriculum-canary-quarantine.json").exists()
            )
            finalize_canary_restore(runtime, restored)
            self.assertFalse(guard.exists())
            self.assertEqual(
                json.loads((runtime / "corpus.progress.json").read_text())[
                    "durable_next_row"
                ],
                100,
            )

    def test_low_space_wbrain_restore_discards_only_rejected_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.wbrain"
            snapshot.write_bytes(b"rejected-container")
            guard = brain / "brain.last-good.wbrain"
            guard.write_bytes(b"accepted-container")
            last_good = {
                "phase": "corpus",
                "row": 100,
                "snapshot": str(snapshot),
                "guard": str(guard),
                "storage": "wbrain",
                "guard_mode": "copy",
            }
            (brain / "brain.last-good.json").write_text(
                json.dumps(last_good), encoding="utf-8"
            )
            (runtime / "corpus.progress.json").write_text(
                json.dumps({"ram_next_row": 200, "durable_next_row": 200}),
                encoding="utf-8",
            )
            (runtime / "curriculum-canary-quarantine.json").write_text(
                json.dumps({
                    "phase": "corpus",
                    "candidate_row": 200,
                    "last_good": last_good,
                }),
                encoding="utf-8",
            )

            calls = 0

            def require_after_candidate_removed(*_args, **_kwargs) -> None:
                nonlocal calls
                calls += 1
                if snapshot.exists():
                    raise RuntimeError("insufficient disk headroom")

            with patch(
                "scripts.programming_curriculum_supervisor."
                "require_snapshot_copy_headroom",
                side_effect=require_after_candidate_removed,
            ):
                restored = restore_canary_quarantine(runtime, finalize=False)

            self.assertEqual(calls, 2)
            self.assertEqual(restored["row"], 100)
            self.assertEqual(snapshot.read_bytes(), b"accepted-container")
            self.assertEqual(guard.read_bytes(), b"accepted-container")
            self.assertTrue(
                (runtime / "curriculum-canary-quarantine.json").exists()
            )

    def test_guarded_block_target_survives_worker_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 100,
            }), encoding="utf-8")
            phase = Phase("corpus", "script", Path("corpus.jsonl"), 1000)
            self.assertEqual(guarded_block_target(runtime, phase, 100, 200), 300)
            self.assertEqual(guarded_block_target(runtime, phase, 175, 200), 300)
            self.assertEqual(guarded_block_target(runtime, phase, 299, 200), 300)

    def test_exact_guarded_boundary_is_admitted_after_supervisor_restart(
            self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            (brain / "brain.last-good.json").write_text(json.dumps({
                "phase": "corpus", "row": 100,
            }), encoding="utf-8")
            phase = Phase("corpus", "script", Path("corpus.jsonl"), 1000)
            self.assertFalse(guarded_admission_due(
                runtime, phase, 299, 200
            ))
            self.assertTrue(guarded_admission_due(
                runtime, phase, 300, 200
            ))
            self.assertTrue(guarded_admission_due(
                runtime, phase, 301, 200
            ))
            self.assertFalse(guarded_admission_due(
                runtime, phase, phase.rows, 200
            ))

    def test_seed_stage_transaction_resolves_without_duplicate_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.bin"
            snapshot.write_bytes(b"accepted")
            guard_seed_stage(runtime, "multilanguage")
            snapshot.unlink()
            snapshot.write_bytes(b"unaccepted")
            self.assertEqual(
                resolve_seed_guard(runtime, {"completed_seed_stages": []}),
                ("multilanguage", "restored"),
            )
            self.assertEqual(snapshot.read_bytes(), b"accepted")

            guard_seed_stage(runtime, "multilanguage")
            snapshot.unlink()
            snapshot.write_bytes(b"accepted-candidate")
            self.assertEqual(
                resolve_seed_guard(
                    runtime, {"completed_seed_stages": ["multilanguage"]}
                ),
                ("multilanguage", "committed"),
            )
            self.assertEqual(snapshot.read_bytes(), b"accepted-candidate")

    def test_seed_stage_wbrain_guard_restores_independent_container(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            brain = runtime / "brain"
            brain.mkdir()
            snapshot = brain / "brain.wbrain"
            snapshot.write_bytes(b"accepted-container")
            guard_seed_stage(runtime, "multilanguage")
            guard = brain / "seed.last-good.wbrain"
            self.assertTrue(guard.is_file())
            self.assertFalse(guard.samefile(snapshot))
            snapshot.write_bytes(b"rejected-container")
            self.assertEqual(
                resolve_seed_guard(runtime, {"completed_seed_stages": []}),
                ("multilanguage", "restored"),
            )
            self.assertEqual(snapshot.read_bytes(), b"accepted-container")
            self.assertFalse(guard.exists())

    def test_reproducible_trainer_covers_proven_seed_curriculum(self) -> None:
        self.assertEqual(SEED_STAGES[0].name, "foundation-python-debug")
        self.assertIn("semantic-routing", [stage.name for stage in SEED_STAGES])
        source = (ROOT / "scripts/programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--include-seed-corpora"', source)
        self.assertIn('"--repeats", str(phase.repeats)', source)
        trainer = (ROOT / "scripts/train_programming_brain.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("programming_experiential_generalization.py", trainer)
        self.assertIn("programming_multidomain_synthesis.py", trainer)
        self.assertIn("programming_parameterized_fulfillment.py", trainer)
        self.assertIn('"--repeats", "1"', trainer)
        self.assertIn('"--auto-quarantine-recovery"', trainer)
        self.assertIn('"--replay-deferred"', trainer)
        self.assertIn('"--node-bin", str(args.node_bin.resolve())', trainer)
        self.assertIn(
            "stop_runtime_node(runtime, args.endpoint)", trainer
        )
        self.assertIn(
            "process = start_node(args, runtime, identity)", trainer
        )

    def test_reproducible_trainer_separates_forward_and_deferred_replay(self) -> None:
        args = SimpleNamespace(
            endpoint="http://127.0.0.1:18095",
            runtime=Path("runtime"),
            corpus_root=Path("corpus"),
            batch_size=256,
            lock_chunk_size=12,
            checkpoint_rows=131072,
            gate_rows=131072,
            canary_rows=16384,
            max_live_lock_seconds=8.0,
            poll_seconds=2.0,
            max_restarts=10,
            node_bin=Path("brain-server.exe"),
        )
        forward, replay = curriculum_commands(args)
        self.assertIn("--auto-quarantine-recovery", forward)
        self.assertIn("--forward-harvest", forward)
        self.assertNotIn("--replay-deferred", forward)
        self.assertIn("--replay-deferred", replay)
        self.assertNotIn("--auto-quarantine-recovery", replay)

        self.assertEqual(
            forward[forward.index("--lock-chunk-size") + 1], "12"
        )
        self.assertEqual(
            forward[forward.index("--poll-seconds") + 1], "2.0"
        )
        self.assertEqual(
            forward[forward.index("--max-restarts") + 1], "10"
        )
        parameterized = parameterized_admission_command(args)
        self.assertIn("programming_parameterized_fulfillment.py",
                      parameterized[1])
        self.assertEqual(
            parameterized[parameterized.index("--repeats") + 1], "1"
        )
        qualifications = dict(qualification_commands(args))
        self.assertEqual(
            set(qualifications),
            {
                "multidomain-holdout",
                "domain-transfer-holdout",
                "state-contract-holdout",
                "cross-project-composition",
                "polyglot-composition",
                "composition-matrix",
                "mobile-runtime",
            },
        )
        self.assertIn(
            "--ablations", qualifications["domain-transfer-holdout"]
        )
        self.assertEqual(
            qualification_state_signature({
                "tick": 7,
                "pool_count": 3,
                "total_neurons": 11,
                "total_concepts": 5,
                "total_binding": 4,
                "binding_pool_id": 0,
                "resident_terminals": 999,
            }),
            {
                "tick": 7,
                "pool_count": 3,
                "total_neurons": 11,
                "total_concepts": 5,
                "total_binding": 4,
                "binding_pool_id": 0,
            },
        )

    def test_deferred_intervals_are_a_successful_forward_handoff_only(self) -> None:
        self.assertEqual(deferred_handoff_exit_code(True), 0)
        self.assertEqual(deferred_handoff_exit_code(False), 1)

    def test_persistent_service_hands_deferred_tail_to_replay(self) -> None:
        source = (
            ROOT / "scripts" / "aws" / "run_programming_curriculum_service.sh"
        ).read_text(encoding="utf-8")
        unit = (
            ROOT / "scripts" / "aws" / "wizard-curriculum-supervisor.service"
        ).read_text(encoding="utf-8")
        forward_position = source.index(
            '"${common[@]}" --auto-quarantine-recovery --forward-harvest'
        )
        expected_handoff = source.index(
            '[[ "${forward_state}" != "deferred_intervals_pending" ]]'
        )
        replay_position = source.index('"${common[@]}" --replay-deferred')
        self.assertLess(forward_position, expected_handoff)
        self.assertLess(expected_handoff, replay_position)
        self.assertIn('exit "${forward_rc}"', source)
        self.assertIn(
            '[[ "${replay_state}" == "deferred_replay_failed" ]]', source
        )
        self.assertIn("exit 42", source)
        self.assertIn("RestartPreventExitStatus=42", unit)
        self.assertIn(
            'supervisor_pid_file="${runtime}/curriculum-service-supervisor.pid"',
            source,
        )
        self.assertIn("supervisor_pid_is_live", source)
        self.assertIn('run_supervisor_stage forward', source)
        self.assertIn('run_supervisor_stage replay', source)
        self.assertIn(
            'echo "Adopting ${adopted_stage} supervisor PID ${adopted_pid}."',
            source,
        )
        self.assertIn("--batch-size 8", source)
        self.assertIn("--lock-chunk-size 8", source)
        self.assertIn("--min-free-memory-gb 8", source)
        self.assertNotIn("--batch-size 32", source)
        self.assertNotIn("--lock-chunk-size 32", source)

    def test_persistent_service_stop_preserves_only_the_brain_node(self) -> None:
        unit = (
            ROOT / "scripts" / "aws" / "wizard-curriculum-supervisor.service"
        ).read_text(encoding="utf-8")
        stop = (
            ROOT / "scripts" / "aws" / "stop_programming_curriculum_service.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "ExecStop=/bin/bash /srv/wizard/project/scripts/aws/"
            "stop_programming_curriculum_service.sh",
            unit,
        )
        self.assertIn("scripts/programming_curriculum_supervisor.py", stop)
        self.assertIn("tools.training_standard.drive_corpora_brain", stop)
        self.assertIn("kill -TERM", stop)
        self.assertNotIn("w1z4rd_brain_server", stop)
        self.assertIn('"${runtime}/curriculum-supervisor.pid"', stop)
        self.assertIn(
            '"${runtime}/curriculum-service-supervisor.pid"',
            stop,
        )
        self.assertIn(
            '"${runtime}/curriculum-service-supervisor.stage"',
            stop,
        )
        self.assertIn("Curriculum processes survived SIGKILL", stop)

    def test_worker_cooperatively_pauses_only_after_durable_progress(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            control = root / "phase.pause"
            progress = root / "phase.progress.json"
            control.write_text("request-token\n", encoding="ascii")
            worker = threading.Thread(
                target=honor_pause_control,
                args=(control, progress, 321, 321),
                daemon=True,
            )
            worker.start()
            acknowledgement = pause_ack_path(control)
            deadline = time.monotonic() + 2.0
            while not acknowledgement.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            payload = json.loads(
                acknowledgement.read_text(encoding="utf-8")
            )
            self.assertEqual(payload["token"], "request-token")
            self.assertEqual(payload["ram_next_row"], 321)
            self.assertEqual(payload["durable_next_row"], 321)
            self.assertTrue(worker.is_alive())
            control.unlink()
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())
            self.assertFalse(acknowledgement.exists())

    def test_continuous_canary_uses_cooperative_idle_retry(self) -> None:
        source = (
            ROOT / "scripts" / "programming_curriculum_supervisor.py"
        ).read_text(encoding="utf-8")
        driver = (
            ROOT / "tools" / "training_standard" / "drive_corpora_brain.py"
        ).read_text(encoding="utf-8")
        self.assertIn("request_worker_pause(", source)
        self.assertIn("continuous_canary_cooperative_retry", source)
        self.assertIn('command.extend(["--pause-control-path"', source)
        self.assertIn("honor_pause_control(", driver)
        self.assertIn("cooperatively pause after a durable batch", driver)

    def test_reproducible_trainer_finalizes_zero_resident_wbrain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            before = {
                "tick": 7,
                "pool_count": 3,
                "total_neurons": 11,
                "total_concepts": 5,
                "total_binding": 4,
                "binding_pool_id": 0,
                "resident_terminals": 99,
            }
            after = {**before, "resident_terminals": 0}
            with (
                patch(
                    "scripts.train_programming_brain.request",
                    side_effect=[before, {"ok": True}, after],
                ),
                patch(
                    "scripts.train_programming_brain.checkpoint"
                ) as checkpoint_call,
            ):
                report = finalize_production_brain(
                    "http://127.0.0.1:18095", runtime
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["resident_terminals"], 0)
            checkpoint_call.assert_called_once()
            persisted = json.loads(
                (runtime / "benchmarks/production-finalization.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["state_before"],
                             persisted["state_after"])

    def test_attached_bounded_worker_is_gated_before_training_resumes(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument("--attach-phase", default="")', source)
        self.assertIn(
            "admission = admit_midphase_candidate(\n"
            "                args, attach_phase, runtime, status_path,",
            source,
        )
        self.assertIn(
            'admission not in {"admitted", "deferred", "recovered"}',
            source,
        )
        self.assertIn('parser.add_argument(\n        "--gate-only-phase"', source)
        self.assertIn('"state": "gate_only_complete"', source)

    def test_bulk_size_adapts_to_measured_live_lock_window(self) -> None:
        self.assertEqual(
            responsive_batch_size(
                32, {"last_batch_size": 32, "last_batch_seconds": 16}, 8
            ),
            16,
        )
        self.assertEqual(
            responsive_batch_size(
                32, {"last_batch_size": 32, "last_batch_seconds": 6.5}, 8
            ),
            32,
        )
        self.assertEqual(
            responsive_batch_size(
                32,
                {
                    "last_batch_size": 32, "last_batch_seconds": 6.5,
                    "max_batch_size": 32, "max_batch_seconds": 12.0,
                },
                8,
            ),
            21,
        )

    def test_lock_scope_grows_on_repeated_live_success_and_reduces_on_breach(self) -> None:
        current = 1
        streak = 0
        for _ in range(7):
            current, streak, change = adapt_lock_chunk_size(
                32, current, 0.1, 8.0, streak
            )
            self.assertEqual(change, "unchanged")
        self.assertEqual((current, streak), (1, 7))
        self.assertEqual(
            adapt_lock_chunk_size(32, current, 0.1, 8.0, streak),
            (2, 0, "increased"),
        )
        self.assertEqual(
            adapt_lock_chunk_size(32, 4, 16.0, 8.0, 5),
            (2, 0, "reduced"),
        )
        self.assertEqual(
            adapt_lock_chunk_size(32, 4, 3.0, 8.0, 5),
            (4, 0, "unchanged"),
        )

    def test_bulk_size_calibration_survives_phase_transition(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            runtime = Path(raw)
            (runtime / "completed.progress.json").write_text(json.dumps({
                "max_batch_size": 24, "max_batch_seconds": 16.0,
            }), encoding="utf-8")
            self.assertEqual(
                runtime_responsive_batch_size(runtime, 32, {}, 8.0), 12
            )

    def test_slow_batch_ledger_preserves_exact_ranges_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            progress = Path(raw) / "phase.progress.json"
            first = {
                "logical_start_row": 100,
                "logical_end_row": 356,
                "max_lock_seconds": 47.9,
            }
            second = {
                "logical_start_row": 356,
                "logical_end_row": 612,
                "max_lock_seconds": 9.2,
            }
            ledger = append_slow_batch_event(progress, first)
            self.assertEqual(append_slow_batch_event(progress, second), ledger)
            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows, [first, second])

    def test_direct_pretrain_records_slow_batch_corpus_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / "corpus.jsonl"
            corpus.write_text(
                "\n".join([
                    json.dumps({"prompt": "one", "response": "first"}),
                    json.dumps({"prompt": "two", "response": "second"}),
                ]) + "\n",
                encoding="utf-8",
            )
            progress = root / "phase.progress.json"
            script = SimpleNamespace(
                id="fixture", category="test", phase="train",
                inputs=[SimpleNamespace(kind="corpus", path="corpus.jsonl")],
            )
            with patch(
                "tools.training_standard.drive_corpora_brain.post_pretrain_batch",
                return_value=(True, "", 9.5, 1, 1, {"frame_lookup": 8}),
            ), patch(
                "tools.training_standard.drive_corpora_brain.post_checkpoint",
                return_value=(True, {}),
            ):
                drive_one(
                    script, 1, root, smoke=False, direct_pretrain=True,
                    batch_size=2, lock_chunk_size=2, progress_path=progress,
                    checkpoint_rows=100, wal_durable=True,
                    max_live_batch_seconds=8.0, inter_post_sleep=0.0,
                )
            ledger = progress.with_name("phase.progress.slow-batches.jsonl")
            event = json.loads(ledger.read_text(encoding="utf-8"))
            self.assertEqual(event["logical_start_row"], 0)
            self.assertEqual(event["logical_end_row"], 2)
            self.assertEqual(event["submitted_episodes"], 2)
            self.assertEqual(event["lock_chunk_size_before"], 2)
            self.assertEqual(event["lock_chunk_size_after"], 1)
            self.assertEqual(event["max_lock_chunk_index"], 1)
            self.assertEqual(event["max_lock_chunk_len"], 1)
            self.assertEqual(event["max_lock_logical_rows"], [1])
            self.assertEqual(event["max_lock_profile_ns"], {"frame_lookup": 8})

    def test_direct_pretrain_explicit_input_overrides_registry_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            corpus = root / "restored" / "corpus.jsonl"
            corpus.parent.mkdir()
            corpus.write_text(
                "\n".join([
                    json.dumps({"prompt": "one", "response": "first"}),
                    json.dumps({"prompt": "two", "response": "second"}),
                ]) + "\n",
                encoding="utf-8",
            )
            progress = root / "phase.progress.json"
            script = SimpleNamespace(
                id="fixture", category="test", phase="train",
                inputs=[
                    SimpleNamespace(
                        kind="corpus",
                        path=r"D:\training\canonical-corpus.jsonl",
                    ),
                ],
            )
            with patch(
                "tools.training_standard.drive_corpora_brain.post_pretrain_batch",
                return_value=(True, "", 0.1, 0, 2, {}),
            ) as pretrain, patch(
                "tools.training_standard.drive_corpora_brain.post_checkpoint",
                return_value=(True, {}),
            ):
                summary = drive_one(
                    script, 1, root, smoke=False, direct_pretrain=True,
                    input_path=str(corpus), batch_size=2, lock_chunk_size=2,
                    progress_path=progress, checkpoint_rows=100,
                    wal_durable=True, max_live_batch_seconds=8.0,
                    inter_post_sleep=0.0,
                )
            self.assertEqual(summary["pairs"], 2)
            self.assertEqual(summary["posted_ok"], 2)
            pretrain.assert_called_once()
            state = json.loads(progress.read_text(encoding="utf-8"))
            self.assertEqual(state["ram_next_row"], 2)
            self.assertEqual(state["durable_next_row"], 2)


if __name__ == "__main__":
    unittest.main()
