"""Cross-layer invariants for the persistent programming-brain runtime."""

from __future__ import annotations

import json
import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.programming_integrated_retention import mutation_enabled
from scripts.programming_curriculum_supervisor import (
    Phase,
    accept_last_good_guard,
    append_deferred_event,
    assert_training_not_quarantined,
    ensure_last_good_guard,
    guarded_block_target,
    deferred_interval_id,
    latest_passing_canary_row,
    phase_offsets,
    preserve_deferred_base,
    publish,
    recall_command,
    restore_canary_quarantine,
    responsive_batch_size,
    runtime_responsive_batch_size,
    topology_delta,
    unresolved_deferred_intervals,
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
    guard_seed_stage,
    resolve_seed_guard,
)
from scripts.programming_exec_env import benchmark_tool_env, isolated_tool_env
from scripts.programming_corpus_recall import accepted_responses, sample_window
from tools.training_standard.drive_corpora_brain import (
    append_slow_batch_event,
    checkpoint_due,
    drive_one,
    row_is_skipped,
)


class ProgrammingRuntimeContractTests(unittest.TestCase):
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
            self.assertIn("stop_runtime_node(runtime)", source)
            self.assertIn("restored = restore_canary_quarantine(runtime)", source)
            self.assertIn("start_runtime_node(runtime, args.node_bin, args.endpoint)", source)
            self.assertIn('"W1Z4RD_TICK_HOUSEKEEPING": "lazy"', source)
            self.assertIn('"W1Z4RD_DEFER_PROMOTION": "1"', source)
            self.assertIn("args.restart_node_after_attach", source)
            self.assertIn('"--skip-range"', source)

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

    def test_shared_compiler_cache_stays_inside_repository_runtime(self) -> None:
        environment = benchmark_tool_env()
        runtime = ROOT / "runtime" / "benchmark-tool-cache"
        for key in ("GOCACHE", "GOMODCACHE", "DOTNET_CLI_HOME", "NUGET_PACKAGES"):
                self.assertTrue(Path(environment[key]).is_relative_to(runtime))

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
            self.assertEqual(recorded["guard_mode"], "copy")

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
        self.assertIn("run_midphase_gate(args, phase, runtime, ram_after)", source)
        self.assertIn('"--no-checkpoint"', source)
        self.assertIn('"--gate-rows", type=int, default=131072', source)
        self.assertIn('"--checkpoint-rows", type=int, default=131072', source)

    def test_dedicated_corpus_supervisor_preserves_live_inference_windows(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--batch-size", type=int, default=256', source)
        self.assertIn('"--inter-batch-yield-seconds", type=float, default=0.0', source)
        self.assertIn('"--max-batch-seconds", str(args.max_live_lock_seconds)', source)
        driver = (ROOT / "tools/training_standard/drive_corpora_brain.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("adaptive_batch_reductions", driver)
        self.assertIn("current_lock_chunk_size = scaled", driver)
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
        self.assertIn('"suspect_end_row": candidate_row', source)
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
            self.assertEqual(metadata["guard_mode"], "copy")
            accept_last_good_guard(runtime)
            self.assertFalse(guard.exists())

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
            restored = restore_canary_quarantine(runtime)
            self.assertEqual(restored["snapshot"], str(snapshot))
            self.assertEqual(snapshot.read_bytes(), b"accepted-container")
            self.assertFalse(guard.exists())
            self.assertFalse((brain / "brain.wal").exists())

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
        self.assertIn('"--auto-quarantine-recovery"', trainer)
        self.assertIn('"--node-bin", str(args.node_bin.resolve())', trainer)

    def test_attached_bounded_worker_is_gated_before_training_resumes(self) -> None:
        source = (ROOT / "scripts" / "programming_curriculum_supervisor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument("--attach-phase", default="")', source)
        self.assertIn("run_midphase_gate(args, attach_phase, runtime, attached_ram)", source)
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
                return_value=(True, "", 9.5),
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


if __name__ == "__main__":
    unittest.main()
