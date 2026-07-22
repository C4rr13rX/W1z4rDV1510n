# Reproducing the programming brain

`scripts/train_programming_brain.py` is the canonical entry point for training a fresh Wizard Vision coding brain with the process proven by the integrated programming run. It preserves atom-level grounding; it does not introduce a tokenizer or bypass the configured neural pools.

## One-command run

Build the current release brain server, then choose a new runtime directory and an unused port:

```powershell
cargo build --release -p w1z4rdv1510n-node --bin w1z4rd_brain_server
python scripts/train_programming_brain.py `
  --runtime runtime/brains/programming-reproduction-001 `
  --port 18601
```

The default corpus root is `D:\w1z4rdv1510n-data\training`. Override it with `--corpus-root` when the generated corpora live elsewhere.

Use `--resume` with the same runtime after an interruption. The trainer records only accepted stages. Each seed stage is protected using the authoritative brain representation: immutable atomically replaced `brain.bin` checkpoints may use an NTFS hard link, while mutable `brain.wbrain` containers always use an independent copy published by atomic rename. On an owned-node resume, an interrupted transaction is either committed when its durable state record exists or restored before the stage is retried. This avoids silently training a partially completed stage twice and prevents a `.wbrain` candidate from mutating its own rollback guard.

Use `--dry-run` to inspect the complete command plan without creating a runtime or contacting a node. Use `--seed-only` to stop after the curated enterprise curriculum and its strict gate. `--external-node` is available for an intentionally pre-launched node, but the normal owned-node mode provides safer restart and rollback behavior.

For a controlled deployment or experience-admission window, stop the supervisor near a block target but allow its corpus worker to reach the exact durable boundary. Then run `programming_curriculum_supervisor.py --gate-only-phase <phase>` with the normal endpoint/runtime/gate options. It executes the same authoritative midphase or completion gate, releases the matching last-good guard only on success, and exits without starting another worker. Resume the ordinary supervisor only after the protected maintenance operation is complete.

## Encoded curriculum

The seed curriculum is ordered as follows:

1. toddler, K-12, Python generation, and failure/repair/success debugging;
2. executable multilingual generation;
3. Python enterprise behaviors;
4. executable multi-file Python projects;
5. platform engineering;
6. native-language enterprise behaviors;
7. TypeScript enterprise behaviors;
8. cross-language transfer;
9. semantic-routing reinforcement.

After every seed expansion, the trainer runs the complete foundational/Python/debug retention gate and checkpoints only a passing candidate. After all seed stages it runs strict enterprise retention, including execution, zero-shot composition, semantic stress, OOV honesty, tick immutability, and stable-topology immutability.

The corpus supervisor then trains:

1. TheAlgorithms/Python canonical algorithms, four dense repetitions;
2. GSM8K;
3. MathInstruct;
4. MetaMathQA;
5. CodeSearchNet Python;
6. five-way CodeSearchNet Python paraphrases;
7. scientific Jupyter source;
8. four-way scientific Jupyter paraphrases;
9. partial-context scientific Jupyter examples.

Corpus processing is resumable and WAL-durable. The default production schedule uses a 131,072-row guarded admission block with non-halting 16,384-row canaries. Every small batch is flushed to the WAL before acknowledgement. Canaries detect drift while training continues and quarantine only the suspect interval; the stopped-worker block gate then requires distributed corpus recall, foundational retention, executable transfer, strict enterprise behavior, OOV honesty, and read-only tick/topology invariants before releasing the independent last-known-good guard. Lock scope adapts downward when observed brain-lock time exceeds the configured ceiling, and the safest measured setting carries forward into later corpus phases.

After the corpus curriculum, the trainer admits two experience gates. The first exercises an unseen environment rule through baseline failure, verified repair, successful execution, and held-out structural transfer. The second composes a never-trained class from twelve independently trained disciplines, executes it, causally ablates every premise, and tests a contradictory no-retry policy. Both mutations are transactionally guarded and must preserve foundation and enterprise retention before their checkpoints are accepted.

## Authoritative artifacts

The runtime contains:

- `programming-training.state.json`: accepted seed and corpus milestones;
- `brain.identity.toml` and `brain.deployment.toml`: frozen configuration copies;
- `seed/*.json`: seed-stage and gate reports;
- `logs/*.log`: command-level execution logs;
- `*.progress.json`: RAM and durable corpus offsets plus batch telemetry;
- `*.slow-batches.jsonl`: append-only row ranges and payload/lock evidence for every transaction exceeding the responsiveness ceiling;
- `*.retention-gate.json` and `*.completion-gate.json`: admission evidence;
- `benchmarks/experiential-generalization.json` and `benchmarks/multidomain-synthesis.json`: post-corpus experience and causal-integration admission evidence;
- `brain/brain.wbrain`: authoritative neuron-addressed container for current runtimes;
- `brain/brain.wal`: durable mutations not yet incorporated into the accepted container state;
- `brain/brain.bin`: optional legacy checkpoint input, not authoritative after `.wbrain` migration;
- `brain/*.last-good.*`: unresolved rollback state, present only while a candidate is under review.

Do not declare a reproduced brain equivalent merely because the process exits successfully. Verify the final state file, every completion gate, the strict enterprise report, execution results, OOV honesty, and the final brain identity together.
