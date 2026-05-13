"""W1z4rD V1510n training standard.

A declarative schema + runner that turns each phase of training into a
verifiable contract.  Every training script in the registry promises:

  - WHAT it trains (inputs)
  - WHAT it expects to be able to answer afterwards (benchmarks)
  - WHAT prior scripts' benchmarks it must not regress (regression_protects)

The runner:
  1. records training events to data/training_events.jsonl
  2. probes the wizard node with each benchmark prompt
  3. scores responses against the benchmark's must_include / must_be_valid
  4. compares fresh scores against the last passing run; if a benchmark
     that used to pass now fails, writes a regression_alert with the
     trail of scripts run between the last pass and now

Lives in scripts/tooling — no Rust changes.  Django reads
training_events.jsonl to render the live-training panel in wizard chat.
"""
