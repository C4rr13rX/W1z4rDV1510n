"""training_standard/generate — synthetic generators on top of ingest output.

Generators are NOT ingest scripts.  They consume one or more JSONL
corpora produced by tools/training_standard/ingest/* and produce new
rows via deliberate transformations:

  - paraphrase.py        N prompt variants per source row (verb/intent
                         swap, style shift, question form)
  - partial_context.py   strip N% of [ctx] atoms so the brain learns
                         graceful degradation when the user
                         underspecifies
  - test_cycle.py        (planned) reconstruct test-fail → fix → pass
                         multi-turn dialogs from CommitPackFT commits
                         tagged as test/fix pairs
  - eem_to_code.py       (planned) take an EEM equation + applicable
                         code and emit "implement equation X" rows

Generators MUST:
  - Preserve license from the source row (no quietly relabeling).
  - Update source_hash to (prompt + "|" + response) so paraphrase
    variants don't dedup against each other.
  - Record provenance: source becomes
    "<original_source>|gen=<generator_name>:<variant_id>"
"""
