"""training_standard/ingest — external-corpus → Row[] adapters.

Each module here implements one source (StackExchange, CodeSearchNet,
The Stack v2, etc.).  Adapters never invent data: every emitted Row
carries license + source provenance + source_hash, validated by
training_standard.row.RowWriter.

Conventions:
  - Streaming I/O only — corpora can be hundreds of GB.
  - All ingest scripts accept --limit for smoke tests, --out for the
    destination JSONL, and --skip-sandbox for fast iteration.
  - Default sandbox backend is whatever W1Z4RD_SANDBOX is set to.
  - Provenance: every row's `source` field is the upstream's stable
    identifier (e.g. "stackoverflow:Q12345:A67890",
    "codesearchnet:python:django/repo/path/to/file.py:funcname").
"""
