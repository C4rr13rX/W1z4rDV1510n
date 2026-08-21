"""Tests for compiling a directory of mixed documents into a corpus.

The tool exists so a folder of arbitrary documents can be turned into the
JSONL the curriculum already consumes. The properties that matter are that it
pairs honestly, refuses what it cannot pair rather than inventing data, and
emits rows that satisfy the same contract every other ingest module does.
"""
import json

from tools.training_standard.ingest import document_directory as dd

REQUIRED_FIELDS = {"prompt", "response", "ctx", "license", "source",
                   "source_hash", "script_id"}


def build_source(tmp_path):
    src = tmp_path / "docs"
    (src / "sub").mkdir(parents=True)
    (src / "guide.md").write_text(
        "# Installing the widget\n"
        "Run the installer and accept the defaults. The service starts\n"
        "automatically and listens on port 8080 when installation ends.\n\n"
        "## Configuring retries\n"
        "Set retry_limit in config.toml. Values above ten are clamped, and\n"
        "zero disables retries entirely for that endpoint.\n",
        encoding="utf-8")
    (src / "helpers.py").write_text(
        '\ndef normalise_email(value):\n'
        '    """Lowercase an email address and strip surrounding whitespace."""\n'
        '    return str(value).strip().lower()\n',
        encoding="utf-8")
    (src / "sub" / "pairs.jsonl").write_text(
        json.dumps({"question": "How do I restart the service safely?",
                    "answer": "Stop it, wait for the port to clear, then "
                              "start it again."}) + "\n",
        encoding="utf-8")
    (src / "notes.bin").write_bytes(b"\x00\x01\x02binary")
    return src


def compile_dir(tmp_path, src):
    out = tmp_path / "corpus.jsonl"
    stats = dd.build(src, out, "cc0-1.0", "docs", "domain_docs_001",
                     "implement")
    rows = [json.loads(line) for line in
            out.read_text(encoding="utf-8").splitlines() if line.strip()]
    return stats, rows


def test_mixed_directory_compiles_to_the_row_contract(tmp_path):
    stats, rows = compile_dir(tmp_path, build_source(tmp_path))

    assert stats.rows == len(rows) > 0
    for row in rows:
        assert set(row) == REQUIRED_FIELDS
        assert row["prompt"].strip() and row["response"].strip()
        assert row["license"] == "cc0-1.0"
        assert row["script_id"] == "domain_docs_001"
        assert set(row["ctx"]) == {"lang", "intent", "source"}
    # Provenance names the file and how the pair was derived.
    assert any(r["source"].endswith("guide.md#section") for r in rows)
    assert any(r["source"].endswith("pairs.jsonl#record") for r in rows)


def test_every_pairing_strategy_contributes(tmp_path):
    _, rows = compile_dir(tmp_path, build_source(tmp_path))
    kinds = {row["source"].rsplit("#", 1)[1] for row in rows}
    assert {"section", "record", "definition"} <= kinds


def test_binary_and_unpairable_files_are_ignored_not_guessed(tmp_path):
    src = build_source(tmp_path)
    (src / "opaque.md").write_text("no headings, just a loose sentence.\n",
                                   encoding="utf-8")
    _, rows = compile_dir(tmp_path, src)

    assert not any("notes.bin" in row["source"] for row in rows)
    assert not any("opaque.md" in row["source"] for row in rows)


def test_a_syntax_error_does_not_discard_the_whole_file(tmp_path):
    """One bad definition must not lose the good ones beside it."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "mixed.py").write_text(
        '\ndef normalise_email(value):\n'
        '    """Lowercase an email address and strip surrounding whitespace."""\n'
        '    return str(value).strip().lower()\n\n\n'
        'def broken(:\n'
        '    pass\n',
        encoding="utf-8")

    _, rows = compile_dir(tmp_path, src)

    assert len(rows) == 1
    assert "normalise_email" in rows[0]["response"]
    assert "broken" not in rows[0]["response"]


def test_python_responses_that_do_not_parse_are_refused(tmp_path):
    """The salvage must not lower the bar on what is emitted."""
    src = tmp_path / "docs"
    src.mkdir()
    (src / "only_broken.py").write_text(
        'def broken(:\n'
        '    """This claims to do something useful."""\n'
        '    pass\n',
        encoding="utf-8")

    stats, rows = compile_dir(tmp_path, src)

    assert rows == []
    assert stats.rows == 0


def test_identical_pairs_are_deduplicated(tmp_path):
    src = tmp_path / "docs"
    src.mkdir()
    record = json.dumps({"prompt": "How do I restart the service safely?",
                         "response": "Stop it, wait for the port to clear, "
                                     "then start it again."})
    (src / "a.jsonl").write_text(record + "\n" + record + "\n",
                                 encoding="utf-8")

    stats, rows = compile_dir(tmp_path, src)

    assert len(rows) == 1
    assert stats.skipped_duplicate == 1


def test_a_non_permissive_license_is_rejected(tmp_path):
    """Licence validation is the writer's, and must not be bypassed."""
    src = build_source(tmp_path)
    out = tmp_path / "corpus.jsonl"
    try:
        dd.build(src, out, "gpl-3.0", "docs", "domain_docs_001", "implement")
    except Exception as exc:  # RowRejected, surfaced by the writer
        assert "gpl-3.0" in str(exc).lower()
    else:
        raise AssertionError("a copyleft licence must not be accepted")
