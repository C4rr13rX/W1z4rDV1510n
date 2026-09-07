"""Go must be really validated and honestly licensed before it is trained.

The brain answered "golang transactional outbox" with order_service.js -- a
JavaScript file -- because "transactional outbox" had only ever been trained
in JavaScript. Teaching it Go is the fix, but a Go corpus is only worth
training if each row is genuinely Go and genuinely usable commercially.
"""
from __future__ import annotations

import shutil

import pytest

from tools.training_standard.row import (
    NON_COMMERCIAL_LICENSES,
    PERMISSIVE_LICENSES,
)
from tools.training_standard.sandbox.local_backend import LocalSandbox

GOOD_GO = (
    "func Dedup(ids []string) []string {\n"
    "\tseen := map[string]bool{}\n"
    "\tout := []string{}\n"
    "\tfor _, id := range ids {\n"
    "\t\tif !seen[id] {\n"
    "\t\t\tseen[id] = true\n"
    "\t\t\tout = append(out, id)\n"
    "\t\t}\n"
    "\t}\n"
    "\treturn out\n"
    "}\n"
)

BROKEN_GO = (
    "func Broken(ids []string) []string {\n"
    "\tfor _, id := range ids {\n"
    "\treturn out\n"
)


@pytest.mark.skipif(shutil.which("gofmt") is None, reason="gofmt not installed")
def test_go_rows_are_really_parsed_not_waved_through() -> None:
    """An unsupported language falls through to CheckResult.passed().

    Without a Go entry every Go row would be accepted unchecked while the
    ingest reported a clean run -- the corpus would look validated and be
    nothing of the kind.
    """
    sandbox = LocalSandbox()
    assert sandbox.check("go", GOOD_GO).ok
    assert not sandbox.check("go", BROKEN_GO).ok


@pytest.mark.skipif(shutil.which("gofmt") is None, reason="gofmt not installed")
def test_a_bare_go_function_is_package_wrapped_before_parsing() -> None:
    """gofmt parses a FILE, so a bare function fails on the package clause.

    Verified on the training host: unwrapped, a VALID function and a BROKEN
    one both exit 2 with "expected 'package', found 'func'". Rejecting every
    row for the same spurious reason would look exactly like a corpus with no
    usable Go in it.
    """
    sandbox = LocalSandbox()
    assert not GOOD_GO.lstrip().startswith("package ")
    assert sandbox.check("go", GOOD_GO).ok

    # An snippet that already declares its package must not be double-wrapped.
    assert sandbox.check("go", "package main\n\nfunc F() {}\n").ok


def test_permissive_mixed_is_commercially_usable_and_distinct() -> None:
    """CodeSearchNet is permissive-only upstream but has no per-row licence.

    Stamping every row "mit" claims terms the corpus does not establish. This
    id records what is actually known -- permissive, exact terms in the repo
    named by `source` -- without widening the allowlist to unknown provenance.
    """
    assert "permissive-mixed" in PERMISSIVE_LICENSES
    assert "permissive-mixed" not in NON_COMMERCIAL_LICENSES
    # The non-commercial set stays intact: this must never become a bypass.
    for forbidden in ("cc-by-nc", "cc-by-nc-sa-4.0", "cc-by-nd"):
        assert forbidden in NON_COMMERCIAL_LICENSES
        assert forbidden not in PERMISSIVE_LICENSES


def _row(license: str):
    # Hash the licence into the source id: the writer dedups on source_hash,
    # so a fixture that reuses one hash silently drops every row after the
    # first and the test reads as a licence rejection.
    from tools.training_standard.row import Row, hash_source
    return Row(prompt="p", response="r", ctx="", license=license,
               source=f"repo:path#{license}", source_hash=hash_source(license),
               script_id="sid")


def test_copyleft_is_refused_for_its_own_reason_not_as_non_commercial() -> None:
    """Copyleft and non-commercial fail for different reasons.

    Non-commercial material is deferrable: train during development, retrain
    without it before shipping. Copyleft is not. Its obligations can attach to
    what the trained system produces, and this brain cannot be un-trained
    selectively -- concepts emerge by Hebbian collapse across everything
    observed, so there is no operation that subtracts one corpus afterwards.
    A single "non-permissive" message invites the wrong remedy.
    """
    from tools.training_standard.row import (
        COPYLEFT_LICENSES, NON_COMMERCIAL_LICENSES, PERMISSIVE_LICENSES,
        RowRejected, _validate,
    )
    for lic in ("gpl-3.0", "agpl-3.0", "lgpl-2.1"):
        assert lic in COPYLEFT_LICENSES
        assert lic not in PERMISSIVE_LICENSES
        with pytest.raises(RowRejected, match="copyleft"):
            _validate(_row(lic))

    with pytest.raises(RowRejected, match="non-commercial"):
        _validate(_row("cc-by-nc"))

    # The categories stay disjoint: a reader must not be able to satisfy one
    # by looking up the other.
    assert not (COPYLEFT_LICENSES & NON_COMMERCIAL_LICENSES)
    assert not (COPYLEFT_LICENSES & PERMISSIVE_LICENSES)


def test_unestablished_provenance_is_refused_even_though_it_looks_open() -> None:
    """CodeSearchNet is the case this exists for.

    Its dataset card states: "each repository has its own license.
    Example-wise license information is not (yet) included in this dataset:
    you will need to find out yourself which license the code is using." It
    applied no licence filter, so it carries copyleft and unlicensed code
    beside permissive code with no per-row record of which is which.
    """
    from tools.training_standard.row import (
        PERMISSIVE_LICENSES, RowRejected, UNKNOWN_PROVENANCE_LICENSES,
        _validate,
    )
    for lic in ("other", "unknown", "codesearchnet"):
        assert lic in UNKNOWN_PROVENANCE_LICENSES
        assert lic not in PERMISSIVE_LICENSES
        with pytest.raises(RowRejected, match="provenance"):
            _validate(_row(lic))


def test_permissive_mixed_does_not_cover_an_unfiltered_corpus() -> None:
    """`permissive-mixed` records an upstream guarantee, not a hope.

    It is only for sources whose upstream filter is itself permissive-only.
    CodeSearchNet has no such filter, so its rows must not borrow this id --
    the comment beside it says so, and this pins the behaviour.
    """
    from tools.training_standard.row import PERMISSIVE_LICENSES, _validate
    assert "permissive-mixed" in PERMISSIVE_LICENSES
    _validate(_row("permissive-mixed"))  # valid when the guarantee is real

    source = (__import__("pathlib").Path(__file__).parents[1]
              / "tools/training_standard/row.py").read_text(encoding="utf-8")
    assert "CodeSearchNet does NOT qualify" in source


def test_an_architecture_corpus_may_carry_what_a_commercial_one_cannot() -> None:
    """The licence question belongs to the CORPUS, not to every row.

    This brain is an architecture proof and is never shipped; a commercial
    brain is a separate future build. Refusing all non-permissive material
    outright would block the architecture work for a reason that does not
    apply to it. Mixing the tiers silently would make the commercial build
    impossible to certify. Tagging the corpus keeps both options open.
    """
    import json
    import tempfile
    from pathlib import Path

    from tools.training_standard.row import (
        RowWriter, RowRejected, TIER_ARCHITECTURE, TIER_COMMERCIAL,
    )

    out = Path(tempfile.mkdtemp())

    # A commercial corpus refuses copyleft, naming the reason.
    with RowWriter(out / "c.jsonl", script_id="sid", source="t") as writer:
        with pytest.raises(RowRejected, match="copyleft"):
            writer.write(_row("gpl-3.0"))
        assert writer.write(_row("mit"))

    # An architecture corpus accepts it.
    with RowWriter(out / "a.jsonl", script_id="sid", source="t",
                   tier=TIER_ARCHITECTURE) as writer:
        assert writer.write(_row("gpl-3.0"))
        assert writer.write(_row("cc-by-nc"))
        # ...but an unrecognised string is still a typo or an unreviewed
        # source, and must not pass silently in either tier.
        with pytest.raises(RowRejected, match="unrecognised"):
            writer.write(_row("made-up-licence"))

    # The tier is recorded per corpus, which is the only granularity a
    # Hebbian brain can honour: once trained, rows cannot be separated again.
    assert json.loads(
        (out / "a.jsonl.manifest").read_text())["tier"] == TIER_ARCHITECTURE
    assert json.loads(
        (out / "c.jsonl.manifest").read_text())["tier"] == TIER_COMMERCIAL


def test_commercial_is_the_default_so_a_corpus_opts_in_to_the_looser_tier() -> None:
    """Forgetting the tag must fail closed, not ship copyleft by accident."""
    import tempfile
    from pathlib import Path

    from tools.training_standard.row import RowRejected, RowWriter

    out = Path(tempfile.mkdtemp())
    with RowWriter(out / "d.jsonl", script_id="sid", source="t") as writer:
        with pytest.raises(RowRejected):
            writer.write(_row("agpl-3.0"))

    with pytest.raises(ValueError, match="unknown corpus tier"):
        RowWriter(out / "e.jsonl", script_id="sid", source="t", tier="whatever")
