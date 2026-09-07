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
