"""training_standard/row.py — the canonical training-row format.

Every ingest/* and generate/* script writes rows in this shape so the
runner, the brain ingest path, and the eval harness all agree.

A row is one training observation.  For Hebbian ingest the brain reads
`text` as a single paired_text observation (prompt and response are
joined with `[/user][asst]` so the brain can learn the boundary as
ordinary atoms — no special-cased separators, in line with the
no-deterministic-NLP rule).

Provenance is non-negotiable: license, source, source_hash.  Rows
without those three fields are rejected by the writer.  This is how we
keep `Permissive-only` enforceable downstream and how we prove no
test-set leakage when we run integration evals.

The context header [ctx ...] is part of the prompt text, not a side
channel.  See training plan §4 — the brain learns the bracketed atoms
as ordinary tokens that gate which mini-columns activate.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Iterator

# Permissive licenses we accept.  Everything else is rejected by the
# writer.  Case-insensitive comparison; SPDX identifiers preferred.
PERMISSIVE_LICENSES = frozenset({
    "mit", "apache-2.0", "apache 2.0", "apache2", "apache",
    "bsd-2-clause", "bsd-3-clause", "bsd",
    "isc", "0bsd", "unlicense", "cc0-1.0", "cc0", "public-domain",
    "zlib", "wtfpl", "mpl-2.0",  # MPL is weak-copyleft but permissive
                                 # for our distribution model (we ship
                                 # weights, not source).
    # Creative Commons attribution licences. Commercial use IS permitted;
    # the obligation is attribution, which `source` already records per row.
    # Added 2026-08-22 for the LibreTexts textbooks: of 183 books, 45 are
    # CC BY or public domain and 18 are CC BY-SA.
    "cc-by-4.0", "cc-by-3.0", "cc-by-2.0", "cc-by",
    # Share-alike. Commercially usable, but derivatives inherit the licence,
    # so keep it distinguishable from plain CC BY rather than folding both
    # into one id -- a downstream consumer has to be able to tell them apart.
    "cc-by-sa-4.0", "cc-by-sa-3.0", "cc-by-sa",
    # A corpus whose permissive-only status is guaranteed UPSTREAM but which
    # carries no per-row licence field, so the row records the guarantee
    # rather than a specific SPDX id it cannot evidence. The exact terms live
    # with the project named in each row's `source`.
    #
    # Only for sources whose upstream filter is ITSELF permissive-only, and
    # only when that filter is documented. It is not a way to wave through a
    # corpus of unknown provenance -- the point is to be honest about which
    # guarantee is doing the work.
    #
    # CodeSearchNet does NOT qualify, despite being the obvious candidate.
    # Its dataset card states plainly: "each repository has its own license.
    # Example-wise license information is not (yet) included in this dataset:
    # you will need to find out yourself which license the code is using."
    # There is no upstream permissive filter to lean on, so CSN rows need
    # their licence resolved per repository before they can be trained.
    "permissive-mixed",
})

#: Licences that forbid commercial use or derivatives outright. Never added to
#: PERMISSIVE_LICENSES; listed so the distinction is greppable and so a future
#: reader does not "helpfully" complete the CC family. Measured 2026-08-22,
#: 110 of 183 LibreTexts books are NC -- the majority -- so this is the common
#: case, not an edge case.
NON_COMMERCIAL_LICENSES = frozenset({
    "cc-by-nc", "cc-by-nc-4.0", "cc-by-nc-3.0",
    "cc-by-nc-sa", "cc-by-nc-sa-4.0", "cc-by-nc-sa-3.0",
    "cc-by-nc-nd", "cc-by-nc-nd-4.0", "cc-by-nc-nd-3.0",
    "cc-by-nd", "cc-by-nd-4.0",
})

#: Strong copyleft. A THIRD category, deliberately not folded into
#: NON_COMMERCIAL_LICENSES, because the two are refused for different reasons
#: and only one of them is a "later" problem.
#:
#: Non-commercial material is a licensing decision that can be deferred: train
#: on it during development, retrain without it before shipping. Copyleft
#: cannot be deferred that way. Its obligations can attach to what the trained
#: system PRODUCES, and this brain cannot be un-trained selectively -- concepts
#: emerge by Hebbian collapse across everything observed, so a GPL-derived
#: function contributes to bindings shared with permissive material and there
#: is no operation that subtracts it afterwards. The only remedy is rebuilding
#: from a clean corpus and re-running the whole curriculum.
#:
#: Kept greppable so a future reader does not "helpfully" add GPL to the
#: permissive set on the grounds that it is open source. It is; that is not
#: the property being tested here.
COPYLEFT_LICENSES = frozenset({
    "gpl", "gpl-2.0", "gpl-3.0", "gplv2", "gplv3",
    "agpl", "agpl-3.0", "agplv3",
    "lgpl", "lgpl-2.1", "lgpl-3.0",
    "cc-by-sa-2.0",  # older CC-SA predating our reviewed 3.0/4.0 entries
    "osl-3.0", "epl-1.0", "epl-2.0", "cddl-1.0", "ms-pl", "sspl-1.0",
})

#: Provenance is unknown or explicitly not established by the source. Refused
#: for training regardless of how the material looks, because "probably fine"
#: is not a licence.
#:
#: CodeSearchNet is the case this exists for. Its own dataset card states:
#: "each repository has its own license. Example-wise license information is
#: not (yet) included in this dataset: you will need to find out yourself
#: which license the code is using." It applied no licence filter, so it
#: contains copyleft and unlicensed code alongside permissive code, with no
#: per-row record of which is which.
UNKNOWN_PROVENANCE_LICENSES = frozenset({
    "unknown", "unspecified", "other", "none", "",
    "codesearchnet",  # the corpus, not a licence -- see above
})

# Corpus tiers. The licence question belongs to the CORPUS, not to every
# individual row, because the two brains being built have different rules.
#
# This brain is an architecture proof: its job is to demonstrate the maths
# works at scale, and it is never shipped. A commercial brain is a separate
# future build. Refusing all non-permissive material outright would block the
# architecture work for a reason that does not apply to it; mixing the tiers
# silently would make the commercial build impossible to certify.
#
# So: tag the corpus, keep the tiers separable, and let the build choose.
# COMMERCIAL corpora are usable by any brain. ARCHITECTURE corpora may carry
# material that cannot ship -- non-commercial, copyleft, or unestablished
# provenance -- and a commercial build excludes every corpus not tagged
# COMMERCIAL. That is a file-level filter, which is the only kind this system
# can honour: a trained brain cannot be un-trained selectively, because
# concepts emerge by Hebbian collapse across everything observed.
TIER_COMMERCIAL = "commercial"
TIER_ARCHITECTURE = "architecture"
CORPUS_TIERS = frozenset({TIER_COMMERCIAL, TIER_ARCHITECTURE})

# Recognized intent tags — used in [ctx intent=...].  Open set; the
# brain learns associations regardless.  Listed here for grep-ability.
KNOWN_INTENTS = frozenset({
    "implement", "explain", "debug", "test", "refactor", "review",
    "design", "deploy", "configure", "document", "translate", "answer",
})


@dataclasses.dataclass(frozen=True)
class Row:
    """One training observation.

    prompt        plain-English (or partial-context) user input.
    response      the assistant response we want the brain to recall.
    ctx           dict of metadata atoms; rendered as `[ctx k=v ...]`
                  prepended to prompt.  May be empty (deliberate
                  partial-context rows — see plan §4).
    license       SPDX-ish identifier; must be in PERMISSIVE_LICENSES.
    source        short label, e.g. "stackoverflow:12345" or
                  "codesearchnet:python:funcname".
    source_hash   sha256(canonical-source-text); enables dedup +
                  test-set-leakage detection.
    script_id     the registry script this row belongs to.
    """

    prompt: str
    response: str
    ctx: dict[str, str]
    license: str
    source: str
    source_hash: str
    script_id: str

    def render_text(self) -> str:
        """Render the row as a single string the brain can ingest.

        Format (literal characters — the brain learns these as atoms):
            [ctx k1=v1 k2=v2]
            [user]
            <prompt>
            [/user][asst]
            <response>
        """
        parts = []
        if self.ctx:
            kv = " ".join(f"{k}={v}" for k, v in sorted(self.ctx.items()))
            parts.append(f"[ctx {kv}]")
        parts.append("[user]")
        parts.append(self.prompt.strip())
        parts.append("[/user][asst]")
        parts.append(self.response.strip())
        return "\n".join(parts)

    def to_jsonl_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "ctx": self.ctx,
            "license": self.license,
            "source": self.source,
            "source_hash": self.source_hash,
            "script_id": self.script_id,
        }


class RowRejected(ValueError):
    """Raised by the writer for any row that fails the contract."""


def _validate(row: Row, tier: str = TIER_COMMERCIAL) -> None:
    if not row.prompt.strip():
        raise RowRejected(f"empty prompt (source={row.source})")
    if not row.response.strip():
        raise RowRejected(f"empty response (source={row.source})")
    lic = row.license.strip().lower()
    if lic not in PERMISSIVE_LICENSES and tier == TIER_ARCHITECTURE:
        # An architecture corpus may carry material a commercial brain cannot.
        # It still must be a licence we RECOGNISE: an unrecognised string is a
        # typo or an unreviewed source, and neither should pass silently.
        if lic not in (COPYLEFT_LICENSES | NON_COMMERCIAL_LICENSES
                       | UNKNOWN_PROVENANCE_LICENSES):
            raise RowRejected(
                f"unrecognised license {row.license!r} (source={row.source}). "
                "Even an architecture corpus needs a licence the pipeline has "
                "actually reviewed."
            )
    elif lic not in PERMISSIVE_LICENSES:
        # Name the REASON, not just the refusal. The three categories fail for
        # different reasons and only one of them is deferrable, so a single
        # "non-permissive" message invites the wrong remedy -- most obviously
        # "we are still in development, train on it and retrain later", which
        # is sound for non-commercial material and unsound for copyleft.
        if lic in COPYLEFT_LICENSES:
            raise RowRejected(
                f"copyleft license {row.license!r} (source={row.source}). "
                "Copyleft obligations can attach to what the trained system "
                "produces, and this brain cannot be un-trained selectively: "
                "concepts emerge by Hebbian collapse across everything "
                "observed, so removing this later means rebuilding from a "
                "clean corpus and re-running the entire curriculum."
            )
        if lic in UNKNOWN_PROVENANCE_LICENSES:
            raise RowRejected(
                f"unestablished provenance {row.license!r} "
                f"(source={row.source}). The source does not record which "
                "licence this row is under, so it cannot be shown to be "
                "commercially usable. Resolve the licence per item before "
                "training rather than assuming the corpus is uniform."
            )
        if lic in NON_COMMERCIAL_LICENSES:
            raise RowRejected(
                f"non-commercial license {row.license!r} "
                f"(source={row.source}). Permitted: "
                f"{sorted(PERMISSIVE_LICENSES)}"
            )
        raise RowRejected(
            f"non-permissive license {row.license!r} (source={row.source}). "
            f"Permitted: {sorted(PERMISSIVE_LICENSES)}"
        )
    if not row.source.strip():
        raise RowRejected("empty source provenance")
    if not row.source_hash or len(row.source_hash) < 16:
        raise RowRejected(f"weak source_hash {row.source_hash!r}")
    if not row.script_id.strip():
        raise RowRejected("empty script_id")


def hash_source(text: str) -> str:
    """Canonical source hash — strips trailing whitespace, normalises
    line endings.  Used for dedup and test-set leakage checks."""
    canon = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class RowWriter:
    """Streaming JSONL writer with provenance manifest.

    Usage:
        with RowWriter(out_path, script_id="...", source="...") as w:
            for r in rows:
                w.write(r)

    Maintains in-memory dedup set (by source_hash) and writes a
    manifest sibling file with row count + source-hash list for the
    eval harness to consult.

    Memory cost: ~80 bytes/row for the dedup set.  At 1M rows that's
    ~80MB — acceptable on this host.  If we ever blow past that we
    swap to a sqlite-backed set.
    """

    def __init__(self, out_path: os.PathLike | str, *,
                 script_id: str, source: str,
                 dedup: bool = True,
                 append: bool = False,
                 tier: str = TIER_COMMERCIAL) -> None:
        if tier not in CORPUS_TIERS:
            raise ValueError(
                f"unknown corpus tier {tier!r}; expected one of "
                f"{sorted(CORPUS_TIERS)}"
            )
        self._path = Path(out_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._script_id = script_id
        self._source = source
        self._tier = tier
        self._dedup = dedup
        self._seen: set[str] = set()
        self._count = 0
        self._rejected = 0
        self._dedup_skipped = 0
        mode = "a" if append else "w"
        self._fh = self._path.open(mode, encoding="utf-8")
        # Manifest sibling.
        self._manifest_path = self._path.with_suffix(self._path.suffix + ".manifest")

    def __enter__(self) -> "RowWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, row: Row) -> bool:
        """Write one row.  Returns True if accepted, False if dedup-skipped.
        Raises RowRejected for contract violations."""
        _validate(row, self._tier)
        if self._dedup:
            if row.source_hash in self._seen:
                self._dedup_skipped += 1
                return False
            self._seen.add(row.source_hash)
        self._fh.write(json.dumps(row.to_jsonl_dict(), ensure_ascii=False) + "\n")
        self._count += 1
        return True

    def close(self) -> None:
        if self._fh.closed:
            return
        self._fh.close()
        manifest = {
            # The tier a commercial build filters on. Recorded per corpus
            # because that is the only granularity a Hebbian brain can honour:
            # once trained, rows cannot be separated again.
            "tier": self._tier,
            "script_id": self._script_id,
            "source": self._source,
            "path": str(self._path),
            "row_count": self._count,
            "dedup_skipped": self._dedup_skipped,
            "rejected": self._rejected,
            "source_hash_count": len(self._seen),
        }
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

    @property
    def count(self) -> int:
        return self._count


def iter_jsonl(path: os.PathLike | str) -> Iterator[dict]:
    """Read rows back as dicts.  Skips blank lines silently."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def render_ctx(**kwargs: str) -> dict[str, str]:
    """Convenience for ingest scripts — drop None/empty values, lowercase keys.
    Returns a dict suitable for Row.ctx."""
    return {k.lower(): str(v) for k, v in kwargs.items() if v not in (None, "")}
