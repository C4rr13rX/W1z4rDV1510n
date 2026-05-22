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
})

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


def _validate(row: Row) -> None:
    if not row.prompt.strip():
        raise RowRejected(f"empty prompt (source={row.source})")
    if not row.response.strip():
        raise RowRejected(f"empty response (source={row.source})")
    lic = row.license.strip().lower()
    if lic not in PERMISSIVE_LICENSES:
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
                 append: bool = False) -> None:
        self._path = Path(out_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._script_id = script_id
        self._source = source
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
        _validate(row)
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
