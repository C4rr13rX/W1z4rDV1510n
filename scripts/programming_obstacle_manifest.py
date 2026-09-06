#!/usr/bin/env python3
"""Versioned manifest for the deterministic 1,000-task obstacle course.

`docs/PROGRAMMING_BRAIN_ACCEPTANCE_CONTRACT.md` defines completion for the
senior software-engineer brain as `1000/1000` on a frozen, held-out course
with fixed per-family counts. This module owns the manifest that course is
built from, and it is deliberately the strictest part of the pipeline: every
way a course could quietly become easier than the contract is rejected here
rather than discovered after a run reports a passing score.

The invariants this module enforces, and why each one exists:

- **Exact family counts.** A course that is 1,000 tasks in total but skewed
  toward the families the brain already passes is not the contract's course.
  `build_manifest` refuses any deviation from `FAMILY_TASK_COUNTS`.
- **Behavioural distinctness, not textual distinctness.** 1,000 renamings of
  one task would satisfy a naive uniqueness check while measuring a single
  capability. Distinctness is therefore keyed on the *behaviour contract* --
  the normalized validator plus its fixtures -- so cosmetic clones collide.
- **No network.** The contract forbids network-dependent cases. A validator
  that reaches the network is non-deterministic and can convert an outage
  into a capability failure, so network imports are a static build error.
- **Bounded timeouts.** An unbounded validator turns a hang into an
  indefinite stall rather than a failure.
- **Held-out prompts.** Obstacle prompts and fixtures must never become
  training rows. `held_out_violations` checks a built manifest against the
  actual corpora instead of trusting that separation was maintained.

The module never reports a partial course as complete. `audit_manifest`
exists so that an incomplete course is legible -- it reports exactly which
families are short and by how many -- while `build_manifest` still refuses to
produce a manifest object at all until the full 1,000 are present.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Run directly (`python scripts/programming_obstacle_manifest.py`) the
    # interpreter puts `scripts/` on the path, not the repository root, so
    # the `scripts.` package imports below would not resolve.
    sys.path.insert(0, str(ROOT))
MANIFEST_ROOT = ROOT / "data" / "obstacle_course"

#: Course revision. Freezing a course stamps this into the manifest so a
#: score can never be attributed to a different set of tasks than the one it
#: actually ran against.
COURSE_VERSION = "v1"

#: Exact per-family counts from the acceptance contract. The totals are part
#: of the contract, not a tuning knob: changing one is changing what
#: completion means, so the sum is asserted at import time.
FAMILY_TASK_COUNTS: dict[str, int] = {
    "requirements_api_contracts": 80,
    "algorithms_data_structures": 70,
    "validation_parsing_serialization": 70,
    "databases_migrations_transactions": 80,
    "http_apis_authn_appsec": 90,
    "concurrency_async_distributed": 80,
    "testing_debugging_repair_refactoring": 100,
    "reliability_observability_performance": 80,
    "cicd_containers_packaging_platform": 70,
    "frontend_state_ux_accessibility": 70,
    "polyglot_native_interop": 70,
    "architecture_multifile_integration": 70,
    "scientific_3d_geometry_robotics": 70,
}

TOTAL_TASKS = 1000

assert sum(FAMILY_TASK_COUNTS.values()) == TOTAL_TASKS, (
    "family counts must sum to the contract total"
)

#: Human-readable family titles, kept beside the slugs so a generated report
#: can quote the contract's own wording back when a family fails.
FAMILY_TITLES: dict[str, str] = {
    "requirements_api_contracts": "Requirements and API contracts",
    "algorithms_data_structures": "Algorithms and data structures",
    "validation_parsing_serialization":
        "Validation, parsing, and serialization",
    "databases_migrations_transactions":
        "Databases, migrations, and transactions",
    "http_apis_authn_appsec":
        "HTTP APIs, authentication, and application security",
    "concurrency_async_distributed":
        "Concurrency, asynchronous work, and distributed coordination",
    "testing_debugging_repair_refactoring":
        "Testing, debugging, repair, and refactoring",
    "reliability_observability_performance":
        "Reliability, observability, and performance",
    "cicd_containers_packaging_platform":
        "CI/CD, containers, packaging, and platform engineering",
    "frontend_state_ux_accessibility": "Frontend state, UX, and accessibility",
    "polyglot_native_interop": "Polyglot, native, and interoperability work",
    "architecture_multifile_integration":
        "Architecture, multi-file change, and integration",
    "scientific_3d_geometry_robotics":
        "Scientific computing, 3D geometry, and robotics",
}

#: Modules that make a validator's verdict depend on something outside the
#: workspace. `subprocess` is deliberately absent: compiling a candidate in a
#: sibling toolchain is exactly what the polyglot families must do, and that
#: stays deterministic because the toolchain is pinned and offline.
NETWORK_MODULES = frozenset({
    "socket", "ssl", "urllib", "urllib2", "urllib3", "http", "httplib",
    "http.client", "requests", "httpx", "aiohttp", "ftplib", "smtplib",
    "poplib", "imaplib", "telnetlib", "xmlrpc", "websockets", "websocket",
    "boto3", "botocore", "paramiko", "pycurl",
})

_IMPORT_PATTERN = re.compile(
    r"^\s*(?:from\s+([A-Za-z_][\w.]*)|import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*))",
    re.MULTILINE,
)

#: A validator has to actually execute the candidate's behaviour. The
#: contract is explicit that "identifier or formatting checks cannot
#: substitute for behavior", so a validator that never asserts anything is
#: rejected at build time rather than silently passing every candidate.
_ASSERTION_PATTERN = re.compile(r"\b(assert|raise\s+AssertionError|fail\()")

MAX_TIMEOUT_SECONDS = 900.0
MIN_TIMEOUT_SECONDS = 1.0


class ManifestError(ValueError):
    """A course that would not measure what the contract requires."""


@dataclass(frozen=True)
class Provenance:
    """Where a task's material came from and what may be done with it.

    The contract requires an SPDX-compatible record for repaired-capability
    curriculum and forbids inferring a licence from public availability.
    Obstacle tasks carry the same record so that a task authored from a
    licensed source can be told apart from one written for this repository,
    and so a licence that forbids redistribution can be caught before the
    manifest is published rather than after.
    """

    origin: str
    spdx_license_id: str
    redistribution_permitted: bool
    source_url: str = ""
    notes: str = ""

    def validate(self, task_id: str) -> None:
        if not self.origin.strip():
            raise ManifestError(f"{task_id}: provenance origin is required")
        if not self.spdx_license_id.strip():
            raise ManifestError(
                f"{task_id}: provenance needs an SPDX licence identifier; "
                "public availability is not a licence"
            )
        if not self.redistribution_permitted:
            raise ManifestError(
                f"{task_id}: provenance forbids redistribution, so the task "
                "cannot ship in a published manifest"
            )


@dataclass(frozen=True)
class ObstacleTask:
    """One held-out task with a deterministic behavioural validator.

    `validator` is Python source executed in a bounded, disposable workspace
    with the candidate response already written to disk. It is Python for
    every family, including the polyglot ones, because a single validator
    language keeps the static safety checks above meaningful; a validator for
    a Go or C# task shells out to that pinned toolchain from Python.
    """

    task_id: str
    family: str
    language: str
    toolchain: str
    prompt: str
    validator: str
    timeout_seconds: float
    provenance: Provenance
    fixtures: Mapping[str, str] = field(default_factory=dict)
    response_filename: str = "candidate.py"

    def validate(self) -> None:
        if self.family not in FAMILY_TASK_COUNTS:
            raise ManifestError(
                f"{self.task_id}: unknown capability family {self.family!r}"
            )
        if not _TASK_ID_PATTERN.fullmatch(self.task_id):
            raise ManifestError(
                f"{self.task_id!r}: task id must be "
                "<family>-<4-digit ordinal>"
            )
        if not self.task_id.startswith(f"{self.family}-"):
            raise ManifestError(
                f"{self.task_id}: task id must be prefixed with its family"
            )
        if len(self.prompt.strip()) < 40:
            raise ManifestError(
                f"{self.task_id}: prompt is too short to state a contract"
            )
        if not self.toolchain.strip():
            raise ManifestError(
                f"{self.task_id}: a pinned toolchain version is required; "
                "an unpinned toolchain makes the verdict irreproducible"
            )
        if not (MIN_TIMEOUT_SECONDS <= self.timeout_seconds
                <= MAX_TIMEOUT_SECONDS):
            raise ManifestError(
                f"{self.task_id}: timeout {self.timeout_seconds}s outside "
                f"[{MIN_TIMEOUT_SECONDS}, {MAX_TIMEOUT_SECONDS}]"
            )
        if not _ASSERTION_PATTERN.search(self.validator):
            raise ManifestError(
                f"{self.task_id}: validator asserts nothing, so it would "
                "pass every candidate including an empty one"
            )
        for module in validator_imports(self.validator):
            root = module.split(".")[0]
            if module in NETWORK_MODULES or root in NETWORK_MODULES:
                raise ManifestError(
                    f"{self.task_id}: validator imports {module!r}; a "
                    "network-dependent case cannot be deterministic"
                )
        for name in self.fixtures:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ManifestError(
                    f"{self.task_id}: fixture {name!r} escapes the workspace"
                )
        self.provenance.validate(self.task_id)

    def behavior_digest(self) -> str:
        """Identify the task by what it measures, not by how it is worded.

        Two tasks that execute the same normalized assertions against the
        same fixtures measure one capability however differently their
        prompts read, so they collide here and `build_manifest` rejects the
        pair. This is what stops a course from reaching 1,000 by restating
        one task a thousand times.
        """
        payload = {
            "language": self.language.casefold(),
            "validator": normalize_source(self.validator),
            "fixtures": {
                name: normalize_source(body)
                for name, body in sorted(self.fixtures.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True,
                             ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self) -> dict:
        record = asdict(self)
        record["fixtures"] = dict(sorted(self.fixtures.items()))
        record["behavior_digest"] = self.behavior_digest()
        return record


_TASK_ID_PATTERN = re.compile(r"[a-z0-9_]+-\d{4}")


def validator_imports(source: str) -> set[str]:
    """Collect module names a validator imports, including grouped imports."""
    found: set[str] = set()
    for from_module, import_list in _IMPORT_PATTERN.findall(source):
        if from_module:
            found.add(from_module)
        if import_list:
            for part in import_list.split(","):
                name = part.strip().split(" as ")[0].strip()
                if name:
                    found.add(name)
    return found


def normalize_source(source: str) -> str:
    """Reduce source to the behaviour it specifies.

    Comments, blank lines, indentation width and Unicode presentation forms
    all change the bytes without changing what a validator checks, so they
    are removed before the distinctness digest is taken. Anything that could
    alter an assertion's meaning -- string contents, operators, numbers -- is
    preserved untouched.
    """
    text = unicodedata.normalize("NFC", source).replace("\r\n", "\n")
    lines = []
    for raw in text.split("\n"):
        line = raw.split("#", 1)[0].rstrip() if "#" in raw else raw.rstrip()
        stripped = line.strip()
        if stripped:
            lines.append(re.sub(r"\s+", " ", stripped))
    return "\n".join(lines)


@dataclass(frozen=True)
class ObstacleManifest:
    """A frozen, complete course. Constructing one is the completeness proof."""

    version: str
    tasks: tuple[ObstacleTask, ...]
    digest: str

    def by_family(self) -> dict[str, tuple[ObstacleTask, ...]]:
        grouped: dict[str, list[ObstacleTask]] = {
            family: [] for family in FAMILY_TASK_COUNTS
        }
        for task in self.tasks:
            grouped[task.family].append(task)
        return {family: tuple(items) for family, items in grouped.items()}

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "digest": self.digest,
            "total": len(self.tasks),
            "family_counts": {
                family: len(items)
                for family, items in self.by_family().items()
            },
            "tasks": [task.to_json() for task in self.tasks],
        }


def audit_manifest(tasks: Sequence[ObstacleTask]) -> dict:
    """Report how far a task set is from the contract without judging it done.

    `build_manifest` raises on an incomplete course, which is correct but
    unhelpful while the course is still being authored. This returns the same
    findings as data so progress is legible: which families are short, which
    ids collide, and which pairs measure the same behaviour. `complete` is
    true only when a manifest could actually be built.
    """
    counts: dict[str, int] = {family: 0 for family in FAMILY_TASK_COUNTS}
    unknown_families: list[str] = []
    for task in tasks:
        if task.family in counts:
            counts[task.family] += 1
        else:
            unknown_families.append(task.family)

    shortfalls = {
        family: FAMILY_TASK_COUNTS[family] - counts[family]
        for family in FAMILY_TASK_COUNTS
        if counts[family] != FAMILY_TASK_COUNTS[family]
    }

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    for task in tasks:
        if task.task_id in seen_ids:
            duplicate_ids.append(task.task_id)
        seen_ids.add(task.task_id)

    digests: dict[str, str] = {}
    duplicate_behaviors: list[tuple[str, str]] = []
    for task in tasks:
        digest = task.behavior_digest()
        if digest in digests:
            duplicate_behaviors.append((digests[digest], task.task_id))
        else:
            digests[digest] = task.task_id

    invalid: list[str] = []
    for task in tasks:
        try:
            task.validate()
        except ManifestError as error:
            invalid.append(str(error))

    return {
        "total": len(tasks),
        "required_total": TOTAL_TASKS,
        "family_counts": counts,
        "family_shortfalls": shortfalls,
        "unknown_families": sorted(set(unknown_families)),
        "duplicate_ids": duplicate_ids,
        "duplicate_behaviors": duplicate_behaviors,
        "invalid_tasks": invalid,
        "complete": (
            len(tasks) == TOTAL_TASKS
            and not shortfalls
            and not unknown_families
            and not duplicate_ids
            and not duplicate_behaviors
            and not invalid
        ),
    }


def build_manifest(tasks: Iterable[ObstacleTask],
                   version: str = COURSE_VERSION) -> ObstacleManifest:
    """Build a frozen course, or refuse.

    There is no partial success here on purpose. An obstacle course that is
    short a family, or that reaches its total with restated duplicates, would
    still produce a confident `N/N` line in a report -- and that line is the
    evidence the acceptance contract turns on. The only way to obtain a
    manifest object is to satisfy every invariant.
    """
    ordered = sorted(tasks, key=lambda task: task.task_id)
    report = audit_manifest(ordered)
    if not report["complete"]:
        raise ManifestError(_describe_incomplete(report))

    digest = hashlib.sha256(
        json.dumps([task.to_json() for task in ordered], sort_keys=True,
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return ObstacleManifest(version=version, tasks=tuple(ordered),
                            digest=digest)


def _describe_incomplete(report: Mapping) -> str:
    parts = [
        f"obstacle course incomplete: {report['total']} of "
        f"{report['required_total']} tasks"
    ]
    if report["family_shortfalls"]:
        detail = ", ".join(
            f"{family} short {count}" if count > 0
            else f"{family} over by {-count}"
            for family, count in sorted(report["family_shortfalls"].items())
        )
        parts.append(f"family counts wrong: {detail}")
    if report["unknown_families"]:
        parts.append(f"unknown families: {report['unknown_families']}")
    if report["duplicate_ids"]:
        parts.append(f"duplicate ids: {report['duplicate_ids'][:5]}")
    if report["duplicate_behaviors"]:
        pairs = ", ".join(
            f"{left}~{right}" for left, right in report["duplicate_behaviors"][:5]
        )
        parts.append(f"tasks measuring identical behaviour: {pairs}")
    if report["invalid_tasks"]:
        parts.append(f"invalid tasks: {report['invalid_tasks'][:3]}")
    return "; ".join(parts)


def manifest_path(version: str = COURSE_VERSION) -> Path:
    return MANIFEST_ROOT / version / "manifest.json"


def freeze_manifest(manifest: ObstacleManifest,
                    path: Path | None = None) -> Path:
    """Write the course to disk exactly once.

    The contract requires the course to be frozen before the run that admits
    it. Overwriting an existing manifest in place would let a failing task be
    edited into a passing one between runs while the version string stayed
    the same, so an existing file with a different digest is an error rather
    than something to replace.
    """
    target = path or manifest_path(manifest.version)
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing.get("digest") != manifest.digest:
            raise ManifestError(
                f"{target} already holds course {existing.get('digest')!r}; "
                f"refusing to overwrite it with {manifest.digest!r}. Publish "
                "a new COURSE_VERSION instead of mutating a frozen course."
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest.to_json(), indent=2, sort_keys=True,
                   ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_manifest(path: Path | None = None) -> ObstacleManifest:
    """Load a frozen course and re-verify it still is the course it claims.

    A manifest is read back at run time and at audit time. Recomputing the
    digest here means an edited task file cannot pass itself off as the
    frozen course that a stored result was measured against.
    """
    target = path or manifest_path()
    record = json.loads(target.read_text(encoding="utf-8"))
    tasks = []
    for entry in record["tasks"]:
        provenance = Provenance(**entry["provenance"])
        tasks.append(ObstacleTask(
            task_id=entry["task_id"],
            family=entry["family"],
            language=entry["language"],
            toolchain=entry["toolchain"],
            prompt=entry["prompt"],
            validator=entry["validator"],
            timeout_seconds=float(entry["timeout_seconds"]),
            provenance=provenance,
            fixtures=dict(entry.get("fixtures") or {}),
            response_filename=entry.get("response_filename", "candidate.py"),
        ))
    rebuilt = build_manifest(tasks, version=record["version"])
    if rebuilt.digest != record.get("digest"):
        raise ManifestError(
            f"{target}: stored digest {record.get('digest')!r} does not match "
            f"its own tasks ({rebuilt.digest!r}); the frozen course was edited"
        )
    return rebuilt


def normalize_prompt(prompt: str) -> str:
    """Collapse a prompt to the form both sides of a leak check compare on."""
    return re.sub(r"\s+", " ", prompt.strip().casefold())


def distinctive_spans(prompts: Mapping[str, str],
                      min_span: int = 60) -> dict[str, str]:
    """Pick, per task, a span that identifies only that task.

    A fixed-length prefix is the obvious fingerprint and the wrong one. Tasks
    within a family deliberately share an opening -- "implement a python
    function that ..." -- so a prefix can be common to dozens of prompts, and
    a single training row containing that shared opening would then be
    reported as a leak of every one of them. Hundreds of false leaks are not
    a conservative failure: they bury the real one.

    So each task gets the shortest prefix that no other task shares, and a
    task whose prefix is never unique falls back to its whole prompt. Every
    returned span still matches its own task, so nothing becomes undetectable
    -- the search only stops being ambiguous about which task leaked.
    """
    normalized = {task_id: text for task_id, text in prompts.items()
                  if len(text) >= min_span}
    spans: dict[str, str] = {}
    lengths = (min_span, min_span * 2, min_span * 4)
    for task_id, text in normalized.items():
        chosen = text
        for length in lengths:
            if length >= len(text):
                break
            prefix = text[:length]
            if sum(1 for other in normalized.values()
                   if other.startswith(prefix)) == 1:
                chosen = prefix
                break
        spans[task_id] = chosen
    return spans


def held_out_violations(manifest: ObstacleManifest,
                        corpus_paths: Sequence[Path],
                        *, min_span: int = 60) -> list[dict]:
    """Find obstacle material that leaked into the training corpora.

    The contract states the held-out prompts and fixtures must never become
    training rows, and a leak is invisible in the score it produces: a
    memorised task passes, so leakage inflates exactly the number the
    contract relies on. Whitespace and case are normalized on both sides so a
    row that absorbed a prompt with different formatting is still caught.
    """
    spans = distinctive_spans(
        {task.task_id: normalize_prompt(task.prompt)
         for task in manifest.tasks},
        min_span=min_span,
    )

    violations: list[dict] = []
    for corpus in corpus_paths:
        if not corpus.exists():
            continue
        with corpus.open("r", encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, start=1):
                haystack = re.sub(r"\s+", " ", line.casefold())
                for task_id, span in spans.items():
                    if span in haystack:
                        violations.append({
                            "task_id": task_id,
                            "corpus": str(corpus),
                            "line": number,
                        })
    return violations


#: Names of a normative document a prompt is written against. Two tasks
#: citing the same one are not automatically duplicates -- RFC 9110 covers
#: conditional requests and range requests both -- but they are the pairs
#: worth deciding about deliberately.
#:
#: A pattern earns its place only when naming the document implies measuring
#: that document's behaviour. POSIX was tried and removed: four prompts say
#: "POSIX file paths" while measuring archive determinism, ignore-rule
#: matching, token verification and path containment, so every pair it
#: reported was noise, and a scan that mostly cries wolf gets rubber-stamped.
#: SPDX was removed for the opposite reason -- it appears in provenance
#: metadata, never in a prompt, so the pattern could not fire at all.
#: No capture groups: `findall` must return the whole citation, or the key
#: loses the prefix and RFC 3986 collides with ISO 3986.
_CITATION_PATTERNS = (
    re.compile(r"\bRFC\s?\d{3,5}\b"),
    re.compile(r"\bISO\s?\d{3,5}\b"),
    re.compile(r"\bSemantic Versioning\b", re.IGNORECASE),
    re.compile(r"\bWCAG\b"),
    re.compile(r"\bUnicode\b"),
)

#: The public names a validator demands of the candidate. `_support.require`
#: emits exactly this line, so the set is the task's required surface.
_REQUIRED_SYMBOL = re.compile(r"hasattr\(candidate, '([^']+)'\)")


def capability_overlaps(tasks: Sequence[ObstacleTask]) -> list[dict]:
    """Report tasks that may measure one capability twice.

    `audit_manifest` cannot see this and is not meant to. `behavior_digest`
    hashes the prompt and the validator, so two tasks asking for the same
    behaviour in different words are distinct to it -- a course of a thousand
    tasks covering eight hundred capabilities would still report 1000/1000,
    which is the number the acceptance contract rests on.

    Not hypothetical. On 2026-09-05 the requirements family gained an
    RFC 3986 reference resolver and a SemVer precedence comparison that
    `validation_parsing_serialization` already owned, down to the same base
    URI and the same prerelease chain. Both were found by reading the
    neighbouring family after committing, which is not a control.

    Two signals, each cheap, each a prompt to look rather than a verdict:

    - the public symbols a validator demands, because a duplicated capability
      is usually requested under the same name; and
    - the normative document a prompt cites, counted only across families,
      because one family reusing a specification for two of its own
      behaviours is ordinary.

    Neither alone would have caught both of that day's duplicates. The
    versions pair shared `compare_versions` but cited no RFC; the URI pair
    cited RFC 3986 under two different function names. Together they cover
    both, and a real collision still has to be judged by a person: a
    line-based `apply_patch` over file hunks and a JSON one over a document
    share a name and measure nothing in common.
    """
    by_symbol: dict[str, list[str]] = {}
    by_citation: dict[str, list[tuple[str, str]]] = {}
    for task in tasks:
        for name in sorted(set(_REQUIRED_SYMBOL.findall(task.validator))):
            by_symbol.setdefault(name, []).append(task.task_id)
        cited: set[str] = set()
        for pattern in _CITATION_PATTERNS:
            for hit in pattern.findall(task.prompt):
                cited.add(re.sub(r"\s+", "", hit).upper())
        for key in sorted(cited):
            by_citation.setdefault(key, []).append((task.family, task.task_id))

    overlaps: list[dict] = []
    for name, task_ids in sorted(by_symbol.items()):
        if len(task_ids) > 1:
            overlaps.append({"kind": "symbol", "key": name,
                             "task_ids": sorted(task_ids)})
    for key, entries in sorted(by_citation.items()):
        if len({family for family, _ in entries}) > 1:
            overlaps.append({"kind": "citation", "key": key,
                             "task_ids": sorted(i for _, i in entries)})
    return overlaps


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=COURSE_VERSION)
    parser.add_argument("--freeze", action="store_true",
                        help="write the course once it is complete")
    parser.add_argument("--corpus", action="append", default=[], type=Path,
                        help="training corpus to check for held-out leakage")
    args = parser.parse_args(argv)

    from scripts.programming_obstacle_tasks import load_authored_tasks

    tasks = load_authored_tasks()
    report = audit_manifest(tasks)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["complete"]:
        return 2

    manifest = build_manifest(tasks, version=args.version)
    if args.corpus:
        leaks = held_out_violations(manifest, args.corpus)
        if leaks:
            print(json.dumps({"held_out_violations": leaks[:20]}, indent=2))
            return 3
    if args.freeze:
        print(f"frozen: {freeze_manifest(manifest)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
