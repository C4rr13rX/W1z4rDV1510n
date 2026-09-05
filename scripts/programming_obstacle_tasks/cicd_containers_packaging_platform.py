"""Held-out tasks: CI/CD, containers, packaging, and platform engineering.

What makes this family hard is that almost every defect in it produces a
green pipeline. A build that embeds a timestamp still ships. A layer cache
keyed on filenames still hits. A pipeline that treats a skipped dependency as
a satisfied one still reports success. A retention sweep that deletes the
artifact a deployment is pinned to runs to completion and only fails later,
during an incident, when the rollback target is gone. None of these announce
themselves, which is why the accepting test is usually written and passes.

So the validators here assert the property that separates the correct
mechanism from the plausible one, and several of them are *equalities between
two runs* rather than checks of a single output: the same inputs presented in
a different order, or built a second time, must produce identical bytes.
Reproducibility is not observable from one build, which is exactly why it is
so often absent.

The version-comparison primitive belongs to
``validation_parsing_serialization``; what is under test here is selection --
choosing the release or the wheel a resolver must install given constraints
and platform tags, and refusing when nothing satisfies them. Likewise the
scheduling of parallel work belongs to ``concurrency_async_distributed``:
task 0005 is about how failure and conditions propagate through a job graph,
not about how to run it fast.
"""

from __future__ import annotations

from scripts.programming_obstacle_tasks import task
from scripts.programming_obstacle_tasks._support import LOAD_CANDIDATE, require

FAMILY = "cicd_containers_packaging_platform"

TASKS = [
    task(
        f"{FAMILY}-0001", FAMILY,
        prompt=(
            "Implement a Python function build_archive(entries) returning the "
            "bytes of a reproducible gzip-compressed tar. entries is an "
            "iterable of (path, data, mode) triples where path is a relative "
            "POSIX path, data is bytes, and mode is the permission bits. The "
            "same set of entries must produce byte-identical output no matter "
            "what order they arrive in, what time the build runs, or which "
            "user runs it: sort the members by path, and record for each "
            "member mtime 0, uid 0, gid 0, empty uname and gname, the given "
            "mode, and type REGTYPE. Use tarfile's USTAR format. The gzip "
            "wrapper must also carry no timestamp and no original filename, "
            "so write it with mtime 0. Raise ValueError if two entries share "
            "a path, if a path is absolute or contains a '..' component, or "
            "if a path is empty."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("build_archive") + r'''
import gzip
import io
import struct
import tarfile

entries = [
    ("src/main.py", b"print('hello')\n", 0o644),
    ("README.md", b"# project\n", 0o644),
    ("bin/run.sh", b"#!/bin/sh\nexec true\n", 0o755),
]

first = build_archive(entries)
# The same inputs in a different order are the same archive. This is the
# assertion a single build can never make, and the one reproducibility means.
second = build_archive(list(reversed(entries)))
assert first == second, (
    "input order changed the archive bytes; members must be sorted by path"
)
assert build_archive(iter(entries)) == first, "an iterator input differed"

# The gzip header's MTIME field is bytes 4..8 and is the timestamp that
# quietly defeats most reproducible builds.
assert first[:2] == b"\x1f\x8b", "output is not gzip"
assert struct.unpack("<I", first[4:8])[0] == 0, (
    "the gzip header carries a build timestamp"
)
assert first[3] & 0x08 == 0, "the gzip header carries an original filename"

# The archive still has to be a correct archive.
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(first)), mode="r:") as tar:
    members = tar.getmembers()
    assert [member.name for member in members] == [
        "README.md", "bin/run.sh", "src/main.py"
    ], [member.name for member in members]
    for member in members:
        assert member.mtime == 0, f"{member.name} carries mtime {member.mtime}"
        assert member.uid == 0 and member.gid == 0, member.name
        assert member.uname == "" and member.gname == "", (
            f"{member.name} leaks the building account: "
            f"{member.uname!r}/{member.gname!r}"
        )
        assert member.type == tarfile.REGTYPE, member.name
    modes = {member.name: member.mode for member in members}
    assert modes["bin/run.sh"] == 0o755, modes
    assert modes["README.md"] == 0o644, modes
    payload = tar.extractfile("src/main.py").read()
    assert payload == b"print('hello')\n", payload

# --- stated error behaviour ----------------------------------------------
for bad in (
    [("a.py", b"", 0o644), ("a.py", b"x", 0o644)],
    [("/etc/passwd", b"", 0o644)],
    [("../escape", b"", 0o644)],
    [("pkg/../../escape", b"", 0o644)],
    [("", b"", 0o644)],
):
    try:
        build_archive(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad[0][0]!r}")
''',
    ),
    task(
        f"{FAMILY}-0002", FAMILY,
        prompt=(
            "Implement a Python function select_versions(requirements, "
            "available) that locks a dependency set. requirements is a list "
            "of requirement strings such as 'flask>=2.0,<3.0', 'urllib3', "
            "'requests==2.31.0', 'numpy!=1.24.0,>=1.23', or 'attrs~=23.1.2'. "
            "A package may appear in more than one requirement and every "
            "constraint on it applies. available maps a package name to the "
            "list of its published version strings, in no particular order. "
            "Versions are dot-separated non-negative integers compared "
            "component by component, with a missing trailing component "
            "treated as 0, so 1.4 and 1.4.0 are equal and 1.10 is above 1.9. "
            "The operators are ==, !=, >=, <=, >, < and ~=, where ~=X.Y.Z "
            "means >=X.Y.Z with X.Y held fixed and ~=X.Y means >=X.Y with X "
            "held fixed. Return a dict mapping each required package to the "
            "highest available version satisfying all of its constraints. "
            "Raise ValueError naming the package if it is absent from "
            "available or if no version of it satisfies the constraints, and "
            "raise ValueError for a malformed requirement or operator."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("select_versions") + r'''
available = {
    "flask": ["1.1.4", "2.0.0", "2.3.3", "3.0.0"],
    "urllib3": ["1.26.18", "2.0.7", "2.1.0"],
    "requests": ["2.30.0", "2.31.0", "2.32.0"],
    "numpy": ["1.23.5", "1.24.0", "1.24.4", "1.26.4"],
    "attrs": ["23.1.0", "23.1.2", "23.1.9", "23.2.0", "24.1.0"],
}

locked = select_versions([
    "flask>=2.0,<3.0",
    "urllib3",
    "requests==2.31.0",
    "numpy!=1.24.0,>=1.23",
    "attrs~=23.1.2",
], available)
assert locked == {
    "flask": "2.3.3",
    "urllib3": "2.1.0",
    "requests": "2.31.0",
    "numpy": "1.26.4",
    # ~=23.1.2 holds 23.1 fixed, so 23.2.0 is out of range and 23.1.9 wins.
    "attrs": "23.1.9",
}, locked

# Constraints accumulate across separate requirements for one package.
assert select_versions(["flask>=2.0", "flask<2.3"], available) == {
    "flask": "2.0.0"
}, select_versions(["flask>=2.0", "flask<2.3"], available)

# ~=X.Y holds only the major fixed.
assert select_versions(["attrs~=23.1"], available)["attrs"] == "23.2.0", (
    select_versions(["attrs~=23.1"], available)
)

# Padding, not lexicographic order: 1.10 is above 1.9, and 1.4 == 1.4.0.
padded = {"pkg": ["1.4", "1.4.0", "1.9", "1.10"]}
assert select_versions(["pkg<=1.10"], padded)["pkg"] == "1.10", (
    select_versions(["pkg<=1.10"], padded)
)
assert select_versions(["pkg==1.4.0"], {"pkg": ["1.4"]})["pkg"] == "1.4"
assert select_versions(["pkg>1.9"], padded)["pkg"] == "1.10"

# --- an unsatisfiable lock must be refused, and must name the package -----
for bad, needle in (
    (["flask>=3.1"], "flask"),
    (["flask>=2.0,<2.0"], "flask"),
    (["requests==2.31.0", "requests>=2.32.0"], "requests"),
    (["absent>=1.0"], "absent"),
):
    try:
        select_versions(bad, available)
    except ValueError as error:
        assert needle in str(error), (
            f"{bad} raised {error!r}, which does not name {needle}"
        )
    else:
        raise AssertionError(f"accepted an unsatisfiable requirement {bad}")

for malformed in (["flask=>2.0"], ["flask>>2.0"], [">=2.0"], ["flask>=x.y"],
                  [""], ["flask>=2.0,"]):
    try:
        select_versions(malformed, available)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed requirement {malformed}")
''',
    ),
    task(
        f"{FAMILY}-0003", FAMILY,
        prompt=(
            "Implement two Python functions for OCI image identity. "
            "chain_ids(diff_ids) takes the ordered list of layer diff IDs, "
            "each the string 'sha256:' followed by 64 lowercase hex "
            "characters, and returns the list of layer chain IDs. The first "
            "chain ID equals the first diff ID; each later chain ID is "
            "'sha256:' followed by the hex SHA-256 digest of the UTF-8 bytes "
            "of the previous chain ID, a single space, and the current diff "
            "ID. image_id(config) takes the exact bytes of the image config "
            "JSON and returns 'sha256:' followed by their hex SHA-256 digest; "
            "it must digest the bytes as given without reformatting them. "
            "Both raise ValueError on a malformed digest string, and "
            "chain_ids returns an empty list for no layers."
        ),
        timeout_seconds=30.0,
        validator=LOAD_CANDIDATE + require("chain_ids") + require("image_id") + r'''
import hashlib


def digest(index):
    return "sha256:" + hashlib.sha256(f"layer-{index}".encode()).hexdigest()


layers = [digest(index) for index in range(4)]
chain = chain_ids(layers)
assert len(chain) == 4, chain
assert chain[0] == layers[0], "the first chain ID is the first diff ID"

# Recomputed independently: the chain folds the PREVIOUS CHAIN id, not the
# previous diff id. Using the diff id produces a plausible-looking list of
# digests that no registry agrees with.
expected = layers[0]
for index in range(1, 4):
    expected = "sha256:" + hashlib.sha256(
        f"{expected} {layers[index]}".encode()
    ).hexdigest()
    assert chain[index] == expected, (
        f"chain ID {index} does not fold the previous chain ID: "
        f"{chain[index]} != {expected}"
    )

# A different layer anywhere changes every chain ID from that point on, and
# none before it.
altered = list(layers)
altered[2] = digest(99)
shifted = chain_ids(altered)
assert shifted[:2] == chain[:2], "an unrelated layer's chain ID changed"
assert shifted[2] != chain[2] and shifted[3] != chain[3], (
    "changing a layer left a later chain ID unchanged"
)

assert chain_ids([]) == []
assert chain_ids([layers[0]]) == [layers[0]]

# --- the config is digested as bytes, not as a reparsed object -----------
config = b'{"architecture":"amd64","os":"linux"}'
assert image_id(config) == "sha256:" + hashlib.sha256(config).hexdigest()
spaced = b'{"architecture": "amd64", "os": "linux"}'
assert image_id(spaced) != image_id(config), (
    "whitespace was normalized away; the image ID digests the exact bytes"
)

# --- stated error behaviour ----------------------------------------------
for bad in ("sha256:" + "0" * 63, "sha256:" + "0" * 65, "sha256:" + "G" * 64,
            "sha256:" + "A" * 64, "sha512:" + "0" * 64, "0" * 64, ""):
    try:
        chain_ids([bad])
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed digest {bad!r}")

try:
    image_id("not bytes")
except (ValueError, TypeError):
    pass
else:
    raise AssertionError("image_id accepted a str config")
''',
    ),
    task(
        f"{FAMILY}-0004", FAMILY,
        prompt=(
            "Implement a Python function first_rebuilt(instructions, "
            "previous, current) that finds where a container build's layer "
            "cache stops hitting. instructions is a list of (command, "
            "argument) pairs in Dockerfile order; previous and current are "
            "dicts mapping a build-context path to its bytes, for the earlier "
            "and the present build. Each instruction has a cache key that "
            "chains: key(i) is the hex SHA-256 of the UTF-8 bytes of key(i-1) "
            "followed by a newline, the command, a newline, the argument, and "
            "for a COPY or ADD command a newline and then, for every context "
            "path the argument selects, the path, a space, the hex SHA-256 of "
            "its content and a newline, with those lines in sorted path "
            "order. key(-1) is the empty string. A COPY or ADD argument is "
            "'<src> <dst>'; src selects every context path when it is '.', "
            "otherwise the path equal to src and every path beginning with "
            "src followed by '/'. Every other command's key depends only on "
            "the command and argument text. Return the index of the first "
            "instruction whose key differs between the two builds, or None if "
            "every key matches. Raise ValueError if a COPY or ADD argument "
            "does not have exactly two whitespace-separated fields."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("first_rebuilt") + r'''
instructions = [
    ("FROM", "python:3.13-slim"),
    ("WORKDIR", "/app"),
    ("COPY", "requirements.txt /app/"),
    ("RUN", "pip install -r requirements.txt"),
    ("COPY", "src /app/src"),
    ("CMD", "python -m app"),
]
context = {
    "requirements.txt": b"flask==3.0.0\n",
    "src/app.py": b"print('v1')\n",
    "src/util.py": b"HELPERS = 1\n",
    "docs/readme.md": b"unrelated\n",
}

assert first_rebuilt(instructions, context, dict(context)) is None

# Editing application source must invalidate the COPY at index 4 and nothing
# earlier -- that ordering is the whole reason requirements are copied first.
edited = dict(context, **{"src/app.py": b"print('v2')\n"})
assert first_rebuilt(instructions, context, edited) == 4, (
    first_rebuilt(instructions, context, edited)
)

# A file the instructions never select changes nothing.
unrelated = dict(context, **{"docs/readme.md": b"rewritten\n"})
assert first_rebuilt(instructions, context, unrelated) is None, (
    "a context file no COPY selects invalidated the cache"
)

# Same name, same size, different content: a key built from filenames or
# sizes hits here, and ships the previous build's code.
same_length = dict(context, **{"src/util.py": b"HELPERS = 2\n"})
assert first_rebuilt(instructions, context, same_length) == 4, (
    "content is not part of the cache key"
)

# Adding a file under the copied prefix invalidates; adding one outside does
# not.
added_inside = dict(context, **{"src/new.py": b"\n"})
assert first_rebuilt(instructions, context, added_inside) == 4
added_outside = dict(context, **{"tools/x.py": b"\n"})
assert first_rebuilt(instructions, context, added_outside) is None

# Removing a copied file invalidates.
removed = {k: v for k, v in context.items() if k != "src/util.py"}
assert first_rebuilt(instructions, context, removed) == 4

# An earlier miss wins even when a later one also differs: the answer is the
# first index, because everything after it rebuilds anyway.
both = dict(edited, **{"requirements.txt": b"flask==3.0.1\n"})
assert first_rebuilt(instructions, context, both) == 2, (
    first_rebuilt(instructions, context, both)
)

# A changed instruction invalidates from that point, with no context change.
changed = list(instructions)
changed[3] = ("RUN", "pip install --no-cache-dir -r requirements.txt")
assert first_rebuilt(changed, context, context) is None, (
    "both builds ran the same instruction list here"
)

# 'COPY . /app' selects everything, so any context change invalidates it.
copy_all = [("FROM", "scratch"), ("COPY", ". /app")]
assert first_rebuilt(copy_all, context, unrelated) == 1

for bad in ([("COPY", "src")], [("ADD", "a b c")], [("COPY", "")]):
    try:
        first_rebuilt(bad, context, context)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad}")
''',
    ),
    task(
        f"{FAMILY}-0005", FAMILY,
        prompt=(
            "Implement a Python function evaluate_pipeline(jobs, results) "
            "that decides which CI jobs run and what the pipeline concludes. "
            "jobs is a list of dicts with 'name', 'needs' (a list of job "
            "names, default empty), 'condition' (one of 'on_success', "
            "'on_failure' or 'always', default 'on_success') and "
            "'continue_on_error' (default False). results maps a job name to "
            "'pass' or 'fail', the outcome that job produces if it runs. A "
            "job runs once all of its needs have finished and its condition "
            "holds: 'on_success' requires every need to have an effective "
            "status of success, 'on_failure' requires at least one need with "
            "an effective status of failure, and 'always' requires only that "
            "the needs finished, running even if they were skipped. A job "
            "that did not run is 'skipped', and a skipped need is neither a "
            "success nor a failure, so an 'on_success' job needing it is "
            "skipped too. A job that ran has status 'success' or 'failed' "
            "from results, but its effective status is success when "
            "continue_on_error is set. Return a dict with 'statuses', mapping "
            "every job name to 'success', 'failed' or 'skipped', and "
            "'conclusion', which is 'failed' if any job's status is 'failed' "
            "and its continue_on_error is not set, otherwise 'success'. Raise "
            "ValueError if needs name an unknown job, if the graph has a "
            "cycle, if a job that runs has no entry in results, or if two "
            "jobs share a name."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("evaluate_pipeline") + r'''
jobs = [
    {"name": "build"},
    {"name": "unit", "needs": ["build"]},
    {"name": "lint", "needs": ["build"], "continue_on_error": True},
    {"name": "deploy", "needs": ["unit", "lint"]},
    {"name": "notify", "needs": ["deploy"], "condition": "always"},
    {"name": "rollback", "needs": ["deploy"], "condition": "on_failure"},
]

# --- everything green -----------------------------------------------------
green = evaluate_pipeline(jobs, {
    "build": "pass", "unit": "pass", "lint": "pass",
    "deploy": "pass", "notify": "pass", "rollback": "pass",
})
assert green["statuses"] == {
    "build": "success", "unit": "success", "lint": "success",
    "deploy": "success", "notify": "success", "rollback": "skipped",
}, green
assert green["conclusion"] == "success", green

# --- a continue_on_error job fails without stopping the pipeline ----------
# This is the case a naive implementation gets wrong in the expensive
# direction: it blocks the deploy on an advisory lint job.
advisory = evaluate_pipeline(jobs, {
    "build": "pass", "unit": "pass", "lint": "fail",
    "deploy": "pass", "notify": "pass", "rollback": "pass",
})
assert advisory["statuses"]["lint"] == "failed", advisory
assert advisory["statuses"]["deploy"] == "success", (
    "an advisory failure blocked the deploy"
)
assert advisory["conclusion"] == "success", (
    f"continue_on_error did not absorb the failure: {advisory['conclusion']}"
)

# --- a real failure skips the successors, and the two conditions differ ---
broken = evaluate_pipeline(jobs, {
    "build": "pass", "unit": "fail", "lint": "pass",
    "deploy": "pass", "notify": "pass", "rollback": "pass",
})
assert broken["statuses"] == {
    "build": "success", "unit": "failed", "lint": "success",
    # deploy needs a failed job, so it is skipped; notify runs anyway, and
    # rollback needs a FAILED need -- deploy was skipped, not failed.
    "deploy": "skipped", "notify": "success", "rollback": "skipped",
}, broken
assert broken["conclusion"] == "failed", broken

# --- on_failure fires only for a genuine failure of a need ---------------
pair = [
    {"name": "test"},
    {"name": "cleanup", "needs": ["test"], "condition": "on_failure"},
]
assert evaluate_pipeline(pair, {"test": "fail", "cleanup": "pass"})[
    "statuses"] == {"test": "failed", "cleanup": "success"}
assert evaluate_pipeline(pair, {"test": "pass", "cleanup": "pass"})[
    "statuses"] == {"test": "success", "cleanup": "skipped"}

# A job with no needs and an on_failure condition has nothing that failed.
assert evaluate_pipeline(
    [{"name": "solo", "condition": "on_failure"}], {"solo": "pass"}
)["statuses"] == {"solo": "skipped"}

# --- stated error behaviour ----------------------------------------------
for bad_jobs, bad_results in (
    ([{"name": "a", "needs": ["ghost"]}], {"a": "pass"}),
    ([{"name": "a", "needs": ["b"]}, {"name": "b", "needs": ["a"]}],
     {"a": "pass", "b": "pass"}),
    ([{"name": "a"}, {"name": "a"}], {"a": "pass"}),
    ([{"name": "a"}], {}),
):
    try:
        evaluate_pipeline(bad_jobs, bad_results)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted {bad_jobs} / {bad_results}")
''',
    ),
    task(
        f"{FAMILY}-0006", FAMILY,
        prompt=(
            "Implement a Python function ignored(paths, patterns) that "
            "applies build-context ignore rules. paths is a list of relative "
            "POSIX file paths and patterns is the lines of an ignore file. "
            "Return the sorted list of paths that are excluded. A blank line "
            "or a line whose first character is '#' is skipped; trailing "
            "spaces are stripped. A leading '!' negates, and the LAST pattern "
            "that matches decides. A pattern ending in '/' is a directory "
            "pattern and applies only to a path's directory components; every "
            "other pattern is a file pattern and applies only to the path "
            "itself. A pattern is anchored when it contains a '/' before its "
            "trailing one, and a leading '/' anchors it and is then dropped. "
            "An anchored file pattern is matched against the whole path and an "
            "unanchored one against the path's final component; an anchored "
            "directory pattern is matched against each directory prefix of "
            "the path and an unanchored one against each directory component. "
            "In a pattern '*' matches any run of characters except '/', '?' "
            "matches one character except '/', and a '**' component matches "
            "any number of components, including none, spanning '/'. Decide "
            "the directory patterns first: a path whose directory is excluded "
            "by them is excluded, and the file patterns are not consulted at "
            "all, so a negation naming a file inside an excluded directory "
            "cannot bring it back -- only re-including the directory can."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("ignored") + r'''
paths = [
    "main.py",
    "build/output.bin",
    "build/keep.txt",
    "src/main.py",
    "src/build/tmp.o",
    "src/vendor/lib.py",
    "docs/api.md",
    "docs/img/logo.png",
    ".env",
    "logs/app.log",
    "logs/nested/deep/app.log",
    "notes.txt",
    "a.pyc",
    "src/b.pyc",
]

# An unanchored basename pattern matches at any depth; an anchored one does
# not. Getting this backwards silently ships build artefacts into an image.
assert ignored(paths, ["*.pyc"]) == ["a.pyc", "src/b.pyc"]
assert ignored(paths, ["/a.pyc"]) == ["a.pyc"]

# A directory pattern excludes the whole subtree at any depth.
assert ignored(paths, ["build/"]) == [
    "build/keep.txt", "build/output.bin", "src/build/tmp.o"
], ignored(paths, ["build/"])
# The same word without the trailing slash is a FILE pattern, and no file is
# named 'build'. Collapsing the two excludes a source tree by accident.
assert ignored(paths, ["build"]) == [], ignored(paths, ["build"])
# Anchored, it excludes only the one at the root.
assert ignored(paths, ["/build/"]) == ["build/keep.txt", "build/output.bin"]

# Last match wins, in file order.
assert ignored(paths, ["*.log", "!logs/nested/**"]) == ["logs/app.log"], (
    ignored(paths, ["*.log", "!logs/nested/**"])
)
assert ignored(paths, ["!*.log", "*.log"]) == [
    "logs/app.log", "logs/nested/deep/app.log"
]

# '**' spans any number of components, including none.
assert ignored(paths, ["docs/**/*.md"]) == ["docs/api.md"], (
    ignored(paths, ["docs/**/*.md"])
)
assert sorted(ignored(paths, ["**/vendor/**"])) == ["src/vendor/lib.py"]

# '*' does not cross a separator.
assert ignored(paths, ["docs/*"]) == ["docs/api.md"], ignored(paths, ["docs/*"])
assert ignored(paths, ["?.pyc"]) == ["a.pyc", "src/b.pyc"]

# --- the rule that surprises everyone ------------------------------------
# Once a directory is excluded it is not descended into, so naming a file
# inside it cannot bring it back. Re-including the directory can.
assert ignored(paths, ["build/", "!build/keep.txt"]) == [
    "build/keep.txt", "build/output.bin", "src/build/tmp.o"
], (
    "a file was re-included out of an excluded directory, which the ignore "
    "semantics do not allow"
)
assert ignored(paths, ["build/", "!build/", "*.bin"]) == ["build/output.bin"], (
    ignored(paths, ["build/", "!build/", "*.bin"])
)

# --- comments, blanks, trailing space, and the escaped hash --------------
assert ignored(paths, ["# a comment", "", "   ", "*.env   "]) == [".env"], (
    ignored(paths, ["# a comment", "", "   ", "*.env   "])
)
assert ignored(paths, []) == []
assert ignored([], ["*"]) == []
''',
    ),
    task(
        f"{FAMILY}-0007", FAMILY,
        prompt=(
            "Implement a Python function plan_deletions(artifacts, policy, "
            "referenced, now) choosing which build artifacts a retention "
            "sweep may remove. artifacts is a list of dicts with 'id', "
            "'branch', 'created_unix' and 'tags' (a list of release tags). "
            "policy is a dict with 'keep_per_branch', a non-negative integer, "
            "'keep_tagged', a bool, and 'min_age_seconds', a non-negative "
            "number. referenced is a collection of artifact ids that a live "
            "deployment currently points at, and now is the sweep time. "
            "Return the sorted list of ids to delete. An artifact is kept if "
            "it is among the newest keep_per_branch artifacts on its own "
            "branch, ranking by created_unix descending and breaking ties by "
            "id descending; or it has at least one tag and keep_tagged is "
            "set; or its id is in referenced; or now - created_unix is less "
            "than min_age_seconds. Everything else is deleted. Raise "
            "ValueError if two artifacts share an id, if a required field is "
            "missing, or if any policy number is negative."
        ),
        timeout_seconds=30.0,
        validator=LOAD_CANDIDATE + require("plan_deletions") + r'''
def artifact(identifier, branch, age_seconds, tags=()):
    return {
        "id": identifier,
        "branch": branch,
        "created_unix": 1_000_000 - age_seconds,
        "tags": list(tags),
    }


now = 1_000_000
artifacts = [
    artifact("main-05", "main", 100),
    artifact("main-04", "main", 200),
    artifact("main-03", "main", 300, tags=["v1.2.0"]),
    artifact("main-02", "main", 400),
    artifact("main-01", "main", 500),
    artifact("pr-9-02", "pr-9", 150),
    artifact("pr-9-01", "pr-9", 250),
    artifact("old-01", "release", 900),
]
policy = {"keep_per_branch": 2, "keep_tagged": True, "min_age_seconds": 0}

deleted = plan_deletions(artifacts, policy, {"main-02"}, now)
# main-05/main-04 are the newest two; main-03 is tagged; main-02 is deployed;
# pr-9 keeps both of its two; release keeps its only one.
assert deleted == ["main-01"], deleted

# The deployed artifact is the one a rollback needs. Dropping the reference
# check deletes exactly the thing an incident will ask for.
assert plan_deletions(artifacts, policy, set(), now) == ["main-01", "main-02"], (
    plan_deletions(artifacts, policy, set(), now)
)

# Tag protection off: the tagged release becomes deletable.
untagged_policy = dict(policy, keep_tagged=False)
assert plan_deletions(artifacts, untagged_policy, set(), now) == [
    "main-01", "main-02", "main-03"
], plan_deletions(artifacts, untagged_policy, set(), now)

# A minimum age protects everything younger, including artifacts that every
# other rule would have released -- a sweep that ignores it deletes the build
# a deployment rolled out minutes ago.
sweep_all = {"keep_per_branch": 0, "keep_tagged": False,
             "min_age_seconds": 450}
assert plan_deletions(artifacts, sweep_all, set(), now) == [
    "main-01", "old-01"
], plan_deletions(artifacts, sweep_all, set(), now)
assert plan_deletions(artifacts, dict(policy, min_age_seconds=600),
                      set(), now) == [], (
    plan_deletions(artifacts, dict(policy, min_age_seconds=600), set(), now)
)

# keep_per_branch 0 keeps nothing on age alone.
none_policy = {"keep_per_branch": 0, "keep_tagged": False,
               "min_age_seconds": 0}
assert plan_deletions(artifacts, none_policy, set(), now) == sorted(
    item["id"] for item in artifacts
), plan_deletions(artifacts, none_policy, set(), now)

# --- the tie-break is part of the contract -------------------------------
# Two artifacts built in the same second: without a deterministic tie-break
# two sweeps of the same input disagree about which one survives.
tied = [
    artifact("b", "main", 10),
    artifact("a", "main", 10),
    artifact("c", "main", 20),
]
keep_one = {"keep_per_branch": 1, "keep_tagged": False, "min_age_seconds": 0}
assert plan_deletions(tied, keep_one, set(), now) == ["a", "c"], (
    plan_deletions(tied, keep_one, set(), now)
)
assert plan_deletions(list(reversed(tied)), keep_one, set(), now) == ["a", "c"], (
    "the result depends on input order"
)

# --- stated error behaviour ----------------------------------------------
for bad_artifacts, bad_policy in (
    ([artifact("x", "main", 1), artifact("x", "main", 2)], policy),
    ([{"id": "x", "branch": "main"}], policy),
    (artifacts, dict(policy, keep_per_branch=-1)),
    (artifacts, dict(policy, min_age_seconds=-5)),
):
    try:
        plan_deletions(bad_artifacts, bad_policy, set(), now)
    except ValueError:
        pass
    else:
        raise AssertionError("an invalid retention input was accepted")
''',
    ),
    task(
        f"{FAMILY}-0008", FAMILY,
        prompt=(
            "Implement a Python function best_wheel(filenames, "
            "supported_tags) selecting the wheel an installer should use. "
            "Each filename has the form "
            "distribution-version(-build)?-python-abi-platform.whl, where "
            "build begins with a digit and the last three fields are tag "
            "sets whose members are joined by '.'. A wheel's compatibility "
            "tags are every combination python.abi.platform written as "
            "'python-abi-platform'. supported_tags is the interpreter's "
            "list of acceptable tags in descending order of preference. A "
            "wheel is a candidate if any of its tags appears in "
            "supported_tags, and its rank is the lowest index at which one "
            "does. Return the filename with the lowest rank; break ties by "
            "the higher build number, treating an absent build as lower than "
            "any present one, and then by the filename that sorts first. "
            "Return None if no filename is a candidate. Raise ValueError for "
            "a filename that does not end in '.whl' or that does not have "
            "five or six '-' separated fields, or if a build field does not "
            "start with a digit."
        ),
        timeout_seconds=30.0,
        validator=LOAD_CANDIDATE + require("best_wheel") + r'''
supported = [
    "cp313-cp313-manylinux_2_28_x86_64",
    "cp313-abi3-manylinux_2_28_x86_64",
    "cp313-none-manylinux_2_28_x86_64",
    "py3-none-any",
]

# A compiled wheel for this exact interpreter beats a pure-python fallback,
# whatever order the index lists them in.
wheels = [
    "pkg-1.0.0-py2.py3-none-any.whl",
    "pkg-1.0.0-cp313-cp313-manylinux_2_28_x86_64.whl",
    "pkg-1.0.0-cp313-abi3-manylinux_2_28_x86_64.whl",
]
assert best_wheel(wheels, supported) == (
    "pkg-1.0.0-cp313-cp313-manylinux_2_28_x86_64.whl"
), best_wheel(wheels, supported)
assert best_wheel(list(reversed(wheels)), supported) == (
    "pkg-1.0.0-cp313-cp313-manylinux_2_28_x86_64.whl"
), "the answer depended on the order the candidates were listed"

# A compressed tag set expands: py2.py3-none-any carries py3-none-any.
assert best_wheel(["pkg-1.0.0-py2.py3-none-any.whl"], supported) == (
    "pkg-1.0.0-py2.py3-none-any.whl"
)
assert best_wheel(["pkg-1.0.0-py2-none-any.whl"], supported) is None, (
    "a wheel with no supported tag was selected"
)

# Rank is the BEST tag the wheel carries, not the first one written.
mixed = ["pkg-1.0.0-py3.cp313-none.cp313-any.manylinux_2_28_x86_64.whl"]
assert best_wheel(mixed, supported) == mixed[0]

# --- build number breaks a tie, and absent is lowest ----------------------
tied = [
    "pkg-1.0.0-cp313-abi3-manylinux_2_28_x86_64.whl",
    "pkg-1.0.0-2-cp313-abi3-manylinux_2_28_x86_64.whl",
    "pkg-1.0.0-10-cp313-abi3-manylinux_2_28_x86_64.whl",
]
assert best_wheel(tied, supported) == (
    "pkg-1.0.0-10-cp313-abi3-manylinux_2_28_x86_64.whl"
), best_wheel(tied, supported)
assert best_wheel(tied[:1] + tied[1:2], supported) == (
    "pkg-1.0.0-2-cp313-abi3-manylinux_2_28_x86_64.whl"
)

# A full tie falls back to the filename that sorts first, so two installers
# resolve the same wheel.
same = [
    "zeta-1.0.0-cp313-abi3-manylinux_2_28_x86_64.whl",
    "alpha-1.0.0-cp313-abi3-manylinux_2_28_x86_64.whl",
]
assert best_wheel(same, supported) == same[1]

assert best_wheel([], supported) is None
assert best_wheel(wheels, []) is None

# --- stated error behaviour ----------------------------------------------
for bad in ("pkg-1.0.0-cp313-abi3-any.tar.gz", "pkg-1.0.0-cp313-abi3.whl",
            "pkg-1.0.0-a1-cp313-abi3-any.whl",
            "pkg-1.0.0-1-2-cp313-abi3-any.whl", "pkg.whl"):
    try:
        best_wheel([bad], supported)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed wheel name {bad!r}")
''',
    ),
]
