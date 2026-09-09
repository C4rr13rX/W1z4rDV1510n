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
    # ---------------------------------------------------------------------
    # Ids from 0101 up. The sequential range below 0100 stays free for
    # whoever extends this family from the front, so two sessions appending
    # to opposite ends of the file cannot mint the same id.
    # ---------------------------------------------------------------------
    task(
        f"{FAMILY}-0101", FAMILY,
        prompt=(
            "Implement a Python function canonical_reference(reference) that "
            "expands a container image reference into its canonical parts, "
            "returning a dict with keys 'registry', 'repository', 'tag' and "
            "'digest'. Parse it as a registry does. A leading component is "
            "the registry only when it contains a '.' or a ':' or is exactly "
            "'localhost'; otherwise there is no registry in the string and it "
            "defaults to 'docker.io'. A ':' introduces a tag only when it "
            "appears after the last '/', so a registry port is not a tag. An "
            "'@' introduces a digest, which must be 'sha256:' followed by "
            "exactly 64 lowercase hex characters. On docker.io a repository "
            "with no '/' is prefixed with 'library/'; on any other registry "
            "it is left alone. The tag defaults to 'latest' only when no "
            "digest was given -- with a digest and no tag, 'tag' is None. "
            "Raise ValueError for an empty reference, a repository component "
            "that is not lowercase alphanumeric with '.', '_' or '-' "
            "separators, a malformed tag, or a malformed digest."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("canonical_reference") + r'''
DIGEST = "sha256:" + "3f" * 32

cases = {
    # No registry component at all.
    "nginx": ("docker.io", "library/nginx", "latest", None),
    "nginx:1.25": ("docker.io", "library/nginx", "1.25", None),
    # A bare first component is NOT a registry: it has no dot and no port,
    # so this is a docker.io namespace and the whole string is the
    # repository. Splitting on the first '/' unconditionally is the
    # mistake this case exists to catch.
    "myorg/app": ("docker.io", "myorg/app", "latest", None),
    "myorg/app:v1": ("docker.io", "myorg/app", "v1", None),
    "deep/org/app:v1": ("docker.io", "deep/org/app", "v1", None),
    # 'localhost' is a registry by name even without a dot.
    "localhost/app": ("localhost", "app", "latest", None),
    # A port makes it a registry, and the port's colon is not a tag.
    "localhost:5000/app:dev": ("localhost:5000", "app", "dev", None),
    "example.com:8443/a/b": ("example.com:8443", "a/b", "latest", None),
    "registry.example.com/team/app:v2":
        ("registry.example.com", "team/app", "v2", None),
    # A digest suppresses the default tag rather than joining it.
    "nginx@" + DIGEST: ("docker.io", "library/nginx", None, DIGEST),
    "myorg/app:v1@" + DIGEST: ("docker.io", "myorg/app", "v1", DIGEST),
}
for text, expected in cases.items():
    got = canonical_reference(text)
    assert isinstance(got, dict), f"{text!r} returned {got!r}"
    actual = (got.get("registry"), got.get("repository"),
              got.get("tag"), got.get("digest"))
    assert actual == expected, f"{text!r} -> {actual} != {expected}"

# 'library/' is a docker.io convention, never applied to another registry.
assert canonical_reference("localhost/app")["repository"] == "app"

# --- stated error behaviour ----------------------------------------------
for bad in (
    "",
    "Nginx",                       # uppercase is not a legal repository
    "nginx:",                      # empty tag
    "nginx:BAD!tag",
    "nginx@sha256:abc",            # digest too short
    "nginx@md5:" + "3f" * 32,      # wrong algorithm
    "nginx@sha256:" + "zz" * 32,   # not hex
):
    try:
        canonical_reference(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed reference {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0102", FAMILY,
        prompt=(
            "Implement a Python function resolve_cache(key, restore_keys, "
            "entries) that picks which cache entry a job restores. entries is "
            "a list of dicts with 'key' and 'created_unix'. An entry whose "
            "key equals key exactly is restored, whatever its age. Otherwise "
            "try each restore key in the order given: collect the entries "
            "whose key starts with it, and if there are any, restore the most "
            "recently created one, breaking a tie on created_unix by the "
            "lexicographically smallest key. The first restore key that "
            "matches anything decides the result -- a later restore key is "
            "never consulted, even if it would find something newer. Return "
            "the restored entry's key, or None when nothing matches. Raise "
            "ValueError if key is empty or any restore key is empty."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("resolve_cache") + r'''
entries = [
    {"key": "build-linux-abc123", "created_unix": 100},
    {"key": "build-linux-def456", "created_unix": 300},
    {"key": "build-macos-aaa111", "created_unix": 900},
    {"key": "build-linux-exact", "created_unix": 50},
]

# An exact hit wins even though two prefix matches are newer than it.
assert resolve_cache(
    "build-linux-exact", ["build-linux-", "build-"], entries
) == "build-linux-exact"

# The decisive case: restore-key ORDER outranks recency. 'build-linux-'
# matches first, so the newest LINUX entry is restored -- not the newer
# macOS entry that the broader 'build-' prefix would reach.
assert resolve_cache(
    "build-linux-zzz", ["build-linux-", "build-"], entries
) == "build-linux-def456"

# A restore key that matches nothing falls through to the next one.
assert resolve_cache(
    "build-macos-zzz", ["build-windows-", "build-macos-"], entries
) == "build-macos-aaa111"

# The broad prefix, when it really is first, does reach the newest entry.
assert resolve_cache("build-zzz", ["build-"], entries) == "build-macos-aaa111"

assert resolve_cache("other", ["nope-"], entries) is None
assert resolve_cache("other", [], entries) is None
assert resolve_cache("anything", ["build-"], []) is None

# A tie on creation time resolves to the smaller key, so two runners agree.
tied = [
    {"key": "p-b", "created_unix": 5},
    {"key": "p-a", "created_unix": 5},
]
assert resolve_cache("p-x", ["p-"], tied) == "p-a"
assert resolve_cache("p-x", ["p-"], list(reversed(tied))) == "p-a"

# --- stated error behaviour ----------------------------------------------
for bad_key, bad_restore in (("", ["p-"]), ("k", [""]), ("k", ["p-", ""])):
    try:
        resolve_cache(bad_key, bad_restore, entries)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"accepted empty key/restore key: {bad_key!r} {bad_restore!r}"
        )
''',
    ),
    task(
        f"{FAMILY}-0103", FAMILY,
        prompt=(
            "Implement a Python function build_order(stages, target) that "
            "returns the stages of a multi-stage container build that must "
            "actually be built to produce target, in an order where every "
            "dependency precedes what needs it. stages maps a stage name to a "
            "dict with 'from' (a stage name or an external base image) and "
            "'copy_from' (a list of stage names it copies artifacts out of). "
            "A 'from' value that is not a stage name is an external image and "
            "contributes no dependency. Both 'from' and 'copy_from' create "
            "dependencies: a stage reached only through a copy_from is still "
            "required. Stages that target does not transitively need must not "
            "appear. The order must be deterministic -- whenever several "
            "stages are ready, build the alphabetically smallest first. The "
            "returned list includes target itself. Raise KeyError if target "
            "is not a stage or a copy_from names an unknown stage, and "
            "ValueError if the dependencies form a cycle."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("build_order") + r'''
stages = {
    "base": {"from": "debian:12", "copy_from": []},
    "deps": {"from": "base", "copy_from": []},
    "assets": {"from": "node:20", "copy_from": []},
    # 'assets' is reachable ONLY through a copy_from. A pruner that walks
    # 'from' alone still returns a plausible, correctly ordered list -- and
    # the build fails later, at the COPY.
    "build": {"from": "deps", "copy_from": ["assets"]},
    "runtime": {"from": "base", "copy_from": ["build"]},
    "docs": {"from": "base", "copy_from": []},
    "test": {"from": "build", "copy_from": []},
}

order = build_order(stages, "runtime")
assert isinstance(order, list), order
assert set(order) == {"base", "deps", "assets", "build", "runtime"}, order
assert len(order) == len(set(order)), f"a stage is built twice: {order}"
# Nothing that only depends ON the target may be dragged in.
assert "docs" not in order and "test" not in order, order

# Every dependency precedes its dependant.
for name in order:
    position = order.index(name)
    required = [stages[name]["from"]] + list(stages[name]["copy_from"])
    for dependency in required:
        if dependency in stages:
            assert order.index(dependency) < position, (
                f"{dependency} must be built before {name}: {order}"
            )

# Deterministic, and specifically alphabetical among ready stages.
assert build_order(stages, "runtime") == order
assert order == ["assets", "base", "deps", "build", "runtime"], order

assert build_order(stages, "docs") == ["base", "docs"]
assert build_order(stages, "assets") == ["assets"]
assert build_order(stages, "test") == [
    "assets", "base", "deps", "build", "test"
], build_order(stages, "test")

# --- stated error behaviour ----------------------------------------------
try:
    build_order(stages, "missing")
except KeyError:
    pass
else:
    raise AssertionError("accepted an unknown target stage")

try:
    build_order(
        {"a": {"from": "scratch", "copy_from": ["ghost"]}}, "a"
    )
except KeyError:
    pass
else:
    raise AssertionError("accepted a copy_from naming an unknown stage")

try:
    build_order(
        {
            "a": {"from": "b", "copy_from": []},
            "b": {"from": "a", "copy_from": []},
        },
        "a",
    )
except ValueError:
    pass
else:
    raise AssertionError("accepted a dependency cycle")
''',
    ),
    task(
        f"{FAMILY}-0104", FAMILY,
        prompt=(
            "Implement a Python function rollout_plan(replicas, max_surge, "
            "max_unavailable) that returns the steps of a rolling update as a "
            "list of dicts with keys 'new' and 'old' holding the replica "
            "counts AFTER that step. The rollout starts from new=0, "
            "old=replicas and every step must respect both budgets: new+old "
            "must never exceed replicas+max_surge, and new+old must never "
            "drop below replicas-max_unavailable. Counts never go negative, "
            "new never decreases, old never increases, and every step must "
            "change something. The last step must be new=replicas, old=0. "
            "Raise ValueError if replicas is less than 1, if either budget is "
            "negative, or if both budgets are zero -- with no surge and no "
            "unavailability allowed there is no legal first move, and "
            "returning an empty plan would report success for a rollout that "
            "can never start."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("rollout_plan") + r'''
def check(replicas, max_surge, max_unavailable):
    plan = rollout_plan(replicas, max_surge, max_unavailable)
    label = f"rollout_plan({replicas}, {max_surge}, {max_unavailable})"
    assert isinstance(plan, list) and plan, f"{label} returned {plan!r}"

    ceiling = replicas + max_surge
    floor = replicas - max_unavailable
    new, old = 0, replicas
    for index, step in enumerate(plan):
        assert isinstance(step, dict), f"{label} step {index}: {step!r}"
        following_new, following_old = step["new"], step["old"]
        assert following_new >= 0 and following_old >= 0, (
            f"{label} step {index} went negative: {step}"
        )
        # The two budgets are the whole point of a rolling update: exceed
        # the ceiling and the cluster cannot schedule; cross the floor and
        # the service drops below its promised capacity mid-deploy.
        assert following_new + following_old <= ceiling, (
            f"{label} step {index} exceeds surge budget {ceiling}: {step}"
        )
        assert following_new + following_old >= floor, (
            f"{label} step {index} falls below availability floor {floor}: "
            f"{step}"
        )
        assert following_new >= new, f"{label} step {index} removed new pods"
        assert following_old <= old, f"{label} step {index} added old pods"
        assert (following_new, following_old) != (new, old), (
            f"{label} step {index} made no progress: {step}"
        )
        new, old = following_new, following_old

    assert (new, old) == (replicas, 0), (
        f"{label} ended at new={new} old={old}, not a completed rollout"
    )
    return plan


# Surge-only, availability-only, and both together.
check(3, 1, 0)
check(3, 0, 1)
check(1, 1, 0)
check(1, 0, 1)
check(10, 2, 2)
check(5, 5, 0)
check(2, 0, 2)

# Deterministic: the same inputs plan the same way twice.
assert rollout_plan(10, 2, 2) == rollout_plan(10, 2, 2)

# --- stated error behaviour ----------------------------------------------
for bad in ((3, 0, 0), (0, 1, 1), (-1, 1, 1), (3, -1, 1), (3, 1, -1)):
    try:
        rollout_plan(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted an impossible rollout {bad}")
''',
    ),
    task(
        f"{FAMILY}-0105", FAMILY,
        prompt=(
            "Implement a Python function admit(quota, used, containers) that "
            "decides whether a pod may be admitted against a namespace quota, "
            "returning (True, '') or (False, reason). quota and used are "
            "dicts with 'cpu' and 'memory' quantity strings; containers is a "
            "list of dicts with 'requests' and 'limits' sub-dicts, either of "
            "which may omit a resource. CPU quantities are millicores: '250m' "
            "is 250, '1' is 1000, '1.5' is 1500. Memory quantities are bytes, "
            "where the binary suffixes Ki, Mi, Gi and Ti are powers of 1024 "
            "and the decimal suffixes k, M, G and T are powers of 1000, and a "
            "bare number is already bytes. Reject with a reason if any "
            "container sets a limit below its own request for a resource. "
            "Otherwise reject if used plus the sum of all container requests "
            "would exceed quota for either resource; equalling the quota "
            "exactly is allowed. A missing request counts as zero and a "
            "missing limit is unbounded. Raise ValueError on a malformed "
            "quantity, including a CPU value finer than a whole millicore."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("admit") + r'''
def verdict(result):
    assert isinstance(result, tuple) and len(result) == 2, result
    assert isinstance(result[0], bool), result
    assert isinstance(result[1], str), result
    return result[0]


quota = {"cpu": "2", "memory": "1Gi"}
zero = {"cpu": "0", "memory": "0"}

# Exactly equalling the quota is admitted: 500m + 1500m == 2000m.
assert verdict(admit(
    quota, {"cpu": "500m", "memory": "0"},
    [{"requests": {"cpu": "1500m", "memory": "0"}, "limits": {}}],
)) is True

# One millicore more is not.
assert verdict(admit(
    quota, {"cpu": "500m", "memory": "0"},
    [{"requests": {"cpu": "1501m", "memory": "0"}, "limits": {}}],
)) is False

# The decisive case. 1025Mi is 1074790400 bytes, just over the 1073741824
# bytes of a 1Gi quota, so this pod must be refused. Read 'Mi' as a
# decimal megabyte and it computes 1025000000 -- comfortably inside the
# quota -- and the pod is admitted onto a node that cannot hold it.
assert verdict(admit(
    quota, zero, [{"requests": {"memory": "1025Mi", "cpu": "0"}, "limits": {}}],
)) is False
# And 1024Mi is exactly 1Gi, which fits.
assert verdict(admit(
    quota, zero, [{"requests": {"memory": "1024Mi", "cpu": "0"}, "limits": {}}],
)) is True
# The decimal suffixes really are decimal: 1G is smaller than 1Gi.
assert verdict(admit(
    quota, zero, [{"requests": {"memory": "1G", "cpu": "0"}, "limits": {}}],
)) is True

# Requests accumulate across containers rather than being taken one at a time.
assert verdict(admit(
    quota, zero,
    [
        {"requests": {"cpu": "1", "memory": "0"}, "limits": {}},
        {"requests": {"cpu": "1", "memory": "0"}, "limits": {}},
        {"requests": {"cpu": "1m", "memory": "0"}, "limits": {}},
    ],
)) is False

# A limit below its own request is incoherent, whatever the quota says.
below = admit(
    quota, zero,
    [{"requests": {"cpu": "500m"}, "limits": {"cpu": "250m"}}],
)
assert verdict(below) is False
assert below[1], "a rejection must carry a reason"

# A missing limit is unbounded, and a missing request is zero.
assert verdict(admit(quota, zero, [{"requests": {}, "limits": {}}])) is True
assert verdict(admit(
    quota, zero, [{"requests": {"cpu": "1"}, "limits": {"cpu": "1"}}],
)) is True

# Fractional and suffixed CPU agree.
assert verdict(admit(
    {"cpu": "1500m", "memory": "1Gi"}, zero,
    [{"requests": {"cpu": "1.5"}, "limits": {}}],
)) is True

# --- stated error behaviour ----------------------------------------------
for bad in ("1.0005", "abc", "", "10Xi", "1..0", "-1"):
    try:
        admit({"cpu": "2", "memory": "1Gi"}, zero,
              [{"requests": {"cpu": bad}, "limits": {}}])
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed cpu quantity {bad!r}")

for bad in ("12MB", "1.5Gi", "zz", "1 Gi"):
    try:
        admit({"cpu": "2", "memory": "1Gi"}, zero,
              [{"requests": {"memory": bad}, "limits": {}}])
    except ValueError:
        pass
    else:
        raise AssertionError(f"accepted malformed memory quantity {bad!r}")
''',
    ),
    task(
        f"{FAMILY}-0106", FAMILY,
        prompt=(
            "Implement a Python function expand_matrix(axes, include, "
            "exclude) that expands a CI build matrix. axes maps an axis name "
            "to its list of values; the base combinations are the cartesian "
            "product in axis order with the last axis varying fastest. Then "
            "drop every combination that matches all of the key/value pairs "
            "of any exclude entry. Then apply the include entries in order, "
            "and never exclude anything they produce: for each entry, look at "
            "the keys it shares with the axes -- if it has such keys and at "
            "least one surviving combination matches all of them, merge the "
            "entry's other keys into every combination that matches; "
            "otherwise append the entry itself as a new combination. Return "
            "the list of combinations as dicts. Raise ValueError if axes is "
            "empty, if any axis has no values, or if an exclude entry names a "
            "key that is not an axis."
        ),
        timeout_seconds=60.0,
        validator=LOAD_CANDIDATE + require("expand_matrix") + r'''
axes = {"os": ["linux", "macos"], "python": ["3.11", "3.12"]}

# The bare product, last axis varying fastest.
assert expand_matrix(axes, [], []) == [
    {"os": "linux", "python": "3.11"},
    {"os": "linux", "python": "3.12"},
    {"os": "macos", "python": "3.11"},
    {"os": "macos", "python": "3.12"},
], expand_matrix(axes, [], [])

# An exclude entry matches on the pairs it names and ignores the rest.
assert expand_matrix(axes, [], [{"os": "macos"}]) == [
    {"os": "linux", "python": "3.11"},
    {"os": "linux", "python": "3.12"},
]

# An include that matches an existing combination decorates it in place
# rather than appending a near-duplicate.
decorated = expand_matrix(
    axes, [{"os": "linux", "python": "3.12", "coverage": True}], []
)
assert decorated == [
    {"os": "linux", "python": "3.11"},
    {"os": "linux", "python": "3.12", "coverage": True},
    {"os": "macos", "python": "3.11"},
    {"os": "macos", "python": "3.12"},
], decorated

# A partial include decorates every combination it matches.
both = expand_matrix(axes, [{"os": "macos", "tier": "slow"}], [])
assert [row.get("tier") for row in both] == [None, None, "slow", "slow"], both

# An include naming a value no axis has is a new combination, appended.
extended = expand_matrix(axes, [{"os": "windows", "python": "3.12"}], [])
assert extended[-1] == {"os": "windows", "python": "3.12"}, extended
assert len(extended) == 5, extended

# The decisive case: include runs AFTER exclude and is never subject to it.
# Excluding macos/3.11 and then including it back must leave it present.
readded = expand_matrix(
    axes,
    [{"os": "macos", "python": "3.11"}],
    [{"os": "macos", "python": "3.11"}],
)
assert {"os": "macos", "python": "3.11"} in readded, readded
assert len(readded) == 4, readded

# An include with no axis keys at all is simply appended.
plain = expand_matrix({"os": ["linux"]}, [{"note": "extra"}], [])
assert plain == [{"os": "linux"}, {"note": "extra"}], plain

# --- stated error behaviour ----------------------------------------------
for bad_axes, bad_exclude in (
    ({}, []),
    ({"os": []}, []),
    ({"os": ["linux"]}, [{"nosuch": "x"}]),
):
    try:
        expand_matrix(bad_axes, [], bad_exclude)
    except ValueError:
        pass
    else:
        raise AssertionError(
            f"accepted a malformed matrix: {bad_axes!r} {bad_exclude!r}"
        )
''',
    ),
]
