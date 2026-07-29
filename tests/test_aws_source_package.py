from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

from scripts.aws.package_for_s3 import sha256, tracked_files, write_source_archive


def test_source_archive_is_commit_bound_and_reproducible() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "archive-test@example.invalid"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Archive Test"],
            cwd=root,
            check=True,
        )
        source = root / "source.txt"
        source.write_text("accepted\n", encoding="utf-8")
        subprocess.run(["git", "add", "source.txt"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        first = Path(raw) / "first.tar.gz"
        second = Path(raw) / "second.tar.gz"
        write_source_archive(first, commit, root)
        source.write_text("dirty working tree must not leak\n", encoding="utf-8")
        write_source_archive(second, commit, root)

        assert tracked_files(commit, root) == ["source.txt"]
        assert sha256(first) == sha256(second)
        with tarfile.open(second, "r:gz") as archive:
            extracted = archive.extractfile("source.txt")
            assert extracted is not None
            assert extracted.read().replace(b"\r\n", b"\n") == b"accepted\n"
