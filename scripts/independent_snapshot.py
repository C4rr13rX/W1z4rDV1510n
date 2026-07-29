"""Space-efficient independent snapshots for mutable brain containers.

Hard links are invalid rollback guards for ``.wbrain`` because the container
updates slot records and appends bodies in place.  Linux filesystems such as
XFS can instead clone extents with ``FICLONE``: the destination has a distinct
inode and later writes are copy-on-write.  Other platforms retain the proven
full-copy fallback.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable


# linux/fs.h: _IOW(0x94, 9, int)
FICLONE = 0x40049409


def _preserve_posix_owner(source: Path, destination: Path) -> None:
    """Keep service-readable ownership when an administrator publishes a copy.

    ``shutil.copy2`` and ``copystat`` preserve metadata but deliberately do not
    preserve uid/gid.  Recovery is commonly invoked through an administrative
    service while the brain node runs as an unprivileged account, so inheriting
    the recovery process owner can make an otherwise valid snapshot unreadable.
    """
    if os.name != "posix":
        return
    source_stat = source.stat()
    os.chown(destination, source_stat.st_uid, source_stat.st_gid)


def _clone_reflink(source: Path, destination: Path) -> bool:
    """Create *destination* as an independent COW clone when supported."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        import fcntl

        with source.open("rb") as source_stream:
            with destination.open("xb") as destination_stream:
                fcntl.ioctl(
                    destination_stream.fileno(),
                    FICLONE,
                    source_stream.fileno(),
                )
                destination_stream.flush()
                os.fsync(destination_stream.fileno())
        shutil.copystat(source, destination, follow_symlinks=True)
        if os.path.samefile(source, destination):
            raise RuntimeError("reflink unexpectedly shares the source inode")
        return True
    except (ImportError, OSError, RuntimeError):
        destination.unlink(missing_ok=True)
        return False


def publish_independent_copy(
    source: Path,
    destination: Path,
    *,
    operation: str,
    require_full_copy_headroom: Callable[[Path, int, str], None],
) -> str:
    """Atomically publish an independent rollback file.

    Returns ``"reflink"`` when the filesystem provided copy-on-write extents
    and ``"copy"`` when a full byte copy was required.
    """
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    if _clone_reflink(source, temporary):
        _preserve_posix_owner(source, temporary)
        os.replace(temporary, destination)
        return "reflink"

    require_full_copy_headroom(source, 1, operation)
    shutil.copy2(source, temporary)
    _preserve_posix_owner(source, temporary)
    os.replace(temporary, destination)
    if os.path.samefile(source, destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError("independent snapshot unexpectedly shares source inode")
    return "copy"
