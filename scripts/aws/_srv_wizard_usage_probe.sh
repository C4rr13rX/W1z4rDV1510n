python3 - <<'PY'
"""Break down the 1 TB /srv/wizard volume that is at 100% with 20 KB free.

The first forensics pass ran `du -x` from /srv, which sits on the root
filesystem, so the mount boundary excluded everything and it reported 0 MB --
a vacuous zero of exactly the class CLAUDE.md warns about. This walks the mount
itself and separates the three consumers that can plausibly hold hundreds of
gigabytes: brain checkpoints (.wbrain), staged corpora, and per-interval
runtime artifacts.

Nothing here deletes. It reports sizes, ages and whether each candidate is
referenced by durable interval state, so the reclaim can be argued before it is
executed.
"""
import json
import subprocess

out = {}


def sh(cmd, timeout=240):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return proc.stdout[-7000:] + (
            ("\n[stderr] " + proc.stderr[-600:]) if proc.stderr.strip() else ""
        )
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


out["df"] = sh("df -h /srv/wizard; df -i /srv/wizard")
out["level1"] = sh("du -m -d 1 /srv/wizard 2>/dev/null | sort -rn | head -20")
out["level2"] = sh("du -m -d 2 /srv/wizard 2>/dev/null | sort -rn | head -40")
out["biggest_files"] = sh(
    "find /srv/wizard -xdev -type f -size +200M "
    "-printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' 2>/dev/null | sort -rn | head -40"
)
out["wbrain"] = sh(
    "find /srv/wizard -xdev \\( -name '*.wbrain' -o -name '*.wbrain.*' "
    "-o -name '*.wal' -o -name '*.ckpt*' \\) "
    "-printf '%s\\t%TY-%Tm-%Td %TH:%TM\\t%p\\n' 2>/dev/null | sort -rn | head -30"
)
out["wbrain_dirs"] = sh(
    "find /srv/wizard -xdev -maxdepth 3 -type d \\( -name '*wbrain*' "
    "-o -name '*checkpoint*' -o -name '*snapshot*' \\) 2>/dev/null | head -20"
)
out["corpora"] = sh("du -m -d 2 /srv/wizard/corpora 2>/dev/null | sort -rn | head -25")
out["runtime_level1"] = sh("du -m -d 1 /srv/wizard/runtime 2>/dev/null | sort -rn | head -20")
out["runtime_subdirs"] = sh(
    "du -m -d 1 /srv/wizard/runtime/programming-integrated-20260713 2>/dev/null "
    "| sort -rn | head -25"
)
out["project"] = sh("du -m -d 2 /srv/wizard/project 2>/dev/null | sort -rn | head -20")

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
