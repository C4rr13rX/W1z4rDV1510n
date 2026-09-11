python3 - <<'PY'
"""How many bytes would rolling back to `brain.last-good.wbrain` actually return?

The volume is at 22.84 GB free of 1.0 TB with the burn stopped. `brain.wbrain`
is 961 GB apparent; `brain.last-good.wbrain` is 395 GB apparent and is a
reflink clone, so the two share physical extents and neither file's SIZE says
anything about what deleting it would free. The last lesson on this volume was
exactly that: nine deferred directories holding ~560 GB of apparent `st_blocks`
returned 0.00 GB, because the bytes were shared.

So measure the sharing directly. `filefrag -v` gives every extent's physical
block range; the reclaim from discarding `brain.wbrain` is the total of its
physical blocks that appear in NO other file -- here approximated as the blocks
not covered by `last-good`, which is the only known sharer. That is a lower
bound on the reclaim and an upper bound on the risk.

Also checks the two things that decide whether the rollback is even legitimate:

  * is `last-good` an independently valid container, or merely a clone whose
    header points into blocks `brain.wbrain` owns (reflink means independently
    valid, but the header must parse to prove it);
  * does the supervisor already own a rollback-to-last-good path, in which case
    this is the system's designed behaviour rather than a hand edit.

Nothing here deletes or modifies anything.
"""
import json
import os
import re
import subprocess
import shutil
import time

R = "/srv/wizard/runtime/programming-integrated-20260713"
BRAIN = f"{R}/brain/brain.wbrain"
GOOD = f"{R}/brain/brain.last-good.wbrain"
out = {"now": time.time()}


def sh(cmd, timeout=600):
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return (proc.stdout + proc.stderr)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


EXT = re.compile(
    r"^\s*\d+:\s*(\d+)\.\.\s*(\d+):\s*(\d+)\.\.\s*(\d+):\s*(\d+):"
)


def extents(path):
    """[(physical_start_block, length_blocks)] from `filefrag -v`."""
    text = sh(f"filefrag -v -b4096 {path} 2>&1")
    spans = []
    for line in text.splitlines():
        m = EXT.match(line)
        if not m:
            continue
        phys_start = int(m.group(3))
        length = int(m.group(5))
        spans.append((phys_start, length))
    return spans


def merge(spans):
    spans = sorted(spans)
    merged = []
    for start, length in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], start + length)
        else:
            merged.append([start, start + length])
    return merged


def total(merged):
    return sum(end - start for start, end in merged)


def subtract(a, b):
    """Blocks in merged-list `a` not covered by merged-list `b`."""
    result = []
    j = 0
    for start, end in a:
        cur = start
        while cur < end:
            while j < len(b) and b[j][1] <= cur:
                j += 1
            if j >= len(b) or b[j][0] >= end:
                result.append((cur, end))
                break
            if b[j][0] > cur:
                result.append((cur, b[j][0]))
            cur = max(cur, b[j][1])
        # `j` must not advance past spans a later extent still needs.
        while j > 0 and b[j - 1][1] > start:
            j -= 1
    return sum(end - start for start, end in result)


for label, path in (("brain", BRAIN), ("last_good", GOOD)):
    try:
        st = os.stat(path)
        out[f"{label}_apparent_gb"] = round(st.st_size / 2**30, 2)
        out[f"{label}_st_blocks_gb"] = round(st.st_blocks * 512 / 2**30, 2)
        out[f"{label}_nlink"] = st.st_nlink
        out[f"{label}_mtime"] = time.ctime(st.st_mtime)
    except OSError as exc:
        out[f"{label}_error"] = str(exc)

t0 = time.time()
brain_spans = merge(extents(BRAIN))
good_spans = merge(extents(GOOD))
out["filefrag_seconds"] = round(time.time() - t0, 1)
out["brain_extent_count"] = len(brain_spans)
out["last_good_extent_count"] = len(good_spans)
if brain_spans and good_spans:
    out["brain_physical_gb"] = round(total(brain_spans) * 4096 / 2**30, 2)
    out["last_good_physical_gb"] = round(total(good_spans) * 4096 / 2**30, 2)
    out["brain_unique_gb"] = round(
        subtract(brain_spans, good_spans) * 4096 / 2**30, 2
    )
    out["last_good_unique_gb"] = round(
        subtract(good_spans, brain_spans) * 4096 / 2**30, 2
    )

# Is `last-good` an independently parseable container?
out["good_header"] = sh(f"head -c 64 {GOOD} | xxd | head -4 2>&1")
out["brain_header"] = sh(f"head -c 64 {BRAIN} | xxd | head -4 2>&1")

# Does the supervisor own a rollback path already?
out["rollback_sites"] = sh(
    "grep -n 'last-good\\|last_good' "
    "/srv/wizard/project/scripts/programming_curriculum_supervisor.py "
    "2>&1 | head -40"
)

out["free_gb"] = round(shutil.disk_usage(R).free / 2**30, 2)
out["biggest_files"] = sh(
    "find /srv/wizard/runtime -xdev -type f -size +5G "
    "-printf '%s\\t%p\\n' 2>/dev/null | sort -rn | head -20"
)
out["census"] = sh("pgrep -af w1z4rd_brain_server 2>&1")[:400]

print("PROBE_JSON " + json.dumps(out, sort_keys=True, default=str))
PY
