python3 - <<'PY'
"""Is a compaction sawtooth even executable on this host, and what would it free?

Independent of how much of the container is live, three facts have to hold
before compaction can be proposed as the reclaim:

  1. a `wbrain_compact` binary exists and supports `--in-place` (the
     out-of-place form needs a full scratch copy, which is what made the
     earlier arithmetic lose);
  2. the brain is stoppable at a known boundary -- it is, 768 recorded
     `settled_node_memory_recycle` events stop and restart the node, and the
     tool requires the brain not be running;
  3. the reclaim is predicted from BLOCK SHARING, not file size. `brain.wbrain`
     was just restored by rolling back to `brain.last-good.wbrain`, so the two
     are reflink clones of each other and compacting only one frees nothing --
     the other still pins every shared extent. This repository has already
     measured a 560 GB delete return 0.00 GB for exactly this reason.

Read-only.
"""
import json
import os
import pathlib
import subprocess
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
BRAIN = R / "brain" / "brain.wbrain"
GUARD = R / "brain" / "brain.last-good.wbrain"
out = {"now": time.time()}


def sh(cmd, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return {"rc": p.returncode,
                "out": (p.stdout or "")[-2500:],
                "err": (p.stderr or "")[-800:]}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


out["which"] = sh("ls -l /srv/wizard/project/bin/wbrain_compact "
                  "/srv/wizard/project/target/release/wbrain_compact "
                  "$(command -v wbrain_compact) 2>&1 | head -20")
out["usage"] = sh("for p in /srv/wizard/project/bin/wbrain_compact "
                  "/srv/wizard/project/target/release/wbrain_compact; do "
                  "[ -x \"$p\" ] && echo \"== $p\" && \"$p\" 2>&1 | head -12; done")

st = os.statvfs(R)
out["disk"] = {"free_gb": round(st.f_bavail * st.f_frsize / 1e9, 2),
               "total_gb": round(st.f_blocks * st.f_frsize / 1e9, 2)}
for name, p in (("brain", BRAIN), ("guard", GUARD)):
    if p.exists():
        s = p.stat()
        out[name] = {"size_gb": round(s.st_size / 1e9, 2),
                     "allocated_gb": round(s.st_blocks * 512 / 1e9, 2),
                     "nlink": s.st_nlink, "inode": s.st_ino,
                     "mtime_age_h": round((time.time() - s.st_mtime) / 3600, 2)}


def extents(path, cap=400000):
    """Sorted physical intervals from filefrag's documented columns."""
    p = subprocess.run(["filefrag", "-v", str(path)], capture_output=True,
                       text=True, timeout=1800)
    iv = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = [x.strip() for x in line.split(":")]
        if len(parts) < 4:
            continue
        try:
            phys = parts[2]
            start = int(phys.split("..")[0].strip().rstrip("."))
            length = int(parts[3].split()[0])
        except Exception:  # noqa: BLE001
            continue
        iv.append((start, start + length))
        if len(iv) >= cap:
            break
    iv.sort()
    merged = []
    for s, e in iv:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def overlap(a, b):
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if s < e:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


try:
    ba = extents(BRAIN)
    ga = extents(GUARD)
    blocks = 4096
    bt = sum(e - s for s, e in ba)
    gt = sum(e - s for s, e in ga)
    sh_ = overlap(ba, ga)
    out["sharing"] = {
        "brain_extents": len(ba), "guard_extents": len(ga),
        "brain_alloc_gb": round(bt * blocks / 1e9, 2),
        "guard_alloc_gb": round(gt * blocks / 1e9, 2),
        "shared_gb": round(sh_ * blocks / 1e9, 2),
        "brain_unique_gb": round((bt - sh_) * blocks / 1e9, 2),
        "guard_unique_gb": round((gt - sh_) * blocks / 1e9, 2),
        "note": "compacting only one file returns its UNIQUE blocks; shared "
                "extents stay pinned by the other name",
    }
except Exception as exc:  # noqa: BLE001
    out["sharing"] = "%s: %s" % (type(exc).__name__, exc)

print("PROBE_JSON " + json.dumps(out, default=str))
PY
