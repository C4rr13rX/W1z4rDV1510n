python3 - <<'PY'
"""Is the go-systems block advancing monotonically, and which suite is the 12th?

Two questions, both of which a spot reading answers wrongly.

(1) The block is at row 50224 of 131072 and gaining ~3 rows/s averaged over
the brain's 4.6 h lifetime -- at that rate it would finish in under 8 h, which
does not explain a 101.6 h drought. The drought is only explained if the block
RESETS. Every ledger record carries `trained_rows`, so the whole go-systems row
series is recoverable: a monotone series means "slow", a sawtooth means "can
never finish". Those need opposite fixes, so the series is printed, not
summarised to a single rate.

(2) `enterprise_gate_confirmation` reports first=11, confirm=11, total=12,
passed=false, 127 times. Eleven suites pass twice and the block is still
rejected, so one suite fails deterministically. The ledger records only the
count; the per-suite verdict is in the gate artifact next to it.
"""
import collections
import json
import pathlib
import time

R = pathlib.Path("/srv/wizard/runtime/programming-integrated-20260713")
out = {"now": time.time()}

# --- (1) the row series, per phase, straight from every record that has one.
series = collections.defaultdict(list)
kind_rows = collections.defaultdict(list)
try:
    with (R / "curriculum-health.jsonl").open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                ev = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            ts = ev.get("updated_unix")
            rows = ev.get("trained_rows")
            phase = str(ev.get("phase") or "")
            if ts is None or rows is None:
                continue
            series[phase].append((float(ts), int(rows), str(ev.get("kind") or "")))
except Exception as exc:  # noqa: BLE001
    out["series_err"] = str(exc)

go = sorted(series.get("go-systems", []))
out["go_systems_points"] = len(go)
if go:
    out["go_first_unix"] = go[0][0]
    out["go_span_h"] = round((go[-1][0] - go[0][0]) / 3600.0, 2)
    # Every point where the row went DOWN: that is a reset, and its size is
    # how much work was thrown away.
    resets = []
    for (t0, r0, _), (t1, r1, k1) in zip(go, go[1:]):
        if r1 < r0:
            resets.append({"at_h_ago": round((out["now"] - t1) / 3600.0, 2),
                           "from_row": r0, "to_row": r1, "lost": r0 - r1,
                           "kind": k1})
    out["go_resets"] = resets[-25:]
    out["go_reset_count"] = len(resets)
    out["go_max_row_ever"] = max(r for _, r, _ in go)
    # Decimated series so the shape is visible without dumping thousands.
    step = max(1, len(go) // 60)
    out["go_series"] = [
        {"h_ago": round((out["now"] - t) / 3600.0, 2), "row": r, "kind": k}
        for t, r, k in go[::step]
    ][-60:]
    out["go_last_10"] = [
        {"h_ago": round((out["now"] - t) / 3600.0, 2), "row": r, "kind": k}
        for t, r, k in go[-10:]
    ]

out["phases_seen"] = sorted(
    {p: len(v) for p, v in series.items()}.items(), key=lambda kv: -kv[1])[:12]

# --- (2) the gate artifacts: per-suite verdicts.
for name in ("jupyter-scientific-para4.enterprise-gate.json",
             "jupyter-scientific-para4.typescript-gate.json",
             "jupyter-scientific-para4.completion-gate.json"):
    p = R / name
    if not p.is_file():
        continue
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        out.setdefault("gate_files", {})[name] = {"err": str(exc)}
        continue
    entry = {"age_h": round((out["now"] - p.stat().st_mtime) / 3600.0, 2),
             "top_keys": sorted(data.keys()) if isinstance(data, dict) else None}

    # Pull out anything that looks like a per-suite verdict, whatever it is
    # nested inside -- the shape is not known in advance.
    verdicts = []

    def walk(node, path=""):
        if isinstance(node, dict):
            keys = set(node.keys())
            if {"passed"} & keys and ("suite" in keys or "name" in keys or "id" in keys):
                verdicts.append({
                    "at": path[:120],
                    "suite": node.get("suite") or node.get("name") or node.get("id"),
                    "passed": node.get("passed"),
                    "detail": json.dumps(
                        {k: v for k, v in node.items()
                         if k not in ("passed", "suite", "name", "id")},
                        default=str)[:400],
                })
            for k, v in node.items():
                if isinstance(v, bool) and k not in ("passed", "ok"):
                    verdicts.append({"at": path[:120], "suite": k, "passed": v,
                                     "detail": "bool-leaf"})
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node[:80]):
                walk(v, f"{path}[{i}]")

    walk(data)
    entry["verdicts"] = verdicts[:60]
    entry["failing"] = [v for v in verdicts if v.get("passed") is False][:30]
    if isinstance(data, dict):
        entry["raw_head"] = json.dumps(data, default=str)[:2500]
    out.setdefault("gate_files", {})[name] = entry

print("PROBE_JSON " + json.dumps(out, default=str))
PY
