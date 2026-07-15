#!/usr/bin/env python3
"""Probe Multiscale Robot World readiness without permitting confident drift."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path


PROMPT = (
    "Build Phase 1 of a CPU-first multiscale physical-world platform beginning "
    "in deep space. Use strict TypeScript OOP for an independently testable "
    "SI-units physics kernel with particles and aggregates, gravity, "
    "electrostatics, deterministic symplectic integration, collisions, "
    "conservation diagnostics, explicit validity domains, uncertainty and "
    "error budgets, and hierarchical refine/coarsen chunking. Keep Three.js "
    "out of the physics kernel. Add a Three.js instanced, LOD, origin-rebased "
    "adapter; CPU-worker transferable typed-array budgets for an i5 with 32GB "
    "RAM; deterministic tests for inverse-square laws, conservation drift, "
    "collisions, replay and coarse-graining equivalence; and documentation "
    "with 2022 CODATA citations, equations, limitations, performance budgets "
    "and a versioned roadmap."
)


def chat(endpoint: str) -> tuple[dict, float]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/brain/chat",
        data=json.dumps({"text": PROMPT}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def safe_manifest(reply: str) -> dict[str, str]:
    try:
        value = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return {}
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, dict) or not files:
        return {}
    output: dict[str, str] = {}
    for name, source in files.items():
        if (not isinstance(name, str) or not isinstance(source, str)
                or not name or not source or name.startswith(("/", "\\"))
                or ".." in Path(name).parts):
            return {}
        output[name.replace("\\", "/")] = source
    return output


def structural_checks(files: dict[str, str]) -> dict[str, bool]:
    names = "\n".join(files).lower()
    source = "\n".join(files.values()).lower()
    physics_source = "\n".join(
        body for name, body in files.items()
        if any(part in name.lower() for part in ("physics", "kernel", "gravity", "particle"))
    ).lower()
    return {
        "strict_typescript": "tsconfig" in names and "strict" in source,
        "si_and_codata": "codata" in source and any(term in source for term in ("si unit", "meter", "kilogram")),
        "gravity_and_electrostatics": "gravit" in source and "electrostat" in source,
        "deterministic_integrator": any(term in source for term in ("symplectic", "velocity verlet", "leapfrog")),
        "collision_and_conservation": "collision" in source and "conservation" in source,
        "validity_and_error_budgets": "validity" in source and "error budget" in source,
        "hierarchical_chunking": "refine" in source and "coarsen" in source,
        "renderer_separated": "three" in source and "three" not in physics_source,
        "cpu_transfer_budget": "worker" in source and "array" in source and "budget" in source,
        "deterministic_tests": "test" in names and "determin" in source and "inverse" in source,
        "documentation_and_roadmap": "readme" in names and "roadmap" in source and "limitation" in source,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/capstone-readiness.json"))
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    result, elapsed = chat(args.endpoint)
    reply = str(result.get("reply") or "")
    grounding = result.get("grounding") or {}
    files = safe_manifest(reply)
    checks = structural_checks(files) if files else {}
    structurally_ready = bool(checks) and all(checks.values())
    honest_oov = not reply and bool(grounding.get("outside_grounding"))
    safe = structurally_ready or honest_oov
    status = (
        "structurally_ready" if structurally_ready
        else "honest_oov" if honest_oov
        else "unsafe_cross_domain_answer"
    )
    report = {
        "passed": safe and (structurally_ready or not args.require_ready),
        "status": status,
        "structurally_ready": structurally_ready,
        "honest_oov": honest_oov,
        "latency_seconds": round(elapsed, 4),
        "reply_chars": len(reply),
        "files": sorted(files),
        "checks": checks,
        "grounding": grounding,
        "intent_diagnostics": result.get("intent_diagnostics"),
        "reply_preview": reply[:1000],
        "updated_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"], "status": status,
        "structurally_ready": structurally_ready, "honest_oov": honest_oov,
    }))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
