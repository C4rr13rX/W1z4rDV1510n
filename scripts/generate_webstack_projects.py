#!/usr/bin/env python3
"""Generate project-level web-stack episodes as file manifests.

`webstack_units.jsonl` teaches one unit per episode: a request for a Django
model recalls that class and nothing else. That is the right shape for a
request that names one thing, and measured 2026-08-20 it works -- 7/7 novel
phrasings recall the correct single unit.

It is the wrong shape for "build a scientific calculator web application",
which needs several files at once. The router already composes those, but only
from responses shaped as `{"files": {name: source}}` -- and no corpus on the
host contained a single manifest row, so composition had nothing to merge.

This emits the multi-file half of that pair, reusing the SAME unit sources so
the two corpora cannot drift apart. The output shape therefore follows the
request rather than a mode flag:

    "Write a Django AppConfig class ..."      -> one class, bare source
    "Build the Django backend for ..."        -> a manifest of backend files
    "Build a scientific calculator web app"   -> a manifest of the whole stack

Every record is authored here and published under CC0-1.0, like the units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_webstack_corpus import DJANGO, PYTHON_CORE, THREEJS, VUE, record
from webstack_units_extra import DJANGO_EXTRA, THREEJS_EXTRA, VUE_EXTRA

LICENSE = "CC0-1.0"
SCRIPT_ID = "programming_webstack_projects_001"

#: unit key -> path the file takes inside a project tree.
FILE_PATHS = {
    "model": "calculator/models.py",
    "serializer": "calculator/serializers.py",
    "view": "calculator/views.py",
    "urls": "calculator/urls.py",
    "surface_view": "calculator/surface_views.py",
    "settings": "config/settings.py",
    "appconfig": "calculator/apps.py",
    "tests": "calculator/tests.py",
    "evaluator": "calculator/evaluator.py",
    "sampling": "calculator/sampling.py",
    "keypad": "src/components/CalculatorKeypad.vue",
    "api_client": "src/api/calculator.js",
    "entry": "src/main.js",
    "store": "src/stores/calculator.js",
    "plot_controls": "src/components/PlotControls.vue",
    "viewport": "src/components/SurfaceViewport.vue",
    "scene": "src/three/SceneHost.js",
    "surface": "src/three/buildSurfaceMesh.js",
    "axes": "src/three/addAxes.js",
    "frame": "src/three/frameObject.js",
}

#: Which units make up each project, and how it may be asked for.
PROJECTS = [
    (
        "django-backend",
        ["model", "serializer", "view", "urls", "surface_view",
         "appconfig", "evaluator", "sampling"],
        "python",
        [
            "Build the Django backend for a scientific calculator with expression "
            "evaluation and surface sampling.",
            "Create the Django REST server for a calculator: models, serializers, "
            "views, urls and a safe evaluator.",
            "Write the server side of a Django calculator API.",
        ],
    ),
    (
        "vue-frontend",
        ["entry", "store", "keypad", "plot_controls", "api_client"],
        "javascript",
        [
            "Build the Vue frontend for a scientific calculator with a keypad and "
            "plot controls.",
            "Create the Vue client for a calculator: entry point, store, keypad and "
            "an API client.",
            "Write the browser side of a Vue calculator application.",
        ],
    ),
    (
        "threejs-viewport",
        ["scene", "surface", "axes", "frame", "viewport"],
        "javascript",
        [
            "Build the three.js 3D viewport that plots a surface from sampled "
            "z values.",
            "Create the three.js rendering layer for a calculator: scene, surface "
            "mesh, axes and camera framing.",
            "Write the 3D charting side of a three.js calculator.",
        ],
    ),
    (
        "full-stack",
        list(FILE_PATHS),
        "python",
        [
            "Build a scientific calculator web application with a Django REST "
            "backend, a Vue frontend and three.js 3D charting.",
            "Create a full scientific calculator that charts variables in 3D space "
            "using Django, Vue and three.js.",
            "Build a Django plus Vue plus three.js scientific calculator that plots "
            "expressions in 3D.",
        ],
    ),
]


def unit_sources() -> dict[str, str]:
    """Every authored unit, keyed the same way both corpora key them."""
    sources: dict[str, str] = {}
    for group in (DJANGO, DJANGO_EXTRA, PYTHON_CORE, VUE, VUE_EXTRA,
                  THREEJS, THREEJS_EXTRA):
        for _prompt, response, unit in group:
            sources[unit] = response.strip() + "\n"
    return sources


def manifest_for(units: list[str], sources: dict[str, str]) -> str:
    """A project as the router's file-manifest shape, key order stable."""
    files = {FILE_PATHS[unit]: sources[unit] for unit in units}
    return json.dumps({"files": files}, indent=2, sort_keys=True) + "\n"


def build_records() -> list[dict]:
    sources = unit_sources()
    missing = [unit for unit in FILE_PATHS if unit not in sources]
    if missing:
        raise SystemExit(f"no source for units: {missing}")
    records: list[dict] = []
    for name, units, lang, prompts in PROJECTS:
        body = manifest_for(units, sources)
        for index, prompt in enumerate(prompts):
            row = record(prompt, body, lang, "implement", name,
                         variant=None if index == 0 else index - 1)
            row["script_id"] = SCRIPT_ID
            records.append(row)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = build_records()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    projects = {row["source"].split("|", 1)[0] for row in records}
    print(f"wrote {len(records)} rows ({len(projects)} projects) to {args.output}")
    for name, units, _lang, prompts in PROJECTS:
        print(f"  {name:18} {len(units):2} files, {len(prompts)} phrasings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
