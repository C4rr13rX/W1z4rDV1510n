#!/usr/bin/env python3
"""Live fail→repair training for a never-observed multi-file synthesis."""
from __future__ import annotations

import argparse
import base64
import difflib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

from programming_project_eval import b64, request


def fragment(file: str, role: str, after: list[str], source: str,
             evidence_id: str | None = None) -> str:
    value: dict[str, object] = {"file": file, "role": role, "after": after,
                                "source": source}
    if evidence_id:
        value["evidence_id"] = evidence_id
    return json.dumps({"code_fragment": value}, sort_keys=True, separators=(",", ":"))


BASE = [
    ("Write a Python multi-file inventory domain module exception declaration.",
     fragment("domain.py", "exception", [],
              "class InsufficientStock(ValueError):\n    pass\n\n")),
    ("Write a Python multi-file inventory domain module reserve stock function with an insufficient stock guard.",
     fragment("domain.py", "reserve", ["exception"],
              "def reserve_stock(available, quantity):\n"
              "    if quantity < 1:\n        raise ValueError('quantity must be positive')\n"
              "    if quantity > available:\n        raise InsufficientStock('insufficient stock')\n"
              "    return available - quantity\n")),
    ("Write a Python multi-file inventory service module service class.",
     fragment("service.py", "class", ["import"],
              "class InventoryService:\n"
              "    def __init__(self, stock):\n        self.stock = dict(stock)\n\n")),
    ("Write a Python multi-file inventory service module reserve method with an insufficient stock guard.",
     fragment("service.py", "method", ["class"],
              "    def reserve(self, sku, quantity):\n"
              "        if sku not in self.stock:\n            raise KeyError(sku)\n"
              "        self.stock[sku] = reserve_stock(self.stock[sku], quantity)\n"
              "        return self.stock[sku]\n")),
]

IMPORT_PROMPT = "Write a Python multi-file inventory service module import statement."
BAD_IMPORT = fragment("service.py", "import", ["domain.py::reserve"],
                      "from domain import reserve_inventory, InsufficientStock\n\n", "bad_import")
GOOD_IMPORT = fragment("service.py", "import", ["domain.py::reserve"],
                       "from domain import reserve_stock, InsufficientStock\n\n", "good_import")
REJECT_BAD = json.dumps({"fragment_outcome": {"evidence_id": "bad_import", "confirmed": False}},
                        sort_keys=True, separators=(",", ":"))
CONFIRM_GOOD = json.dumps({"fragment_outcome": {"evidence_id": "good_import", "confirmed": True}},
                          sort_keys=True, separators=(",", ":"))

PROMPTS = [
    ("canonical", "Build a Python multi-file inventory project with a domain module, service module, exception declaration, import statement, service class, reserve method, and insufficient stock guard."),
    ("heldout", "Create a Python multi-file inventory project with a domain file, service file, exception class, import line, service class, and reservation method that rejects insufficient stock."),
]
OOV = "Connect the inventory project to an unspecified proprietary warehouse protocol."


def observe_pair(endpoint: str, prompt: str, action: str, repeats: int) -> None:
    for _ in range(repeats):
        request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(prompt)})
        request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(prompt)})
        request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(action)})
        request(endpoint, "/brain/tick", {})


def execute(response: str) -> tuple[bool, str, dict[str, str]]:
    try:
        files = json.loads(response)["files"]
        if not isinstance(files, dict) or not files:
            return False, "invalid/empty files", {}
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}", {}
    with tempfile.TemporaryDirectory(prefix="wv-multifile-repair-") as raw:
        root = Path(raw)
        for name, source in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(source, str):
                return False, f"unsafe file {name!r}", {}
            (root / name).write_text(source, encoding="utf-8")
        harness = root / "integration.py"
        harness.write_text(
            "from domain import InsufficientStock\nfrom service import InventoryService\n"
            "s=InventoryService({'A':5}); assert s.reserve('A',2)==3\n"
            "try:\n s.reserve('A',4); raise AssertionError('over-reserved')\n"
            "except InsufficientStock: pass\nassert s.stock['A']==3\nprint('PASS')\n",
            encoding="utf-8")
        runner = ("import runpy,sys; sys.path.insert(0,'.'); "
                  "runpy.run_path('integration.py', run_name='__main__')")
        run = subprocess.run(["python", "-I", "-c", runner], cwd=root,
                             capture_output=True, text=True, timeout=15)
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-1200:], files


def environment() -> str:
    return json.dumps({"language": "python", "version": platform.python_version(),
                       "operating_system": platform.system(), "architecture": platform.machine(),
                       "logical_cpus": os.cpu_count()}, sort_keys=True, separators=(",", ":"))


def train(endpoint: str, repeats: int) -> dict:
    for prompt, action in BASE:
        observe_pair(endpoint, prompt, action, repeats)
    observe_pair(endpoint, IMPORT_PROMPT, BAD_IMPORT, repeats)
    # Execute the deliberately broken candidate directly. This keeps the
    # failure fixture stable even when a previous run has already learned the
    # rejection marker and correctly suppresses it during ordinary chat.
    base_sources = [json.loads(action)["code_fragment"] for _, action in BASE]
    bad = json.loads(BAD_IMPORT)["code_fragment"]
    broken_files = {
        "domain.py": "".join(f["source"] for f in base_sources if f["file"] == "domain.py"),
        "service.py": bad["source"] + "".join(
            f["source"] for f in base_sources if f["file"] == "service.py"),
    }
    broken_reply = json.dumps({"files": broken_files}, sort_keys=True, separators=(",", ":"))
    broken_ok, console_before, broken_files = execute(broken_reply)
    if broken_ok:
        raise RuntimeError("broken import unexpectedly executed")

    good_source = json.loads(GOOD_IMPORT)["code_fragment"]["source"]
    bad_source = json.loads(BAD_IMPORT)["code_fragment"]["source"]
    delta = "".join(difflib.unified_diff(bad_source.splitlines(True),
                                          good_source.splitlines(True),
                                          fromfile="bad_import", tofile="good_import"))
    episode = {
        1: IMPORT_PROMPT,
        2: json.dumps(broken_files, sort_keys=True),
        3: console_before,
        5: environment(),
        6: json.dumps({"status": "failure", "error_class": "ImportError"}),
        7: delta,
        8: "PASS",
        9: json.dumps({"transition": "failure_to_success", "repair": "symbol_name"}),
        10: bad_source,
        11: json.dumps({"kind": "replace_import_symbol", "from": "reserve_inventory",
                        "to": "reserve_stock"}),
        12: IMPORT_PROMPT,
    }
    for _ in range(repeats):
        for pool_id, frame in episode.items():
            request(endpoint, "/brain/observe", {"pool_id": pool_id, "frame": b64(frame)})
        request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(GOOD_IMPORT)})
        request(endpoint, "/brain/tick", {})
    observe_pair(endpoint, IMPORT_PROMPT, REJECT_BAD, 3)
    observe_pair(endpoint, IMPORT_PROMPT, GOOD_IMPORT, repeats)
    observe_pair(endpoint, IMPORT_PROMPT, CONFIRM_GOOD, 3)
    return {"failed_before": True, "console_before": console_before[-500:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/multifile_repair_synthesis.json"))
    args = parser.parse_args()
    training = {} if args.no_train else train(args.endpoint, args.repeats)
    rows = []
    for kind, prompt in PROMPTS:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        passed, detail, files = execute(str(result.get("reply") or ""))
        rows.append({"kind": kind, "executes": passed, "files": sorted(files),
                     "detail": "" if passed else detail})
    unknown = request(args.endpoint, "/brain/chat", {"text": OOV})
    honest = not unknown.get("reply") and bool((unknown.get("grounding") or {}).get("outside_grounding"))
    report = {"passed": sum(row["executes"] for row in rows), "total": len(rows),
              "oov_honest": honest, "training": training, "results": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "oov_honest": honest, "failed_before": training.get("failed_before")}))
    return 0 if report["passed"] == report["total"] and honest else 1


if __name__ == "__main__":
    raise SystemExit(main())
