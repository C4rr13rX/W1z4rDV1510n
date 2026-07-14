#!/usr/bin/env python3
"""Train and execute multi-file enterprise project answers from the persistent brain."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def manifest(files: dict[str, str]) -> str:
    return json.dumps({"files": files}, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    paraphrase: str
    response: str
    integration_test: str


CASES = [
    Case(
        "multifile_inventory",
        "Build a Python multi-file inventory project with separate domain and service files that prevents over-reservation.",
        "Create Python code in multiple files separating inventory rules from a service that reserves stock safely.",
        manifest({
            "domain.py":
                "class OutOfStock(ValueError):\n    pass\n\n"
                "def reserve_stock(available, quantity):\n"
                "    if quantity < 1:\n        raise ValueError(\"quantity must be positive\")\n"
                "    if quantity > available:\n        raise OutOfStock(\"insufficient stock\")\n"
                "    return available - quantity\n",
            "service.py":
                "from domain import reserve_stock\n\n"
                "class InventoryService:\n"
                "    def __init__(self, stock):\n        self.stock = dict(stock)\n\n"
                "    def reserve(self, sku, quantity):\n"
                "        if sku not in self.stock:\n            raise KeyError(sku)\n"
                "        self.stock[sku] = reserve_stock(self.stock[sku], quantity)\n"
                "        return self.stock[sku]\n",
        }),
        "from domain import OutOfStock\nfrom service import InventoryService\n"
        "service = InventoryService({'A': 5})\nassert service.reserve('A', 2) == 3\n"
        "try:\n    service.reserve('A', 4)\n    raise AssertionError('over-reserved')\nexcept OutOfStock:\n    pass\n"
        "assert service.stock['A'] == 3\n",
    ),
    Case(
        "sqlite_transaction",
        "Implement a Python SQLite repository file with an atomic transfer transaction that rolls back on invalid accounts or insufficient funds.",
        "Write Python database transaction code for an all-or-nothing balance transfer using SQLite.",
        manifest({
            "repository.py":
                "import sqlite3\n\n"
                "def transfer(db_path, source_id, target_id, amount):\n"
                "    if amount <= 0:\n        raise ValueError(\"amount must be positive\")\n"
                "    with sqlite3.connect(db_path) as connection:\n"
                "        debited = connection.execute(\"UPDATE accounts SET balance = balance - ? WHERE id = ? AND balance >= ?\", (amount, source_id, amount))\n"
                "        if debited.rowcount != 1:\n            raise ValueError(\"invalid source or insufficient funds\")\n"
                "        credited = connection.execute(\"UPDATE accounts SET balance = balance + ? WHERE id = ?\", (amount, target_id))\n"
                "        if credited.rowcount != 1:\n            raise ValueError(\"invalid target\")\n",
        }),
        "import sqlite3, tempfile\nfrom pathlib import Path\nfrom repository import transfer\n"
        "db = str(Path(tempfile.mkdtemp()) / 'accounts.db')\n"
        "with sqlite3.connect(db) as c:\n    c.execute('CREATE TABLE accounts(id TEXT PRIMARY KEY, balance INTEGER NOT NULL)')\n    c.executemany('INSERT INTO accounts VALUES(?,?)', [('a',100),('b',20)])\n"
        "transfer(db, 'a', 'b', 30)\n"
        "with sqlite3.connect(db) as c:\n    assert c.execute('SELECT balance FROM accounts WHERE id=\"a\"').fetchone()[0] == 70\n    assert c.execute('SELECT balance FROM accounts WHERE id=\"b\"').fetchone()[0] == 50\n"
        "try:\n    transfer(db, 'a', 'missing', 10)\n    raise AssertionError('invalid target committed')\nexcept ValueError:\n    pass\n"
        "with sqlite3.connect(db) as c:\n    assert c.execute('SELECT balance FROM accounts WHERE id=\"a\"').fetchone()[0] == 70\n",
    ),
    Case(
        "bounded_async",
        "Implement a Python async concurrency module with bounded_map that limits parallel workers and preserves result order.",
        "Write Python asynchronous code using a semaphore for bounded concurrency while returning outputs in input order.",
        manifest({
            "concurrency.py":
                "import asyncio\n\n"
                "async def bounded_map(worker, items, limit):\n"
                "    if limit < 1:\n        raise ValueError(\"limit must be positive\")\n"
                "    semaphore = asyncio.Semaphore(limit)\n"
                "    async def run(item):\n"
                "        async with semaphore:\n            return await worker(item)\n"
                "    return await asyncio.gather(*(run(item) for item in items))\n",
        }),
        "import asyncio\nfrom concurrency import bounded_map\n"
        "async def main():\n"
        "    active = 0\n    peak = 0\n"
        "    async def worker(value):\n"
        "        nonlocal active, peak\n        active += 1\n        peak = max(peak, active)\n"
        "        await asyncio.sleep(0.01 * (4 - value))\n        active -= 1\n        return value * 2\n"
        "    result = await bounded_map(worker, [1,2,3], 2)\n"
        "    assert result == [2,4,6]\n    assert peak <= 2\n"
        "asyncio.run(main())\n",
    ),
    Case(
        "default_deny_authorization",
        "Implement a Python authorization module with default-deny RBAC: admins may act, and owners may only read their own resource.",
        "Create Python access-control code that denies by default, permits administrators, and permits owner reads only.",
        manifest({
            "authorization.py":
                "def is_authorized(principal, action, owner_id):\n"
                "    if not isinstance(principal, dict):\n        return False\n"
                "    roles = set(principal.get(\"roles\", []))\n"
                "    if \"admin\" in roles:\n        return True\n"
                "    principal_id = principal.get(\"id\")\n"
                "    return principal_id is not None and action == \"read\" and principal_id == owner_id\n",
        }),
        "from authorization import is_authorized\n"
        "assert is_authorized({'id':'x','roles':['admin']}, 'delete', 'y')\n"
        "assert is_authorized({'id':'u','roles':['user']}, 'read', 'u')\n"
        "assert not is_authorized({'id':'u','roles':['user']}, 'write', 'u')\n"
        "assert not is_authorized({'id':'u','roles':['user']}, 'read', 'other')\n"
        "assert not is_authorized({}, 'read', None)\nassert not is_authorized(None, 'read', None)\n",
    ),
]

OOV = [
    "Build a Python multi-region deployment project for an unspecified cloud platform.",
    "Implement Python integration files for an undocumented identity provider.",
    "Create a Python database migration for a schema that has not been provided.",
]


def b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).rstrip(b"=").decode()


def request(endpoint: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(endpoint.rstrip("/") + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for case in CASES:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(case.response)})
            request(endpoint, "/brain/tick", {})


def execute_project(response: str, integration_test: str) -> tuple[bool, str]:
    try:
        payload = json.loads(response)
        files = payload["files"]
        if not isinstance(files, dict) or not files:
            return False, "manifest has no files"
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}"
    with tempfile.TemporaryDirectory(prefix="wv-project-") as raw:
        root = Path(raw)
        for name, content in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(content, str):
                return False, f"unsafe manifest path: {name!r}"
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        (root / "integration_test.py").write_text(integration_test + "\nprint('PASS')\n",
                                                   encoding="utf-8")
        try:
            runner = ("import runpy,sys; sys.path.insert(0,'.'); "
                      "runpy.run_path('integration_test.py', run_name='__main__')")
            run = subprocess.run([sys.executable, "-I", "-c", runner], cwd=root,
                                 capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-700:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/projects.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    results = []
    for case in CASES:
        for kind, prompt in (("trained", case.prompt), ("paraphrase", case.paraphrase)):
            result = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(result.get("reply") or "")
            passed, detail = execute_project(reply, case.integration_test)
            results.append({"name": case.name, "kind": kind, "nonempty": bool(reply),
                            "exact": reply == case.response, "executes": passed,
                            "detail": "" if passed else detail})
    oov = []
    for prompt in OOV:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not result.get("reply") and bool((result.get("grounding") or {}).get("outside_grounding"))
        oov.append({"prompt": prompt, "honest": honest, "reply": result.get("reply")})
    summary = {kind: {"executes": sum(row["executes"] for row in results if row["kind"] == kind),
                      "total": len(CASES)} for kind in ("trained", "paraphrase")}
    summary["oov_honesty"] = {"passed": sum(row["honest"] for row in oov), "total": len(oov)}
    report = {"summary": summary, "results": results, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in results) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
