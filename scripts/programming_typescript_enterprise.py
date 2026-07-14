#!/usr/bin/env python3
"""Strictly compile and execute the persistent brain's TypeScript curriculum."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from programming_project_eval import b64, manifest, request


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    paraphrase: str
    response: str
    integration: str


CASES = [
    Case(
        "idempotent_orders",
        "Implement a TypeScript idempotent order command service that validates input and replays the first result for a repeated key.",
        "Create typed order-handling code that prevents duplicate TypeScript commands when clients retry with the same idempotency key.",
        manifest({"orders.ts":
            "export type OrderInput = Readonly<{ sku: string; quantity: number }>;\n"
            "export type Order = Readonly<OrderInput & { id: number }>;\n"
            "export class OrderService {\n"
            "  private readonly orders: Order[] = [];\n"
            "  private readonly responses = new Map<string, Order>();\n"
            "  create(input: OrderInput, key: string): Order {\n"
            "    if (!key) throw new Error('idempotency key required');\n"
            "    const previous = this.responses.get(key); if (previous) return previous;\n"
            "    if (!input.sku || !Number.isInteger(input.quantity) || input.quantity < 1) throw new Error('invalid order');\n"
            "    const order = Object.freeze({ id: this.orders.length + 1, ...input });\n"
            "    this.orders.push(order); this.responses.set(key, order); return order;\n"
            "  }\n  count(): number { return this.orders.length; }\n}\n"}),
        "import { OrderService } from './orders';\n"
        "const service = new OrderService();\n"
        "const first = service.create({sku:'A',quantity:2},'key');\n"
        "const replay = service.create({sku:'B',quantity:9},'key');\n"
        "if (first !== replay || service.count() !== 1) throw new Error('duplicate');\n"
        "console.log('PASS');\n",
    ),
    Case(
        "async_retry",
        "Implement a TypeScript asynchronous retry helper with bounded maxAttempts that returns on success and rethrows the final error.",
        "Write TypeScript async retry code that limits attempts, stops after success, and preserves the last exception.",
        manifest({"retry.ts":
            "export async function retry<T>(operation: () => Promise<T>, maxAttempts: number): Promise<T> {\n"
            "  if (!Number.isInteger(maxAttempts) || maxAttempts < 1) throw new RangeError('maxAttempts');\n"
            "  for (let attempt = 1; ; attempt += 1) {\n"
            "    try { return await operation(); }\n"
            "    catch (error: unknown) { if (attempt >= maxAttempts) throw error; }\n"
            "  }\n}\n"}),
        "import { retry } from './retry';\n"
        "async function main(): Promise<void> {\n"
        "  let calls=0; const value=await retry(async()=>{calls++; if(calls<3) throw new Error('x'); return 7;},3);\n"
        "  if(value!==7 || calls!==3) throw new Error('retry failed');\n"
        "  let final: unknown; try { await retry(async()=>{throw new TypeError('final');},2); } catch(e: unknown) { final=e; }\n"
        "  if(!(final instanceof TypeError) || final.message!=='final') throw new Error('wrong error');\n"
        "  console.log('PASS');\n}\nvoid main();\n",
    ),
    Case(
        "optimistic_store",
        "Implement a TypeScript in-memory store with optimistic concurrency that rejects stale expected versions.",
        "Create typed version-checked storage so stale TypeScript writers cannot replace newer state.",
        manifest({"store.ts":
            "export type Entry<T> = Readonly<{ value: T; version: number }>;\n"
            "export class VersionedStore<T> {\n"
            "  private readonly values = new Map<string, Entry<T>>();\n"
            "  put(key: string, value: T, expectedVersion: number): Entry<T> {\n"
            "    const actual = this.values.get(key)?.version ?? 0;\n"
            "    if (actual !== expectedVersion) throw new Error('stale write');\n"
            "    const next = Object.freeze({value, version: actual + 1}); this.values.set(key,next); return next;\n"
            "  }\n  get(key: string): Entry<T> | undefined { return this.values.get(key); }\n}\n"}),
        "import { VersionedStore } from './store';\n"
        "const store=new VersionedStore<string>(); store.put('k','a',0); store.put('k','b',1);\n"
        "let stale=false; try { store.put('k','old',1); } catch { stale=true; }\n"
        "if(!stale || store.get('k')?.value!=='b') throw new Error('optimistic concurrency failed');\n"
        "console.log('PASS');\n",
    ),
]

OOV = [
    "Implement a TypeScript client for an undocumented binary protocol.",
    "Create a TypeScript migration whose source and target schemas have not been provided.",
    "Build TypeScript production retry timing without any latency requirements or service objectives.",
]


def execute(response: str, integration: str) -> tuple[bool, str]:
    try:
        files = json.loads(response)["files"]
        if not isinstance(files, dict) or not files:
            return False, "empty manifest"
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}"
    with tempfile.TemporaryDirectory(prefix="wv-typescript-") as raw:
        root = Path(raw)
        for name, content in {**files, "integration.ts": integration}.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(content, str):
                return False, f"unsafe file: {name!r}"
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        compiler = shutil.which("tsc") or shutil.which("tsc.cmd")
        if compiler is None:
            return False, "TypeScript compiler is unavailable"
        compile_run = subprocess.run(
            [compiler, "--strict", "--target", "ES2022", "--module", "commonjs",
             "--outDir", "dist", *sorted(files), "integration.ts"],
            cwd=root, capture_output=True, text=True, timeout=30,
        )
        if compile_run.returncode != 0:
            return False, compile_run.stderr or compile_run.stdout
        run = subprocess.run(
            ["node", "dist/integration.js"], cwd=root,
            capture_output=True, text=True, timeout=15,
        )
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-900:]


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for case in CASES:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(case.response)})
            request(endpoint, "/brain/tick", {})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runtime/benchmarks/typescript_enterprise.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    rows = []
    for case in CASES:
        for kind, prompt in (("trained", case.prompt), ("paraphrase", case.paraphrase)):
            response = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(response.get("reply") or "")
            passed, detail = execute(reply, case.integration)
            rows.append({"name": case.name, "kind": kind, "executes": passed,
                         "exact": reply == case.response, "detail": "" if passed else detail})
    oov = []
    for prompt in OOV:
        response = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not response.get("reply") and bool((response.get("grounding") or {}).get("outside_grounding"))
        oov.append({"prompt": prompt, "honest": honest, "reply": response.get("reply")})
    summary = {kind: {"executes": sum(row["executes"] for row in rows if row["kind"] == kind),
                      "total": len(CASES)} for kind in ("trained", "paraphrase")}
    summary["oov_honesty"] = {"passed": sum(row["honest"] for row in oov), "total": len(oov)}
    report = {"summary": summary, "results": rows, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in rows) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
