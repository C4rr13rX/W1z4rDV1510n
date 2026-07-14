#!/usr/bin/env python3
"""Compile and execute multi-file enterprise projects across native runtimes."""
from __future__ import annotations

import argparse
import json
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
    test_files: dict[str, str]
    command: tuple[str, ...]
    after_command: tuple[str, ...] | None = None


CASES = [
    Case(
        "javascript_outbox",
        "Build a JavaScript order service with idempotent commands and a transactional outbox event.",
        "Create Node.js code that prevents duplicate orders and records one outbox event for reliable publication.",
        manifest({"order_service.js":
            "class OrderService {\n"
            "  constructor() { this.orders = []; this.outbox = []; this.responses = new Map(); }\n"
            "  create(input, key) {\n"
            "    if (!key) throw new Error('idempotency key required');\n"
            "    if (this.responses.has(key)) return this.responses.get(key);\n"
            "    if (!input || !input.sku || !Number.isInteger(input.quantity) || input.quantity < 1) throw new Error('invalid order');\n"
            "    const order = Object.freeze({ id: this.orders.length + 1, sku: input.sku, quantity: input.quantity });\n"
            "    this.orders.push(order);\n"
            "    this.outbox.push({ type: 'OrderCreated', orderId: order.id });\n"
            "    this.responses.set(key, order);\n    return order;\n  }\n}\nmodule.exports = { OrderService };\n"}),
        {"integration.js":
            "const assert = require('assert'); const {OrderService}=require('./order_service');\n"
            "const s=new OrderService(); const a=s.create({sku:'A',quantity:2},'k');\n"
            "const b=s.create({sku:'B',quantity:9},'k'); assert.deepStrictEqual(a,b);\n"
            "assert.strictEqual(s.orders.length,1); assert.strictEqual(s.outbox.length,1); console.log('PASS');\n"},
        ("node", "integration.js"),
    ),
    Case(
        "go_deduplication",
        "Implement a Go concurrency-safe deduplicator that accepts each work key only once.",
        "Write Golang code using synchronization to suppress duplicate work across concurrent callers.",
        manifest({"dedup.go":
            "package main\nimport \"sync\"\n"
            "type Deduplicator struct { mu sync.Mutex; seen map[string]struct{} }\n"
            "func NewDeduplicator() *Deduplicator { return &Deduplicator{seen: make(map[string]struct{})} }\n"
            "func (d *Deduplicator) AddIfNew(key string) bool {\n"
            " d.mu.Lock(); defer d.mu.Unlock(); if _, ok := d.seen[key]; ok { return false }; d.seen[key]=struct{}{}; return true\n}\n"}),
        {"integration.go":
            "package main\nimport (\"fmt\"; \"sync\"; \"sync/atomic\")\n"
            "func main(){ d:=NewDeduplicator(); var accepted int32; var wg sync.WaitGroup; for i:=0;i<50;i++ { wg.Add(1); go func(){ defer wg.Done(); if d.AddIfNew(\"job\") { atomic.AddInt32(&accepted,1) } }() }; wg.Wait(); if accepted != 1 { panic(accepted) }; fmt.Println(\"PASS\") }\n"},
        ("go", "run", "integration.go", "dedup.go"),
    ),
    Case(
        "csharp_async_retry",
        "Implement a C# asynchronous retry policy that stops on success and rethrows after maxAttempts.",
        "Write C sharp async retry code that bounds attempts and preserves the final exception.",
        manifest({"RetryPolicy.cs":
            "using System; using System.Threading.Tasks;\n"
            "public static class RetryPolicy {\n"
            " public static async Task<T> ExecuteAsync<T>(Func<Task<T>> operation, int maxAttempts) {\n"
            "  if (maxAttempts < 1) throw new ArgumentOutOfRangeException(nameof(maxAttempts));\n"
            "  for (int attempt=1;;attempt++) { try { return await operation(); } catch when (attempt < maxAttempts) { } }\n"
            " }\n}\n"}),
        {"Eval.csproj":
            "<Project Sdk=\"Microsoft.NET.Sdk\"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>",
         "Program.cs":
            "using System; using System.Threading.Tasks; class Program { static async Task Main(){ int calls=0; var value=await RetryPolicy.ExecuteAsync(async()=>{ await Task.Yield(); calls++; if(calls<3) throw new InvalidOperationException(); return 7; },3); if(value!=7||calls!=3) throw new Exception(); calls=0; try { await RetryPolicy.ExecuteAsync<int>(()=>{ calls++; throw new ApplicationException(\"final\"); },2); } catch(ApplicationException e) { if(e.Message!=\"final\"||calls!=2) throw; Console.WriteLine(\"PASS\"); return; } throw new Exception(); } }"},
        ("dotnet", "run", "--project", "Eval.csproj", "--nologo"),
    ),
    Case(
        "java_optimistic_store",
        "Implement a Java in-memory store with optimistic concurrency using expected versions to reject stale writes.",
        "Create Java code that version-checks updates so stale writers cannot overwrite newer state.",
        manifest({"VersionedStore.java":
            "import java.util.*;\npublic final class VersionedStore {\n"
            " public record Entry(String value, long version) {}\n private final Map<String,Entry> data=new HashMap<>();\n"
            " public synchronized Entry put(String key,String value,long expectedVersion){ Entry current=data.get(key); long actual=current==null?0:current.version(); if(actual!=expectedVersion) throw new ConcurrentModificationException(); Entry next=new Entry(value,actual+1); data.put(key,next); return next; }\n"
            " public synchronized Entry get(String key){ return data.get(key); }\n}\n"}),
        {"Integration.java":
            "public class Integration { public static void main(String[] a){ VersionedStore s=new VersionedStore(); var first=s.put(\"k\",\"a\",0); if(first.version()!=1) throw new RuntimeException(); s.put(\"k\",\"b\",1); try { s.put(\"k\",\"stale\",1); throw new RuntimeException(); } catch(java.util.ConcurrentModificationException ok){} if(!s.get(\"k\").value().equals(\"b\")) throw new RuntimeException(); System.out.println(\"PASS\"); } }"},
        ("javac", "VersionedStore.java", "Integration.java"), ("java", "Integration"),
    ),
    Case(
        "rust_atomic_ledger",
        "Build a Rust ledger with an atomic transfer that preserves balances when validation fails.",
        "Create Rust code for all-or-nothing account transfers that reject missing accounts and insufficient funds.",
        manifest({"ledger.rs":
            "use std::collections::HashMap;\npub struct Ledger { balances: HashMap<String,i64> }\n"
            "impl Ledger { pub fn new(entries: &[(&str,i64)]) -> Self { Self { balances: entries.iter().map(|(k,v)|(k.to_string(),*v)).collect() } }\n"
            " pub fn balance(&self,id:&str)->Option<i64>{self.balances.get(id).copied()}\n"
            " pub fn transfer(&mut self,from:&str,to:&str,amount:i64)->Result<(),String>{ if amount<=0{return Err(\"invalid amount\".into())} let source=self.balance(from).ok_or(\"missing source\")?; let target=self.balance(to).ok_or(\"missing target\")?; if source<amount{return Err(\"insufficient funds\".into())} self.balances.insert(from.into(),source-amount); self.balances.insert(to.into(),target+amount); Ok(()) } }\n"}),
        {"main.rs":
            "mod ledger; use ledger::Ledger; fn main(){ let mut l=Ledger::new(&[(\"a\",100),(\"b\",20)]); l.transfer(\"a\",\"b\",30).unwrap(); assert_eq!(l.balance(\"a\"),Some(70)); assert!(l.transfer(\"a\",\"missing\",10).is_err()); assert_eq!(l.balance(\"a\"),Some(70)); assert!(l.transfer(\"a\",\"b\",100).is_err()); assert_eq!(l.balance(\"b\"),Some(50)); println!(\"PASS\"); }"},
        ("rustc", "main.rs", "-o", "eval.exe"), ("eval.exe",),
    ),
]

OOV = [
    "Build a JavaScript transactional outbox for an unspecified message broker contract.",
    "Implement a Go deduplicator whose required retention duration and storage are not provided.",
    "Create a Rust ledger integration for an undocumented settlement protocol.",
]


def execute(case: Case, response: str) -> tuple[bool, str]:
    try:
        files = json.loads(response)["files"]
        if not isinstance(files, dict) or not files: return False, "empty manifest"
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}"
    with tempfile.TemporaryDirectory(prefix=f"wv-native-{case.name}-") as raw:
        root = Path(raw)
        for name, content in {**files, **case.test_files}.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(content, str):
                return False, f"unsafe file: {name!r}"
            target = root.joinpath(*relative.parts); target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            run = subprocess.run(case.command, cwd=root, capture_output=True, text=True, timeout=45)
            if run.returncode == 0 and case.after_command:
                command = list(case.after_command)
                if command[0] == "eval.exe": command[0] = str(root / "eval.exe")
                run = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=15)
        except (subprocess.TimeoutExpired, FileNotFoundError) as error:
            return False, str(error)
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-900:]


def train(endpoint: str, repeats: int) -> None:
    for _ in range(repeats):
        for case in CASES:
            request(endpoint, "/brain/observe", {"pool_id": 1, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 12, "frame": b64(case.prompt)})
            request(endpoint, "/brain/observe", {"pool_id": 4, "frame": b64(case.response)})
            request(endpoint, "/brain/tick", {})


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--endpoint",default="http://127.0.0.1:18600")
    parser.add_argument("--repeats",type=int,default=6); parser.add_argument("--no-train",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("runtime/benchmarks/native_enterprise.json")); args=parser.parse_args()
    if not args.no_train: train(args.endpoint,args.repeats)
    results=[]
    for case in CASES:
        for kind,prompt in (("trained",case.prompt),("paraphrase",case.paraphrase)):
            reply=str(request(args.endpoint,"/brain/chat",{"text":prompt}).get("reply") or "")
            passed,detail=execute(case,reply); results.append({"name":case.name,"kind":kind,"exact":reply==case.response,"executes":passed,"detail":"" if passed else detail})
    oov=[]
    for prompt in OOV:
        result=request(args.endpoint,"/brain/chat",{"text":prompt}); honest=not result.get("reply") and bool((result.get("grounding") or {}).get("outside_grounding")); oov.append({"prompt":prompt,"honest":honest})
    summary={kind:{"executes":sum(r["executes"] for r in results if r["kind"]==kind),"total":len(CASES)} for kind in ("trained","paraphrase")}; summary["oov_honesty"]={"passed":sum(r["honest"] for r in oov),"total":len(oov)}
    report={"summary":summary,"results":results,"oov":oov}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(summary))
    return 0 if all(r["executes"] for r in results) and all(r["honest"] for r in oov) else 1


if __name__ == "__main__": raise SystemExit(main())
