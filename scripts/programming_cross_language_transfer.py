#!/usr/bin/env python3
"""Executable benchmark for language x behavior integration.

Each language and each enterprise behavior already occurs elsewhere in the
curriculum, but these exact pairings do not.  Run with --no-train to measure
zero-shot integration, then train the pairings and use the held-out prompts to
measure whether raw-character semantic routing generalizes beyond memorization.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from programming_exec_env import benchmark_tool_env, run_tool

from programming_project_eval import b64, manifest, request


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    heldout: str
    response: str
    harness: dict[str, str]
    compile: tuple[str, ...]
    run: tuple[str, ...] | None = None


CASES = [
    Case(
        "javascript_authorization",
        "Build JavaScript default-deny authorization where administrators may do anything and owners may only read their own records.",
        "Create Node.js access rules granting superusers every operation while ordinary identities can only inspect objects they own.",
        manifest({"authorization.js":
            "function isAuthorized(principal, action, ownerId) {\n"
            "  if (!principal || typeof principal !== 'object') return false;\n"
            "  const roles = new Set(Array.isArray(principal.roles) ? principal.roles : []);\n"
            "  if (roles.has('admin')) return true;\n"
            "  return principal.id != null && action === 'read' && principal.id === ownerId;\n"
            "}\nmodule.exports = { isAuthorized };\n"}),
        {"integration.js":
            "const assert=require('assert'); const {isAuthorized}=require('./authorization');\n"
            "assert(isAuthorized({id:'a',roles:['admin']},'delete','x'));\n"
            "assert(isAuthorized({id:'u',roles:[]},'read','u'));\n"
            "assert(!isAuthorized({id:'u',roles:[]},'write','u'));\n"
            "assert(!isAuthorized(null,'read',null)); console.log('PASS');\n"},
        ("node", "integration.js"),
    ),
    Case(
        "go_atomic_transfer",
        "Build a Go in-memory ledger whose transfer changes both balances atomically and leaves them untouched on validation failure.",
        "Write Golang account movement so a missing account or insufficient funds can never leave only one side updated.",
        manifest({"ledger.go":
            "package main\nimport (\"errors\"; \"sync\")\n"
            "type Ledger struct { mu sync.Mutex; balances map[string]int64 }\n"
            "func NewLedger(v map[string]int64) *Ledger { c:=map[string]int64{}; for k,n:=range v { c[k]=n }; return &Ledger{balances:c} }\n"
            "func (l *Ledger) Balance(id string) (int64,bool) { l.mu.Lock(); defer l.mu.Unlock(); n,ok:=l.balances[id]; return n,ok }\n"
            "func (l *Ledger) Transfer(from,to string, amount int64) error { l.mu.Lock(); defer l.mu.Unlock(); if amount<=0{return errors.New(\"invalid amount\")}; a,ok:=l.balances[from]; if !ok{return errors.New(\"missing source\")}; b,ok:=l.balances[to]; if !ok{return errors.New(\"missing target\")}; if a<amount{return errors.New(\"insufficient funds\")}; l.balances[from]=a-amount; l.balances[to]=b+amount; return nil }\n"}),
        {"integration.go":
            "package main\nimport \"fmt\"\nfunc main(){ l:=NewLedger(map[string]int64{\"a\":10,\"b\":2}); if l.Transfer(\"a\",\"b\",4)!=nil{panic(\"valid\")}; a,_:=l.Balance(\"a\"); if a!=6{panic(a)}; if l.Transfer(\"a\",\"missing\",2)==nil{panic(\"missing\")}; a,_=l.Balance(\"a\"); if a!=6{panic(\"partial\")}; fmt.Println(\"PASS\") }\n"},
        ("go", "run", "integration.go", "ledger.go"),
    ),
    Case(
        "rust_idempotent_commands",
        "Build a Rust order service where an idempotency key returns the original order and never creates a duplicate.",
        "Create Rust command handling that remembers a request key so retries produce one stored order and the first result.",
        manifest({"orders.rs":
            "use std::collections::HashMap;\n#[derive(Clone,Debug,PartialEq)] pub struct Order { pub id: usize, pub sku: String }\n"
            "pub struct OrderService { pub orders: Vec<Order>, responses: HashMap<String,Order> }\n"
            "impl OrderService { pub fn new()->Self{Self{orders:Vec::new(),responses:HashMap::new()}} pub fn create(&mut self,sku:&str,key:&str)->Result<Order,String>{ if key.is_empty(){return Err(\"idempotency key required\".into())} if let Some(v)=self.responses.get(key){return Ok(v.clone())} if sku.is_empty(){return Err(\"invalid sku\".into())} let order=Order{id:self.orders.len()+1,sku:sku.into()}; self.orders.push(order.clone()); self.responses.insert(key.into(),order.clone()); Ok(order) } }\n"}),
        {"main.rs":
            "mod orders; use orders::OrderService; fn main(){let mut s=OrderService::new(); let a=s.create(\"A\",\"k\").unwrap(); let b=s.create(\"B\",\"k\").unwrap(); assert_eq!(a,b); assert_eq!(s.orders.len(),1); println!(\"PASS\");}\n"},
        ("rustc", "main.rs", "-o", "eval.exe"), ("eval.exe",),
    ),
    Case(
        "java_redacted_logging",
        "Build Java structured audit logging with a correlation identifier and recursive redaction of password and token fields.",
        "Create Java machine-readable event records carrying request tracing while cleansing nested credentials.",
        manifest({"AuditLog.java":
            "import java.util.*; public final class AuditLog { private static final Set<String> SECRET=Set.of(\"password\",\"token\",\"secret\"); private AuditLog(){} public static Map<String,Object> event(String name,String correlationId,Map<String,Object> data){ Map<String,Object> out=new LinkedHashMap<>(); out.put(\"event\",name); out.put(\"correlationId\",correlationId); out.put(\"data\",redact(data)); return out; } @SuppressWarnings(\"unchecked\") private static Object redact(Object value){ if(value instanceof Map<?,?> map){ Map<String,Object> copy=new LinkedHashMap<>(); for(var e:map.entrySet()){String k=String.valueOf(e.getKey()); copy.put(k,SECRET.contains(k.toLowerCase())?\"[REDACTED]\":redact(e.getValue()));} return copy;} if(value instanceof List<?> list){List<Object> copy=new ArrayList<>(); for(Object v:list)copy.add(redact(v)); return copy;} return value; } }\n"}),
        {"Integration.java":
            "import java.util.*; public class Integration { public static void main(String[] x){var nested=new LinkedHashMap<String,Object>(); nested.put(\"token\",\"raw\"); var data=new LinkedHashMap<String,Object>(); data.put(\"nested\",nested); var e=AuditLog.event(\"login\",\"req-7\",data); String s=e.toString(); if(!s.contains(\"req-7\")||s.contains(\"raw\"))throw new RuntimeException(s); System.out.println(\"PASS\");}}\n"},
        ("javac", "AuditLog.java", "Integration.java"), ("java", "Integration"),
    ),
]

OOV = [
    "Connect the Java audit logger to an unspecified proprietary telemetry protocol.",
    "Persist the Rust idempotency table using an undocumented database schema.",
]


def execute(case: Case, response: str) -> tuple[bool, str]:
    try:
        files = json.loads(response)["files"]
        if not isinstance(files, dict) or not files:
            return False, "empty manifest"
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        return False, f"invalid manifest: {error}"
    with tempfile.TemporaryDirectory(prefix=f"wv-transfer-{case.name}-") as raw:
        root = Path(raw)
        environment = benchmark_tool_env()
        for name, content in {**files, **case.harness}.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(content, str):
                return False, f"unsafe file: {name!r}"
            target = root.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        try:
            run = run_tool(case.compile, cwd=root, env=environment, timeout=90)
            if run.returncode == 0 and case.run:
                command = list(case.run)
                if command[0] == "eval.exe":
                    command[0] = str(root / "eval.exe")
                run = run_tool(command, cwd=root, env=environment, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runtime/benchmarks/cross_language_transfer.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    rows = []
    for case in CASES:
        for kind, prompt in (("canonical", case.prompt), ("heldout", case.heldout)):
            result = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(result.get("reply") or "")
            passed, detail = execute(case, reply)
            rows.append({"name": case.name, "kind": kind, "executes": passed,
                         "outside_grounding": bool((result.get("grounding") or {}).get("outside_grounding")),
                         "detail": "" if passed else detail})
    oov = []
    for prompt in OOV:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not result.get("reply") and bool((result.get("grounding") or {}).get("outside_grounding"))
        oov.append({"prompt": prompt, "honest": honest})
    summary = {kind: {"executes": sum(r["executes"] for r in rows if r["kind"] == kind),
                      "total": len(CASES)} for kind in ("canonical", "heldout")}
    summary["oov_honesty"] = {"passed": sum(r["honest"] for r in oov), "total": len(oov)}
    report = {"summary": summary, "results": rows, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(r["executes"] for r in rows) and all(r["honest"] for r in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
