#!/usr/bin/env python3
"""Train and execution-test one atom-grounded task across installed languages."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    language: str
    prompt: str
    paraphrase: str
    response: str
    filename: str
    command: tuple[str, ...]
    harness: str = ""


CASES = [
    Case("python", "Write a Python function named square that returns its input multiplied by itself.",
         "Create Python code computing the second power of a supplied number.",
         "def square(n):\n    return n * n", "main.py", ("python", "-I", "main.py"),
         "\nassert square(7) == 49\nprint('PASS')\n"),
    Case("javascript", "Write a JavaScript function named square that returns its input multiplied by itself.",
         "Create Node.js code computing the second power of a supplied number.",
         "function square(n) {\n  return n * n;\n}", "main.js", ("node", "main.js"),
         "\nif (square(7) !== 49) throw new Error('wrong result');\nconsole.log('PASS');\n"),
    Case("csharp", "Write a C# method named Square that returns its integer input multiplied by itself.",
         "Create a C sharp function computing the second power of a supplied integer.",
         "static int Square(int n)\n{\n    return n * n;\n}", "Program.cs",
         ("dotnet", "run", "--project", "Eval.csproj", "--nologo")),
    Case("go", "Write a Go function named square that returns its integer input multiplied by itself.",
         "Create a Golang function computing the second power of a supplied integer.",
         "func square(n int) int {\n\treturn n * n\n}", "main.go", ("go", "run", "main.go")),
    Case("rust", "Write a Rust function named square that returns its integer input multiplied by itself.",
         "Create Rust code computing the second power of a supplied integer.",
         "fn square(n: i32) -> i32 {\n    n * n\n}", "main.rs", ("rustc", "main.rs", "-o", "eval.exe")),
    Case("java", "Write a Java method named square that returns its integer input multiplied by itself.",
         "Create Java code computing the second power of a supplied integer.",
         "static int square(int n) {\n    return n * n;\n}", "Main.java", ("javac", "Main.java")),
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


def execute(case: Case, code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(prefix=f"wv-{case.language}-") as raw:
        work = Path(raw)
        if case.language == "csharp":
            (work / "Eval.csproj").write_text(
                '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType>'
                '<TargetFramework>net8.0</TargetFramework></PropertyGroup></Project>', encoding="utf-8")
            source = ("using System; class Program {\n" + code
                      + "\nstatic void Main() { if (Square(7) != 49) throw new Exception(); Console.WriteLine(\"PASS\"); }\n}")
        elif case.language == "go":
            source = ("package main\nimport \"fmt\"\n" + code
                      + "\nfunc main() { if square(7) != 49 { panic(\"wrong result\") }; fmt.Println(\"PASS\") }\n")
        elif case.language == "rust":
            source = code + "\nfn main() { assert_eq!(square(7), 49); println!(\"PASS\"); }\n"
        elif case.language == "java":
            source = ("class Main {\n" + code
                      + "\npublic static void main(String[] args) { if (square(7) != 49) throw new RuntimeException(); System.out.println(\"PASS\"); }\n}")
        else:
            source = code + case.harness
        (work / case.filename).write_text(source, encoding="utf-8")
        try:
            first = subprocess.run(case.command, cwd=work, capture_output=True, text=True, timeout=30)
            if first.returncode != 0:
                return False, (first.stderr or first.stdout)[-400:]
            if case.language == "rust":
                first = subprocess.run([str(work / "eval.exe")], cwd=work,
                                       capture_output=True, text=True, timeout=5)
            elif case.language == "java":
                first = subprocess.run(["java", "Main"], cwd=work,
                                       capture_output=True, text=True, timeout=5)
            return first.returncode == 0 and "PASS" in first.stdout, (first.stderr or first.stdout)[-400:]
        except (subprocess.TimeoutExpired, FileNotFoundError) as error:
            return False, str(error)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/multilanguage.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)
    results = []
    for case in CASES:
        for kind, prompt in (("trained", case.prompt), ("paraphrase", case.paraphrase)):
            reply = str(request(args.endpoint, "/brain/chat", {"text": prompt}).get("reply") or "")
            passed, detail = execute(case, reply)
            results.append({"language": case.language, "kind": kind, "nonempty": bool(reply),
                            "exact": reply == case.response, "executes": passed,
                            "detail": "" if passed else detail})
    summary = {kind: {"executes": sum(r["executes"] for r in results if r["kind"] == kind),
                      "total": len(CASES)} for kind in ("trained", "paraphrase")}
    report = {"summary": summary, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
