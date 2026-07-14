#!/usr/bin/env python3
"""Execution-backed continual-learning curriculum for enterprise Python behaviors."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    paraphrase: str
    response: str
    assertions: str


CASES = [
    Case(
        "input_validation",
        "Implement a Python function validate_user that validates required user fields id and email and normalizes them.",
        "Write Python input validation for required user identifiers and email addresses, returning normalized values.",
        "def validate_user(payload):\n"
        "    missing = [field for field in (\"id\", \"email\") if not payload.get(field)]\n"
        "    if missing:\n"
        "        raise ValueError(\"missing required fields: \" + \", \".join(missing))\n"
        "    return {\"id\": str(payload[\"id\"]), \"email\": str(payload[\"email\"]).strip().lower()}",
        "assert validate_user({'id': 7, 'email': ' A@EXAMPLE.COM '}) == {'id': '7', 'email': 'a@example.com'}\n"
        "try:\n    validate_user({'id': 1})\n    raise AssertionError('missing email accepted')\nexcept ValueError as e:\n    assert 'email' in str(e)\n",
    ),
    Case(
        "bounded_retry",
        "Implement a Python function retry_call that retries an operation after transient failures up to max_attempts.",
        "Write Python bounded retry behavior that returns on success and re-raises after all attempts fail.",
        "def retry_call(operation, max_attempts):\n"
        "    if max_attempts < 1:\n"
        "        raise ValueError(\"max_attempts must be positive\")\n"
        "    for attempt in range(max_attempts):\n"
        "        try:\n"
        "            return operation()\n"
        "        except Exception:\n"
        "            if attempt + 1 == max_attempts:\n"
        "                raise",
        "calls = []\n"
        "def flaky():\n    calls.append(1)\n    if len(calls) < 3: raise RuntimeError('transient')\n    return 'ok'\n"
        "assert retry_call(flaky, 3) == 'ok' and len(calls) == 3\n"
        "calls.clear()\n"
        "try:\n    retry_call(lambda: (_ for _ in ()).throw(RuntimeError('x')), 2)\n    raise AssertionError('not raised')\nexcept RuntimeError:\n    pass\n",
    ),
    Case(
        "json_aggregation",
        "Implement a Python function aggregate_orders that parses a JSON order array and returns totals by customer.",
        "Create Python code to summarize JSON orders into per-customer totals.",
        "def aggregate_orders(payload):\n"
        "    import json\n"
        "    orders = json.loads(payload) if isinstance(payload, str) else payload\n"
        "    totals = {}\n"
        "    for order in orders:\n"
        "        customer = str(order[\"customer\"])\n"
        "        totals[customer] = totals.get(customer, 0) + order[\"amount\"]\n"
        "    return totals",
        "import json\n"
        "orders = [{'customer':'a','amount':4},{'customer':'b','amount':3},{'customer':'a','amount':6}]\n"
        "assert aggregate_orders(json.dumps(orders)) == {'a':10,'b':3}\n"
        "assert aggregate_orders([]) == {}\n",
    ),
    Case(
        "secret_redaction",
        "Implement a Python function redact_secrets that recursively redacts password, token, and api_key values.",
        "Build Python code to recursively mask secrets in nested dictionaries and lists.",
        "def redact_secrets(value):\n"
        "    secret_keys = {\"password\", \"token\", \"api_key\"}\n"
        "    if isinstance(value, dict):\n"
        "        return {key: (\"[REDACTED]\" if str(key).lower() in secret_keys else redact_secrets(item)) for key, item in value.items()}\n"
        "    if isinstance(value, list):\n"
        "        return [redact_secrets(item) for item in value]\n"
        "    return value",
        "original = {'user':'a','password':'p','nested':[{'TOKEN':'t','ok':1}]}\n"
        "result = redact_secrets(original)\n"
        "assert result == {'user':'a','password':'[REDACTED]','nested':[{'TOKEN':'[REDACTED]','ok':1}]}\n"
        "assert original['password'] == 'p'\n",
    ),
    Case(
        "batching",
        "Implement a Python function make_batches that chunks items into batches of a positive size.",
        "Write a Python batching function that splits records into fixed-size chunks and validates the size.",
        "def make_batches(items, size):\n"
        "    if size < 1:\n"
        "        raise ValueError(\"size must be positive\")\n"
        "    return [items[index:index + size] for index in range(0, len(items), size)]",
        "assert make_batches([1,2,3,4,5], 2) == [[1,2],[3,4],[5]]\n"
        "assert make_batches([], 3) == []\n"
        "try:\n    make_batches([1], 0)\n    raise AssertionError('zero accepted')\nexcept ValueError:\n    pass\n",
    ),
]

OOV = [
    "Implement a Python distributed consensus protocol with Byzantine fault tolerance.",
    "Write Python code that migrates an unknown production database schema without downtime.",
    "Create Python code for a proprietary payment provider whose API has not been specified.",
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


def executes(code: str, assertions: str) -> tuple[bool, str]:
    if not code:
        return False, "empty"
    with tempfile.TemporaryDirectory(prefix="wv-enterprise-") as raw:
        path = Path(raw) / "evaluate.py"
        path.write_text(code + "\n" + assertions + "\nprint('PASS')\n", encoding="utf-8")
        try:
            run = subprocess.run([sys.executable, "-I", str(path)], capture_output=True,
                                 text=True, timeout=8)
        except subprocess.TimeoutExpired:
            return False, "timeout"
        return run.returncode == 0 and "PASS" in run.stdout, (run.stderr or run.stdout)[-500:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:18600")
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("runtime/benchmarks/enterprise.json"))
    args = parser.parse_args()
    if not args.no_train:
        train(args.endpoint, args.repeats)

    results = []
    for case in CASES:
        for kind, prompt in (("trained", case.prompt), ("paraphrase", case.paraphrase)):
            result = request(args.endpoint, "/brain/chat", {"text": prompt})
            reply = str(result.get("reply") or "")
            passed, detail = executes(reply, case.assertions)
            results.append({"name": case.name, "kind": kind, "nonempty": bool(reply),
                            "exact": reply == case.response, "executes": passed,
                            "detail": "" if passed else detail})
    oov = []
    for prompt in OOV:
        result = request(args.endpoint, "/brain/chat", {"text": prompt})
        honest = not result.get("reply") and bool((result.get("grounding") or {}).get("outside_grounding"))
        oov.append({"prompt": prompt, "honest": honest, "reply": result.get("reply")})
    summary = {
        kind: {"executes": sum(row["executes"] for row in results if row["kind"] == kind),
               "total": len(CASES)} for kind in ("trained", "paraphrase")
    }
    summary["oov_honesty"] = {"passed": sum(row["honest"] for row in oov), "total": len(oov)}
    report = {"summary": summary, "results": results, "oov": oov}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if all(row["executes"] for row in results) and all(row["honest"] for row in oov) else 1


if __name__ == "__main__":
    raise SystemExit(main())
