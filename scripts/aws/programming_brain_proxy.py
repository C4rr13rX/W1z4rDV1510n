#!/usr/bin/env python3
"""Loopback-only HTTP proxy to the private AWS programming brain via SSM.

The AWS service remains bound to 127.0.0.1. Requests are relayed with the
existing, least-privilege SendCommand channel; request content is never logged.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from scripts.aws.bootstrap_training_host import send_and_wait
except ModuleNotFoundError:
    from bootstrap_training_host import send_and_wait


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE = "i-0d7a6deeb0ead2dfc"
ALLOWED_PATHS = {"/health", "/brain/chat", "/chat"}
MAX_REQUEST_BYTES = 12 * 1024


def relay_command(path: str, body: bytes) -> str:
    encoded_path = base64.b64encode(path.encode("utf-8")).decode("ascii")
    encoded_body = base64.b64encode(body).decode("ascii")
    return f"""python3 - <<'PY'
import base64, json, urllib.error, urllib.request
path = base64.b64decode({encoded_path!r}).decode('utf-8')
body = base64.b64decode({encoded_body!r})
request = urllib.request.Request(
    'http://127.0.0.1:18095' + path,
    data=body if body else None,
    headers={{'Content-Type': 'application/json'}},
    method='POST' if body else 'GET',
)
try:
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        content_type = response.headers.get('Content-Type', 'application/json')
        payload = response.read()
except urllib.error.HTTPError as exc:
    status = exc.code
    content_type = exc.headers.get('Content-Type', 'application/json')
    payload = exc.read()
print(json.dumps({{
    'status': status,
    'content_type': content_type,
    'body': base64.b64encode(payload).decode('ascii'),
}}, separators=(',', ':')))
PY"""


def decode_relay_output(output: str) -> tuple[int, str, bytes]:
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("AWS programming brain returned no response")
    payload = json.loads(lines[-1])
    status = int(payload["status"])
    content_type = str(payload.get("content_type") or "application/json")
    body = base64.b64decode(payload.get("body") or "", validate=True)
    return status, content_type, body


class ProgrammingBrainProxy(BaseHTTPRequestHandler):
    server_version = "WizardProgrammingBrainProxy/1"

    def log_message(self, message: str, *args: object) -> None:
        # Log method/status only; never request bodies or query data.
        sys.stderr.write("[programming-brain-proxy] " + message % args + "\n")

    def _reject(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _relay(self, body: bytes) -> None:
        if self.path not in ALLOWED_PATHS:
            self._reject(404, "unsupported route")
            return
        try:
            invocation = send_and_wait(
                self.server.aws_profile,
                self.server.instance_id,
                [relay_command(self.path, body)],
                180,
                comment="Relay private Wizard programming brain request",
            )
            status, content_type, response_body = decode_relay_output(
                str(invocation.get("StandardOutputContent") or "")
            )
        except Exception as exc:
            self.log_error("relay failed: %s", type(exc).__name__)
            self._reject(502, "private programming brain relay failed")
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def do_GET(self) -> None:  # noqa: N802
        self._relay(b"")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/brain/chat", "/chat"}:
            self._reject(405, "POST is only supported for chat routes")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reject(400, "invalid content length")
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._reject(413, "request body must be between 1 byte and 12 KiB")
            return
        self._relay(self.rfile.read(length))


def serve(profile: str, instance_id: str, port: int) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), ProgrammingBrainProxy)
    server.aws_profile = profile
    server.instance_id = instance_id
    print(f"Programming brain proxy listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="FountainServer")
    parser.add_argument("--instance-id", default=DEFAULT_INSTANCE)
    parser.add_argument("--port", type=int, default=18096)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    serve(args.profile, args.instance_id, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
