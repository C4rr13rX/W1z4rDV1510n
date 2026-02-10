from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

from .base import Provider


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _read_json(resp) -> Any:
    raw = resp.read()
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


class RestProvider(Provider):
    """
    Minimal REST wrapper for providers where users can supply:
    - base_url
    - auth header name/prefix
    - token env var name

    This gives users immediate "firepower" (they can hit provider APIs directly via /quantum/experiment),
    even if higher-level workflows aren't implemented yet.
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        priority: int = 50,
        token_env: Optional[str] = None,
        auth_header: str = "Authorization",
        auth_prefix: str = "",
        cfg: Optional[dict] = None,
    ):
        super().__init__(name=name, priority=priority, cfg=cfg)
        self.base_url = base_url.rstrip("/")
        self.token_env = token_env
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix

    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError(f"{self.name}: branch_scoring not implemented; enable dwave/braket annealer strategy or use local fallback")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        t = str(task_obj.get("type", "")).strip()
        if t != "raw_rest":
            raise RuntimeError(f"{self.name}: unsupported experiment type {t!r}; use type='raw_rest'")
        method = str(task_obj.get("method", "GET")).upper()
        path = str(task_obj.get("path", "")).strip()
        if not path.startswith("/"):
            raise RuntimeError(f"{self.name}: raw_rest requires absolute path starting with '/'")
        query = task_obj.get("query")
        body = task_obj.get("json")
        headers = dict(task_obj.get("headers") or {})
        return self._request(method, path, query=query, json_body=body, headers=headers, timeout_secs=timeout_secs)

    def _auth_header_value(self) -> Optional[Tuple[str, str]]:
        if not self.token_env:
            return None
        tok = os.environ.get(self.token_env, "").strip()
        if not tok:
            return None
        return (self.auth_header, f"{self.auth_prefix}{tok}")

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Any = None,
        json_body: Any = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_secs: int = 60,
    ) -> Dict[str, Any]:
        url = self.base_url + path
        if query is not None:
            if not isinstance(query, dict):
                raise RuntimeError("query must be an object")
            url = url + "?" + urllib.parse.urlencode({k: str(v) for k, v in query.items()})
        data = None
        req_headers = {"Accept": "application/json"}
        if headers:
            for k, v in headers.items():
                req_headers[str(k)] = str(v)
        if json_body is not None:
            data = _json_bytes(json_body)
            req_headers["Content-Type"] = "application/json"
        auth = self._auth_header_value()
        if auth:
            req_headers[auth[0]] = auth[1]

        req = urllib.request.Request(url=url, data=data, method=method, headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=max(int(timeout_secs), 1)) as resp:
                status = getattr(resp, "status", 200)
                out = _read_json(resp)
                return {"status": status, "url": url, "json": out}
        except urllib.error.HTTPError as e:
            try:
                out = json.loads(e.read().decode("utf-8"))
            except Exception:
                out = {"error": str(e)}
            return {"status": int(getattr(e, "code", 0) or 0), "url": url, "json": out}

