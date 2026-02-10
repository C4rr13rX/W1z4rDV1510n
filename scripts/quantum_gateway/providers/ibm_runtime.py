from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .rest_base import RestProvider


class IbmRuntimeProvider(RestProvider):
    def __init__(self, *args, service_crn_env: str = "IBM_QUANTUM_SERVICE_CRN", **kwargs):
        super().__init__(*args, **kwargs)
        self.service_crn_env = service_crn_env

    def _auth_header_value(self) -> Optional[Tuple[str, str]]:
        # IBM docs use Authorization: Bearer <token>
        return super()._auth_header_value()

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
        headers = dict(headers or {})
        crn = os.environ.get(self.service_crn_env, "").strip()
        if crn:
            headers.setdefault("Service-CRN", crn)
        return super()._request(
            method,
            path,
            query=query,
            json_body=json_body,
            headers=headers,
            timeout_secs=timeout_secs,
        )


def build_provider(cfg: dict):
    # IBM Quantum Platform (Qiskit Runtime REST).
    # Base URL: https://quantum.cloud.ibm.com/api/v1/
    # Auth: Authorization: Bearer <token>, plus Service-CRN header for many flows.
    return IbmRuntimeProvider(
        name="ibm_runtime",
        base_url=str(cfg.get("base_url", "https://quantum.cloud.ibm.com/api/v1")),
        priority=int(cfg.get("priority", 50)),
        token_env=str(cfg.get("token_env", "IBM_QUANTUM_TOKEN")),
        auth_header=str(cfg.get("auth_header", "Authorization")),
        auth_prefix=str(cfg.get("auth_prefix", "Bearer ")),
        service_crn_env=str(cfg.get("service_crn_env", "IBM_QUANTUM_SERVICE_CRN")),
        cfg=cfg,
    )

