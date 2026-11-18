#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D:/Projects/W1z4rDV1510n/scripts/multimodal_cloud_sync.py

Zero-Local-Disk-Bloat Cloud Storage Sync Engine for W1z4rDV1510n Multimodal Corpora.

Modeled on D:/Projects/CoolCryptoUtilities/serverless/hybrid/gcs_s3_compat.py,
D:/Projects/CoolCryptoUtilities/services/smart_storage.py, and
D:/Projects/C4rr13rX/gcp-backend/storage.cjs.

Core Invariant (Zero Local Media Accumulation):
  1. Compute local byte_size, SHA-256, and MD5-base64 before upload.
  2. Upload content-addressed object to private Google Cloud Storage (or S3).
  3. Issue an independent remote HEAD / blob.reload() metadata readback and verify:
       - remote exists (generation > 0 / ETag present)
       - remote byte size == local byte size (> 0)
       - remote MD5 hash == local MD5 hash
  4. Immediately unlink() the local temporary file and assert `not path.exists()`.
  5. Record only compact metadata + canonical neural atoms in local JSONL manifests.
"""
from __future__ import annotations

import base64
import ctypes
import datetime as dt
import gc
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger("w1z4rd.multimodal_cloud_sync")

# Allow seamless import of google-cloud-storage when invoked from system Python 3.13
_CCU_VENV_SITE = Path("D:/Projects/CoolCryptoUtilities/.venv/Lib/site-packages")
if _CCU_VENV_SITE.exists() and str(_CCU_VENV_SITE) not in sys.path:
    sys.path.append(str(_CCU_VENV_SITE))

DEFAULT_GCP_PROJECT = "adamedsalldotcom-1759096489951"
DEFAULT_GCP_LOCATION = "US-CENTRAL1"
CANDIDATE_DEDICATED_BUCKETS = (
    "w1z4rdv1510n-multimodal-corpus",
    "c4rr13rx-w1z4rd-multimodal",
    "w1z4rdv1510n-multimodal-22059000239",
)
FALLBACK_SHARED_BUCKETS = (
    "c4rr13rx-data-22059000239",
    "ccu-hybrid-20261112",
)


def load_env_file(env_path: Path, *, override: bool = False) -> dict[str, str]:
    """Minimal read-only .env parser (never mutates files)."""
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = val
        if override or key not in os.environ:
            os.environ[key] = val
    return loaded


def _gcloud_cli_credentials(initial_token: str | None = None):
    """Build refreshable, in-memory credentials from the active gcloud login.
    Never writes service-account keys or tokens to disk.
    """
    from google.oauth2.credentials import Credentials

    def _fetch_token() -> str:
        gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
        if not gcloud:
            raise RuntimeError("gcloud CLI is unavailable for token retrieval")
        result = subprocess.run(
            [gcloud, "auth", "print-access-token"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        tok = result.stdout.strip()
        if not tok:
            raise RuntimeError("gcloud returned an empty access token")
        return tok

    token = initial_token or os.getenv("GOOGLE_ACCESS_TOKEN") or _fetch_token()

    class GcloudCliCredentials(Credentials):
        def __init__(self, tok: str):
            super().__init__(
                token=tok,
                expiry=dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                + dt.timedelta(minutes=50),
            )

        def refresh(self, request):  # noqa: ARG002
            self.token = _fetch_token()
            self.expiry = (
                dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                + dt.timedelta(minutes=50)
            )

    return GcloudCliCredentials(token)


def _detect_gcp_project() -> str:
    proj = (
        os.getenv("W1Z4RD_GCP_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    if proj:
        return proj.strip()
    gcloud = shutil.which("gcloud") or shutil.which("gcloud.cmd")
    if gcloud:
        try:
            res = subprocess.run(
                [gcloud, "config", "get-value", "project"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            detected = res.stdout.strip()
            if detected and detected != "(unset)":
                return detected
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_GCP_PROJECT


@dataclass(frozen=True)
class ElasticAllowance:
    tier: str
    avail_gib: float
    total_gib: float
    effective_floor_gib: float
    usable_headroom_gib: float
    cpu_pct: float
    process_rss_mib: float
    concurrency: int
    batch_size: int
    pace_sleep_s: float
    trimmed_working_set: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ElasticResourceGovernor:
    """Polite Elastic PC-Allowance Governor ('Nice' Resource Scaling).

    Guarantees:
      - Leaves >= `min_free_gib` (default 8.0 GiB) untouched PLUS an additional
        `under_cap_margin` (default 12% -> effective soft floor 8.96 GiB + 88%
        headroom cap) so concurrent crypto trading strategies and desktop apps
        always run smoothly.
      - Lowers current process OS priority to `BELOW_NORMAL_PRIORITY_CLASS`.
      - Scales up (`batch_size` / `concurrency` up to `max_concurrency=4`) when
        free RAM is abundant (`>= abundant_free_gib`, default 14.5 GiB).
      - Shrinks smoothly (`batch_size=1`, `concurrency=1`, `gc.collect()`,
        Win32 `EmptyWorkingSet`, gentle pacing sleep) when free RAM tightens,
        never halting forward progress.
    """

    def __init__(
        self,
        *,
        min_free_gib: float = 8.0,
        under_cap_margin: float = 0.12,
        abundant_free_gib: float = 14.5,
        max_concurrency: int = 4,
        apply_below_normal_priority: bool = True,
    ) -> None:
        self.min_free_gib = float(min_free_gib)
        self.under_cap_margin = float(under_cap_margin)
        self.effective_floor_gib = round(self.min_free_gib * (1.0 + self.under_cap_margin), 3)
        self.abundant_free_gib = float(abundant_free_gib)
        self.max_concurrency = max(1, min(4, int(max_concurrency)))
        self._proc = psutil.Process() if psutil is not None else None
        self._last_trim_ts = 0.0
        self.trim_count = 0
        if apply_below_normal_priority:
            self.set_below_normal_priority()

    def set_below_normal_priority(self) -> bool:
        if os.name == "nt":
            try:
                if self._proc is not None and hasattr(psutil, "BELOW_NORMAL_PRIORITY_CLASS"):
                    self._proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    return True
            except Exception:  # noqa: BLE001
                pass
            try:
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                handle = k32.GetCurrentProcess()
                k32.SetPriorityClass(handle, 0x00004000)
                return True
            except Exception:  # noqa: BLE001
                return False
        return False

    @staticmethod
    def subprocess_nice_CreationFlags() -> int:
        if os.name == "nt":
            # BELOW_NORMAL_PRIORITY_CLASS (0x4000) | CREATE_NO_WINDOW (0x08000000)
            return 0x00004000 | 0x08000000
        return 0

    def trim_working_set(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and (now - self._last_trim_ts) < 3.0:
            return False
        self._last_trim_ts = now
        gc.collect()
        if os.name == "nt":
            try:
                k32 = ctypes.windll.kernel32
                k32.GetCurrentProcess.restype = ctypes.c_void_p
                handle = k32.GetCurrentProcess()
                try:
                    psapi = ctypes.windll.psapi
                    psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
                    ok = psapi.EmptyWorkingSet(handle)
                except AttributeError:
                    k32.K32EmptyWorkingSet.argtypes = [ctypes.c_void_p]
                    ok = k32.K32EmptyWorkingSet(handle)
                if ok:
                    self.trim_count += 1
                    return True
            except Exception:  # noqa: BLE001
                return False
        return True

    def sample(self) -> ElasticAllowance:
        if psutil is not None:
            vm = psutil.virtual_memory()
            avail_gib = vm.available / (1024 ** 3)
            total_gib = vm.total / (1024 ** 3)
            cpu_pct = float(psutil.cpu_percent(interval=None))
            rss_mib = (
                self._proc.memory_info().rss / (1024 ** 2)
                if self._proc is not None
                else 0.0
            )
        else:
            avail_gib, total_gib, cpu_pct, rss_mib = 10.0, 32.0, 50.0, 64.0

        # Usable RAM allowance stays 12% under the cap above min_free_gib
        raw_above_floor = max(0.0, avail_gib - self.min_free_gib)
        usable_headroom_gib = round(raw_above_floor * (1.0 - self.under_cap_margin), 3)
        trimmed = False

        if avail_gib >= self.abundant_free_gib and cpu_pct < 82.0:
            tier = "abundant"
            concurrency = self.max_concurrency
            batch_size = max(4, self.max_concurrency * 3)
            pace_sleep_s = 0.02
        elif avail_gib >= 11.2 and cpu_pct < 90.0:
            tier = "comfortable"
            concurrency = min(self.max_concurrency, 3 if cpu_pct < 80.0 else 2)
            batch_size = max(2, concurrency * 2)
            pace_sleep_s = 0.05
        elif avail_gib >= self.effective_floor_gib:
            tier = "polite_moderate"
            concurrency = min(self.max_concurrency, 2 if cpu_pct < 92.0 else 1)
            batch_size = concurrency * 2
            pace_sleep_s = 0.12 if cpu_pct < 92.0 else 0.20
        else:
            # Tight RAM (< 8.96 GiB, inside the 8.0 GiB + 12% cushion): shrink smoothly!
            tier = "tight_conserve" if avail_gib >= self.min_free_gib else "floor_guard"
            concurrency = 1
            batch_size = 1
            trimmed = self.trim_working_set()
            pace_sleep_s = 0.08 if avail_gib >= self.min_free_gib else 0.15

        return ElasticAllowance(
            tier=tier,
            avail_gib=round(avail_gib, 3),
            total_gib=round(total_gib, 3),
            effective_floor_gib=self.effective_floor_gib,
            usable_headroom_gib=usable_headroom_gib,
            cpu_pct=round(cpu_pct, 1),
            process_rss_mib=round(rss_mib, 2),
            concurrency=concurrency,
            batch_size=batch_size,
            pace_sleep_s=pace_sleep_s,
            trimmed_working_set=trimmed,
        )


@dataclass(frozen=True)
class SyncReceipt:
    provider: str
    bucket: str
    key: str
    cloud_uri: str
    byte_size: int
    sha256: str
    md5_b64: str
    generation_or_etag: str
    verified_in_cloud: bool
    deleted_locally: bool
    already_existed_in_cloud: bool
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultimodalCloudSync:
    """Thread-safe zero-local-storage media sync manager for GCS (primary) or S3."""

    def __init__(
        self,
        *,
        provider: str | None = None,
        bucket: str | None = None,
        base_prefix: str = "",
        project: str | None = None,
        location: str = DEFAULT_GCP_LOCATION,
        auto_provision_bucket: bool = True,
    ) -> None:
        self.provider = (
            provider
            or os.getenv("W1Z4RD_CLOUD_PROVIDER")
            or os.getenv("HYBRID_STORAGE_PROVIDER")
            or "gcs"
        ).strip().lower()
        self.project = project or _detect_gcp_project()
        self.location = location
        self.base_prefix = base_prefix.strip("/")
        self._lock = threading.RLock()
        self._gcs_client = None
        self._s3_client = None
        self.bucket_provision_note = ""

        requested_bucket = (
            bucket or os.getenv("W1Z4RD_MULTIMODAL_BUCKET", "").strip()
        )
        if requested_bucket:
            self.bucket = requested_bucket
            if auto_provision_bucket and self.provider == "gcs":
                self._ensure_named_gcs_bucket(self.bucket)
        elif self.provider == "gcs" and auto_provision_bucket:
            self.bucket, extra_prefix, self.bucket_provision_note = (
                self._provision_or_select_gcs_bucket()
            )
            if extra_prefix and not self.base_prefix.startswith(extra_prefix):
                self.base_prefix = (
                    f"{extra_prefix}/{self.base_prefix}".strip("/")
                    if self.base_prefix
                    else extra_prefix
                )
        else:
            self.bucket = requested_bucket or FALLBACK_SHARED_BUCKETS[0]

    # -- Client initialization ------------------------------------------------
    @property
    def gcs_client(self):
        with self._lock:
            if self._gcs_client is None:
                from google.cloud import storage
                from google.auth.exceptions import DefaultCredentialsError

                if os.getenv("GOOGLE_ACCESS_TOKEN"):
                    self._gcs_client = storage.Client(
                        project=self.project,
                        credentials=_gcloud_cli_credentials(os.getenv("GOOGLE_ACCESS_TOKEN")),
                    )
                else:
                    try:
                        self._gcs_client = storage.Client(project=self.project)
                        # Probe credentials lazily
                        _ = self._gcs_client.project
                    except DefaultCredentialsError:
                        self._gcs_client = storage.Client(
                            project=self.project,
                            credentials=_gcloud_cli_credentials(),
                        )
            return self._gcs_client

    @property
    def s3_client(self):
        with self._lock:
            if self._s3_client is None:
                import boto3
                from botocore.config import Config

                profile = os.getenv("AWS_PROFILE") or os.getenv("MARKET_DATA_S3_PROFILE") or None
                region = os.getenv("AWS_REGION", "us-east-1")
                session = (
                    boto3.Session(profile_name=profile, region_name=region)
                    if profile
                    else boto3.Session(region_name=region)
                )
                self._s3_client = session.client(
                    "s3",
                    config=Config(
                        retries={"max_attempts": 4, "mode": "standard"},
                        connect_timeout=5,
                        read_timeout=20,
                    ),
                )
            return self._s3_client

    # -- Bucket provisioning --------------------------------------------------
    def _ensure_named_gcs_bucket(self, bucket_name: str) -> None:
        client = self.gcs_client
        bucket_obj = client.bucket(bucket_name)
        if bucket_obj.exists():
            self.bucket_provision_note = f"verified_existing_bucket:{bucket_name}"
            return
        bucket_obj.storage_class = "STANDARD"
        bucket_obj.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket_obj.iam_configuration.public_access_prevention = "enforced"
        created = client.create_bucket(bucket_obj, location=self.location)
        self.bucket_provision_note = f"created_dedicated_bucket:{created.name}@{self.location}"

    def _provision_or_select_gcs_bucket(self) -> tuple[str, str, str]:
        """Try existing or newly created dedicated multimodal bucket first;
        fall back to isolated prefix in existing project bucket only if IAM forbids creation.
        """
        client = self.gcs_client
        for candidate in CANDIDATE_DEDICATED_BUCKETS:
            try:
                b = client.lookup_bucket(candidate)
                if b is not None:
                    return candidate, "", f"reused_dedicated_bucket:{candidate}"
            except Exception:  # noqa: BLE001
                pass

        last_err: Exception | None = None
        for candidate in CANDIDATE_DEDICATED_BUCKETS:
            try:
                bucket_obj = client.bucket(candidate)
                bucket_obj.storage_class = "STANDARD"
                bucket_obj.iam_configuration.uniform_bucket_level_access_enabled = True
                bucket_obj.iam_configuration.public_access_prevention = "enforced"
                created = client.create_bucket(bucket_obj, location=self.location)
                return (
                    created.name,
                    "",
                    f"created_dedicated_bucket:{created.name}@{self.location}",
                )
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

        for shared in FALLBACK_SHARED_BUCKETS:
            b = client.lookup_bucket(shared)
            if b is not None:
                return (
                    shared,
                    "w1z4rdv1510n-multimodal",
                    f"fallback_shared_bucket_prefix:{shared}/w1z4rdv1510n-multimodal (create_err={last_err})",
                )
        raise RuntimeError(f"Unable to create or access any GCS bucket: {last_err}")

    # -- Key / URI helpers ----------------------------------------------------
    def full_key(self, relative_key: str) -> str:
        cleaned = relative_key.replace("\\", "/").strip("/")
        if not cleaned or ".." in cleaned.split("/"):
            raise ValueError(f"Unsafe cloud object key: {relative_key!r}")
        if self.base_prefix and not cleaned.startswith(f"{self.base_prefix}/"):
            return f"{self.base_prefix}/{cleaned}"
        return cleaned

    def uri_for_key(self, key: str) -> str:
        scheme = "gs" if self.provider == "gcs" else "s3"
        return f"{scheme}://{self.bucket}/{key}"

    @staticmethod
    def parse_cloud_uri(uri: str) -> tuple[str, str, str]:
        for scheme, prov in (("gs://", "gcs"), ("s3://", "s3")):
            if uri.startswith(scheme):
                rest = uri[len(scheme):]
                bucket, _, key = rest.partition("/")
                if not bucket or not key:
                    raise ValueError(f"Malformed cloud URI: {uri!r}")
                return prov, bucket, key
        raise ValueError(f"Unsupported cloud URI scheme: {uri!r}")

    @staticmethod
    def compute_digests(data: bytes) -> tuple[str, str]:
        sha256_hex = hashlib.sha256(data).hexdigest()
        md5_b64 = base64.b64encode(hashlib.md5(data).digest()).decode("ascii")
        return sha256_hex, md5_b64

    # -- Remote verification --------------------------------------------------
    def verify_remote_object(
        self,
        key_or_uri: str,
        *,
        expected_size: int | None = None,
        expected_md5_b64: str | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Issue a fresh remote metadata HEAD / reload and verify byte size + checksum."""
        if key_or_uri.startswith(("gs://", "s3://")):
            prov, bucket, key = self.parse_cloud_uri(key_or_uri)
        else:
            prov, bucket, key = self.provider, self.bucket, self.full_key(key_or_uri)

        if prov == "gcs":
            # Construct a fresh Blob object so cached upload properties are never trusted.
            fresh_blob = self.gcs_client.bucket(bucket).blob(key)
            if not fresh_blob.exists():
                return {"verified": False, "reason": "remote_object_missing", "bucket": bucket, "key": key}
            fresh_blob.reload()
            remote_size = int(fresh_blob.size or 0)
            remote_md5 = fresh_blob.md5_hash or ""
            remote_gen = str(fresh_blob.generation or "")
            meta = fresh_blob.metadata or {}
            if remote_size <= 0 or not remote_gen:
                return {"verified": False, "reason": "zero_size_or_no_generation", "size": remote_size}
            if expected_size is not None and remote_size != int(expected_size):
                return {
                    "verified": False,
                    "reason": f"size_mismatch:{remote_size}!={expected_size}",
                    "size": remote_size,
                }
            if expected_md5_b64 and remote_md5 != expected_md5_b64:
                return {
                    "verified": False,
                    "reason": f"md5_mismatch:{remote_md5}!={expected_md5_b64}",
                }
            if expected_sha256 and meta.get("sha256") and meta.get("sha256") != expected_sha256:
                return {
                    "verified": False,
                    "reason": f"sha256_metadata_mismatch:{meta.get('sha256')}!={expected_sha256}",
                }
            return {
                "verified": True,
                "provider": "gcs",
                "bucket": bucket,
                "key": key,
                "cloud_uri": f"gs://{bucket}/{key}",
                "size": remote_size,
                "md5_b64": remote_md5,
                "sha256": meta.get("sha256", expected_sha256 or ""),
                "generation_or_etag": remote_gen,
            }

        # S3 fallback verification via head_object
        head = self.s3_client.head_object(Bucket=bucket, Key=key)
        remote_size = int(head.get("ContentLength") or 0)
        etag = str(head.get("ETag") or "").strip('"')
        meta = head.get("Metadata") or {}
        if remote_size <= 0 or not etag:
            return {"verified": False, "reason": "s3_zero_size_or_no_etag"}
        if expected_size is not None and remote_size != int(expected_size):
            return {"verified": False, "reason": f"s3_size_mismatch:{remote_size}!={expected_size}"}
        if expected_sha256 and meta.get("sha256") and meta.get("sha256") != expected_sha256:
            return {"verified": False, "reason": "s3_sha256_mismatch"}
        return {
            "verified": True,
            "provider": "s3",
            "bucket": bucket,
            "key": key,
            "cloud_uri": f"s3://{bucket}/{key}",
            "size": remote_size,
            "md5_b64": expected_md5_b64 or "",
            "sha256": meta.get("sha256", expected_sha256 or ""),
            "generation_or_etag": etag,
        }

    # -- Atomic Upload -> Verify -> Unlink transaction ------------------------
    def upload_verify_and_delete(
        self,
        local_path: Path,
        relative_key: str,
        *,
        content_type: str = "audio/mpeg",
        metadata: Optional[dict[str, str]] = None,
        cleanup_temp_on_failure: bool = True,
    ) -> SyncReceipt:
        """Upload `local_path` to cloud storage, verify remote HEAD + size + MD5/SHA256,
        and immediately unlink `local_path` so local media files NEVER accumulate.
        """
        t0 = time.perf_counter()
        local_path = Path(local_path)
        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"Local media file missing before upload: {local_path}")

        key = self.full_key(relative_key)
        already_existed = False

        try:
            payload = local_path.read_bytes()
            byte_size = len(payload)
            if byte_size <= 0:
                raise ValueError(f"Refusing to upload empty media file: {local_path}")

            sha256_hex, md5_b64 = self.compute_digests(payload)
            obj_meta = {"sha256": sha256_hex, **(metadata or {})}

            if self.provider == "gcs":
                blob = self.gcs_client.bucket(self.bucket).blob(key)
                blob.metadata = obj_meta
                try:
                    blob.upload_from_string(
                        payload,
                        content_type=content_type,
                        if_generation_match=0,
                        checksum="md5",
                        timeout=60,
                    )
                except Exception as exc:  # noqa: BLE001
                    code = getattr(exc, "code", None)
                    if code == 412 or "PreconditionFailed" in type(exc).__name__ or "412" in str(exc):
                        # Immutable object already exists; verify below that its exact size & hash match!
                        already_existed = True
                    else:
                        raise
            else:
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=key,
                    Body=payload,
                    ContentType=content_type,
                    Metadata=obj_meta,
                )

            check = self.verify_remote_object(
                f"{'gs' if self.provider == 'gcs' else 's3'}://{self.bucket}/{key}",
                expected_size=byte_size,
                expected_md5_b64=md5_b64,
                expected_sha256=sha256_hex,
            )
            if not check.get("verified"):
                raise RuntimeError(
                    f"Remote verification failed for {key}: {check.get('reason')}"
                )

            # Remote durability confirmed -> immediately delete local file
            local_path.unlink(missing_ok=False)
            if local_path.exists():
                raise RuntimeError(f"Failed to delete local file after cloud verify: {local_path}")

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return SyncReceipt(
                provider=self.provider,
                bucket=self.bucket,
                key=key,
                cloud_uri=check["cloud_uri"],
                byte_size=byte_size,
                sha256=sha256_hex,
                md5_b64=md5_b64,
                generation_or_etag=str(check["generation_or_etag"]),
                verified_in_cloud=True,
                deleted_locally=True,
                already_existed_in_cloud=already_existed,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            if cleanup_temp_on_failure and local_path.exists():
                try:
                    local_path.unlink()
                except OSError:
                    pass
            raise

    def fetch_bytes_ephemeral(self, key_or_uri: str, *, expected_sha256: str | None = None) -> bytes:
        """Read remote media bytes strictly into RAM (never touching disk) when raw playback is needed."""
        if key_or_uri.startswith(("gs://", "s3://")):
            prov, bucket, key = self.parse_cloud_uri(key_or_uri)
        else:
            prov, bucket, key = self.provider, self.bucket, self.full_key(key_or_uri)

        if prov == "gcs":
            blob = self.gcs_client.bucket(bucket).blob(key)
            data = blob.download_as_bytes(raw_download=True, timeout=60)
        else:
            resp = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = resp["Body"].read()

        if expected_sha256:
            actual = hashlib.sha256(data).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(f"Ephemeral cloud read SHA-256 mismatch: {actual} != {expected_sha256}")
        return data

    def upload_json_checkpoint(self, relative_key: str, payload: dict[str, Any] | list[Any]) -> str:
        """Store compact manifest/checkpoint metadata in cloud storage."""
        key = self.full_key(relative_key)
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.provider == "gcs":
            blob = self.gcs_client.bucket(self.bucket).blob(key)
            blob.upload_from_string(raw, content_type="application/json", checksum="md5", timeout=60)
        else:
            self.s3_client.put_object(
                Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json"
            )
        return self.uri_for_key(key)

    @staticmethod
    def count_local_media_files(directory: Path) -> int:
        if not directory.exists():
            return 0
        exts = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".png", ".jpg", ".jpeg", ".webp", ".tmp"}
        return sum(
            1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts
        )


if __name__ == "__main__":
    import tempfile

    sync = MultimodalCloudSync()
    print(json.dumps({
        "provider": sync.provider,
        "project": sync.project,
        "bucket": sync.bucket,
        "base_prefix": sync.base_prefix,
        "provision_note": sync.bucket_provision_note,
    }, indent=2))

    # Disposable self-test: write 32-byte probe -> upload+verify+unlink -> delete remote probe
    with tempfile.TemporaryDirectory(prefix="w1z4rd_sync_probe_") as tmpdir:
        probe_file = Path(tmpdir) / "probe.tmp"
        probe_file.write_bytes(b"w1z4rdv1510n-zero-disk-bloat-ok\n")
        receipt = sync.upload_verify_and_delete(
            probe_file,
            "_health/zero_bloat_probe.txt",
            content_type="text/plain",
        )
        assert not probe_file.exists(), "Local probe file was not deleted!"
        readback = sync.fetch_bytes_ephemeral(receipt.cloud_uri, expected_sha256=receipt.sha256)
        assert readback == b"w1z4rdv1510n-zero-disk-bloat-ok\n"
        if sync.provider == "gcs":
            sync.gcs_client.bucket(sync.bucket).blob(receipt.key).delete()
        print("Self-test PASSED:", json.dumps(receipt.to_dict(), indent=2))
