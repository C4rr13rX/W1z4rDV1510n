#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_multimodal_obstacle_course.py
=================================
End-to-end Multimodal, Meta-Stream Pool Erection, Minimal-Neuron Compositional
Synthesis, and 12-Discipline Anti-Forgetting Obstacle Course for W1z4rD V1510n.

Supports THREE non-mocked execution backends:
  1. `local-rust-brain` (default when `w1z4rd_brain_server.exe` is present):
     Spawns an isolated, disposable Rust `w1z4rd_brain_server.exe` subprocess
     configured with the dynamically erected `multisensory_crossmodal.identity.toml`
     at `BELOW_NORMAL_PRIORITY_CLASS`, exercising real Rust `Brain` pools,
     binding indexes, OOV precision gates, and `TransientWorkspace` EEM composition.
  2. `endpoint` (`--endpoint http://127.0.0.1:8090` or `:8095`):
     Connects over HTTP to a running W1z4rD node or brain server, auto-detecting
     `/brain/*` and `/multi_pool/*` + `/sensor/observe*` surfaces.
  3. `simulator` (`--mode simulator`):
     Pure-Python faithful W1z4rD `Fabric` + `Pool` + `BindingIndex` +
     `TransientWorkspace` substrate for zero-binary environments and unit tests.

Polite Elastic PC-Allowance ("Nice" Resource Scaling):
  - Runs at `psutil.BELOW_NORMAL_PRIORITY_CLASS` (including child micro-brains).
  - Preserves >= 8.0 GiB of system RAM untouched (+ 12% under-cap cushion).
  - Expands episode batch chunks when free RAM >= 14.5 GiB and smoothly shrinks
    working buffers + `gc.collect()` when RAM tightens, never stopping execution.

Usage:
  python scripts/run_multimodal_obstacle_course.py
  python scripts/run_multimodal_obstacle_course.py --mode simulator
  python scripts/run_multimodal_obstacle_course.py --endpoint http://127.0.0.1:8090
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psutil

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from curate_multisensory_crossmodal_corpus import (  # noqa: E402
    TWELVE_DISCIPLINES,
    CanonicalMultisensoryCodec,
    MetaStreamPoolErector,
    PoliteElasticGovernor,
    curate_all,
)

# Pool IDs matching `multisensory_crossmodal.identity.toml`
POOL_TEXT = 1
POOL_VISUAL = 2
POOL_AUDIO = 3
POOL_ACTION = 4
POOL_TURN = 5
POOL_SOCIAL = 6
POOL_SENSOR_IMU = 7
POOL_SENSOR_THERMAL = 8
POOL_SIGNAL_RF = 9
POOL_SENSOR_KINEMATICS = 10
POOL_SENSOR_BIOCHEM = 11
POOL_COMPOSITIONAL_ROLE = 12


def b64u_encode(text_or_bytes: str | bytes) -> str:
    raw = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def b64u_decode(encoded: str | None) -> str | None:
    if not encoded:
        return None
    pad = "=" * ((4 - len(encoded) % 4) % 4)
    return base64.urlsafe_b64decode(encoded + pad).decode("utf-8", errors="replace")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_brain_server_executable() -> Path | None:
    candidates = [
        ROOT / "target" / "release" / "w1z4rd_brain_server.exe",
        ROOT / "target" / "release" / "deps" / "w1z4rd_brain_server.exe",
        ROOT / "target" / "debug" / "w1z4rd_brain_server.exe",
        ROOT / "target" / "release" / "w1z4rd_brain_server",
        ROOT / "target" / "debug" / "w1z4rd_brain_server",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


# ============================================================================
# 1. Unified Substrate Adapter (Real Rust HTTP Server / Live Node / Simulator)
# ============================================================================

class BrainHttpBackend:
    """Talks to a real W1z4rD HTTP endpoint (`w1z4rd_brain_server` or `w1z4rd_node`)."""

    def __init__(
        self,
        endpoint: str,
        governor: PoliteElasticGovernor,
        owned_process: subprocess.Popen[Any] | None = None,
        temp_dir: tempfile.TemporaryDirectory[str] | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.governor = governor
        self.owned_process = owned_process
        self.temp_dir = temp_dir
        self.has_multi_pool_api = self._probe_multi_pool()

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> Any:
        url = f"{self.endpoint}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _probe_multi_pool(self) -> bool:
        try:
            res = self._request("GET", "/multi_pool/stats", timeout=3.0)
            return isinstance(res, dict)
        except Exception:
            return False

    def stats(self) -> dict[str, Any]:
        return self._request("GET", "/brain/stats")

    def pretrain_bindings_polite(self, episodes: list[list[tuple[int, str]]]) -> int:
        """Batch-ingest binding episodes using elastic governor batch sizing."""
        accepted = 0
        idx = 0
        while idx < len(episodes):
            snap = self.governor.breathe()
            chunk_len = max(1, min(snap.batch_size, 32))
            chunk = episodes[idx : idx + chunk_len]
            payload = {
                "episodes": [
                    {
                        "frames": [
                            {"pool_id": pid, "frame": b64u_encode(val)}
                            for pid, val in ep
                        ]
                    }
                    for ep in chunk
                ],
                "lock_chunk_size": max(1, min(chunk_len, 12)),
            }
            res = self._request("POST", "/brain/pretrain_bindings", payload, timeout=120.0)
            if not res.get("ok"):
                raise RuntimeError(f"/brain/pretrain_bindings failed: {res}")
            accepted += int(res.get("accepted", len(chunk)))
            idx += chunk_len
        return accepted

    def predict(self, query_pool: int, frame: str, target_pool: int = POOL_ACTION) -> dict[str, Any]:
        res = self._request("POST", "/brain/predict", {
            "query_pool": query_pool,
            "target_pool": target_pool,
            "frame": b64u_encode(frame),
        })
        return {
            "answer": b64u_decode(res.get("answer")),
            "outside_grounding": bool(res.get("outside_grounding", True)),
            "confidence": float(res.get("integrated_confidence", 0.0)),
            "known_atom_count": int(res.get("known_atom_count", 0)),
        }

    def predict_multi(self, streams: list[tuple[int, str]], target_pool: int = POOL_ACTION) -> dict[str, Any]:
        res = self._request("POST", "/brain/predict/multi", {
            "target_pool": target_pool,
            "streams": [
                {"pool_id": pid, "frame": b64u_encode(val)}
                for pid, val in streams
            ],
        })
        ans = b64u_decode(res.get("answer"))
        return {
            "answer": ans,
            "outside_grounding": ans is None,
            "confidence": float(res.get("integrated_confidence", 0.0)),
        }

    def chat(self, text: str) -> dict[str, Any]:
        res = self._request("POST", "/brain/chat", {"text": text})
        reply = str(res.get("reply") or "")
        grounding = res.get("grounding") or {}
        return {
            "reply": reply,
            "outside_grounding": bool(grounding.get("outside_grounding")) or not reply.strip(),
        }

    def logic_consolidate_relation(
        self,
        predicate: str,
        args: list[tuple[str, str]],
        confidence: float = 0.98,
        source: str = "confirmed_sensor_episode",
    ) -> bool:
        res = self._request("POST", "/brain/logic/consolidate", {
            "outcome_confirmed": True,
            "relation": {
                "predicate": predicate,
                "arguments": [{"kind": k, "value": v} for k, v in args],
                "confidence": confidence,
                "provenance": [source],
            },
        })
        return bool(res.get("consolidated"))

    def logic_consolidate_rule(
        self,
        name: str,
        premises: list[tuple[str, list[tuple[str, str, str]]]],
        conclusion: tuple[str, list[tuple[str, str, str]]],
    ) -> bool:
        def encode_arg(mode: str, a: str, b: str) -> dict[str, Any]:
            if mode == "var":
                return {"Variable": {"name": a, "kind": b}}
            return {"Constant": {"kind": a, "value": b}}

        res = self._request("POST", "/brain/logic/consolidate", {
            "outcome_confirmed": True,
            "rule": {
                "name": name,
                "premises": [
                    {
                        "predicate": pred,
                        "arguments": [encode_arg(m, x, y) for m, x, y in pargs],
                    }
                    for pred, pargs in premises
                ],
                "conclusion": {
                    "predicate": conclusion[0],
                    "arguments": [encode_arg(m, x, y) for m, x, y in conclusion[1]],
                },
            },
        })
        return bool(res.get("consolidated"))

    def logic_compose(self, predicate: str | None = None, max_rounds: int = 8) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"max_rounds": max_rounds}
        if predicate is not None:
            payload["predicate"] = predicate
        res = self._request("POST", "/brain/logic/compose", payload)
        return list(res.get("facts") or [])

    def multi_pool_fanout_probe(self) -> dict[str, Any] | None:
        """Exercise `/multi_pool/*` when connected to a full `:8090` node."""
        if not self.has_multi_pool_api:
            return None
        for pool in ("mp_text", "mp_social", "mp_visual", "mp_sensor"):
            self._request("POST", "/multi_pool/register", {"pool_id": pool})
        self._request("POST", "/multi_pool/train_fanout", {
            "src_pool": "mp_text",
            "src": "emergency_rotary_halt",
            "targets": [
                {"tgt_pool": "mp_social", "tgt": "soc:i:halt|soc:t:urgent"},
                {"tgt_pool": "mp_visual", "tgt": "vis:c:red|vis:s:circle"},
                {"tgt_pool": "mp_sensor", "tgt": "sen:imu:2g_braking"},
            ],
            "passes": 3,
            "lr": 1.0,
        })
        ask_res = self._request("POST", "/multi_pool/ask", {
            "src_pool": "mp_text",
            "text": "emergency_rotary_halt",
        })
        return {"multi_pool_ask": ask_res, "multi_pool_stats": self._request("GET", "/multi_pool/stats")}

    def process_rss_mib(self) -> float:
        if self.owned_process is not None and self.owned_process.poll() is None:
            try:
                return round(psutil.Process(self.owned_process.pid).memory_info().rss / (1024.0 ** 2), 2)
            except Exception:
                return 0.0
        return 0.0

    def close(self) -> None:
        if self.owned_process is not None:
            try:
                self.owned_process.kill()
                self.owned_process.wait(timeout=10)
            except Exception:
                pass
        if self.temp_dir is not None:
            try:
                self.temp_dir.cleanup()
            except Exception:
                pass


class LocalSimulatorBackend:
    """Pure-Python faithful W1z4rD Fabric + BindingIndex + TransientWorkspace backend."""

    def __init__(self, governor: PoliteElasticGovernor, pool_ids: list[int]) -> None:
        self.governor = governor
        self.has_multi_pool_api = False
        self.pools: dict[int, set[str]] = {0: set(), **{pid: set() for pid in pool_ids}}
        self.episodes: list[dict[int, str]] = []
        self.relations: list[dict[str, Any]] = []
        self.rules: list[dict[str, Any]] = []
        self.tick = 0

    @staticmethod
    def _atoms(pool_id: int, frame: str) -> set[str]:
        return {f"p{pool_id}:b:{b:02x}" for b in frame.encode("utf-8")} | {
            f"p{pool_id}:tok:{tok}" for tok in frame.split("|") if tok
        }

    def stats(self) -> dict[str, Any]:
        total_atoms = sum(len(v) for k, v in self.pools.items() if k != 0)
        return {
            "tick": self.tick,
            "pool_count": len(self.pools) - 1,
            "total_neurons": total_atoms + len(self.episodes),
            "total_concepts": len(self.episodes),
            "total_binding": len(self.episodes),
            "total_terminals": total_atoms * 2,
        }

    def pretrain_bindings_polite(self, episodes: list[list[tuple[int, str]]]) -> int:
        idx = 0
        while idx < len(episodes):
            snap = self.governor.breathe()
            chunk = episodes[idx : idx + snap.batch_size]
            for ep in chunk:
                ep_map: dict[int, str] = {}
                for pid, frame in ep:
                    self.pools.setdefault(pid, set()).update(self._atoms(pid, frame))
                    ep_map[pid] = frame
                self.episodes.append(ep_map)
                self.tick += 1
            idx += snap.batch_size
        return len(episodes)

    def predict(self, query_pool: int, frame: str, target_pool: int = POOL_ACTION) -> dict[str, Any]:
        q_atoms = self._atoms(query_pool, frame)
        known = q_atoms & self.pools.get(query_pool, set())
        if not q_atoms or not known:
            return {"answer": None, "outside_grounding": True, "confidence": 0.0, "known_atom_count": 0}

        best_score = 0.0
        best_ans: str | None = None
        for ep in self.episodes:
            if query_pool not in ep or target_pool not in ep:
                continue
            if ep[query_pool] == frame:
                return {
                    "answer": ep[target_pool],
                    "outside_grounding": False,
                    "confidence": 1.0,
                    "known_atom_count": len(known),
                }
            cand_atoms = self._atoms(query_pool, ep[query_pool])
            inter = len(q_atoms & cand_atoms)
            precision = inter / max(1, len(q_atoms))
            recall = inter / max(1, len(cand_atoms))
            score = 2 * precision * recall / max(1e-9, precision + recall)
            if precision >= 0.70 and score > best_score:
                best_score = score
                best_ans = ep[target_pool]

        if best_ans is None or best_score < 0.70:
            return {"answer": None, "outside_grounding": True, "confidence": best_score, "known_atom_count": len(known)}
        return {"answer": best_ans, "outside_grounding": False, "confidence": best_score, "known_atom_count": len(known)}

    def predict_multi(self, streams: list[tuple[int, str]], target_pool: int = POOL_ACTION) -> dict[str, Any]:
        best_exact = -1
        best_score = 0.0
        best_ans: str | None = None
        for ep in self.episodes:
            if target_pool not in ep:
                continue
            exact_matches = 0
            pool_scores: list[float] = []
            valid = True
            for pid, frame in streams:
                if pid not in ep:
                    valid = False
                    break
                if ep[pid] == frame:
                    exact_matches += 1
                    pool_scores.append(1.0)
                else:
                    qa = self._atoms(pid, frame)
                    ca = self._atoms(pid, ep[pid])
                    sc = len(qa & ca) / max(1, len(qa | ca))
                    if sc < 0.99:
                        valid = False
                        break
                    pool_scores.append(sc)
            if not valid or not pool_scores:
                continue
            mean_sc = sum(pool_scores) / len(pool_scores)
            if (exact_matches, mean_sc) > (best_exact, best_score):
                best_exact = exact_matches
                best_score = mean_sc
                best_ans = ep[target_pool]
        return {
            "answer": best_ans,
            "outside_grounding": best_ans is None,
            "confidence": round(best_score, 4),
        }

    @staticmethod
    def _motifs3(text: str) -> set[bytes]:
        b = text.lower().encode("utf-8")
        if len(b) < 3:
            return {b} if b else set()
        return {b[i : i + 3] for i in range(len(b) - 2)}

    def chat(self, text: str) -> dict[str, Any]:
        """Mirror `brain_api.rs` `/brain/chat` 3-byte motif gate (`UNLABELED_RECALL_MIN_SCORE = 0.45`)."""
        q_m = self._motifs3(text)
        if not q_m:
            return {"reply": "", "outside_grounding": True}
        best_score = 0.0
        best_ans = ""
        for ep in self.episodes:
            if POOL_TEXT not in ep or POOL_ACTION not in ep:
                continue
            if ep[POOL_TEXT] == text:
                return {"reply": ep[POOL_ACTION], "outside_grounding": False}
            c_m = self._motifs3(ep[POOL_TEXT])
            sc = len(q_m & c_m) / max(1, len(q_m | c_m))
            if sc > best_score:
                best_score = sc
                best_ans = ep[POOL_ACTION]
        if best_score < 0.45:
            return {"reply": "", "outside_grounding": True}
        return {"reply": best_ans, "outside_grounding": False}

    def logic_consolidate_relation(
        self,
        predicate: str,
        args: list[tuple[str, str]],
        confidence: float = 0.98,
        source: str = "confirmed_sensor_episode",
    ) -> bool:
        self.relations.append({
            "predicate": predicate,
            "arguments": [{"kind": k, "value": v} for k, v in args],
            "confidence": confidence,
            "provenance": [source],
        })
        return True

    def logic_consolidate_rule(
        self,
        name: str,
        premises: list[tuple[str, list[tuple[str, str, str]]]],
        conclusion: tuple[str, list[tuple[str, str, str]]],
    ) -> bool:
        self.rules.append({"name": name, "premises": premises, "conclusion": conclusion})
        return True

    def logic_compose(self, predicate: str | None = None, max_rounds: int = 8) -> list[dict[str, Any]]:
        facts = {
            (r["predicate"], tuple((a["kind"], a["value"]) for a in r["arguments"])): dict(r)
            for r in self.relations
        }
        for _ in range(max_rounds):
            before = len(facts)
            snapshot = list(facts.values())
            for rule in self.rules:
                matches = self._join_premises(rule["premises"], snapshot, 0, {}, 1.0, set())
                c_pred, c_args = rule["conclusion"]
                for bindings, conf, prov in matches:
                    inst: list[dict[str, str]] = []
                    ok = True
                    for mode, x, y in c_args:
                        if mode == "const":
                            inst.append({"kind": x, "value": y})
                        elif x in bindings:
                            inst.append({"kind": bindings[x][0], "value": bindings[x][1]})
                        else:
                            ok = False
                            break
                    if not ok:
                        continue
                    key = (c_pred, tuple((a["kind"], a["value"]) for a in inst))
                    if key not in facts:
                        facts[key] = {
                            "predicate": c_pred,
                            "arguments": inst,
                            "confidence": conf,
                            "provenance": sorted(prov | {f"rule:{rule['name']}"}),
                        }
            if len(facts) == before:
                break
        out = list(facts.values())
        if predicate is not None:
            out = [f for f in out if f["predicate"] == predicate]
        return out

    def _join_premises(
        self,
        premises: list[Any],
        facts: list[dict[str, Any]],
        idx: int,
        bindings: dict[str, tuple[str, str]],
        conf: float,
        prov: set[str],
    ) -> list[tuple[dict[str, tuple[str, str]], float, set[str]]]:
        if idx == len(premises):
            return [(bindings, conf, prov)]
        pred, pargs = premises[idx]
        results: list[tuple[dict[str, tuple[str, str]], float, set[str]]] = []
        for f in facts:
            if f["predicate"] != pred or len(f["arguments"]) != len(pargs):
                continue
            nxt = dict(bindings)
            matched = True
            for (mode, x, y), val in zip(pargs, f["arguments"]):
                vk, vv = val["kind"], val["value"]
                if mode == "const":
                    if (x, y) != (vk, vv):
                        matched = False
                        break
                else:
                    if vk != y:
                        matched = False
                        break
                    if x in nxt and nxt[x] != (vk, vv):
                        matched = False
                        break
                    nxt[x] = (vk, vv)
            if matched:
                results.extend(
                    self._join_premises(
                        premises,
                        facts,
                        idx + 1,
                        nxt,
                        min(conf, float(f["confidence"])),
                        prov | set(f.get("provenance", [])),
                    )
                )
        return results

    def multi_pool_fanout_probe(self) -> None:
        return None

    def process_rss_mib(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


def spawn_disposable_rust_brain(
    executable: Path,
    identity_path: Path,
    governor: PoliteElasticGovernor,
) -> BrainHttpBackend:
    """Launch an isolated disposable `w1z4rd_brain_server.exe` at BELOW_NORMAL priority."""
    temp_dir = tempfile.TemporaryDirectory(prefix="w1z4rd_mm_obstacle_")
    data_dir = Path(temp_dir.name) / "brain_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port()

    env = os.environ.copy()
    env.update({
        "W1Z4RDV1510N_DATA_DIR": str(data_dir),
        "W1Z4RD_NODE_BRAIN_DIR": str(data_dir),
        "W1Z4RD_BRAIN_IDENTITY": str(identity_path),
        "W1Z4RD_BRAIN_PORT": str(port),
        "W1Z4RD_BRAIN_BIND": "127.0.0.1",
        # Enforce strict per-pool atom/sequence grounding on /brain/predict/multi
        # so ungrounded sensory noise (even sharing modality prefixes) is cleanly refused.
        "BRAIN_MULTI_MIN_POOL_SCORE": "0.99",
        # Politeness is enforced smoothly by PoliteElasticGovernor without retry-stalling
        # this tiny ~20 MiB disposable micro-brain.
        "W1Z4RD_TIER_MIN_SYS_AVAIL_MB": "0",
    })
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    if os.name == "nt" and hasattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS"):
        flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS

    stdout_log = (Path(temp_dir.name) / "brain.stdout.log").open("w", encoding="utf-8")
    stderr_log = (Path(temp_dir.name) / "brain.stderr.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(executable)],
        cwd=str(ROOT),
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
        creationflags=flags,
    )
    governor.apply_process_niceness(proc.pid)
    endpoint = f"http://127.0.0.1:{port}"
    backend = BrainHttpBackend(endpoint=endpoint, governor=governor, owned_process=proc, temp_dir=temp_dir)

    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stderr_log.flush()
            err_txt = (Path(temp_dir.name) / "brain.stderr.log").read_text(encoding="utf-8", errors="replace")
            backend.close()
            raise RuntimeError(f"Disposable w1z4rd_brain_server exited early ({proc.returncode}): {err_txt[-500:]}")
        try:
            backend.stats()
            return backend
        except Exception:
            time.sleep(0.2)

    backend.close()
    raise TimeoutError(f"Disposable w1z4rd_brain_server did not become ready on {endpoint}")


# ============================================================================
# 2. The Four Obstacle Courses
# ============================================================================

def episode_to_pool_frames(ep: dict[str, Any]) -> list[tuple[int, str]]:
    """Map one curated 5-modality episode into all specialized sensory + action pools."""
    wf = ep["wire_frames"]
    sen = ep["sensor"]
    return [
        (POOL_TEXT, ep["query"]),
        (POOL_VISUAL, wf["visual"]),
        (POOL_AUDIO, wf["audio"]),
        (POOL_SOCIAL, wf["social"]),
        (POOL_SENSOR_IMU, f"sen:imu:{sen['imu_accel']}"),
        (POOL_SENSOR_THERMAL, f"sen:thm:{sen['thermal']}"),
        (POOL_SIGNAL_RF, f"sig:elc:{sen['electrical']}|sig:band:{ep['audio']['band']}"),
        (POOL_SENSOR_KINEMATICS, f"sen:trq:{sen['torque']}"),
        (POOL_SENSOR_BIOCHEM, f"sen:bio:{sen['env_biochem']}"),
        (POOL_ACTION, ep["action"]),
    ]


def run_course_2_meta_stream_pool_erection(
    data_dir: Path,
    backend: BrainHttpBackend | LocalSimulatorBackend,
    governor: PoliteElasticGovernor,
) -> dict[str, Any]:
    """Course 2: Automated Pool Erection from Meta-Stream Discovery."""
    snap = governor.breathe()
    envelopes = [
        json.loads(line)
        for line in (data_dir / "meta_stream_discovery_feed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    erector = MetaStreamPoolErector(
        spawn_support_threshold=3,
        novelty_threshold=0.28,
        max_dynamic_pools=snap.max_dynamic_pools,
        max_neurons_per_pool=snap.max_neurons_per_pool,
        max_fanout_per_pool=8,
        hebbian_window_secs=2,
        prune_idle_secs=100,
    )
    for env in envelopes:
        erector.observe_envelope(env["stream_header"], env["features"], env["timestamp_unix"])

    summaries = erector.erected_summaries()
    erected_names = {s.pool_name for s in summaries}
    expected_pools = {
        "social_pragmatics",
        "sensor_imu",
        "sensor_thermal",
        "signal_rf",
        "visual_diagram",
        "audio_prosody",
        "sensor_kinematics",
        "sensor_biochem",
    }
    rejected_noise = set(erector.unspawned_noise_streams())

    # Verify OrganicEncoder basis deduplication: 68 valid envelopes -> far fewer basis neurons
    total_valid_obs = sum(s.observation_count for s in summaries)
    total_basis_neurons = sum(s.basis_neuron_count for s in summaries)
    dedup_ratio = round(total_valid_obs / max(1, total_basis_neurons), 2)
    fanout_bounded = all(len(s.co_occurring_pools) <= erector.max_fanout_per_pool for s in summaries)
    cross_wired = all(len(s.co_occurring_pools) >= 3 for s in summaries)

    # Verify idle pool pruning lifecycle on a disposable stream
    idle_erector = MetaStreamPoolErector(spawn_support_threshold=2, prune_idle_secs=50)
    idle_erector.observe_envelope("sensor:legacy_sonar:ping", [1.0, 0.2], 10)
    idle_erector.observe_envelope("sensor:legacy_sonar:ping", [1.0, 0.2], 11)
    spawned_before_prune = len(idle_erector.erected_summaries()) == 1
    pruned_ids = idle_erector.prune_idle(100)
    idle_pruned_ok = spawned_before_prune and ("sensor_legacy_sonar" in pruned_ids)

    # Verify live Brain substrate actually has all erected pools active & wired
    st = backend.stats()
    live_pool_count = int(st.get("pool_count", 0))
    live_pools_ok = live_pool_count >= 12

    checks = {
        "all_8_novel_modalities_erected": expected_pools.issubset(erected_names),
        "transient_noise_streams_rejected": {"noise_glitch_spike", "debug_ephemeral_probe"}.issubset(rejected_noise),
        "basis_deduplication_active": total_basis_neurons <= (total_valid_obs // 2),
        "bounded_fanout_respected": fanout_bounded,
        "hebbian_cross_pool_wiring_formed": cross_wired,
        "idle_pool_pruning_verified": idle_pruned_ok,
        "live_brain_pools_instantiated": live_pools_ok,
    }
    passed = sum(1 for v in checks.values() if v)
    return {
        "course": "Course 2 — Automated Pool Erection from Meta-Stream Discovery",
        "passed": passed == len(checks),
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "erected_pool_names": sorted(erected_names),
        "rejected_low_support_streams": sorted(rejected_noise),
        "total_stream_observations": total_valid_obs,
        "total_basis_neurons_spawned": total_basis_neurons,
        "basis_deduplication_ratio_x": dedup_ratio,
        "live_brain_pool_count": live_pool_count,
    }


def run_course_1_crossmodal_hebbian_cofiring(
    episodes: list[dict[str, Any]],
    oov_queries: list[str],
    backend: BrainHttpBackend | LocalSimulatorBackend,
    governor: PoliteElasticGovernor,
) -> dict[str, Any]:
    """Course 1: Synchronized 5-Modality Co-Firing, Partial-Cue Recall & OOV Abstention."""
    governor.breathe()
    # Train all synchronized 5-modality episodes into the Brain
    binding_batches = [episode_to_pool_frames(ep) for ep in episodes]
    accepted = backend.pretrain_bindings_polite(binding_batches)

    base_eps = [ep for ep in episodes if ep.get("variant_index", 0) == 0]

    # 1A: Partial-Cue Recall: `visual + sensor -> action & text` (NO text, audio, or social supplied!)
    vis_sensor_probes = base_eps[:12]
    vis_sensor_hits = 0
    for ep in vis_sensor_probes:
        sen = ep["sensor"]
        streams = [
            (POOL_VISUAL, ep["wire_frames"]["visual"]),
            (POOL_SENSOR_IMU, f"sen:imu:{sen['imu_accel']}"),
            (POOL_SENSOR_THERMAL, f"sen:thm:{sen['thermal']}"),
            (POOL_SENSOR_KINEMATICS, f"sen:trq:{sen['torque']}"),
            (POOL_SENSOR_BIOCHEM, f"sen:bio:{sen['env_biochem']}"),
        ]
        pred_act = backend.predict_multi(streams, target_pool=POOL_ACTION)
        pred_txt = backend.predict_multi(streams, target_pool=POOL_TEXT)
        if pred_act["answer"] == ep["action"] and pred_txt["answer"] is not None:
            vis_sensor_hits += 1

    # 1B: Partial-Cue Recall: `audio + social -> action & visual` (NO text, visual, or sensor supplied!)
    # Select episodes whose `(audio, social)` pair is uniquely identifying across base_eps
    aud_soc_counts: dict[tuple[str, str], int] = {}
    for ep in base_eps:
        sig = (ep["wire_frames"]["audio"], ep["wire_frames"]["social"])
        aud_soc_counts[sig] = aud_soc_counts.get(sig, 0) + 1
    aud_soc_probes = [
        ep for ep in base_eps
        if aud_soc_counts[(ep["wire_frames"]["audio"], ep["wire_frames"]["social"])] == 1
    ][:12]

    aud_soc_hits = 0
    for ep in aud_soc_probes:
        streams = [
            (POOL_AUDIO, ep["wire_frames"]["audio"]),
            (POOL_SOCIAL, ep["wire_frames"]["social"]),
        ]
        pred_act = backend.predict_multi(streams, target_pool=POOL_ACTION)
        pred_vis = backend.predict_multi(streams, target_pool=POOL_VISUAL)
        if pred_act["answer"] == ep["action"] and pred_vis["answer"] == ep["wire_frames"]["visual"]:
            aud_soc_hits += 1

    # 1C: Cross-Modal Disambiguation: episodes sharing `shape == "circle"` disambiguated by social + sensor
    circle_eps = [ep for ep in base_eps if ep["visual"]["shape"] == "circle"]
    disambig_hits = 0
    for ep in circle_eps:
        sen = ep["sensor"]
        streams = [
            (POOL_VISUAL, ep["wire_frames"]["visual"]),
            (POOL_SOCIAL, ep["wire_frames"]["social"]),
            (POOL_SENSOR_IMU, f"sen:imu:{sen['imu_accel']}"),
            (POOL_SENSOR_THERMAL, f"sen:thm:{sen['thermal']}"),
        ]
        pred = backend.predict_multi(streams, target_pool=POOL_ACTION)
        if pred["answer"] == ep["action"]:
            disambig_hits += 1

    # 1D: OOV Abstention Honesty across ungrounded text (`/brain/chat` motif gate per
    #     science_scope_benchmark.py) & ungrounded sensory noise (`/brain/predict/multi`)
    oov_abstained = 0
    for q in oov_queries:
        res = backend.chat(q)
        if res["outside_grounding"] and not res["reply"].strip():
            oov_abstained += 1

    sensor_oov_cues = [
        (POOL_VISUAL, "vis:c:ultraviolet_x99|vis:s:hypercube_11d|vis:m:warp_jump|vis:l:non_euclidean"),
        (POOL_AUDIO, "aud:w:tachyon_chirp|aud:b:999ghz|aud:c:hyperbolic|aud:p:alien_glyph"),
        (POOL_SOCIAL, "soc:i:mind_meld|soc:t:martian|soc:u:telepathy|soc:c:hive|soc:p:cosmic"),
        (POOL_SENSOR_IMU, "sen:imu:warp_factor_9_impossible_99999g"),
    ]
    sensor_oov_abstained = 0
    for pid, cue in sensor_oov_cues:
        res = backend.predict_multi([(pid, cue)], target_pool=POOL_ACTION)
        if res["outside_grounding"] and res["answer"] is None:
            sensor_oov_abstained += 1

    mp_probe = backend.multi_pool_fanout_probe()

    total_oov = len(oov_queries) + len(sensor_oov_cues)
    total_oov_ok = oov_abstained + sensor_oov_abstained
    passed = (
        vis_sensor_hits == len(vis_sensor_probes)
        and aud_soc_hits == len(aud_soc_probes)
        and disambig_hits == len(circle_eps)
        and total_oov_ok == total_oov
    )

    return {
        "course": "Course 1 — Cross-Modal Hebbian Co-Firing & OOV Abstention",
        "passed": passed,
        "episodes_trained": accepted,
        "visual_plus_sensor_recall": {"hits": vis_sensor_hits, "total": len(vis_sensor_probes), "rate": round(vis_sensor_hits / len(vis_sensor_probes), 4)},
        "audio_plus_social_recall": {"hits": aud_soc_hits, "total": len(aud_soc_probes), "rate": round(aud_soc_hits / len(aud_soc_probes), 4)},
        "ambiguous_shape_crossmodal_disambiguation": {"hits": disambig_hits, "total": len(circle_eps), "rate": round(disambig_hits / max(1, len(circle_eps)), 4)},
        "oov_abstention_honesty": {
            "abstained": total_oov_ok,
            "total": total_oov,
            "honesty_rate": round(total_oov_ok / total_oov, 4),
            "hallucination_rate": round(1.0 - (total_oov_ok / total_oov), 4),
        },
        "node_multi_pool_fanout_probe": mp_probe is not None,
    }


def run_course_3_minimal_neuron_and_compositional_synthesis(
    data_dir: Path,
    episodes: list[dict[str, Any]],
    backend: BrainHttpBackend | LocalSimulatorBackend,
    governor: PoliteElasticGovernor,
) -> dict[str, Any]:
    """Course 3: Minimal-Neuron Non-Embellished Reconstructibility & Compositional Synthesis."""
    governor.breathe()
    manifest = json.loads((data_dir / "curation_manifest.json").read_text(encoding="utf-8"))
    comp_pack = json.loads((data_dir / "compositional_synthesis_curriculum.json").read_text(encoding="utf-8"))
    economy = manifest["neuron_economy_proof"]

    # 3A: Live Brain Reconstructibility WITHOUT Embellishment
    base_eps = [ep for ep in episodes if ep.get("variant_index", 0) == 0][:12]
    exact_reconstructed = 0
    zero_embellished = 0
    for ep in base_eps:
        vis_res = backend.predict(POOL_TEXT, ep["query"], target_pool=POOL_VISUAL)
        aud_res = backend.predict(POOL_TEXT, ep["query"], target_pool=POOL_AUDIO)
        soc_res = backend.predict(POOL_TEXT, ep["query"], target_pool=POOL_SOCIAL)
        if not (vis_res["answer"] and aud_res["answer"] and soc_res["answer"]):
            continue
        rec_vis = CanonicalMultisensoryCodec.reconstruct_visual(
            CanonicalMultisensoryCodec.deserialize_frame(vis_res["answer"])
        )
        rec_aud = CanonicalMultisensoryCodec.reconstruct_audio(
            CanonicalMultisensoryCodec.deserialize_frame(aud_res["answer"])
        )
        rec_soc = CanonicalMultisensoryCodec.reconstruct_social(
            CanonicalMultisensoryCodec.deserialize_frame(soc_res["answer"])
        )
        if rec_vis == ep["visual"] and rec_aud == ep["audio"] and rec_soc == ep["social"]:
            exact_reconstructed += 1
        if (
            set(rec_vis.keys()) == set(ep["visual"].keys())
            and set(rec_aud.keys()) == set(ep["audio"].keys())
            and set(rec_soc.keys()) == set(ep["social"].keys())
        ):
            zero_embellished += 1

    # 3B: Compositional Hebbian Synthesis from Separately Learned Component Experiences
    # Train each component experience SEPARATELY into the Brain fabric.
    # Role target is stored as a first-class Action concept (`role:<name>` in POOL_ACTION)
    # gated by `(POOL_COMPOSITIONAL_ROLE, "mode:component_role")` so it never collides
    # with operational action episodes.
    comp_episodes: list[list[tuple[int, str]]] = []
    primitive_to_role_bindings: list[list[tuple[int, str]]] = []
    role_ctx = (POOL_COMPOSITIONAL_ROLE, "mode:component_role")
    for comp in comp_pack["component_experiences"]:
        role_target = (POOL_ACTION, f"role:{comp['role']}")
        frames: list[tuple[int, str]] = [role_ctx, role_target]
        if comp["visual_atoms"]:
            v_wire = CanonicalMultisensoryCodec.serialize_frame(comp["visual_atoms"])
            frames.append((POOL_VISUAL, v_wire))
            primitive_to_role_bindings.append([(POOL_VISUAL, v_wire), role_ctx, role_target])
        if comp["social_atoms"]:
            s_wire = CanonicalMultisensoryCodec.serialize_frame(comp["social_atoms"])
            frames.append((POOL_SOCIAL, s_wire))
            primitive_to_role_bindings.append([(POOL_SOCIAL, s_wire), role_ctx, role_target])
        if comp["audio_atoms"]:
            a_wire = CanonicalMultisensoryCodec.serialize_frame(comp["audio_atoms"])
            frames.append((POOL_AUDIO, a_wire))
            primitive_to_role_bindings.append([(POOL_AUDIO, a_wire), role_ctx, role_target])
        if comp["sensor_atoms"]:
            for satom in comp["sensor_atoms"]:
                pid = (
                    POOL_SENSOR_KINEMATICS if ":trq:" in satom
                    else POOL_SENSOR_THERMAL if ":thm:" in satom
                    else POOL_SENSOR_IMU if ":imu:" in satom
                    else POOL_SIGNAL_RF if ":elc:" in satom
                    else POOL_SENSOR_BIOCHEM
                )
                frames.append((pid, satom))
                primitive_to_role_bindings.append([(pid, satom), role_ctx, role_target])
        comp_episodes.append(frames)

    backend.pretrain_bindings_polite(comp_episodes + primitive_to_role_bindings)

    # Register compositional pairing schemas in EEM (`TransientWorkspace` rules)
    for rule_spec in comp_pack["synthesis_rules"]:
        backend.logic_consolidate_rule(
            name=rule_spec["rule_name"],
            premises=[
                ("active_component_role", [("const", "session", rule_spec["rule_name"]), ("const", "role", rule_spec["left_role"])]),
                ("active_component_role", [("const", "session", rule_spec["rule_name"]), ("const", "role", rule_spec["right_role"])]),
            ],
            conclusion=(
                "synthesized_capability",
                [("const", "session", rule_spec["rule_name"]), ("const", "capability", rule_spec["expected_synthesis"])],
            ),
        )

    def detect_grounded_roles_from_stimulus(stim: dict[str, list[str]]) -> set[str]:
        """Probe the Brain fabric across sensory pools to see which learned component roles fire."""
        fired_roles: set[str] = set()
        candidate_probes: list[tuple[int, str]] = []
        if stim.get("visual_atoms"):
            va = stim["visual_atoms"]
            candidate_probes.append((POOL_VISUAL, CanonicalMultisensoryCodec.serialize_frame(va)))
            colors = [x for x in va if x.startswith("vis:c:")]
            shapes = [x for x in va if x.startswith("vis:s:")]
            motions = [x for x in va if x.startswith("vis:m:")]
            layouts = [x for x in va if x.startswith("vis:l:")]
            for c in colors:
                for s in shapes:
                    candidate_probes.append((POOL_VISUAL, f"{c}|{s}"))
                    for m in motions:
                        candidate_probes.append((POOL_VISUAL, f"{c}|{s}|{m}"))
            for l_tok in layouts:
                candidate_probes.append((POOL_VISUAL, l_tok))
        if stim.get("social_atoms"):
            candidate_probes.append((POOL_SOCIAL, CanonicalMultisensoryCodec.serialize_frame(stim["social_atoms"])))
        if stim.get("audio_atoms"):
            candidate_probes.append((POOL_AUDIO, CanonicalMultisensoryCodec.serialize_frame(stim["audio_atoms"])))
        for satom in stim.get("sensor_atoms", []):
            pid = (
                POOL_SENSOR_KINEMATICS if ":trq:" in satom
                else POOL_SENSOR_THERMAL if ":thm:" in satom
                else POOL_SENSOR_IMU if ":imu:" in satom
                else POOL_SIGNAL_RF if ":elc:" in satom
                else POOL_SENSOR_BIOCHEM
            )
            candidate_probes.append((pid, satom))

        for pid, wire in candidate_probes:
            res = backend.predict_multi(
                [(pid, wire), (POOL_COMPOSITIONAL_ROLE, "mode:component_role")],
                target_pool=POOL_ACTION,
            )
            ans = res.get("answer") or ""
            if not res["outside_grounding"] and ans.startswith("role:"):
                fired_roles.add(ans[5:])
        return fired_roles

    synthesis_hits = 0
    synthesis_details: list[dict[str, Any]] = []
    for rule_spec in comp_pack["synthesis_rules"]:
        session_tag = rule_spec["rule_name"]
        roles = detect_grounded_roles_from_stimulus(rule_spec["novel_combined_stimulus"])
        for role in sorted(roles):
            backend.logic_consolidate_relation(
                "active_component_role",
                [("session", session_tag), ("role", role)],
                confidence=0.95,
                source=f"neural_cofire:{session_tag}",
            )
        composed = backend.logic_compose(predicate="synthesized_capability", max_rounds=4)
        session_caps = [
            f["arguments"][1]["value"]
            for f in composed
            if len(f.get("arguments", [])) == 2 and f["arguments"][0]["value"] == session_tag
        ]
        ok = (rule_spec["expected_synthesis"] in session_caps) and (len(session_caps) == 1)
        if ok:
            synthesis_hits += 1
        synthesis_details.append({
            "rule": session_tag,
            "recovered_component_roles": sorted(roles),
            "synthesized": session_caps,
            "expected": rule_spec["expected_synthesis"],
            "passed": ok,
        })

    # Negative controls: ungrounded OOV or single partial primitive must NOT synthesize any capability
    false_syntheses = 0
    for ctrl in comp_pack["ungrounded_controls"]:
        ctrl_tag = ctrl["control_id"]
        roles = detect_grounded_roles_from_stimulus(ctrl)
        for role in sorted(roles):
            backend.logic_consolidate_relation(
                "active_component_role",
                [("session", ctrl_tag), ("role", role)],
                confidence=0.95,
                source=f"ctrl:{ctrl_tag}",
            )
        composed = backend.logic_compose(predicate="synthesized_capability", max_rounds=4)
        ctrl_caps = [
            f["arguments"][1]["value"]
            for f in composed
            if len(f.get("arguments", [])) == 2 and f["arguments"][0]["value"] == ctrl_tag
        ]
        if ctrl_caps:
            false_syntheses += 1

    brain_stats = backend.stats()
    total_neurons = int(brain_stats.get("total_neurons", 0))
    neurons_per_episode = round(total_neurons / max(1, len(episodes)), 2)

    passed = (
        economy["neuron_reduction_factor_x"] >= 20.0
        and exact_reconstructed == len(base_eps)
        and zero_embellished == len(base_eps)
        and synthesis_hits == len(comp_pack["synthesis_rules"])
        and false_syntheses == 0
    )

    return {
        "course": "Course 3 — Minimal-Neuron Reconstructibility & Compositional Hebbian Synthesis",
        "passed": passed,
        "neuron_economy": {
            **economy,
            "live_brain_total_neurons": total_neurons,
            "live_brain_neurons_per_episode": neurons_per_episode,
        },
        "live_non_embellished_reconstruction": {
            "exact_matches": exact_reconstructed,
            "zero_embellished": zero_embellished,
            "total": len(base_eps),
            "exact_rate": round(exact_reconstructed / len(base_eps), 4),
        },
        "compositional_hebbian_synthesis": {
            "synthesized_correctly": synthesis_hits,
            "total_novel_combinations": len(comp_pack["synthesis_rules"]),
            "false_syntheses_on_controls": false_syntheses,
            "controls_tested": len(comp_pack["ungrounded_controls"]),
            "details": synthesis_details,
        },
    }


def seed_foundational_anchors_and_measure_baseline(
    anchors: list[dict[str, str]],
    backend: BrainHttpBackend | LocalSimulatorBackend,
) -> int:
    """Seed foundational anchors BEFORE any multimodal/discipline training."""
    anchor_bindings = [
        [(POOL_TEXT, item["query"]), (POOL_ACTION, item["answer"])]
        for item in anchors
    ]
    backend.pretrain_bindings_polite(anchor_bindings)
    hits = 0
    for item in anchors:
        res = backend.predict_multi([(POOL_TEXT, item["query"])], target_pool=POOL_ACTION)
        if res["answer"] == item["answer"] and not res["outside_grounding"]:
            hits += 1
    return hits


def run_course_4_twelve_disciplines_and_antiforget(
    data_dir: Path,
    episodes: list[dict[str, Any]],
    baseline_anchor_hits: int,
    backend: BrainHttpBackend | LocalSimulatorBackend,
    governor: PoliteElasticGovernor,
) -> dict[str, Any]:
    """Course 4: 12-Discipline Supplement, Multi-Hop Derivation & Anti-Forgetting."""
    governor.breathe()
    pack = json.loads((data_dir / "twelve_discipline_obstacle_pack.json").read_text(encoding="utf-8"))
    anchors = pack["foundational_anchors"]
    chains = pack["multihop_chains"]
    oov_queries = pack["oov_outside_queries"]

    # Supplement with 12-Discipline Text->Explanation bindings + Multi-Hop Causal Chains
    base_eps = [ep for ep in episodes if ep.get("variant_index", 0) == 0]
    disc_qa_bindings = [
        [(POOL_TEXT, f"Explain [{ep['discipline']}]: {ep['query']}"), (POOL_ACTION, ep["text"])]
        for ep in base_eps
    ]
    hop_bindings: list[list[tuple[int, str]]] = []
    for ch in chains:
        for src_state, dst_state in ch["hops"]:
            hop_bindings.append([(POOL_TEXT, src_state), (POOL_ACTION, dst_state)])
            backend.logic_consolidate_relation(
                "causal_step",
                [("state", src_state), ("state", dst_state)],
                confidence=0.97,
                source=ch["chain_id"],
            )

    # Transitive causal chaining rule in EEM (`causal_path(A, C) :- causal_step(A, B), causal_path(B, C)`)
    backend.logic_consolidate_rule(
        name="causal_base_step",
        premises=[("causal_step", [("var", "A", "state"), ("var", "B", "state")])],
        conclusion=("causal_path", [("var", "A", "state"), ("var", "B", "state")]),
    )
    backend.logic_consolidate_rule(
        name="causal_transitive_chain",
        premises=[
            ("causal_path", [("var", "A", "state"), ("var", "B", "state")]),
            ("causal_step", [("var", "B", "state"), ("var", "C", "state")]),
        ],
        conclusion=("causal_path", [("var", "A", "state"), ("var", "C", "state")]),
    )

    backend.pretrain_bindings_polite(disc_qa_bindings + hop_bindings)

    # Stage 4.3: Evaluate ALL 12 Disciplines (both Q->Explanation and Multi-Sensory->Action)
    per_discipline: dict[str, dict[str, int]] = {
        d: {"text_hits": 0, "multisensory_hits": 0, "total": 0}
        for d in TWELVE_DISCIPLINES
    }
    for ep in base_eps:
        d = ep["discipline"]
        per_discipline[d]["total"] += 1
        q_res = backend.predict(POOL_TEXT, f"Explain [{d}]: {ep['query']}", target_pool=POOL_ACTION)
        if q_res["answer"] == ep["text"]:
            per_discipline[d]["text_hits"] += 1

        sen = ep["sensor"]
        ms_res = backend.predict_multi([
            (POOL_VISUAL, ep["wire_frames"]["visual"]),
            (POOL_AUDIO, ep["wire_frames"]["audio"]),
            (POOL_SOCIAL, ep["wire_frames"]["social"]),
            (POOL_SENSOR_IMU, f"sen:imu:{sen['imu_accel']}"),
            (POOL_SENSOR_THERMAL, f"sen:thm:{sen['thermal']}"),
        ], target_pool=POOL_ACTION)
        if ms_res["answer"] == ep["action"]:
            per_discipline[d]["multisensory_hits"] += 1

    disciplines_passed = sum(
        1 for d, sc in per_discipline.items()
        if sc["total"] > 0 and sc["text_hits"] == sc["total"] and sc["multisensory_hits"] == sc["total"]
    )

    # Stage 4.4: Evaluate 6 Multi-Hop Cross-Discipline Derivations (3 hops each)
    composed_paths = backend.logic_compose(predicate="causal_path", max_rounds=8)
    eem_pairs = {
        (f["arguments"][0]["value"], f["arguments"][1]["value"])
        for f in composed_paths
        if len(f.get("arguments", [])) == 2
    }

    chain_hits = 0
    chain_details: list[dict[str, Any]] = []
    for ch in chains:
        current = ch["seed"]
        walked: list[str] = [current]
        for _ in range(len(ch["hops"])):
            nxt = backend.predict_multi([(POOL_TEXT, current)], target_pool=POOL_ACTION)
            if nxt["outside_grounding"] or not nxt["answer"]:
                break
            current = nxt["answer"]
            walked.append(current)
        fabric_ok = walked[-1] == ch["expected_terminal"] and len(walked) == (len(ch["hops"]) + 1)
        eem_ok = (ch["seed"], ch["expected_terminal"]) in eem_pairs
        if fabric_ok and eem_ok:
            chain_hits += 1
        chain_details.append({
            "chain_id": ch["chain_id"],
            "disciplines": ch["disciplines"],
            "hops_walked": len(walked) - 1,
            "terminal_reached": walked[-1],
            "expected_terminal": ch["expected_terminal"],
            "fabric_chain_ok": fabric_ok,
            "eem_workspace_ok": eem_ok,
        })

    # Stage 4.5: Anti-Forgetting Check on Foundational Anchors + OOV Scope Refusal
    retained_anchor_hits = 0
    for item in anchors:
        res = backend.predict_multi([(POOL_TEXT, item["query"])], target_pool=POOL_ACTION)
        if res["answer"] == item["answer"] and not res["outside_grounding"]:
            retained_anchor_hits += 1

    oov_refusals = 0
    for q in oov_queries:
        res = backend.chat(q)
        if res["outside_grounding"] and not res["reply"].strip():
            oov_refusals += 1

    forgetting_count = max(0, baseline_anchor_hits - retained_anchor_hits)
    passed = (
        baseline_anchor_hits == len(anchors)
        and retained_anchor_hits == len(anchors)
        and forgetting_count == 0
        and disciplines_passed == len(TWELVE_DISCIPLINES)
        and chain_hits == len(chains)
        and oov_refusals == len(oov_queries)
    )

    return {
        "course": "Course 4 — 12-Discipline Supplement, Multi-Hop Derivation & Anti-Forgetting",
        "passed": passed,
        "disciplines_passed": disciplines_passed,
        "disciplines_total": len(TWELVE_DISCIPLINES),
        "per_discipline": per_discipline,
        "multihop_derivations": {
            "chains_passed": chain_hits,
            "chains_total": len(chains),
            "details": chain_details,
        },
        "anti_forgetting_anchors": {
            "baseline_hits": baseline_anchor_hits,
            "post_supplement_retained_hits": retained_anchor_hits,
            "anchors_total": len(anchors),
            "retention_rate": round(retained_anchor_hits / len(anchors), 4),
            "forgetting_rate": round(forgetting_count / len(anchors), 4),
        },
        "outside_scope_abstention": {
            "refusals": oov_refusals,
            "total": len(oov_queries),
            "honesty_rate": round(oov_refusals / len(oov_queries), 4),
        },
    }


# ============================================================================
# 3. Orchestrator Entry Point
# ============================================================================

def run_obstacle_suite(
    mode: str = "auto",
    endpoint: str | None = None,
    executable: Path | None = None,
    data_dir: Path = ROOT / "data" / "multimodal_streams",
    report_path: Path | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    governor = PoliteElasticGovernor(min_free_gib=8.0, expand_gib=14.5, cushion_fraction=0.12)
    governor.sample()

    data_dir = data_dir.resolve()
    if not (data_dir / "curation_manifest.json").is_file():
        curate_all(data_dir)

    identity_path = data_dir / "multisensory_crossmodal.identity.toml"
    episodes = [
        json.loads(line)
        for line in (data_dir / "crossmodal_episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    disc_pack = json.loads((data_dir / "twelve_discipline_obstacle_pack.json").read_text(encoding="utf-8"))

    resolved_mode = mode
    if endpoint:
        resolved_mode = "endpoint"
    elif mode == "auto":
        exe = executable or find_brain_server_executable()
        resolved_mode = "local-rust-brain" if exe is not None else "simulator"

    backend: BrainHttpBackend | LocalSimulatorBackend
    backend_label: str
    if resolved_mode == "endpoint":
        assert endpoint is not None
        backend = BrainHttpBackend(endpoint=endpoint, governor=governor)
        backend_label = f"live_http_endpoint({endpoint})"
    elif resolved_mode == "local-rust-brain":
        exe = (executable or find_brain_server_executable())
        if exe is None or not exe.is_file():
            raise FileNotFoundError("w1z4rd_brain_server executable not found; pass --executable or --mode simulator")
        backend = spawn_disposable_rust_brain(exe, identity_path, governor)
        backend_label = f"disposable_rust_brain_server({exe.name})"
    elif resolved_mode == "simulator":
        backend = LocalSimulatorBackend(governor=governor, pool_ids=list(range(1, 13)))
        backend_label = "local_python_w1z4rd_simulator"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    try:
        # 0. Seed foundational anchors FIRST (before Course 1, Course 3, and Course 4
        #    supplements) so Course 4 measures true session-wide anti-forgetting!
        baseline_anchor_hits = seed_foundational_anchors_and_measure_baseline(
            disc_pack["foundational_anchors"], backend
        )
        c2 = run_course_2_meta_stream_pool_erection(data_dir, backend, governor)
        c1 = run_course_1_crossmodal_hebbian_cofiring(
            episodes, disc_pack["oov_outside_queries"], backend, governor
        )
        c3 = run_course_3_minimal_neuron_and_compositional_synthesis(
            data_dir, episodes, backend, governor
        )
        c4 = run_course_4_twelve_disciplines_and_antiforget(
            data_dir, episodes, baseline_anchor_hits, backend, governor
        )

        final_stats = backend.stats()
        brain_rss_mib = backend.process_rss_mib()
        runner_rss_mib = round(psutil.Process().memory_info().rss / (1024.0 ** 2), 2)
        governor.sample()
        gc.collect()
    finally:
        backend.close()

    courses = [c1, c2, c3, c4]
    courses_passed = sum(1 for c in courses if c["passed"])
    report = {
        "schema_version": "1.0.0",
        "timestamp_unix": round(time.time(), 3),
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "backend": backend_label,
        "mode": resolved_mode,
        "all_passed": courses_passed == len(courses),
        "courses_passed": courses_passed,
        "courses_total": len(courses),
        "final_brain_stats": final_stats,
        "resource_politeness": {
            **governor.summary(),
            "runner_rss_mib": runner_rss_mib,
            "disposable_brain_rss_mib": brain_rss_mib,
            "combined_rss_mib": round(runner_rss_mib + brain_rss_mib, 2),
        },
        "courses": {
            "course_1_crossmodal_cofiring": c1,
            "course_2_meta_stream_pool_erection": c2,
            "course_3_minimal_neuron_and_synthesis": c3,
            "course_4_twelve_disciplines_and_antiforget": c4,
        },
    }

    if report_path is None:
        report_path = data_dir / "obstacle_course_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "local-rust-brain", "simulator", "endpoint"),
        default="auto",
    )
    parser.add_argument("--endpoint", type=str, default=None, help="Live node/brain URL, e.g. http://127.0.0.1:8090")
    parser.add_argument("--executable", type=Path, default=None, help="Path to w1z4rd_brain_server binary")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "multimodal_streams")
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "data" / "multimodal_streams" / "obstacle_course_report.json",
    )
    args = parser.parse_args(argv)

    report = run_obstacle_suite(
        mode=args.mode,
        endpoint=args.endpoint,
        executable=args.executable,
        data_dir=args.data_dir,
        report_path=args.report,
    )
    print(json.dumps({
        "all_passed": report["all_passed"],
        "courses_passed": f"{report['courses_passed']}/{report['courses_total']}",
        "backend": report["backend"],
        "elapsed_seconds": report["elapsed_seconds"],
        "final_brain_stats": report["final_brain_stats"],
        "resource_politeness": report["resource_politeness"],
        "course_verdicts": {k: v["passed"] for k, v in report["courses"].items()},
        "report_path": str(args.report.resolve()),
    }, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
