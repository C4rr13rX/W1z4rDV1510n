from __future__ import annotations

from typing import Any, Dict

from .base import Provider


class BraketProvider(Provider):
    def supports(self, purpose: str) -> bool:
        caps = (self.cfg or {}).get("capabilities") or {}
        if isinstance(caps, dict) and purpose in caps:
            return bool(caps.get(purpose))
        return purpose == "experiment"

    def score_branches(self, request_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        raise RuntimeError("braket: branch_scoring not implemented yet (best path is mapping to QUBO for D-Wave via Braket)")

    def run_experiment(self, task_obj: dict, timeout_secs: int) -> Dict[str, Any]:
        """
        Supported task shapes (when amazon-braket-sdk is installed):
        - type=braket_circuit: {device_arn, shots, circuit: ...}
        - type=braket_openqasm: {device_arn, shots, openqasm: 'OPENQASM 3; ...'}

        This is intentionally minimal; advanced workflows should be built on top.
        """
        try:
            from braket.aws import AwsDevice  # type: ignore
            from braket.circuits import Circuit  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "braket: missing dependency amazon-braket-sdk; pip install amazon-braket-sdk"
            ) from e

        t = str(task_obj.get("type", "")).strip()
        device_arn = str(task_obj.get("device_arn", "")).strip()
        if not device_arn:
            raise RuntimeError("braket: missing device_arn")
        shots = int(task_obj.get("shots", 1000))
        device = AwsDevice(device_arn)

        if t == "braket_circuit":
            circ = task_obj.get("circuit")
            if not isinstance(circ, dict):
                raise RuntimeError("braket: circuit must be an object compatible with Braket's Circuit JSON IR")
            # Best-effort: allow users to pass in OpenQASM via braket_openqasm instead.
            raise RuntimeError("braket: braket_circuit is not implemented; use braket_openqasm for now")

        if t == "braket_openqasm":
            qasm = str(task_obj.get("openqasm", ""))
            if not qasm.strip():
                raise RuntimeError("braket: missing openqasm")
            # Circuit.from_ir supports OpenQASM 3 in newer SDK versions; keep defensive.
            try:
                circuit = Circuit().from_ir(qasm)  # type: ignore[attr-defined]
                task = device.run(circuit, shots=shots)
            except Exception:
                # Fall back to submitting raw OpenQASM program if available.
                try:
                    from braket.ir.openqasm import Program  # type: ignore

                    program = Program(source=qasm)
                    task = device.run(program, shots=shots)
                except Exception as e:
                    raise RuntimeError("braket: failed to submit OpenQASM program") from e

            result = task.result()
            counts = getattr(result, "measurement_counts", None)
            meta = {
                "provider": "braket",
                "device_arn": device_arn,
                "task_id": str(getattr(task, "id", "")),
            }
            out = {"counts": dict(counts) if counts is not None else None, "metadata": meta}
            return out

        raise RuntimeError(f"braket: unsupported experiment type {t!r}")


def build_provider(cfg: dict):
    return BraketProvider(name="braket", priority=int(cfg.get("priority", 50)), cfg=cfg)

