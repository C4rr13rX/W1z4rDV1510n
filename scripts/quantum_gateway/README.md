# Quantum Gateway

This folder contains a small HTTP service that plugs into W1z4rDV1510n's existing **remote quantum endpoint** support (`compute.quantum_endpoints`).

The Rust core already knows how to call an HTTP endpoint via `QuantumHttpExecutor`. The gateway makes that endpoint useful by:
- Adding a provider registry for **Amazon Braket**, **Azure Quantum**, **IBM Qiskit Runtime**, **IonQ**, **D-Wave Leap (SAPI)**, **Google Quantum Engine**, **Pasqal**, **Xanadu XCC**, **AQT**, **Quandela/Perceval**, **Scaleway QaS (Quandela)**, **Quantum Inspire**, **qBraid**, and **Strangeworks Azure**.
- Enforcing **outcome protection** (schema checks + probability normalization + divergence rejection + caching).
- Exposing a direct power-user endpoint (`/quantum/experiment`) for raw API access.

## Run it

```powershell
python scripts/quantum_gateway/gateway.py
```

Defaults:
- Listen: `127.0.0.1:5050` (override with `W1Z4RDV1510N_QUANTUM_GATEWAY_ADDR`)
- Config: `scripts/quantum_gateway/gateway_config.json` (override with `W1Z4RDV1510N_QUANTUM_GATEWAY_CONFIG`)

## Secure it (optional)

If you set `W1Z4RDV1510N_QUANTUM_GATEWAY_TOKEN`, the gateway will require:

`Authorization: Bearer <token>`

(configurable in `gateway_config.json` under `auth`).

## Connect the Rust node/runtime

Add an endpoint in `node_config.json` (or `node_config_example.json`):

```json
{
  "compute": {
    "allow_quantum": true,
    "quantum_endpoints": [
      {
        "name": "local-gateway",
        "url": "http://127.0.0.1:5050/quantum/submit",
        "timeout_secs": 30,
        "provider": "gateway",
        "priority": 10,
        "auth_env": "W1Z4RDV1510N_QUANTUM_GATEWAY_TOKEN",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer "
      }
    ]
  }
}
```

Then enable remote quantum usage in the run config:

```json
{
  "quantum": { "enabled": true, "remote_enabled": true }
}
```

## Direct experiments (raw REST)

`POST /quantum/experiment`

Example (IonQ list backends, assuming your IonQ config enables it and `IONQ_API_KEY` is set):

```json
{
  "provider": "ionq",
  "task": { "type": "raw_rest", "method": "GET", "path": "/backends" },
  "timeout_secs": 30
}
```

This is intentionally generic: you can hit *any* documented provider endpoint without waiting for a bespoke adapter.

## Notes / next upgrades

- The current `/quantum/submit` implementation supports:
  - `BRANCH_SCORING` (provider orchestration + outcome protection; providers are opt-in)
  - `QUANTUM_CALIBRATION` (local heuristic calibrator; no external creds needed)
- Higher-level provider-specific adapters (e.g., mapping branch scoring to QUBO sampling on D-Wave/Braket) are the next high-leverage step.

