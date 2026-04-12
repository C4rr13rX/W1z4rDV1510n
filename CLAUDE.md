# W1z4rD V1510n — Claude Code Project Config

## Project Overview
Distributed AI/neural computing node system with cluster, P2P gossip, wallet, and dashboard.
Owner: C4rr13rX (c4rr13rX@gmail.com) | Repo: https://github.com/C4rr13rX/W1z4rDV1510n

## Workspace Structure
- `crates/core` — neural fabric, Hebbian learning, neuro API (port 8080), sensor streams
- `crates/cluster` — P2P cluster ring, OTP join, gossip, heartbeat/election
- `crates/node` — main node binary (`w1z4rdv1510n-node`), node API (port 8090), all HTTP routes
- `crates/dashboard` — egui/eframe desktop GUI (`w1z4rd-dashboard`)
- `crates/experimental-hw` — GPU/hardware experiments

## Build
```bash
export PATH="$PATH:/c/Users/Node/.cargo/bin:/c/Users/Node/AppData/Local/Microsoft/WinGet/Packages/BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe/mingw64/bin"
cargo build --release --workspace
```
Toolchain: `stable-x86_64-pc-windows-gnu` (requires WinLibs MinGW-w64 for `dlltool.exe`).

## Run
```bash
# Node (from install dir so it picks up node_config.json)
cd /c/W1z4rD && ./w1z4rd_node.exe

# Dashboard
./target/release/w1z4rd-dashboard.exe
```
Install dir: `C:\W1z4rD\` — always launch node from there (config is relative to CWD).

## Deploy after build
```bash
# Copy fresh node binary (requires admin)
powershell -Command "Start-Process powershell -Verb RunAs -ArgumentList '-Command Copy-Item -Force C:\\Users\\Node\\W1z4rDV1510n\\target\\release\\w1z4rdv1510n-node.exe C:\\W1z4rD\\w1z4rd_node.exe' -Wait"
```

## Key Ports
| Port  | Service         |
|-------|-----------------|
| 8080  | Neuro API       |
| 8090  | Node API        |
| 51611 | Cluster (SIGIL) |

## Node Modes
- `SENSOR` — local AI/streaming mode, wallet optional (set in `node_config.json`)
- `PRODUCTION` — full Web3 mode, wallet required

## Important Notes
- Always commit and push after any code changes
- Kill old processes before deploying new binary (port conflicts cause silent API thread death)
- `node_config.json` in `C:\W1z4rD\` has `data.enabled: false` and `wallet.prompt_on_load: false`
- The GNU toolchain requires WinLibs PATH to be set or dlltool errors occur
- Avira AV may quarantine Rust build artifacts — exclusions are set in Windows Defender
