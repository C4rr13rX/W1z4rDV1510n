# W1z4rD V1510n — Node 3 Installer
# Installs and configures a second PC to join the cluster as node-3.
#
# Requirements:
#   - Run as Administrator on the TARGET PC
#   - Copy this script + the bin\ folder from the main PC, OR provide -SourceDir
#
# Usage:
#   .\install_node3.ps1
#   .\install_node3.ps1 -SourceDir "\\192.168.1.84\W1z4rD\bin"
#   .\install_node3.ps1 -InstallDir "C:\W1z4rD" -NodeHost "192.168.1.84"
#
# What this does:
#   1. Creates C:\W1z4rD (or -InstallDir)
#   2. Copies w1z4rd_node.exe + w1z4rd_dashboard.exe
#   3. Writes node_config.json pointing bootstrap_peers at the main node
#   4. Opens firewall ports 8088, 8090, 51611
#   5. Creates start_node.bat and start_dashboard.bat
#   6. Registers a scheduled task to auto-start the node on login
#   7. Launches the node + dashboard

param(
    [string]$InstallDir = "C:\W1z4rD",
    [string]$SourceDir  = "",          # path to a bin\ folder; auto-detects if blank
    [string]$NodeHost   = "192.168.1.84",  # main node IP
    [string]$NodePort   = "8088",          # main node P2P gossip port
    [string]$ApiPort    = "8090",          # local API port for this node
    [switch]$NoAutoStart,
    [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"

# ── Elevation check ───────────────────────────────────────────────────────────
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "`n[W1z4rD] Restarting as Administrator..." -ForegroundColor Yellow
    $ps_args = "-File `"$PSCommandPath`" -NodeHost `"$NodeHost`" -NodePort `"$NodePort`" -ApiPort `"$ApiPort`""
    if ($InstallDir) { $ps_args += " -InstallDir `"$InstallDir`"" }
    if ($SourceDir)  { $ps_args += " -SourceDir `"$SourceDir`"" }
    if ($NoAutoStart){ $ps_args += " -NoAutoStart" }
    if ($SkipLaunch) { $ps_args += " -SkipLaunch" }
    Start-Process powershell -Verb RunAs -ArgumentList $ps_args
    exit
}

function Banner($msg) {
    Write-Host ""
    Write-Host "  ════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host "   W1z4rD V1510n  —  $msg" -ForegroundColor Cyan
    Write-Host "  ════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host ""
}
function Step($n, $msg)  { Write-Host "  [$n] $msg" -ForegroundColor White }
function OK($msg)        { Write-Host "      OK  $msg" -ForegroundColor Green }
function WARN($msg)      { Write-Host "      !!  $msg" -ForegroundColor Yellow }
function ERR($msg)       { Write-Host "      XX  $msg" -ForegroundColor Red }

Banner "Node 3 Installer"

# ── Detect this PC's LAN IP ───────────────────────────────────────────────────
$ThisPCIP = (
    Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.IPAddress -notlike "172.*" -and
        $_.IPAddress -notlike "192.168.56.*"   # skip VirtualBox host-only
    } |
    Sort-Object PrefixLength |
    Select-Object -First 1 -ExpandProperty IPAddress
)
if (-not $ThisPCIP) {
    # Fallback: first non-loopback IPv4
    $ThisPCIP = (
        Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notlike "127.*" } |
        Select-Object -First 1 -ExpandProperty IPAddress
    )
}
Write-Host "  This PC LAN IP : $ThisPCIP" -ForegroundColor Cyan

# ── Locate source binaries ────────────────────────────────────────────────────
Step 0 "Locating source binaries"

if ($SourceDir -eq "") {
    # If this script is in a bin\ or scripts\ folder next to the binaries, try siblings
    $script_dir = Split-Path -Parent $PSCommandPath
    $candidates = @(
        (Join-Path $script_dir "..\bin"),            # project\scripts\ → project\bin\
        (Join-Path $script_dir "bin"),               # standalone
        $script_dir,                                  # binaries in same folder as script
        "\\$NodeHost\W1z4rD\bin"                     # UNC share from main PC
    )
    foreach ($c in $candidates) {
        $c = [System.IO.Path]::GetFullPath($c)
        if (Test-Path (Join-Path $c "w1z4rd_node.exe")) {
            $SourceDir = $c
            OK "Found binaries at $SourceDir"
            break
        }
    }
}

if ($SourceDir -eq "" -or -not (Test-Path (Join-Path $SourceDir "w1z4rd_node.exe"))) {
    ERR "Cannot find w1z4rd_node.exe. Copy the bin\ folder from the main PC alongside this script."
    ERR "Or run:  .\install_node3.ps1 -SourceDir '\\MAINPC\share\bin'"
    exit 1
}

# ── Step 1: Create install directory ─────────────────────────────────────────
Step 1 "Creating install directory: $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir            | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\data"     | Out-Null
New-Item -ItemType Directory -Force -Path "$InstallDir\logs"     | Out-Null
OK "Directory ready"

# ── Step 2: Copy binaries ─────────────────────────────────────────────────────
Step 2 "Installing binaries"
Stop-Process -Name "w1z4rd_node" -Force -ErrorAction SilentlyContinue
Copy-Item -Force (Join-Path $SourceDir "w1z4rd_node.exe") "$InstallDir\w1z4rd_node.exe"
OK "w1z4rd_node.exe installed"

$dash_src = Join-Path $SourceDir "w1z4rd_dashboard.exe"
if (Test-Path $dash_src) {
    Copy-Item -Force $dash_src "$InstallDir\w1z4rd_dashboard.exe"
    OK "w1z4rd_dashboard.exe installed"
}

# ── Step 3: Write node config ─────────────────────────────────────────────────
Step 3 "Writing node_config.json (bootstrap → $NodeHost`:$NodePort, advertise → $ThisPCIP`:8088)"

$config = @"
{
  "node_id": "node-3",
  "node_mode": "SENSOR",
  "node_role": "WORKER",
  "network": {
    "listen_addr": "0.0.0.0:8088",
    "advertise_addr": "$ThisPCIP`:8088",
    "bootstrap_peers": ["$NodeHost`:$NodePort"],
    "max_peers": 128,
    "gossip_protocol": "w1z4rdv1510n-gossip",
    "security": {
      "max_message_bytes": 262144,
      "max_messages_per_rpc": 128,
      "max_pending_incoming": 64,
      "max_pending_outgoing": 64,
      "max_established_incoming": 256,
      "max_established_outgoing": 256,
      "max_established_total": 512,
      "max_established_per_peer": 8
    }
  },
  "openstack": { "enabled": false, "mode": "LOCAL_CONTROL_PLANE", "region": "RegionOne", "interface": "public" },
  "wallet": {
    "enabled": true,
    "path": "$InstallDir\\wallet.json",
    "auto_create": true,
    "encrypted": false,
    "passphrase_env": "",
    "prompt_on_load": false
  },
  "data": {
    "enabled": false,
    "storage_path": "$InstallDir\\data",
    "host_storage": false,
    "max_payload_bytes": 524288,
    "chunk_size_bytes": 32768,
    "replication_factor": 1,
    "receipt_quorum": 1,
    "require_manifest_signature": false,
    "require_receipt_signature": false,
    "max_pending_chunks": 64,
    "maintenance_enabled": false,
    "maintenance_interval_secs": 300,
    "retention_days": 30,
    "max_storage_bytes": 0,
    "max_repair_requests_per_tick": 0,
    "storage_reward_enabled": false,
    "storage_reward_base": 0.0,
    "storage_reward_per_mb": 0.0
  },
  "streaming": {
    "enabled": true,
    "ultradian_node": false,
    "run_config_path": "$InstallDir\\run_config.json",
    "publish_streams": true,
    "publish_shares": true,
    "consume_streams": true,
    "consume_shares": true,
    "stream_payload_kind": "stream.envelope.v1",
    "share_payload_kind": "neural.fabric.v1",
    "min_cpu_cores": 2,
    "min_memory_gb": 4.0
  },
  "knowledge": {
    "enabled": true,
    "persist_state": true,
    "state_path": "$InstallDir\\knowledge_state.json",
    "queue": {
      "min_votes": 2,
      "min_confidence": 0.85,
      "max_pending": 256,
      "candidate_limit": 6,
      "reward_base": 0.8,
      "reward_per_candidate": 0.1
    }
  },
  "blockchain": {
    "enabled": false,
    "chain_id": "w1z4rdv1510n-l1",
    "consensus": "poa",
    "bootstrap_peers": [],
    "node_role": "WORKER",
    "reward_policy": {
      "sensor_reward_weight": 1.0, "compute_reward_weight": 1.0,
      "energy_efficiency_weight": 1.0, "uptime_reward_weight": 1.0
    },
    "energy_efficiency": { "target_watts": 150.0, "efficiency_baseline": 1.0 },
    "attestation": { "endpoint": "", "required": false },
    "require_sensor_attestation": false
  },
  "compute": { "allow_gpu": true, "allow_quantum": false, "quantum_endpoints": [] },
  "cluster": { "enabled": true, "mode": "join", "min_nodes": 1, "openstack_minimal": false },
  "ledger": { "enabled": false, "backend": "local", "endpoint": "" },
  "sensors": [],
  "chain_spec": {
    "genesis_path": "$InstallDir\\chain\\genesis.json",
    "reward_contract_path": "$InstallDir\\chain\\reward_contract.json",
    "bridge_contract_path": "$InstallDir\\chain\\bridge_contract.json",
    "token_standard_path": "$InstallDir\\chain\\token_standard.json"
  },
  "energy_reporting": { "enabled": false, "sample_interval_secs": 30 }
}
"@
$config | Set-Content "$InstallDir\node_config.json" -Encoding UTF8
OK "node_config.json written"

# ── Step 4: Firewall rules ────────────────────────────────────────────────────
Step 4 "Opening firewall ports"

$rules = @(
    @{ Port = 8088;  Name = "W1z4rD P2P Gossip";  Desc = "Cluster peer-to-peer gossip" },
    @{ Port = 8090;  Name = "W1z4rD Node API";     Desc = "Node REST API for dashboard and tools" },
    @{ Port = 51611; Name = "W1z4rD SIGIL";         Desc = "Cluster ring, heartbeat, and election port" }
)
foreach ($r in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
    if ($existing) {
        WARN "Rule '$($r.Name)' already exists — skipping"
    } else {
        New-NetFirewallRule `
            -DisplayName $r.Name -Description $r.Desc `
            -Direction Inbound -Protocol TCP `
            -LocalPort $r.Port -Action Allow -Profile Private,Domain `
            | Out-Null
        OK "Port $($r.Port) open"
    }
}

# ── Step 5: Start scripts ─────────────────────────────────────────────────────
Step 5 "Creating start scripts"

$start_bat = "$InstallDir\start_node.bat"
@"
@echo off
cd /d "$InstallDir"
echo Starting W1z4rD node-3 (bootstrapping to $NodeHost`:$NodePort)...
start "" /B w1z4rd_node.exe --config node_config.json --addr 0.0.0.0:$ApiPort >> logs\node.log 2>&1
echo Node started. API on http://localhost:$ApiPort
echo Log: $InstallDir\logs\node.log
pause
"@ | Set-Content $start_bat -Encoding ASCII
OK "start_node.bat created"

$start_dash = "$InstallDir\start_dashboard.bat"
@"
@echo off
cd /d "$InstallDir"
start "" w1z4rd_dashboard.exe --node http://localhost:$ApiPort --api http://localhost:$ApiPort
"@ | Set-Content $start_dash -Encoding ASCII
OK "start_dashboard.bat created"

$start_all = "$InstallDir\start_all.bat"
@"
@echo off
cd /d "$InstallDir"
echo Starting W1z4rD node-3...
start "" /B w1z4rd_node.exe --config node_config.json --addr 0.0.0.0:$ApiPort >> logs\node.log 2>&1
timeout /t 3 /nobreak >nul
echo Opening dashboard...
start "" w1z4rd_dashboard.exe --node http://localhost:$ApiPort --api http://localhost:$ApiPort
echo Done. Node log: $InstallDir\logs\node.log
"@ | Set-Content $start_all -Encoding ASCII
OK "start_all.bat created (recommended: run this to start everything)"

# ── Step 6: Scheduled task (auto-start) ──────────────────────────────────────
if (-not $NoAutoStart) {
    Step 6 "Setting up auto-start on login"
    $task_name = "W1z4rD V1510n Node-3"
    Unregister-ScheduledTask -TaskName $task_name -Confirm:$false -ErrorAction SilentlyContinue
    $action   = New-ScheduledTaskAction `
        -Execute "$InstallDir\w1z4rd_node.exe" `
        -Argument "--config `"$InstallDir\node_config.json`" --addr 0.0.0.0:$ApiPort" `
        -WorkingDirectory $InstallDir
    $trigger  = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $task_name -Action $action -Trigger $trigger `
        -Settings $settings -RunLevel Highest -Force | Out-Null
    OK "Scheduled task '$task_name' registered (runs at login)"
} else {
    WARN "Skipped auto-start (-NoAutoStart)"
}

# ── Step 7: Launch ────────────────────────────────────────────────────────────
if (-not $SkipLaunch) {
    Step 7 "Starting node and dashboard"

    $node_proc = Start-Process -FilePath "$InstallDir\w1z4rd_node.exe" `
        -ArgumentList "--config `"$InstallDir\node_config.json`" --addr 0.0.0.0:$ApiPort" `
        -WorkingDirectory $InstallDir `
        -RedirectStandardOutput "$InstallDir\logs\node.log" `
        -RedirectStandardError  "$InstallDir\logs\node_err.log" `
        -WindowStyle Hidden -PassThru
    OK "Node started (PID $($node_proc.Id))"

    Start-Sleep -Seconds 3

    # Quick health check
    try {
        $resp = Invoke-WebRequest -Uri "http://localhost:$ApiPort/health" -TimeoutSec 5 -UseBasicParsing
        $health = $resp.Content | ConvertFrom-Json
        OK "Node healthy — id=$($health.node_id) status=$($health.status)"
    } catch {
        WARN "Health check timed out — node may still be starting (check logs\node.log)"
    }

    if (Test-Path "$InstallDir\w1z4rd_dashboard.exe") {
        Start-Process "$InstallDir\w1z4rd_dashboard.exe" `
            -ArgumentList "--node http://localhost:$ApiPort --api http://localhost:$ApiPort" `
            -WorkingDirectory $InstallDir
        OK "Dashboard launched"
    }
} else {
    WARN "Skipped launch (-SkipLaunch). Run start_all.bat to start the node."
}

# ── Done ──────────────────────────────────────────────────────────────────────
Banner "Installation Complete"
Write-Host "  Install dir : $InstallDir" -ForegroundColor Cyan
Write-Host "  Bootstrap   : $NodeHost`:$NodePort" -ForegroundColor White
Write-Host "  Node API    : http://localhost:$ApiPort" -ForegroundColor White
Write-Host "  Start all   : $start_all" -ForegroundColor White
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "   1. Confirm node appears in the Dashboard > Cluster tab" -ForegroundColor White
Write-Host "   2. On the MAIN PC: accept/approve node-3 in the cluster panel" -ForegroundColor White
Write-Host "   3. Training data will sync from the main node automatically" -ForegroundColor White
Write-Host ""
