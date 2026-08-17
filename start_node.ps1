# Start the W1z4rD node with polite resource behavior.
#
# Politeness contract:
#  - W1Z4RD_TIER_MIN_SYS_AVAIL_MB: the node must leave at least this much
#    system RAM for everything else. As available RAM approaches the floor,
#    the tier orchestrator evicts neurons to the SSD cold tier aggressively
#    (demand paging brings them back when predictions need them).
#  - W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS: learned atoms/concepts are snapshotted
#    to brain-data\brain.bin on this cadence so restarts never lose the fabric.
param(
    [string]$Addr = "127.0.0.1:8090",
    [int]$MinSysAvailMb = 4096,
    [int]$CheckpointSecs = 900,
    # Explicit brain directory. Without it the node resolves
    # <W1Z4RDV1510N_DATA_DIR>/brain when that env var is set (bin\start_node.cmd
    # and the supervisor both set it), so the same binary would load a
    # different brain depending on which launcher started it. Pin it here so
    # every entry point agrees on where the fabric lives.
    [string]$BrainDir = "$PSScriptRoot\brain-data",
    # Specialized-pool brain: ohlcv(1)/news(2)/outcome(3) sensory+action pools
    # with prediction-error feedback loops. Pass empty strings to fall back to
    # the default multimodal topology.
    [string]$Identity = "brains\market_small.identity.toml",
    [string]$Deployment = "brains\market_django.deployment.toml"
)

$env:W1Z4RD_TIER_MIN_SYS_AVAIL_MB = "$MinSysAvailMb"
$env:W1Z4RD_BRAIN_AUTOCHECKPOINT_SECS = "$CheckpointSecs"
if ($BrainDir) { $env:W1Z4RD_NODE_BRAIN_DIR = $BrainDir }
if ($Identity)   { $env:W1Z4RD_BRAIN_IDENTITY   = (Join-Path $PSScriptRoot $Identity) }
if ($Deployment) { $env:W1Z4RD_BRAIN_DEPLOYMENT = (Join-Path $PSScriptRoot $Deployment) }

$root = $PSScriptRoot
$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Start-Process -FilePath (Join-Path $root "bin\w1z4rd_node.exe") `
    -ArgumentList "--config","node_config.json","api","--addr",$Addr `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "node-$stamp.out.log") `
    -RedirectStandardError  (Join-Path $logDir "node-$stamp.err.log")
Write-Host "w1z4rd_node starting on $Addr (RAM floor ${MinSysAvailMb}MB, checkpoint every ${CheckpointSecs}s, logs: logs\node-$stamp.*.log)"
