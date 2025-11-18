param(
    [string]$ConfigPath,
    [string]$LogDir,
    [switch]$DryRun,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir "..") | Select-Object -First 1 | ForEach-Object { $_.Path }
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $repoRoot "data\\synthetic_run_config.json"
}
if (-not $LogDir) {
    $LogDir = Join-Path $repoRoot "logs"
}

if (-not (Test-Path $ConfigPath)) {
    throw "Config file not found: $ConfigPath"
}

$configFullPath = Resolve-Path $ConfigPath | Select-Object -First 1 | ForEach-Object { $_.Path }
$logDirFull = Resolve-Path $LogDir -ErrorAction SilentlyContinue
if (-not $logDirFull) {
    $logDirFull = New-Item -ItemType Directory -Path $LogDir -Force | Select-Object -ExpandProperty FullName
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDirFull "predict_state_$timestamp.log"
$args = @("run", "--", "--config", $configFullPath)

if ($DryRun) {
    Write-Host "[DRY-RUN] cargo $($args -join ' ')" -ForegroundColor Cyan
    Write-Host "[DRY-RUN] log file would be: $logPath" -ForegroundColor Cyan
    return
}

$cmdCommand = "cd /d `"$repoRoot`" && cargo $($args -join ' ') > `"$logPath`" 2>&1"
$process = Start-Process -FilePath "cmd.exe" `
    -ArgumentList @("/c", $cmdCommand) `
    -WindowStyle Hidden `
    -PassThru

$runMetadata = @{
    pid          = $process.Id
    config_path  = $configFullPath
    log_path     = $logPath
    started_at   = (Get-Date).ToString("o")
    command_line = "cargo $($args -join ' ')"
}

$metadataPath = Join-Path $logDirFull "latest_run.json"
$runMetadata | ConvertTo-Json | Set-Content -Path $metadataPath

if (-not $Quiet) {
    Write-Host ("Started predict_state (PID {0}) logging to {1}" -f $process.Id, $logPath)
    Write-Host ("Metadata saved to {0}" -f $metadataPath)
}
