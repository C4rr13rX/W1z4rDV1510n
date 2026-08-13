param(
    [string]$BrainDir = "brain-data",
    [string]$StagingDir = "runtime/market-brain-wbrain-migration",
    [string]$NodeAddress = "127.0.0.1:8090",
    [int]$RequiredAvailableMb = 8192,
    [int]$DiskReserveGb = 8,
    [double]$EstimatedExpansionFactor = 1.25,
    [int]$CheckpointTimeoutSeconds = 1800,
    [switch]$Execute,
    [switch]$SkipBuild
)

# Prepare a neuron-addressable copy of the local market brain without ever
# publishing a partial container into the live brain directory. The legacy
# node is checkpointed and stopped only after every preflight passes, and is
# always restarted from the untouched live directory in finally. Promotion is
# intentionally a separate admission step after cold-open and behavior gates.

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_memory.ps1")

$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$brain = [System.IO.Path]::GetFullPath((Join-Path $repo $BrainDir))
$staging = [System.IO.Path]::GetFullPath((Join-Path $repo $StagingDir))
$source = Join-Path $brain "brain.bin"
$stagedSource = Join-Path $staging "brain.bin"
$destination = Join-Path $staging "brain.wbrain"
$manifestPath = Join-Path $staging "source-manifest.json"
$logPath = Join-Path $staging "migration.log"
$pidPath = Join-Path $staging "brain-migrate.pid"
$monitorPidPath = Join-Path $staging "brain-migrate-monitor.pid"
$identity = Join-Path $repo "brains\market_small.identity.toml"
$deployment = Join-Path $repo "brains\market_django.deployment.toml"

function Assert-RepositoryPath([string]$Path, [string]$Label) {
    $prefix = $repo + [System.IO.Path]::DirectorySeparatorChar
    if (-not $Path.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "$Label must stay inside the repository: $Path"
    }
}

function Get-MarketNodeMatches {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -ieq "w1z4rd_node.exe" -and
        [string]$_.CommandLine -like "*api --addr $NodeAddress*"
    })
}

function Wait-Http([string]$Url, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 5
            if ($response.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 2
    }
    return $false
}

function Get-SourceManifest {
    $cold = Join-Path $brain "cold"
    $files = @((Get-Item -LiteralPath $source))
    if (Test-Path -LiteralPath $cold) {
        $files += @(Get-ChildItem -LiteralPath $cold -File | Sort-Object Name)
    }
    $rows = @($files | ForEach-Object {
        [ordered]@{
            relative_path = [System.IO.Path]::GetRelativePath($brain, $_.FullName)
            length = [int64]$_.Length
            last_write_utc_ticks = [int64]$_.LastWriteTimeUtc.Ticks
        }
    })
    [ordered]@{
        schema = 1
        brain_dir = $brain
        files = $rows
        total_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    }
}

function Stage-ImmutableSource($Manifest) {
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    if (Test-Path -LiteralPath $manifestPath) {
        $existing = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        $expected = $Manifest | ConvertTo-Json -Depth 6 -Compress
        $actual = $existing | ConvertTo-Json -Depth 6 -Compress
        if ($actual -ne $expected) {
            throw "Staged migration belongs to a different legacy snapshot; preserve it for diagnosis and choose a new StagingDir."
        }
        return
    }
    foreach ($file in $Manifest.files) {
        $from = Join-Path $brain $file.relative_path
        $to = Join-Path $staging $file.relative_path
        New-Item -ItemType Directory -Path (Split-Path -Parent $to) -Force | Out-Null
        New-Item -ItemType HardLink -Path $to -Target $from | Out-Null
    }
    $Manifest | ConvertTo-Json -Depth 6 |
        Set-Content -LiteralPath $manifestPath -Encoding utf8
}

Assert-RepositoryPath $brain "BrainDir"
Assert-RepositoryPath $staging "StagingDir"
if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
    throw "Legacy market brain not found: $source"
}
if (-not (Test-Path -LiteralPath $identity -PathType Leaf)) {
    throw "Market identity not found: $identity"
}
if (-not (Test-Path -LiteralPath $deployment -PathType Leaf)) {
    throw "Market deployment not found: $deployment"
}

$manifest = Get-SourceManifest
$existingBytes = if (Test-Path -LiteralPath $destination) {
    [int64](Get-Item -LiteralPath $destination).Length
} else { 0L }
$estimatedBytes = [int64][Math]::Ceiling(
    [double]$manifest.total_bytes * [Math]::Max(1.0, $EstimatedExpansionFactor)
)
$remainingBytes = [Math]::Max(0L, $estimatedBytes - $existingBytes)
$requiredFreeBytes = $remainingBytes + ([int64]$DiskReserveGb * 1GB)
$drive = [System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($staging))
$availableMb = Get-WizardAvailableMemoryMb
$matches = @(Get-MarketNodeMatches)
$preflight = [ordered]@{
    ready = ($availableMb -ge $RequiredAvailableMb -and
             $drive.AvailableFreeSpace -ge $requiredFreeBytes -and
             $matches.Count -eq 1)
    execute_requested = [bool]$Execute
    live_brain_dir = $brain
    staging_dir = $staging
    source_bytes = [int64]$manifest.total_bytes
    existing_destination_bytes = $existingBytes
    estimated_destination_bytes = $estimatedBytes
    required_free_bytes = $requiredFreeBytes
    available_disk_bytes = [int64]$drive.AvailableFreeSpace
    required_available_memory_mb = $RequiredAvailableMb
    available_memory_mb = [int64]$availableMb
    verified_node_count = $matches.Count
    verified_node_pid = if ($matches.Count -eq 1) { [int]$matches[0].ProcessId } else { $null }
    live_source_untouched = $true
    promotion_performed = $false
}
$preflight | ConvertTo-Json -Depth 5

if (-not $Execute) { return }
if (-not $preflight.ready) {
    throw "Market brain migration preflight failed; no process or brain file was changed."
}

if (-not $SkipBuild) {
    & (Join-Path $repo "scripts\run_cargo_bounded.ps1") `
        -MinimumAvailableMb $RequiredAvailableMb -BuildJobs 1 `
        -CargoArgs @("build", "-p", "w1z4rdv1510n-node", "--bin", "w1z4rd_brain_migrate")
    if ($LASTEXITCODE -ne 0) { throw "Bounded migrator build failed: $LASTEXITCODE" }
}
$migrator = Join-Path $repo "target\debug\w1z4rd_brain_migrate.exe"
if (-not (Test-Path -LiteralPath $migrator -PathType Leaf)) {
    throw "Migration executable not found: $migrator"
}

$nodeWasStopped = $false
$migrationSucceeded = $false
New-Item -ItemType Directory -Path $staging -Force | Out-Null
try {
    # A successful checkpoint is required before the exact live owner stops.
    Invoke-WebRequest -UseBasicParsing -Method Post `
        -Uri "http://$NodeAddress/neuro/checkpoint" -ContentType "application/json" `
        -Body "{}" -TimeoutSec $CheckpointTimeoutSeconds | Out-Null
    $manifest = Get-SourceManifest

    Invoke-WebRequest -UseBasicParsing -Method Post `
        -Uri "http://$NodeAddress/shutdown" -ContentType "application/json" `
        -Body "{}" -TimeoutSec 30 | Out-Null
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline -and @(Get-MarketNodeMatches).Count -gt 0) {
        Start-Sleep -Seconds 2
    }
    if (@(Get-MarketNodeMatches).Count -ne 0) {
        throw "Verified node did not stop gracefully; refusing forced termination."
    }
    $nodeWasStopped = $true

    Stage-ImmutableSource $manifest
    $env:W1Z4RD_BRAIN_IDENTITY = $identity
    $env:W1Z4RD_BRAIN_DEPLOYMENT = $deployment
    $stdout = Join-Path $staging "brain-migrate.stdout.log"
    $stderr = Join-Path $staging "brain-migrate.stderr.log"
    $migration = Start-Process -FilePath $migrator -ArgumentList @($staging) `
        -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -PassThru
    try { $migration.PriorityClass = "BelowNormal" } catch {}
    Set-Content -LiteralPath $pidPath -Value $migration.Id -Encoding ascii

    $monitorArgs = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        (Join-Path $repo "scripts\monitor_migration_memory.ps1"),
        "-PidFile", $pidPath, "-LogPath", (Join-Path $staging "memory.log"),
        "-AbortPrivateMb", "8192"
    )
    $monitor = Start-Process -FilePath "powershell.exe" -ArgumentList $monitorArgs `
        -WorkingDirectory $repo -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath $monitorPidPath -Value $monitor.Id -Encoding ascii
    $migration.WaitForExit()
    if ($migration.ExitCode -ne 0) {
        throw "Migration exited $($migration.ExitCode); partial staging remains resumable."
    }
    if (-not (Test-Path -LiteralPath $destination -PathType Leaf)) {
        throw "Migrator reported success without publishing $destination"
    }
    $migrationSucceeded = $true
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "{0:o} migration_complete bytes={1} live_source_untouched=true promotion_performed=false" -f `
        (Get-Date), (Get-Item -LiteralPath $destination).Length
    )
} finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
    if ($nodeWasStopped) {
        & (Join-Path $repo "start_node.ps1") | Out-Host
        if (-not (Wait-Http "http://$NodeAddress/health" 180)) {
            throw "Untouched legacy market node did not recover after migration attempt."
        }
    }
}

if ($migrationSucceeded) {
    Write-Host "Staged migration completed and legacy production node recovered." -ForegroundColor Green
    Write-Host "No promotion occurred. Validate the staged .wbrain with the current node binary before switch-over."
}
