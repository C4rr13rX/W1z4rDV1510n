param(
    [string]$BrainDir = "runtime/brains/programming-integrated-20260713/brain",
    [int]$RequiredAvailableMb = 8192,
    [int]$StableSamples = 3,
    [int]$PollSeconds = 15,
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_memory.ps1")
$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$brain = [System.IO.Path]::GetFullPath((Join-Path $repo $BrainDir))
$source = Join-Path $brain "brain.bin"
$log = Join-Path $brain "brain-migration-admission.log"
$watcherPid = Join-Path $brain "brain-migration-admission.pid"

if (-not $brain.StartsWith($repo + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Brain directory must stay inside the repository: $brain"
}
if (-not (Test-Path -LiteralPath $source)) {
    throw "Legacy brain checkpoint not found: $source"
}
Set-Content -LiteralPath $watcherPid -Value $PID -Encoding ascii

function Write-AdmissionLog([string]$Message) {
    Add-Content -LiteralPath $log -Value ("{0:o} {1}" -f (Get-Date), $Message) -Encoding utf8
}

try {
    $stable = 0
    while ($true) {
        $migrationPidFile = Join-Path $brain "brain-migrate.pid"
        if (Test-Path -LiteralPath $migrationPidFile) {
            $existingPid = [int](Get-Content -LiteralPath $migrationPidFile -Raw)
            if ($null -ne (Get-Process -Id $existingPid -ErrorAction SilentlyContinue)) {
                Write-AdmissionLog "migration pid=$existingPid already running; watcher exiting"
                break
            }
        }

        $available = Get-WizardAvailableMemoryMb
        if ($available -ge $RequiredAvailableMb) {
            $stable += 1
        } else {
            $stable = 0
        }
        if ($stable -lt [Math]::Max(1, $StableSamples)) {
            Start-Sleep -Seconds ([Math]::Max(5, $PollSeconds))
            continue
        }

        Write-AdmissionLog ("admitted migration after {0} stable samples; available_mb={1:N0}" -f $stable, $available)
        if (-not $SkipBuild) {
            Push-Location $repo
            try {
                & (Join-Path $repo "scripts/run_cargo_bounded.ps1") `
                    -CargoArgs @("build", "-p", "w1z4rdv1510n-node", "--bin", "w1z4rd_brain_migrate") `
                    *>> $log
                if ($LASTEXITCODE -ne 0) {
                    Write-AdmissionLog "bounded build returned $LASTEXITCODE; returning to admission wait"
                    $stable = 0
                    continue
                }
            } catch {
                Write-AdmissionLog "bounded build refused: $($_.Exception.Message)"
                $stable = 0
                continue
            } finally {
                Pop-Location
            }
        }

        $exe = Join-Path $repo "target/debug/w1z4rd_brain_migrate.exe"
        $stdout = Join-Path $brain "brain-migrate.stdout.log"
        $stderr = Join-Path $brain "brain-migrate.stderr.log"
        $migration = Start-Process -FilePath $exe -ArgumentList @($brain) `
            -WorkingDirectory $repo -WindowStyle Hidden `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
        try { $migration.PriorityClass = "BelowNormal" } catch {}
        Set-Content -LiteralPath $migrationPidFile -Value $migration.Id -Encoding ascii

        $memoryLog = Join-Path $brain "brain-migrate-memory.log"
        $monitorScript = Join-Path $repo "scripts/monitor_migration_memory.ps1"
        $monitorArgs = @(
            "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $monitorScript,
            "-PidFile", $migrationPidFile, "-LogPath", $memoryLog,
            "-AbortPrivateMb", "8192"
        )
        $monitor = Start-Process -FilePath "powershell.exe" -ArgumentList $monitorArgs `
            -WorkingDirectory $repo -WindowStyle Hidden -PassThru
        Set-Content -LiteralPath (Join-Path $brain "brain-migrate-monitor.pid") `
            -Value $monitor.Id -Encoding ascii

        Start-Sleep -Seconds 10
        if ($null -ne (Get-Process -Id $migration.Id -ErrorAction SilentlyContinue)) {
            Write-AdmissionLog "migration pid=$($migration.Id) monitor_pid=$($monitor.Id) running"
            break
        }
        Write-AdmissionLog "migration exited during admission; returning to wait"
        $stable = 0
    }
} finally {
    Remove-Item -LiteralPath $watcherPid -Force -ErrorAction SilentlyContinue
}
