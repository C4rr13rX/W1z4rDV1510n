param(
    [Parameter(Mandatory = $true)]
    [string]$PidFile,
    [int]$IntervalSeconds = 5,
    [int]$TrimIntervalSeconds = 30,
    [int]$PauseAvailableMb = 4608,
    [int]$ResumeAvailableMb = 5120,
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$pidPath = [System.IO.Path]::GetFullPath($PidFile)
if (-not (Test-Path -LiteralPath $pidPath)) {
    throw "Migration PID file not found: $pidPath"
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

public static class WizardWorkingSet {
    [DllImport("psapi.dll", SetLastError = true)]
    public static extern bool EmptyWorkingSet(IntPtr process);

    [DllImport("ntdll.dll")]
    public static extern int NtSuspendProcess(IntPtr process);

    [DllImport("ntdll.dll")]
    public static extern int NtResumeProcess(IntPtr process);
}
'@

$migrationPid = [int](Get-Content -LiteralPath $pidPath)
$interval = [Math]::Max(1, $IntervalSeconds)
$trimInterval = [Math]::Max($interval, $TrimIntervalSeconds)
$resumeFloor = [Math]::Max($PauseAvailableMb + 512, $ResumeAvailableMb)
$lastLog = [DateTime]::MinValue
$lastTrim = [DateTime]::MinValue
$suspended = $false

# A prior monitor may have been terminated while it owned a suspension.
# Normalize this migration process to running before taking ownership.
$initialProcess = Get-Process -Id $migrationPid -ErrorAction SilentlyContinue
if ($null -ne $initialProcess) {
    [void][WizardWorkingSet]::NtResumeProcess($initialProcess.Handle)
}

try {
    while ($true) {
        $process = Get-Process -Id $migrationPid -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            break
        }

        $now = Get-Date
        if (-not $suspended -and ($now - $lastTrim).TotalSeconds -ge $trimInterval) {
            [void][WizardWorkingSet]::EmptyWorkingSet($process.Handle)
            $lastTrim = $now
        }
        $availableMb = (
            Get-Counter "\Memory\Available MBytes" -ErrorAction Stop
        ).CounterSamples[0].CookedValue
        $stateChanged = $false
        if (-not $suspended -and $availableMb -le $PauseAvailableMb) {
            $status = [WizardWorkingSet]::NtSuspendProcess($process.Handle)
            if ($status -ne 0) {
                throw "NtSuspendProcess failed with NTSTATUS $status"
            }
            $suspended = $true
            [void][WizardWorkingSet]::EmptyWorkingSet($process.Handle)
            $stateChanged = $true
        } elseif ($suspended -and $availableMb -ge $resumeFloor) {
            $status = [WizardWorkingSet]::NtResumeProcess($process.Handle)
            if ($status -ne 0) {
                throw "NtResumeProcess failed with NTSTATUS $status"
            }
            $suspended = $false
            $stateChanged = $true
        }

        if ($LogPath -and (
            $stateChanged -or ($now - $lastLog).TotalSeconds -ge 60
        )) {
            $state = if ($suspended) { "paused" } else { "running" }
            $line = "{0:o} pid={1} state={2} working_set_mb={3:N0} available_mb={4:N0}" -f `
                $now, $migrationPid, $state, ($process.WorkingSet64 / 1MB), $availableMb
            Add-Content -LiteralPath $LogPath -Value $line -Encoding utf8
            $lastLog = $now
        }
        Start-Sleep -Seconds $interval
    }
} finally {
    if ($suspended) {
        $process = Get-Process -Id $migrationPid -ErrorAction SilentlyContinue
        if ($null -ne $process) {
            [void][WizardWorkingSet]::NtResumeProcess($process.Handle)
        }
    }
}
