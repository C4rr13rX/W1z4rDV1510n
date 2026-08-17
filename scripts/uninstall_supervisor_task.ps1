<#
    Remove the W1z4rD supervisor Scheduled Task registered by
    install_supervisor_task.ps1.

    Leaves any running supervisor process alone unless -StopProcess is
    given, so you can swap the task definition without interrupting a
    live training run.
#>
param(
    [string]$TaskName = "W1z4rDSupervisor",
    [string]$TaskPath = "\W1z4rD\",
    [switch]$StopProcess
)

$ErrorActionPreference = "Stop"

if (Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
    Write-Host "Unregistered scheduled task '$TaskPath$TaskName'." -ForegroundColor Green
} else {
    Write-Host "No scheduled task named '$TaskPath$TaskName'."
}

if ($StopProcess) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -like "*w1z4rd_supervisor.py*" }
    foreach ($p in $procs) {
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped supervisor PID $($p.ProcessId)."
    }
    if (-not $procs) { Write-Host "No running supervisor process found." }
} else {
    Write-Host "Any running supervisor process was left alone (-StopProcess to kill it)."
}
