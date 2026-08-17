<#
    Register the W1z4rD supervisor as a Windows Scheduled Task.

    Why a Scheduled Task instead of the Startup-folder VBS that
    install_startup.cmd drops:

      * RestartOnFailure -- the VBS launcher fires exactly once at logon.
        If the supervisor dies (or was never started because the machine
        rebooted straight into a locked session) nothing brings it back,
        which is how the stack ended up with a 3-day-stale production
        heartbeat and a supervisor whose last log line was months old.
      * The task also runs at boot, not just at interactive logon.
      * ExecutionTimeLimit 0 -- never kill the long-running watchdog.

    Per-user task, so no admin rights are needed. Remove with
    uninstall_supervisor_task.ps1.
#>
param(
    [string]$TaskName = "W1z4rDSupervisor",
    # Registering at the root path "\" needs elevation on this box; a
    # per-user subfolder does not. Keep the default non-elevated so the
    # installer works from a normal shell.
    [string]$TaskPath = "\W1z4rD\",
    [string]$Python   = "",
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Supervisor  = Join-Path $ProjectRoot "scripts\w1z4rd_supervisor.py"

if (-not (Test-Path -LiteralPath $Supervisor)) {
    throw "supervisor script not found: $Supervisor"
}

# Prefer pythonw.exe so no console window flashes on every logon; fall
# back to python.exe when only that is present.
if (-not $Python) {
    $candidates = @(
        "C:\Python313\pythonw.exe",
        "C:\Python313\python.exe"
    ) + @(
        (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source,
        (Get-Command python.exe  -ErrorAction SilentlyContinue).Source
    )
    $Python = $candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
}
if (-not $Python) { throw "no python interpreter found; pass -Python <path>" }

Write-Host "Project    : $ProjectRoot"
Write-Host "Supervisor : $Supervisor"
Write-Host "Interpreter: $Python"

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument ('"{0}"' -f $Supervisor) -WorkingDirectory $ProjectRoot

# -AtStartup requires elevation (it registers under the local machine), so
# try logon+startup first and fall back to logon-only when not running as
# admin. Logon-only still covers the normal desktop case.
$isAdmin = ([Security.Principal.WindowsPrincipal] `
    [Security.Principal.WindowsIdentity]::GetCurrent()
).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

$logonTrigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
if ($isAdmin) {
    $triggers = @($logonTrigger, (New-ScheduledTaskTrigger -AtStartup))
} else {
    $triggers = @($logonTrigger)
    Write-Host "Not elevated: registering an at-logon trigger only." -ForegroundColor Yellow
    Write-Host "  (Re-run from an admin shell to also start before logon.)" -ForegroundColor DarkGray
}

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
    -MultipleInstances IgnoreNew

# Principal: an explicit S4U/Interactive principal needs elevation to
# register ("Access is denied" from a normal shell), so when we are not
# admin we omit -Principal entirely and let the task inherit the current
# interactive user. That still runs the supervisor at logon under this
# account, which is all it needs.
$principal = $null
if ($isAdmin) {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType S4U -RunLevel Limited
}

if (Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -ErrorAction SilentlyContinue) {
    Write-Host "Existing task found -- unregistering first."
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath -Confirm:$false
}

$desc = "W1z4rD V1510n supervisor: keeps the node, Django panel (:8001), " +
        "R3V3N!R trading services and training alive."
$register = @{
    TaskName    = $TaskName
    TaskPath    = $TaskPath
    Action      = $action
    Trigger     = $triggers
    Settings    = $settings
    Description = $desc
}
if ($principal) { $register['Principal'] = $principal }
Register-ScheduledTask @register -ErrorAction Stop | Out-Null

Write-Host ""
Write-Host "Registered scheduled task '$TaskPath$TaskName'." -ForegroundColor Green
Get-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath |
    Select-Object TaskName, TaskPath, State | Format-Table -AutoSize

if ($StartNow) {
    $already = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
        Where-Object { $_.CommandLine -like "*w1z4rd_supervisor.py*" }
    if ($already) {
        Write-Host "Supervisor already running as PID $($already.ProcessId); leaving it alone."
    } else {
        Start-ScheduledTask -TaskName $TaskName -TaskPath $TaskPath
        Start-Sleep -Seconds 4
        $proc = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" |
            Where-Object { $_.CommandLine -like "*w1z4rd_supervisor.py*" }
        if ($proc) {
            Write-Host "Supervisor started as PID $($proc.ProcessId)." -ForegroundColor Green
        } else {
            Write-Warning "Task started but no supervisor process observed yet; check the log."
        }
    }
}

Write-Host ""
Write-Host "Log    : D:\w1z4rdv1510n-data\training\supervisor.log"
Write-Host "Config : $ProjectRoot\supervisor.toml"
Write-Host "Remove : scripts\uninstall_supervisor_task.ps1"
