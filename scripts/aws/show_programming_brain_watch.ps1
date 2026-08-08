param(
    [string]$SessionId = "019f5dbf-ea56-7410-8735-83ecac08f1a7"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "runtime\programming-brain-codex-watch"
$Watcher = Join-Path $PSScriptRoot "watch_programming_brain.py"
$Activity = Join-Path $RunRoot "activity.log"
$State = Join-Path $RunRoot "state.json"
$PidFile = Join-Path $RunRoot "watch.pid"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Find-Watcher {
    @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(w)?\.exe$' -and
        $_.CommandLine -like '*watch_programming_brain.py*'
    })
}

$running = Find-Watcher
if ($running.Count -eq 0) {
    $python = (Get-Command python).Source
    $arguments = @(
        $Watcher, "--session-id", $SessionId,
        "--poll-seconds", "300",
        "--stability-polls", "2",
        "--retry-cooldown", "1800"
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RunRoot "watch.stdout.log") `
        -RedirectStandardError (Join-Path $RunRoot "watch.stderr.log") `
        -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
    $watcherPid = $process.Id
    $started = $true
} else {
    $watcherPid = $running[0].ProcessId
    $started = $false
}

Clear-Host
Write-Host "WIZARD VISION - PROGRAMMING BRAIN SUPERVISOR" -ForegroundColor Cyan
Write-Host ""
if ($started) {
    Write-Host "Started background watcher PID $watcherPid." -ForegroundColor Green
} else {
    Write-Host "Background watcher is already running as PID $watcherPid." -ForegroundColor Green
}
Write-Host "Closing this window closes only the tail; supervision keeps running."
Write-Host "AWS is polled every five minutes. Codex appears below only on an alarm."
Write-Host ""

if (Test-Path -LiteralPath $State) {
    try {
        $snapshot = Get-Content -LiteralPath $State -Raw | ConvertFrom-Json
        $probe = $snapshot.last_probe
        $status = $probe.status
        Write-Host ("Last state : {0} / {1}" -f $status.phase, $status.state)
        Write-Host ("Rows       : {0} -> guarded target {1}" -f `
            $status.durable_next_row, $status.block_target_row)
        $curriculum = $probe.curriculum
        if ($null -ne $curriculum) {
            Write-Host ("Curriculum : {0:N0} / {1:N0} durable ({2:N0} forward remain)" -f `
                $curriculum.durable_processed_rows, $curriculum.total_rows, `
                $curriculum.forward_remaining_rows)
            Write-Host ("Admission  : {0:N0} accepted; {1:N0} quarantined for replay" -f `
                $curriculum.accepted_rows, $curriculum.deferred_rows)
            Write-Host ("Outstanding: at least {0:N0} rows including known replay" -f `
                $curriculum.minimum_outstanding_rows)
        }
        Write-Host ("Processes  : supervisor={0}, wrapper={1}, worker={2}" -f `
            $probe.supervisor_count, $probe.wrapper_count, $probe.worker_count)
        Write-Host ("Decision   : {0} - {1}" -f `
            $snapshot.last_decision.kind, $snapshot.last_decision.reason)
        Write-Host ""
    } catch {
        Write-Host "The first status snapshot is still being written." -ForegroundColor Yellow
    }
}

if (-not (Test-Path -LiteralPath $Activity)) {
    New-Item -ItemType File -Path $Activity | Out-Null
}
Write-Host "LIVE ACTIVITY" -ForegroundColor Yellow
Write-Host "-------------"
Get-Content -LiteralPath $Activity -Tail 60 -Wait
