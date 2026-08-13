param(
    [string]$AwsProfile = "FountainServer",
    [string]$InstanceId = "i-0d7a6deeb0ead2dfc",
    [int]$Port = 18096
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunRoot = Join-Path $ProjectRoot "runtime\programming-brain-proxy"
$Proxy = Join-Path $PSScriptRoot "programming_brain_proxy.py"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalAddress -in @("127.0.0.1", "::1") } |
    Select-Object -First 1
if ($listener) {
    $process = Get-CimInstance Win32_Process -Filter ("ProcessId={0}" -f $listener.OwningProcess)
    if ($process.CommandLine -notlike "*programming_brain_proxy.py*") {
        throw "Port $Port is already owned by an unrelated process."
    }
    Write-Host "Programming brain proxy already running as PID $($listener.OwningProcess)."
    exit 0
}

$python = (Get-Command python).Source
$process = Start-Process -FilePath $python -ArgumentList @(
    $Proxy, "--profile", $AwsProfile, "--instance-id", $InstanceId, "--port", $Port
) -WorkingDirectory $ProjectRoot -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $RunRoot "proxy.stdout.log") `
  -RedirectStandardError (Join-Path $RunRoot "proxy.stderr.log") -PassThru

for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 250
    $listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -eq "127.0.0.1" } | Select-Object -First 1
    if ($listener) {
        Set-Content -LiteralPath (Join-Path $RunRoot "proxy.pid") -Value $process.Id -Encoding ascii
        Write-Host "Programming brain proxy started as PID $($process.Id) on 127.0.0.1:$Port."
        exit 0
    }
    if ($process.HasExited) {
        throw "Programming brain proxy exited during startup. See runtime/programming-brain-proxy."
    }
}
throw "Programming brain proxy did not listen within five seconds."
