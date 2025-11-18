param(
    [string]$LogPath,
    [string]$MetadataPath,
    [int]$Tail = 40,
    [int]$RefreshIntervalMs = 500,
    [int]$StopAfterSeconds = 0,
    [int]$WaitTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Resolve-Path (Join-Path $scriptDir "..") | Select-Object -First 1 | ForEach-Object { $_.Path }
if (-not $MetadataPath) {
    $MetadataPath = Join-Path $repoRoot "logs\\latest_run.json"
}

function Resolve-AbsolutePath {
    param(
        [string]$PathValue,
        [string]$BasePath
    )
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $PathValue))
}

function Resolve-LogPath {
    param(
        [string]$LogPath,
        [string]$MetadataPath
    )

    if ($LogPath) {
        return Resolve-AbsolutePath -PathValue $LogPath -BasePath $repoRoot
    }

    if (-not (Test-Path $MetadataPath)) {
        throw "Log path not provided and metadata file not found ($MetadataPath)."
    }

    $meta = Get-Content $MetadataPath | ConvertFrom-Json
    if (-not $meta.log_path) {
        throw "Metadata file $MetadataPath does not contain a log_path entry."
    }
    return Resolve-AbsolutePath -PathValue $meta.log_path -BasePath $repoRoot
}

$logFullPath = Resolve-LogPath -LogPath $LogPath -MetadataPath $MetadataPath

$waitDeadline = (Get-Date).AddSeconds($WaitTimeoutSeconds)
while (-not (Test-Path $logFullPath)) {
    if ((Get-Date) -gt $waitDeadline) {
        throw "Timed out waiting for log file $logFullPath"
    }
    Start-Sleep -Milliseconds 500
}

Write-Host ("Tailing log: {0}" -f $logFullPath)
if ($Tail -gt 0) {
    Get-Content -Path $logFullPath -Tail $Tail
}

$cursor = (Get-Item $logFullPath).Length
$deadline = if ($StopAfterSeconds -gt 0) { (Get-Date).AddSeconds($StopAfterSeconds) } else { $null }

while ($true) {
    if ($deadline -and (Get-Date) -ge $deadline) {
        Write-Host "StopAfterSeconds reached; exiting tail."
        break
    }
    Start-Sleep -Milliseconds $RefreshIntervalMs
    $currentLength = (Get-Item $logFullPath).Length
    if ($currentLength -lt $cursor) {
        Write-Warning "Log file size shrank (new run?). Resetting cursor."
        $cursor = 0
    }
    if ($currentLength -le $cursor) {
        continue
    }
    $fs = $null
    $sr = $null
    try {
        $fs = [System.IO.File]::Open($logFullPath, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        $fs.Seek($cursor, [System.IO.SeekOrigin]::Begin) | Out-Null
        $sr = New-Object System.IO.StreamReader($fs)
        while (-not $sr.EndOfStream) {
            $line = $sr.ReadLine()
            if ($line -ne $null) {
                Write-Host $line
            }
        }
        $cursor = $fs.Position
    }
    finally {
        if ($sr) { $sr.Dispose() }
        elseif ($fs) { $fs.Dispose() }
    }
}
