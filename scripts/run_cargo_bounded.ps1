param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$CargoArgs,
    [int]$MinimumAvailableMb = 8192,
    [int]$BuildJobs = 1
)

. (Join-Path $PSScriptRoot "windows_memory.ps1")
$available = Get-WizardAvailableMemoryMb

if ($available -lt $MinimumAvailableMb) {
    throw "Refusing cargo: only $([math]::Round($available)) MB available; minimum is $MinimumAvailableMb MB."
}

$env:CARGO_BUILD_JOBS = [string][math]::Max(1, $BuildJobs)
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'

Write-Host "cargo $($CargoArgs -join ' ') (jobs=$env:CARGO_BUILD_JOBS, available=$([math]::Round($available)) MB)"
& cargo @CargoArgs
exit $LASTEXITCODE
