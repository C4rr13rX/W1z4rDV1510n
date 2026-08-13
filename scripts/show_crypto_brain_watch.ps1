param(
    [int]$StopAfterSeconds = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunRoot = Join-Path $ProjectRoot "runtime\market-evolution"
$Events = Join-Path $RunRoot "events.jsonl"
$TailScript = Join-Path $PSScriptRoot "tail_simulation_logs.ps1"

function Read-JsonIfPresent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

Clear-Host
Write-Host "WIZARD VISION - CRYPTO BRAIN" -ForegroundColor Cyan
Write-Host ""

$supervisor = Read-JsonIfPresent (Join-Path $RunRoot "supervisor_status.json")
$status = Read-JsonIfPresent (Join-Path $RunRoot "status.json")
$admission = Read-JsonIfPresent (Join-Path $RunRoot "ghost_stack_admission.json")
$champion = Read-JsonIfPresent (Join-Path $RunRoot "champion.json")
$outcomePool = Read-JsonIfPresent (Join-Path $RunRoot "genome_outcome_pool.json")

if ($supervisor) {
    Write-Host ("Supervisor : {0} (PID {1}; worker {2})" -f `
        $supervisor.phase, $supervisor.supervisor_pid, $supervisor.worker_pid)
}
if ($status) {
    Write-Host ("Evolution  : generation {0}; {1}" -f $status.generation, $status.phase)
    if ($status.phase -eq "memory_wait") {
        Write-Host ("Memory     : {0:N2} GB free; waits for {1:N2} GB" -f `
            $status.available_memory_gb, $status.required_memory_gb) -ForegroundColor Yellow
    }
}
if ($champion -and $champion.result -and $champion.result.summary) {
    $summary = $champion.result.summary
    Write-Host ("Champion   : {0}; accuracy {1:P2}; PF {2:N3}; coverage {3:P2}" -f `
        $champion.genome_id, $summary.min_accuracy, $summary.min_profit_factor, `
        $summary.min_coverage)
}
if ($outcomePool) {
    Write-Host ("Outcome AI : {0} examples; {1}; proposed={2}" -f `
        $outcomePool.examples, $outcomePool.acquisition_mode, $outcomePool.proposed)
}
if ($admission) {
    $color = if ($admission.admitted) { "Green" } else { "Yellow" }
    Write-Host ("Trading    : {0}; PF {1:N3}/{2:N2}; live={3}" -f `
        $admission.mode, $admission.validated_profit_factor, $admission.threshold, `
        $admission.live_execution_enabled) -ForegroundColor $color
}

Write-Host ""
Write-Host "LIVE EVOLUTION EVENTS (Ctrl+C closes only this tail)" -ForegroundColor Yellow
Write-Host "----------------------------------------------------"
& $TailScript -LogPath $Events -Tail 40 -StopAfterSeconds $StopAfterSeconds
