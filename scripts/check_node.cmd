@echo off
REM ─── W1z4rD V1510n node status check ───────────────────────────────────────
REM Probes the node + Django + brain snapshot, prints a colourised summary,
REM then waits for Q to close.  Refresh-friendly (R reruns the probes).

title W1z4rD V1510n Status
color 0F

:run_check
cls
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "$h = $null; $b = $null; $mp = $null; $dj = $null;" ^
  "try { $h  = Invoke-RestMethod -Uri http://localhost:8090/health -TimeoutSec 3 } catch {}" ^
  "try { $b  = Invoke-RestMethod -Uri http://localhost:8090/brain  -TimeoutSec 5 } catch {}" ^
  "try { $mp = Invoke-RestMethod -Uri http://localhost:8090/multi_pool/stats -TimeoutSec 3 } catch {}" ^
  "try { $dj = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/wizard-chat/status/ -TimeoutSec 3 } catch {}" ^
  "Write-Host '===================================================='   -ForegroundColor Cyan;" ^
  "Write-Host '  W1z4rD V1510n  -  status at ' (Get-Date -Format 'HH:mm:ss') -ForegroundColor Cyan;" ^
  "Write-Host '===================================================='   -ForegroundColor Cyan;" ^
  "Write-Host '';" ^
  "if ($h) {" ^
  "    Write-Host ('[OK]  Node:    online (uptime ' + $h.uptime_secs + 's, id ' + $h.node_id + ')') -ForegroundColor Green" ^
  "} else { Write-Host '[--]  Node:    OFFLINE (port 8090 not responding)' -ForegroundColor Red };" ^
  "if ($dj -and $dj.online) {" ^
  "    Write-Host ('[OK]  Django:  online at ' + $dj.endpoint) -ForegroundColor Green" ^
  "} else { Write-Host '[--]  Django:  OFFLINE (port 8000 not responding)' -ForegroundColor Yellow };" ^
  "Write-Host '';" ^
  "if ($b) {" ^
  "    Write-Host '-- Slow pool --' -ForegroundColor Cyan;" ^
  "    Write-Host ('    active labels:   ' + $b.slow_pool.active_label_count);" ^
  "    Write-Host '-- Multi-pool --' -ForegroundColor Cyan;" ^
  "    $pools = ($b.multi_pool.pools.PSObject.Properties | ForEach-Object { $_.Name + '=' + $_.Value }) -join '  ';" ^
  "    Write-Host ('    pools:           ' + $pools);" ^
  "    Write-Host ('    cross_edges:     ' + $b.multi_pool.cross_edges);" ^
  "    Write-Host '-- Motif hierarchy --' -ForegroundColor Cyan;" ^
  "    Write-Host ('    total motifs:    ' + $b.motifs.total);" ^
  "    Write-Host ('    attractors:      ' + $b.motifs.attractor_count);" ^
  "    if ($b.motifs.by_level.PSObject.Properties.Count -gt 0) {" ^
  "        $lvl = ($b.motifs.by_level.PSObject.Properties | Sort-Object Name | ForEach-Object { 'L' + $_.Name + '=' + $_.Value }) -join '  ';" ^
  "        Write-Host ('    by level:        ' + $lvl)" ^
  "    };" ^
  "    Write-Host '-- Neuromodulators --' -ForegroundColor Cyan;" ^
  "    Write-Host ('    dopamine:        ' + $b.neuromodulators.dopamine);" ^
  "    Write-Host ('    norepinephrine:  ' + $b.neuromodulators.norepinephrine);" ^
  "    Write-Host ('    acetylcholine:   ' + $b.neuromodulators.acetylcholine);" ^
  "    Write-Host ('    serotonin:       ' + $b.neuromodulators.serotonin);" ^
  "    Write-Host '-- Feedback recipes --' -ForegroundColor Cyan;" ^
  "    Write-Host ('    ' + ($b.feedback_recipes -join ', '))" ^
  "} else { Write-Host '-- /brain snapshot unavailable --' -ForegroundColor Red };" ^
  "Write-Host '';" ^
  "Write-Host '----------------------------------------------------'   -ForegroundColor DarkGray;" ^
  "Write-Host '  R = refresh    Q = quit'                             -ForegroundColor DarkGray"

REM ─── Key-press loop ────────────────────────────────────────────────────────
:wait_key
choice /c QR /n /m " "
if errorlevel 2 goto run_check
if errorlevel 1 exit /b 0
goto wait_key
