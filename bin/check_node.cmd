@echo off
REM ─── W1z4rD V1510n node status check ───────────────────────────────────────
REM Probes the node + Django + brain snapshot, prints a colourised summary,
REM then waits for Q to close.  Refresh-friendly (R reruns the probes).
REM
REM Distinguishes three states for each service:
REM   ONLINE                — port listening AND HTTP responding healthily
REM   SLOW                  — port listening but HTTP didn't answer in time
REM   OFFLINE               — port not listening at all
REM
REM The old version conflated "Django can't reach the node" with "Django is
REM down", and used a 3s timeout which was too aggressive when the node was
REM under training load.  Both fixed here.

title W1z4rD V1510n Status
color 0F

:run_check
cls
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'SilentlyContinue';" ^
  "function Probe-Port($p) { try { $c = New-Object System.Net.Sockets.TcpClient; $iar = $c.BeginConnect('127.0.0.1', $p, $null, $null); $ok = $iar.AsyncWaitHandle.WaitOne(1500, $false); if ($ok -and $c.Connected) { $c.Close(); return $true } else { try { $c.Close() } catch {}; return $false } } catch { return $false } }" ^
  "$nodeListen = Probe-Port 8090;" ^
  "$djListen   = Probe-Port 8000;" ^
  "$h = $null; $hErr = '';" ^
  "$b = $null;" ^
  "$dj = $null; $djErr = '';" ^
  "if ($nodeListen) { try { $h  = Invoke-RestMethod -Uri http://127.0.0.1:8090/health -TimeoutSec 15 } catch { $hErr = $_.Exception.Message } }" ^
  "if ($nodeListen) { try { $b  = Invoke-RestMethod -Uri http://127.0.0.1:8090/brain  -TimeoutSec 20 } catch {} }" ^
  "if ($djListen)   { try { $dj = Invoke-RestMethod -Uri http://127.0.0.1:8000/api/wizard-chat/status/ -TimeoutSec 15 } catch { $djErr = $_.Exception.Message } }" ^
  "Write-Host '===================================================='   -ForegroundColor Cyan;" ^
  "Write-Host '  W1z4rD V1510n  -  status at ' (Get-Date -Format 'HH:mm:ss') -ForegroundColor Cyan;" ^
  "Write-Host '===================================================='   -ForegroundColor Cyan;" ^
  "Write-Host '';" ^
  "if ($h) {" ^
  "    Write-Host ('[OK]  Node:    online (uptime ' + $h.uptime_secs + 's, id ' + $h.node_id + ')') -ForegroundColor Green" ^
  "} elseif ($nodeListen) {" ^
  "    Write-Host '[??]  Node:    SLOW (port 8090 listening but no HTTP response in 15s)' -ForegroundColor Yellow;" ^
  "    if ($hErr) { Write-Host ('       last error: ' + $hErr) -ForegroundColor DarkYellow };" ^
  "    Write-Host '       The node process is alive but unresponsive — supervisor should restart it; if not, kill it manually.' -ForegroundColor DarkGray" ^
  "} else {" ^
  "    Write-Host '[--]  Node:    OFFLINE (port 8090 not listening)' -ForegroundColor Red" ^
  "};" ^
  "if ($dj -ne $null) {" ^
  "    $nodeReachable = if ($dj.online) { 'yes' } else { 'no (' + $dj.error + ')' };" ^
  "    Write-Host ('[OK]  Django:  online at ' + $dj.endpoint + '  (node reachable from Django: ' + $nodeReachable + ')') -ForegroundColor Green" ^
  "} elseif ($djListen) {" ^
  "    Write-Host '[??]  Django:  SLOW (port 8000 listening but no HTTP response in 15s)' -ForegroundColor Yellow;" ^
  "    if ($djErr) { Write-Host ('       last error: ' + $djErr) -ForegroundColor DarkYellow }" ^
  "} else {" ^
  "    Write-Host '[--]  Django:  OFFLINE (port 8000 not listening)' -ForegroundColor Yellow" ^
  "};" ^
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
