@echo off
REM Continuous Claude Code supervision of the AWS programming-brain training.
REM
REM The AWS curriculum supervisor is a systemd unit and survives host reboots
REM on its own. THIS is the piece that dies with the workstation: it polls the
REM remote probe and wakes a Claude Code session when a fault needs repair.
REM Started from the Windows Startup folder so a reboot does not silently
REM leave training unwatched -- measured 2026-09-09, the watcher had been down
REM across several restarts while the supervisor ran on.
REM
REM Logs: runtime\programming-brain-watch\activity.log

setlocal
if "%WIZARD_WATCH_MODEL%"=="" set WIZARD_WATCH_MODEL=opus
if "%WIZARD_WATCH_EFFORT%"=="" set WIZARD_WATCH_EFFORT=xhigh

cd /d "%~dp0..\.."
if not exist "runtime\programming-brain-watch" mkdir "runtime\programming-brain-watch"

:loop
REM Restart on crash rather than leaving training unwatched until someone
REM notices. A 30 s pause keeps a hard failure from spinning the CPU.
python scripts\aws\watch_programming_brain.py ^
  --poll-seconds 300 ^
  --stability-polls 2 ^
  --stall-seconds 1800 ^
  --admission-stall-hours 6 ^
  --memory-floor-gb 1.5 ^
  --retry-cooldown 1800 >> "runtime\programming-brain-watch\watcher.out" 2>&1
timeout /t 30 /nobreak >nul
goto loop
