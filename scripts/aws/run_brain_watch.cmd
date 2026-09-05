@echo off
REM Continuous Claude Code supervision of the AWS programming-brain training.
REM
REM Runs on the WORKSTATION (not the AWS host): it polls a compact remote
REM probe over SSM and wakes one Claude Code session when -- and only when --
REM the same actionable fault is seen on consecutive polls.
REM
REM Why this exists: the curriculum can look perfectly healthy while
REM admitting nothing. Measured 2026-09-05 -- eight clean resource cycles,
REM seven intervals advanced, 18,568 accepted episodes, zero failed batches,
REM and no admission for 348 hours, because the gate never ran once.
REM
REM Set CLAUDE_SESSION_ID to keep one continuous session across alarms so the
REM agent retains what it already learned about this brain.

setlocal
if "%WIZARD_WATCH_MODEL%"=="" set WIZARD_WATCH_MODEL=opus
if "%WIZARD_WATCH_EFFORT%"=="" set WIZARD_WATCH_EFFORT=xhigh

cd /d "%~dp0..\.."
python scripts\aws\watch_programming_brain.py ^
  --poll-seconds 300 ^
  --stability-polls 2 ^
  --stall-seconds 1800 ^
  --admission-stall-hours 6 ^
  --memory-floor-gb 1.5 ^
  --retry-cooldown 1800 %*
endlocal
