@echo off
REM ─── W1z4rD V1510n training launcher ────────────────────────────────────────
REM Launches the curriculum DETACHED so closing this window leaves training
REM running.  Resumable — if killed or crashed, re-run with SKIP_CLEAR=1 to
REM pick up from the last completed phase.  Default behaviour clears the pool
REM and starts fresh.

title W1z4rD V1510n Training Launcher

cd /d "D:\Projects\W1z4rDV1510n"

REM Find Git Bash — try the common install locations, then the PATH.
set "BASH_EXE="
if exist "C:\Program Files\Git\bin\bash.exe" set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
if not defined BASH_EXE if exist "C:\Program Files (x86)\Git\bin\bash.exe" set "BASH_EXE=C:\Program Files (x86)\Git\bin\bash.exe"
if not defined BASH_EXE where bash.exe >NUL 2>&1 && for /f "delims=" %%I in ('where bash.exe') do (
    if not defined BASH_EXE set "BASH_EXE=%%I"
)
if not defined BASH_EXE (
    echo ERROR: Git Bash not found.  Install Git for Windows from
    echo        https://git-scm.com/download/win and re-run.
    pause
    exit /b 1
)

REM Sanity-check that the node is up.
curl -s -o NUL -m 3 http://localhost:8090/health
if errorlevel 1 (
    echo.
    echo WARNING: localhost:8090 not responding.  Start the node first
    echo          ^(double-click "Start W1z4rD Node"^) then re-run this.
    echo.
    set /p YN="Continue anyway? [y/N]: "
    if /i not "%YN%"=="y" exit /b 1
)

REM Refuse to launch a second copy on top of an in-flight curriculum.
REM Detect by checking for an existing bash running run_all_training.sh.
for /f "tokens=2 delims=," %%P in (
    'tasklist /v /fi "imagename eq bash.exe" /fo csv 2^>nul ^| findstr /i "run_all_training"'
) do (
    echo.
    echo Training is already running ^(bash PID %%P^).  Tail the log to watch
    echo progress:
    echo     type D:\w1z4rdv1510n-data\training\run_all.log
    echo.
    echo If it's stuck, kill PID %%P first then re-run.
    pause
    exit /b 0
)

echo === W1z4rD V1510n Training (detached) ===
echo Bash      : %BASH_EXE%
echo Project   : %CD%
echo Log       : D:\w1z4rdv1510n-data\training\run_all.log
echo Full log  : D:\w1z4rdv1510n-data\training\run_all_full.log
echo.
echo Resume from a crashed/swapped run by re-running with SKIP_CLEAR=1:
echo     set SKIP_CLEAR=1
echo     bin\start_training.cmd
echo.
echo Launching in 2 seconds (CTRL+C to abort)...
timeout /t 2 /nobreak >NUL

REM Detached launch — `start "" /b` orphans the bash from this cmd so
REM closing this window does NOT kill the curriculum.  Output goes to
REM run_all_full.log; the per-phase summary goes to run_all.log via tee
REM inside the bash script itself.
start "" /b "%BASH_EXE%" -c "cd 'D:/Projects/W1z4rDV1510n' && SKIP_CLEAR='%SKIP_CLEAR%' bash scripts/run_all_training.sh > /d/w1z4rdv1510n-data/training/run_all_full.log 2>&1"

REM Give the bash a moment to start, then verify.
timeout /t 4 /nobreak >NUL

REM Confirm we see a bash running.
tasklist /v /fi "imagename eq bash.exe" /fo csv 2>nul | findstr /i "run_all_training" >NUL
if errorlevel 1 (
    echo.
    echo ERROR: bash did not stay up.  Check D:\w1z4rdv1510n-data\training\run_all_full.log
    echo        for an immediate failure reason.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Training is now running in the background.
echo You can close this window — training will keep going.
echo Watch progress with the "Check W1z4rD Node" desktop icon.
echo ============================================================
echo.
echo This window will close in 8 seconds (or press any key now).
timeout /t 8
