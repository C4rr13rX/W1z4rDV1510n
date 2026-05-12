@echo off
REM ─── W1z4rD V1510n training launcher ────────────────────────────────────────
REM Runs the full 30-phase curriculum.  Resumable — if killed or crashed,
REM re-run with SKIP_CLEAR=1 to pick up from the last completed phase.
REM Without SKIP_CLEAR the pool is cleared and curriculum starts fresh.

title W1z4rD V1510n Training

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

REM Make sure the node is up — training calls HTTP endpoints on localhost:8090.
curl -s -o NUL -m 3 http://localhost:8090/health
if errorlevel 1 (
    echo.
    echo WARNING: localhost:8090 not responding.  Start the node first
    echo          (double-click "Start W1z4rD Node") then re-run this.
    echo.
    set /p YN="Continue anyway? [y/N]: "
    if /i not "%YN%"=="y" exit /b 1
)

echo === W1z4rD V1510n Training ===
echo Bash    : %BASH_EXE%
echo Project : %CD%
echo Log     : D:\w1z4rdv1510n-data\training\run_all.log
echo.
echo If a prior run was interrupted, you can resume with the markers
echo intact by running:
echo     set SKIP_CLEAR=1
echo     bin\start_training.cmd
echo.
echo Starting in 3 seconds (CTRL+C to abort)...
echo ============================
timeout /t 3 /nobreak >NUL

"%BASH_EXE%" scripts/run_all_training.sh
set RC=%ERRORLEVEL%

echo.
echo === Training exited with code %RC% ===
echo Re-run with SKIP_CLEAR=1 to resume from the last completed phase.
pause
