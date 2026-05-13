@echo off
REM ─── W1z4rD V1510n supervisor → startup ─────────────────────────────────────
REM Drops a hidden-window VBScript launcher in the user's Startup folder so
REM the supervisor runs automatically every time you log on.  No admin
REM needed — it's per-user, not system-wide.
REM
REM To remove: run uninstall_startup.cmd.

setlocal

set "PYTHON_EXE=python.exe"
set "SUPERVISOR=D:\Projects\W1z4rDV1510n\scripts\w1z4rd_supervisor.py"
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_PATH=%STARTUP_DIR%\W1z4rD V1510n Supervisor.vbs"

REM Verify Python is reachable.
where %PYTHON_EXE% >NUL 2>&1
if errorlevel 1 (
    echo ERROR: python.exe is not on PATH.  Install Python 3.11+ from
    echo        https://www.python.org/downloads/ and re-run this script.
    pause
    exit /b 1
)

if not exist "%SUPERVISOR%" (
    echo ERROR: supervisor script not found at:
    echo        %SUPERVISOR%
    pause
    exit /b 1
)

REM Make sure the Startup folder exists.
if not exist "%STARTUP_DIR%" mkdir "%STARTUP_DIR%"

REM Write the VBScript launcher.  wscript runs it with no visible window
REM (the 0 in WshShell.Run), so the supervisor lives quietly in the
REM background; you can find it in Task Manager as python.exe with
REM w1z4rd_supervisor in the command line.
> "%VBS_PATH%" echo ' W1z4rD V1510n supervisor - runs at logon, hidden window.
>> "%VBS_PATH%" echo ' Remove this file from the Startup folder to disable auto-start.
>> "%VBS_PATH%" echo Set WshShell = CreateObject("WScript.Shell"^)
>> "%VBS_PATH%" echo WshShell.CurrentDirectory = "D:\Projects\W1z4rDV1510n"
>> "%VBS_PATH%" echo WshShell.Run "%PYTHON_EXE% ""%SUPERVISOR%""", 0, False

echo Installed:
echo   %VBS_PATH%
echo.
echo It will run automatically every time you log in.  Starting it now...
start "" wscript.exe "%VBS_PATH%"
timeout /t 3 /nobreak >NUL

REM Verify a python.exe with the supervisor in its command line is alive.
tasklist /v /fi "imagename eq python.exe" /fo csv 2>NUL | findstr /i "w1z4rd_supervisor" >NUL
if errorlevel 1 (
    echo.
    echo WARNING: supervisor did not appear in tasklist.  It may still be
    echo          starting — check D:\w1z4rdv1510n-data\training\supervisor.log
    echo          in a few seconds.
) else (
    echo.
    echo Supervisor is running.
)

echo.
echo Logs:
echo   D:\w1z4rdv1510n-data\training\supervisor.log
echo.
echo Configure behaviour by editing:
echo   D:\Projects\W1z4rDV1510n\supervisor.toml
echo   ^(copy supervisor.toml.example as a template^)
echo.
pause
