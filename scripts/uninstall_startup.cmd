@echo off
REM ─── W1z4rD V1510n supervisor uninstaller ──────────────────────────────────
REM Removes the Startup-folder VBScript launcher so the supervisor no longer
REM auto-starts at logon.  Optionally also kills any currently-running
REM supervisor instance.

setlocal

set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_PATH=%STARTUP_DIR%\W1z4rD V1510n Supervisor.vbs"

if exist "%VBS_PATH%" (
    del "%VBS_PATH%"
    echo Removed: %VBS_PATH%
) else (
    echo Already absent: %VBS_PATH%
)

echo.
echo Looking for a running supervisor to stop...
REM Kill any python.exe whose command line includes w1z4rd_supervisor.
for /f "tokens=2 delims=," %%P in (
    'tasklist /v /fi "imagename eq python.exe" /fo csv 2^>nul ^| findstr /i "w1z4rd_supervisor"'
) do (
    echo Stopping supervisor PID %%~P
    taskkill /pid %%~P /f >NUL 2>&1
)
echo Done.
pause
