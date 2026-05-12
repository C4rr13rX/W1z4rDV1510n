@echo off
REM ─── W1z4rD V1510n node launcher ────────────────────────────────────────────
REM Sets up the env, runs the node binary in this terminal so you can see logs.
REM Closing this window stops the node.  CTRL+C also stops it cleanly.

title W1z4rD V1510n Node

set "W1Z4RDV1510N_DATA_DIR=D:\w1z4rdv1510n-data"
cd /d "D:\Projects\W1z4rDV1510n"

echo === W1z4rD V1510n Node ===
echo Data dir: %W1Z4RDV1510N_DATA_DIR%
echo Project : %CD%
echo Binary  : bin\w1z4rd_node.exe
echo.
echo Node listens on:
echo   http://localhost:8090   API / /chat / /multi_pool / /neuro/*
echo.
echo Press CTRL+C to stop the node.  Closing this window also stops it.
echo ============================
echo.

bin\w1z4rd_node.exe
set RC=%ERRORLEVEL%

echo.
echo === Node exited with code %RC% ===
pause
