@echo off
:: Start W1z4rD node and initialise a new cluster on this machine.
:: The node API (port 8090) and cluster listener (port 51611) start together.
:: Run this on the coordinator machine first; copy the OTP into start_worker.ps1.

echo.
echo  W1z4rD Cluster Coordinator
echo  Starting node API (8090) + cluster listener (51611)...
echo.

cd /d "%~dp0.."

:: Launch node in background (HTTP API starts automatically on ports 8080 + 8090)
start "W1z4rD Node" cmd /c "set W1Z4RDV1510N_DATA_DIR=D:\w1z4rdv1510n-data && bin\w1z4rd_node.exe"

:: Wait for API to come up
timeout /t 3 /nobreak >nul

:: Init the cluster via REST API and display the OTP
echo  Initialising cluster via REST API...
curl -s -X POST http://127.0.0.1:8090/cluster/init ^
  -H "Content-Type: application/json" ^
  -d "{\"bind\": \"0.0.0.0:51611\", \"otp_ttl_secs\": 300}"

echo.
echo  Copy the OTP above and paste it into start_worker.ps1 on the worker machine.
echo  Node is running in a separate window. Close that window to stop.
echo.
pause
