@echo off
:: Start W1z4rD as cluster coordinator on PEGASUS
:: Run this instead of the plain node when you want cluster mode.
:: The OTP printed here is what you paste into start_worker.ps1 on DESKTOP-6E34B18.

echo.
echo  W1z4rD Cluster Coordinator
echo  Copy the OTP below and run start_worker.ps1 on DESKTOP-6E34B18
echo.

cd /d "%~dp0.."
bin\w1z4rd_node.exe cluster-init --bind 192.168.1.84:51611
