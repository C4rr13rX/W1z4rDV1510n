# W1z4rD Worker — run on DESKTOP-6E34B18
# 1. Start the coordinator on PEGASUS first (scripts\start_cluster.bat)
# 2. Copy the OTP it prints into $OTP below
# 3. Run this script

param(
    [string]$OTP = "",
    [string]$Coordinator = "192.168.1.84:51611"
)

if (-not $OTP) {
    $OTP = Read-Host "Enter OTP from PEGASUS coordinator"
}

Set-Location "C:\W1z4rD"
.\w1z4rd_node.exe cluster-join --coordinator $Coordinator --otp $OTP
