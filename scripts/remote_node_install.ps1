# W1z4rD Node — Remote Install & Cluster Join
# Run this as Administrator on DESKTOP-6E34B18
# -----------------------------------------------
# Replace COORDINATOR_IP and OTP before running.
# Generate a fresh OTP on PEGASUS with:
#   bin\w1z4rd_node.exe cluster-otp

$COORDINATOR_IP = "192.168.1.84"
$OTP            = "PORTAL-4864"   # <-- regenerate if expired
$INSTALL_DIR    = "C:\W1z4rD"

# 1. Create install directory
New-Item -ItemType Directory -Force -Path $INSTALL_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$INSTALL_DIR\logs" | Out-Null

# 2. Download binary and config from PEGASUS
Write-Host "Downloading node binary..."
Invoke-WebRequest -Uri "http://${COORDINATOR_IP}:9999/w1z4rd_node.exe" `
    -OutFile "$INSTALL_DIR\w1z4rd_node.exe" -UseBasicParsing

Write-Host "Downloading node config..."
Invoke-WebRequest -Uri "http://${COORDINATOR_IP}:9999/node_config.json" `
    -OutFile "$INSTALL_DIR\node_config.json" -UseBasicParsing

# 3. Join the cluster
Write-Host "Joining cluster at ${COORDINATOR_IP}:51611 ..."
Set-Location $INSTALL_DIR
.\w1z4rd_node.exe cluster-join --coordinator "${COORDINATOR_IP}:51611" --otp $OTP
