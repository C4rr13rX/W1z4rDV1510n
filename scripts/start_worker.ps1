# W1z4rD Worker — join an existing cluster
# 1. Start the coordinator node on the coordinator machine (start_cluster.bat)
# 2. Copy the OTP printed by the coordinator into $OTP below (or pass as argument)
# 3. Run this script on the worker machine

param(
    [string]$OTP = "",
    [string]$Coordinator = "192.168.1.84:51611",
    [string]$DataDir = "D:\w1z4rdv1510n-data",
    [string]$NodeBin = ".\bin\w1z4rd_node.exe"
)

if (-not $OTP) {
    $OTP = Read-Host "Enter OTP from coordinator"
}

# Set working directory and data path
Set-Location $PSScriptRoot\..

# Launch node in background (HTTP API starts automatically on ports 8080 + 8090)
$env:W1Z4RDV1510N_DATA_DIR = $DataDir
Write-Host ""
Write-Host " Starting W1z4rD node (API on :8090, cluster on :51611)..."
Start-Process -FilePath $NodeBin -NoNewWindow -PassThru | Out-Null

# Wait for API to come up
Start-Sleep -Seconds 3

# Join the cluster via REST API
Write-Host " Joining cluster at $Coordinator..."
$body = @{
    coordinator = $Coordinator
    otp         = $OTP
    bind        = "0.0.0.0:51611"
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post `
    -Uri "http://127.0.0.1:8090/cluster/join" `
    -ContentType "application/json" `
    -Body $body

Write-Host ""
Write-Host " Joined cluster: $($response.cluster_id)  nodes: $($response.node_count)"
Write-Host ""
Write-Host " Check distributed sync status:"
Write-Host "   curl http://127.0.0.1:8090/cluster/sync/status"
Write-Host ""
Write-Host " Node is running in background. Use Stop-Process -Name w1z4rd_node to stop."
Write-Host ""
