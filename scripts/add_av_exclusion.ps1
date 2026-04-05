#Requires -RunAsAdministrator
# Run this once: Right-click → "Run as Administrator"
# Adds Windows Defender (and Avira via WinSC platform) exclusions for this project.

$projectRoot = Split-Path $PSScriptRoot -Parent
$targetDir   = Join-Path $projectRoot "target"

Write-Host "Adding AV exclusions for:" -ForegroundColor Cyan
Write-Host "  $projectRoot"
Write-Host "  $targetDir"

Add-MpPreference -ExclusionPath $projectRoot -Force
Add-MpPreference -ExclusionPath $targetDir   -Force
Add-MpPreference -ExclusionProcess "predict_state.exe"  -Force
Add-MpPreference -ExclusionProcess "w1z4rd_api.exe"     -Force
Add-MpPreference -ExclusionProcess "calibrate_energy.exe" -Force

$prefs = Get-MpPreference
Write-Host "`nExclusions in effect:" -ForegroundColor Green
$prefs.ExclusionPath    | ForEach-Object { Write-Host "  [path]    $_" }
$prefs.ExclusionProcess | ForEach-Object { Write-Host "  [process] $_" }
Write-Host "`nDone. Avira uses the Windows Security Platform API, so these exclusions apply to it too." -ForegroundColor Green
