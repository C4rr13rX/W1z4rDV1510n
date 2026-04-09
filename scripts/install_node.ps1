# W1z4rD V1510n — Node Installer
# Run as Administrator.  Sets up firewall rules, installs binary, and optionally joins a cluster.
#
# Usage:
#   .\install_node.ps1                            # interactive
#   .\install_node.ps1 -InstallDir C:\W1z4rD      # custom install path
#   .\install_node.ps1 -Coordinator 192.168.1.84:51611 -OTP WORD-NNNN  # auto-join

param(
    [string]$InstallDir  = "C:\W1z4rD",
    [string]$Coordinator = "",
    [string]$OTP         = "",
    [switch]$NoAutoStart
)

$ErrorActionPreference = "Stop"

# ── Elevation check ───────────────────────────────────────────────────────────
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "`n[W1z4rD] Restarting as Administrator..." -ForegroundColor Yellow
    $args_str = "-File `"$PSCommandPath`""
    if ($InstallDir)   { $args_str += " -InstallDir `"$InstallDir`"" }
    if ($Coordinator)  { $args_str += " -Coordinator `"$Coordinator`"" }
    if ($OTP)          { $args_str += " -OTP `"$OTP`"" }
    if ($NoAutoStart)  { $args_str += " -NoAutoStart" }
    Start-Process powershell -Verb RunAs -ArgumentList $args_str
    exit
}

function Banner($msg) {
    Write-Host ""
    Write-Host "  ════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host "   W1z4rD V1510n  —  $msg" -ForegroundColor Cyan
    Write-Host "  ════════════════════════════════════════" -ForegroundColor DarkCyan
    Write-Host ""
}

function Step($n, $msg) {
    Write-Host "  [$n] $msg" -ForegroundColor White
}

function OK($msg) {
    Write-Host "      OK  $msg" -ForegroundColor Green
}

function WARN($msg) {
    Write-Host "      !!  $msg" -ForegroundColor Yellow
}

Banner "Node Installer"

# ── Determine source binary ───────────────────────────────────────────────────
$script_dir  = Split-Path -Parent $PSCommandPath
$project_dir = Split-Path -Parent $script_dir
$src_binary  = Join-Path $project_dir "bin\w1z4rd_node.exe"

if (-not (Test-Path $src_binary)) {
    $src_binary = Join-Path $project_dir "target\release\w1z4rdv1510n-node.exe"
}

if (-not (Test-Path $src_binary)) {
    Write-Host "  ERROR: Cannot find w1z4rd_node.exe. Build the project first." -ForegroundColor Red
    exit 1
}

# ── Step 1: Create install directory ─────────────────────────────────────────
Step 1 "Creating install directory: $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
OK "Directory ready"

# ── Step 2: Copy binary ───────────────────────────────────────────────────────
Step 2 "Installing node binary"
$dst_binary = Join-Path $InstallDir "w1z4rd_node.exe"
Stop-Process -Name "w1z4rd_node" -Force -ErrorAction SilentlyContinue
Copy-Item -Force $src_binary $dst_binary
OK "Installed to $dst_binary"

# Also copy dashboard if present
$src_dash = Join-Path $project_dir "bin\w1z4rd_dashboard.exe"
if (-not (Test-Path $src_dash)) {
    $src_dash = Join-Path $project_dir "target\release\w1z4rd-dashboard.exe"
}
if (Test-Path $src_dash) {
    Copy-Item -Force $src_dash (Join-Path $InstallDir "w1z4rd_dashboard.exe")
    OK "Dashboard installed"
}

# ── Step 3: Firewall rules ────────────────────────────────────────────────────
Step 3 "Opening firewall ports"

$rules = @(
    @{ Port = 8080;  Name = "W1z4rD Neuro API";  Desc = "Neural fabric training and prediction API" },
    @{ Port = 8090;  Name = "W1z4rD Node API";   Desc = "Cluster management and dashboard API" },
    @{ Port = 51611; Name = "W1z4rD SIGIL";       Desc = "Cluster ring, heartbeat, and election port" }
)

foreach ($r in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
    if ($existing) {
        WARN "Rule '$($r.Name)' already exists — skipping"
    } else {
        New-NetFirewallRule `
            -DisplayName $r.Name `
            -Description $r.Desc `
            -Direction   Inbound `
            -Protocol    TCP `
            -LocalPort   $r.Port `
            -Action      Allow `
            -Profile     Private,Domain `
            | Out-Null
        OK "Port $($r.Port) open ($($r.Name))"
    }
}

# ── Step 4: Create startup batch file ─────────────────────────────────────────
Step 4 "Creating start script"
$start_bat = Join-Path $InstallDir "start_node.bat"
@"
@echo off
cd /d "$InstallDir"
w1z4rd_node.exe --api-addr 0.0.0.0:8080
"@ | Set-Content $start_bat
OK "Created $start_bat"

# ── Step 5: Scheduled task (optional auto-start) ──────────────────────────────
if (-not $NoAutoStart) {
    Step 5 "Setting up auto-start on login"
    $task_name = "W1z4rD V1510n Node"
    Unregister-ScheduledTask -TaskName $task_name -Confirm:$false -ErrorAction SilentlyContinue
    $action  = New-ScheduledTaskAction -Execute $dst_binary -Argument "--api-addr 0.0.0.0:8080" -WorkingDirectory $InstallDir
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $task_name -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
    OK "Scheduled task '$task_name' registered (runs at login)"
} else {
    WARN "Skipped auto-start task (-NoAutoStart)"
}

# ── Step 6: Optional cluster join ────────────────────────────────────────────
if ($Coordinator -and $OTP) {
    Step 6 "Joining cluster at $Coordinator"
    $join_script = Join-Path $InstallDir "join_cluster.bat"
    @"
@echo off
cd /d "$InstallDir"
w1z4rd_node.exe cluster-join --coordinator $Coordinator --otp $OTP
pause
"@ | Set-Content $join_script
    OK "Created $join_script — run it after the node is started to join the cluster"
    WARN "Or use the Dashboard > Cluster tab to join without CLI"
} elseif ($Coordinator -or $OTP) {
    WARN "Provide both -Coordinator and -OTP to set up a cluster join script"
}

# ── Done ──────────────────────────────────────────────────────────────────────
Banner "Installation Complete"
Write-Host "  Node installed to: $InstallDir" -ForegroundColor Cyan
Write-Host "  Start manually:    $start_bat" -ForegroundColor White
Write-Host "  Dashboard:         $InstallDir\w1z4rd_dashboard.exe" -ForegroundColor White
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Yellow
Write-Host "   1. Start the node (or reboot if auto-start is enabled)" -ForegroundColor White
Write-Host "   2. Open the Dashboard and go to the Cluster tab" -ForegroundColor White
Write-Host "   3. Init a new cluster OR join an existing one with an OTP" -ForegroundColor White
Write-Host ""

$launch = Read-Host "  Launch the dashboard now? [Y/n]"
if ($launch -ne "n" -and $launch -ne "N") {
    $dash = Join-Path $InstallDir "w1z4rd_dashboard.exe"
    if (Test-Path $dash) {
        Start-Process $dash
    } else {
        WARN "Dashboard not found at $dash"
    }
}
