# Run as Administrator — opens the ports needed for cluster P2P on this (main) node.
# Required so node-3 on the other PC can connect in.
#
# Usage (from an Admin PowerShell):
#   .\scripts\open_firewall_cluster.ps1

$rules = @(
    @{ Port = 8088;  Name = "W1z4rD P2P Gossip"; Desc = "Cluster peer-to-peer gossip" },
    @{ Port = 8090;  Name = "W1z4rD Node API";    Desc = "Node REST API (local cluster)" },
    @{ Port = 51611; Name = "W1z4rD SIGIL";        Desc = "Cluster ring, heartbeat, election" }
)

foreach ($r in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $r.Name -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "  [SKIP]  $($r.Name) already exists" -ForegroundColor Yellow
    } else {
        New-NetFirewallRule `
            -DisplayName $r.Name -Description $r.Desc `
            -Direction Inbound -Protocol TCP `
            -LocalPort $r.Port -Action Allow -Profile Private,Domain `
            | Out-Null
        Write-Host "  [OK]    Port $($r.Port) open ($($r.Name))" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Done. Node 3 can now reach this machine on ports 8088, 8090, 51611." -ForegroundColor Cyan
