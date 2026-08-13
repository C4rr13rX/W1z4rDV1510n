from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_market_brain_wbrain_migration.ps1"


def test_market_migration_is_staged_and_non_promoting_by_default():
    source = SCRIPT.read_text(encoding="utf-8")
    assert '[switch]$Execute' in source
    assert 'if (-not $Execute) { return }' in source
    assert 'live_source_untouched = $true' in source
    assert 'promotion_performed = $false' in source
    assert 'New-Item -ItemType HardLink' in source
    assert 'brain.wbrain' in source
    assert 'Move-Item' not in source
    assert 'Remove-Item -LiteralPath $source' not in source


def test_market_migration_fails_closed_on_resources_and_process_identity():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'RequiredAvailableMb = 8192' in source
    assert 'EstimatedExpansionFactor = 1.25' in source
    assert '$files | Measure-Object -Property Length -Sum' in source
    assert '$drive.AvailableFreeSpace -ge $requiredFreeBytes' in source
    assert '$matches.Count -eq 1' in source
    assert 'api --addr $NodeAddress' in source
    assert 'refusing forced termination' in source
    assert 'monitor_migration_memory.ps1' in source


def test_market_migration_recovers_legacy_node_even_after_failure():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'Invoke-WebRequest -UseBasicParsing -Method Post' in source
    assert '/neuro/checkpoint' in source
    assert '/shutdown' in source
    assert 'finally {' in source
    assert 'start_node.ps1' in source
    assert 'Wait-Http "http://$NodeAddress/health" 180' in source
