import json

from scripts.market_corpus_manifest import asset_family, build_manifest, symbol_from_path


def write_bars(path, closes, *, start=1_700_000_000):
    rows = [
        {
            "timestamp": start + index * 3600,
            "open": close,
            "high": close * 1.01,
            "low": close * .99,
            "close": close,
            "volume": 10 + index,
        }
        for index, close in enumerate(closes)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_symbol_and_family_normalization(tmp_path):
    assert symbol_from_path(tmp_path / "0012_weth-usdc.json") == "WETH-USDC"
    assert asset_family("WETH-USDC") == "major"
    assert asset_family("AAVE-WETH") == "defi"
    assert asset_family("PEPE-USDC") == "high_volatility"
    assert asset_family("USDT-USDC") == "stable_cross"


def test_manifest_collapses_versions_and_cross_chain_copies(tmp_path):
    closes = [100 + index for index in range(8)]
    write_bars(tmp_path / "base" / "0001_WETH-USDC.json", closes[:-1])
    write_bars(tmp_path / "base" / "0002_WETH-USDC.json", closes)
    write_bars(tmp_path / "ethereum" / "0001_WETH-USDC.json", closes)
    write_bars(tmp_path / "optimism" / "0001_OP-USDC.json", [20 + index / 10 for index in range(8)])

    manifest = build_manifest(tmp_path, min_rows=5, min_hourly_share=.99)
    summary = manifest["summary"]
    assert summary["files_discovered"] == 4
    assert summary["canonical_chain_pairs"] == 3
    assert summary["selected_independent_series"] == 2
    selected = {item["symbol"] for item in manifest["selected"]}
    assert selected == {"WETH-USDC", "OP-USDC"}
    old = next(item for item in manifest["files"]
               if item["relative_path"] == "base/0001_WETH-USDC.json")
    assert not old["canonical_for_chain_pair"]
    assert old["duplicate_of"] == "base/0002_WETH-USDC.json"


def test_irregular_or_short_series_are_not_selected(tmp_path):
    write_bars(tmp_path / "base" / "0001_LINK-USDC.json", [1, 2, 3])
    write_bars(tmp_path / "base" / "0002_AAVE-USDC.json", range(10))
    rows = json.loads((tmp_path / "base" / "0002_AAVE-USDC.json").read_text())
    rows[5]["timestamp"] += 900
    (tmp_path / "base" / "0002_AAVE-USDC.json").write_text(json.dumps(rows))
    manifest = build_manifest(tmp_path, min_rows=5, min_hourly_share=.99)
    assert manifest["summary"]["selected_independent_series"] == 0
    assert {item["reason"] for item in manifest["files"]} == {
        "too_few_rows", "irregular_cadence"
    }
