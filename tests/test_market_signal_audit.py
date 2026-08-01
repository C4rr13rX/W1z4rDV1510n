from scripts.market_signal_audit import select_primary_assets


def test_primary_asset_selection_prefers_stable_quote_then_longer_series():
    records = [
        {"base_asset": "ETH", "quote_asset": "WBTC", "rows": 5000, "relative_path": "a"},
        {"base_asset": "ETH", "quote_asset": "USDC", "rows": 3000, "relative_path": "b"},
        {"base_asset": "ETH", "quote_asset": "USDT", "rows": 4000, "relative_path": "c"},
        {"base_asset": "BTC", "quote_asset": "USDC", "rows": 2000, "relative_path": "d"},
    ]

    selected = {row["base_asset"]: row for row in select_primary_assets(records)}

    assert selected["ETH"]["relative_path"] == "c"
    assert selected["BTC"]["relative_path"] == "d"
    assert "_selection_rank" not in selected["ETH"]
