from scripts.market_signal_audit import (
    attach_market_breadth, build_rows, derive_supplemental_features, select_primary_assets,
)


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


def test_supplemental_rolling_features_never_read_future_rows():
    rows = []
    for index in range(30):
        rows.append({
            "timestamp": index * 3600, "spot_close": 100 + index,
            "spot_base_volume": 10, "futures_base_volume": 20,
            "spot_quote_volume": 100 + index, "futures_quote_volume": 200 + index,
            "spot_trade_count": 10 + index, "futures_trade_count": 20 + index,
            "spot_taker_buy_base": 6, "futures_taker_buy_base": 9,
            "futures_spot_basis": index / 10_000,
            "premium_close": index / 20_000, "funding_rate": index / 1_000_000,
        })
    before = derive_supplemental_features(rows)
    rows[-1]["funding_rate"] = 999
    after = derive_supplemental_features(rows)
    assert before[20 * 3600] == after[20 * 3600]
    assert before[29 * 3600] != after[29 * 3600]


def test_market_breadth_uses_current_features_not_future_targets():
    rows = [
        {"timestamp": 1, "target": target,
         "features": {"r1": value, "r6": value * 2, "r24": value * 3}}
        for target, value in ((1, .01), (-1, -.02), (0, .03))
    ]
    attach_market_breadth(rows)
    before = [dict(row["features"]) for row in rows]
    rows[0]["target"] = -1
    attach_market_breadth(rows)
    assert [row["features"] for row in rows] == before
    assert rows[0]["features"]["market_breadth_r6"] == 2 / 3


def test_build_rows_retains_bounded_multi_horizon_training_outcomes(monkeypatch):
    bars = [{"timestamp": index * 3600, "close": 100.0 + index}
            for index in range(220)]
    monkeypatch.setattr("scripts.market_signal_audit.continuous_features",
                        lambda *args, **kwargs: {})
    rows = build_rows(
        {"base_asset": "TEST"}, bars, reference=bars,
        reference_times=[row["timestamp"] for row in bars], horizon=12, stride=12,
        auxiliary_horizons=(1, 6, 24),
    )
    assert rows
    assert set(rows[0]["future_returns"]) == {"1", "6", "12", "24"}
    assert rows[-1]["timestamp"] + 24 * 3600 < bars[-1]["timestamp"] + 3600
