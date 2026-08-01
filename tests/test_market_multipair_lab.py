from scripts.market_multipair_lab import (
    cluster_asset_time_rows, derive_cutoff, eligible_indices, evenly_spaced, partition_assets,
    market_rows, progress,
)


def records():
    result = []
    for family, assets in (("major", ("WETH", "SOL", "WBTC")),
                           ("defi", ("AAVE", "LINK", "UNI", "MKR"))):
        for number, asset in enumerate(assets):
            result.append({"family": family, "base_asset": asset,
                           "relative_path": f"base/{number}_{asset}-USDC.json",
                           "end_timestamp": 2_000_000_000 + number * 3600})
    return result


def test_asset_partition_never_leaks_one_base_between_sides():
    training, holdout = partition_assets(records(), .34, "fixed")
    train_assets = {row["base_asset"] for row in training}
    holdout_assets = {row["base_asset"] for row in holdout}
    assert holdout_assets
    assert train_assets.isdisjoint(holdout_assets)
    assert "WETH" in train_assets
    assert "WBTC" in train_assets


def test_global_cutoff_purges_every_training_target():
    bars = [{"timestamp": index * 3600.0} for index in range(1000)]
    train, test = eligible_indices(bars, cutoff=800 * 3600.0, horizon=12)
    assert train[-1] + 12 < test[0]
    assert bars[train[-1] + 12]["timestamp"] < 800 * 3600.0
    assert bars[test[0]]["timestamp"] >= 800 * 3600.0


def test_cutoff_and_spacing_are_deterministic():
    cutoff = derive_cutoff(records(), test_n=40, horizon=12)
    assert isinstance(cutoff, float)
    assert evenly_spaced(list(range(100)), 5) == [0, 25, 50, 74, 99]


def test_chain_copies_collapse_to_one_asset_time_vote():
    common = {"base_asset": "AAVE", "timestamp": 1.0, "return": .02,
              "confidence": .6, "latency_seconds": .1, "momentum_direction": 1}
    rows = [
        {**common, "actual": "updraft", "predicted": "updraft"},
        {**common, "actual": "updraft", "predicted": "updraft"},
        {**common, "actual": "downshift", "predicted": "downshift"},
    ]
    collapsed = cluster_asset_time_rows(rows)
    assert len(collapsed) == 1
    assert collapsed[0]["actual"] == "updraft"
    assert collapsed[0]["predicted"] == "updraft"
    assert collapsed[0]["cluster_members"] == 3


def test_detached_progress_pipe_cannot_abort(monkeypatch):
    def broken(*_args, **_kwargs):
        raise BrokenPipeError("detached")

    monkeypatch.setattr("builtins.print", broken)
    progress("still safe")


def test_market_rows_carry_asset_cluster_identity():
    class Client:
        def predict(self, _streams):
            return "future updraft", .5, .01

    bars = [{"timestamp": index * 3600.0, "open": 100 + index / 100,
             "high": 101 + index / 100, "low": 99 + index / 100,
             "close": 100.5 + index / 100, "volume": 10,
             "buy_volume": 6, "sell_volume": 4} for index in range(600)]
    record = {"chain": "base", "symbol": "AAVE-USDC", "base_asset": "AAVE"}
    rows = market_rows(record, bars, [520], Client(), horizon=12, news=[],
                       reference_bars=None, active_pools={2, 5, 9}, cost_bps=20)
    assert rows[0]["base_asset"] == "AAVE"
