from scripts.market_multipair_lab import (
    derive_cutoff, eligible_indices, evenly_spaced, partition_assets,
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
