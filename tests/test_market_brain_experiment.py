from scripts.market_brain_experiment import (
    NewsItem,
    chronological_fold_indices,
    feature_streams,
    return_label,
    target_label,
)


def bars(count=1000):
    return [
        {
            "timestamp": i * 3600.0,
            "open": 100.0 + i / 10,
            "high": 101.0 + i / 10,
            "low": 99.0 + i / 10,
            "close": 100.5 + i / 10,
            "volume": 1000.0 + i,
            "buy_volume": 510.0 + i / 2,
            "sell_volume": 490.0 + i / 2,
        }
        for i in range(count)
    ]


def test_chronological_folds_are_purged_and_ordered():
    folds = chronological_fold_indices(2000, horizon=12, folds=3, test_n=200)
    assert len(folds) == 3
    for train, test in folds:
        assert train.stop + 12 == test.start
        assert train.start == 512
        assert test.stop - test.start == 200
        assert test.stop + 12 <= 2000


def test_feature_streams_are_separate_and_never_read_future_news():
    data = bars()
    now = data[700]["timestamp"]
    news = [
        NewsItem(now - 60, "known catalyst", 1.0, ("WETH", "CATALYST")),
        NewsItem(now + 60, "future exploit", -1.0, ("WETH", "EXPLOIT")),
    ]
    streams = feature_streams(data, 700, symbol="WETH-USDC", chain="base",
                              horizon=12, news=news)
    assert len(streams) == 10
    assert len({pool for pool, _ in streams}) == 10
    joined = " ".join(frame for _, frame in streams)
    assert "catalyst" in joined
    assert "exploit" not in joined


def test_return_labels_are_byte_disjoint():
    labels = [return_label(value) for value in (-.1, -.02, -.005, 0, .005, .02, .1)]
    assert len(set(labels)) == 7
    for left in labels:
        for right in labels:
            if left != right:
                assert left not in right


def test_direction_target_is_disjoint_and_preserves_flat_band():
    assert target_label(-0.01, "direction3") == "downshift"
    assert target_label(0.0, "direction3") == "sideways"
    assert target_label(0.01, "direction3") == "updraft"
