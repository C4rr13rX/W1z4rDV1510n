import csv
import zipfile

from scripts.market_binance_public import (
    last_complete_month, month_range, normalize_timestamp,
    parse_funding_archive, parse_kline_archive,
)


def write_zip(path, rows):
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(csv_path, arcname="data.csv")
    csv_path.unlink()


def test_month_and_timestamp_normalization():
    assert month_range("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert normalize_timestamp("1735689600000000") == 1735689600
    assert normalize_timestamp("1735689600000") == 1735689600
    assert normalize_timestamp("1735689600") == 1735689600
    assert last_complete_month.__name__ == "last_complete_month"


def test_archive_parsers_accept_headered_and_headerless_rows(tmp_path):
    kline = tmp_path / "kline.zip"
    write_zip(kline, [
        ["open_time", "open", "high", "low", "close", "volume", "close_time",
         "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"],
        ["1735689600000000", "10", "12", "9", "11", "5", "0", "55", "7", "3", "33", "0"],
    ])
    parsed = parse_kline_archive(kline, "spot")
    assert parsed[1735689600]["spot_close"] == 11
    assert parsed[1735689600]["spot_trade_count"] == 7
    funding = tmp_path / "funding.zip"
    write_zip(funding, [
        ["calc_time", "funding_interval_hours", "last_funding_rate"],
        ["1735689600000", "8", "0.0001"],
    ])
    assert parse_funding_archive(funding) == [(1735689600, 0.0001)]
