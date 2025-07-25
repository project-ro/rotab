import polars as pl
import pytest
import datetime

from rotab.core.operation.derive_funcs_polars import (
    log,
    log1p,
    exp,
    sqrt,
    clip,
    round,
    floor,
    ceil,
    abs,
    len,
    startswith,
    endswith,
    lower,
    upper,
    replace_values,
    year,
    month,
    day,
    weekday,
    hour,
    days_between,
    is_null,
    strip,
    not_null,
    min,
    max,
    format_timestamp,
)


def test_log():
    df = pl.DataFrame({"a": [1, 10, 100]})
    out = df.select(log("a")).to_series()
    assert out[0] == pytest.approx(0)


def test_log1p():
    df = pl.DataFrame({"a": [0, 1, 9]})
    out = df.select(log1p("a")).to_series()
    assert out[1] == pytest.approx(pl.Series([0, 0.6931, 2.3025], dtype=pl.Float64)[1], rel=1e-2)


def test_exp():
    df = pl.DataFrame({"a": [0, 1]})
    out = df.select(exp("a")).to_series()
    assert out[1] == pytest.approx(2.7182, rel=1e-2)


def test_sqrt():
    df = pl.DataFrame({"a": [0, 4, 9]})
    out = df.select(sqrt("a")).to_series()
    assert out[1] == 2


def test_clip():
    df = pl.DataFrame({"a": [0, 5, 10]})
    out = df.select(clip("a", 3, 7)).to_series()
    assert out[0] == 3


def test_round():
    df = pl.DataFrame({"a": [1.2345]})
    out = df.select(round("a", 2)).to_series()
    assert out[0] == 1.23


def test_floor():
    df = pl.DataFrame({"a": [1.9]})
    out = df.select(floor("a")).to_series()
    assert out[0] == 1


def test_ceil():
    df = pl.DataFrame({"a": [1.1]})
    out = df.select(ceil("a")).to_series()
    assert out[0] == 2


def test_abs():
    df = pl.DataFrame({"a": [-5]})
    out = df.select(abs("a")).to_series()
    assert out[0] == 5


def test_len():
    df = pl.DataFrame({"a": ["abc", "de", "f"]})
    out = df.select(len("a")).to_series()
    assert out.to_list() == [3, 2, 1]


def test_startswith():
    df = pl.DataFrame({"a": ["apple", "banana"]})
    out = df.select(startswith("a", "a")).to_series()
    assert out[0] is True


def test_endswith():
    df = pl.DataFrame({"a": ["test", "best"]})
    out = df.select(endswith("a", "t")).to_series()
    assert all(out)


def test_lower():
    df = pl.DataFrame({"a": ["ABC"]})
    out = df.select(lower("a")).to_series()
    assert out[0] == "abc"


def test_upper():
    df = pl.DataFrame({"a": ["abc"]})
    out = df.select(upper("a")).to_series()
    assert out[0] == "ABC"


def test_replace_values():
    df = pl.DataFrame({"a": ["abc"]})
    out = df.select(replace_values("a", "a", "z")).to_series()
    assert out[0] == "zbc"


def test_strip():
    df = pl.DataFrame({"a": [" abc "]})
    out = df.select(strip("a")).to_series()
    assert out[0] == "abc"


def test_year():
    df = pl.DataFrame({"a": [datetime.datetime(2020, 5, 1)]})
    out = df.select(year("a")).to_series()
    assert out[0] == 2020


def test_month():
    df = pl.DataFrame({"a": [datetime.datetime(2020, 5, 1)]})
    out = df.select(month("a")).to_series()
    assert out[0] == 5


def test_day():
    df = pl.DataFrame({"a": [datetime.datetime(2020, 5, 15)]})
    out = df.select(day("a")).to_series()
    assert out[0] == 15


def test_hour():
    df = pl.DataFrame({"a": [datetime.datetime(2020, 5, 1, 13)]})
    out = df.select(hour("a")).to_series()
    assert out[0] == 13


def test_weekday():
    df = pl.DataFrame({"a": [datetime.datetime(2020, 5, 1)]})
    out = df.select(weekday("a")).to_series()
    print(out)
    assert out[0] == 5  # Friday (1=Monday, 7=Sunday)


def test_days_between():
    df = pl.DataFrame({"d1": [datetime.datetime(2020, 5, 1)], "d2": [datetime.datetime(2020, 5, 10)]})
    out = df.select(days_between("d1", "d2")).to_series()
    assert out[0] == 9


def test_is_null():
    df = pl.DataFrame({"a": [None, 1]})
    out = df.select(is_null("a")).to_series()
    assert out[0] is True


def test_not_null():
    df = pl.DataFrame({"a": [None, 1]})
    out = df.select(not_null("a")).to_series()
    assert out[1] is True


def test_min():
    df = pl.DataFrame({"a": [1], "b": [2]})
    out = df.select(min("a", "b")).to_series()
    assert out[0] == 1


def test_max():
    df = pl.DataFrame({"a": [1], "b": [2]})
    out = df.select(max("a", "b")).to_series()
    assert out[0] == 2


@pytest.mark.parametrize(
    "input_val,parse_fmt,output_format,input_tz,output_tz,expected",
    [
        ("2025-07-19T14:30:00+09:00", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S", None, "UTC", "2025-07-19T05:30:00"),
        ("2025-07-19T14:30:00", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "Asia/Tokyo", "UTC", "2025-07-19T05:30:00"),
        ("2025-07-19T14:30:00", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S", "UTC", "Asia/Tokyo", "2025-07-19T23:30:00"),
        ("2020-03-12", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "Asia/Tokyo", "UTC", "2020-03-12T00:00:00"),
        ("2020年3月12日", "%Y年%m月%d日", "%Y-%m-%dT%H:%M:%S", "Asia/Tokyo", "UTC", "2020-03-12T00:00:00"),
        ("2025-07-19T14:30:00+09:00", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d", None, "UTC", "2025/07/19"),
        (
            "2025-01-01T08:59:59+09:00",
            "%Y-%m-%dT%H:%M:%S",
            "%Y年%m月",
            None,
            "UTC",
            "2024年12月",
        ),
        ("2025-07-19T14:30:00", "%Y-%m-%dT%H:%M:%S", "%H時%M分", "UTC", "Asia/Tokyo", "23時30分"),
        ("2025-07-19T14:30:00", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M", "Asia/Tokyo", None, "2025/07/19 14:30"),
    ],
)
def test_format_timestamp_cases(input_val, parse_fmt, output_format, input_tz, output_tz, expected):
    df = pl.DataFrame({"ts": [input_val]})
    result = df.with_columns(
        [
            format_timestamp(
                "ts",
                parse_fmt,
                output_format=output_format,
                input_tz=input_tz,
                output_tz=output_tz,
            ).alias("parsed")
        ]
    )
    assert result["parsed"][0] == expected
