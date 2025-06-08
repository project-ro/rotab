import pytest
import math
import datetime
from rotab.core.operation.new_columns_funcs import *


def test_log():
    assert log(100, 10) == 2
    assert log(math.e, math.e) == 1


def test_log1p():
    assert math.isclose(log1p(1), math.log(2))


def test_exp():
    assert math.isclose(exp(1), math.e)


def test_sqrt():
    assert sqrt(4) == 2


def test_clip():
    assert clip(5, 1, 10) == 5
    assert clip(-1, 0, 10) == 0
    assert clip(20, 0, 10) == 10


def test_round():
    assert round(3.14159, 2) == 3.14
    assert round(3.14159) == 3


def test_floor():
    assert floor(2.9) == 2


def test_ceil():
    assert ceil(2.1) == 3


def test_abs():
    assert abs(-5) == 5


def test_len():
    assert len("hello") == 5
    assert len([1, 2, 3]) == 3


def test_startswith():
    assert startswith("hello", "he") is True
    assert startswith("hello", "no") is False


def test_endswith():
    assert endswith("hello", "lo") is True
    assert endswith("hello", "he") is False


def test_lower():
    assert lower("HELLO") == "hello"


def test_upper():
    assert upper("hello") == "HELLO"


def test_replace_values():
    assert replace_values("aabbcc", "b", "x") == "aaxxcc"


def test_format_datetime():
    dt = datetime.datetime(2020, 1, 1, 12, 0)
    assert format_datetime(dt, "%Y-%m-%d") == "2020-01-01"
    assert format_datetime("2020-01-01T12:00:00", "%H") == "12"


def test_year():
    assert year("2020-01-01T12:00:00") == 2020


def test_month():
    assert month("2020-01-01T12:00:00") == 1


def test_day():
    assert day("2020-01-01T12:00:00") == 1


def test_weekday():
    assert weekday("2020-01-01T12:00:00") == 2  # Wednesday


def test_hour():
    assert hour("2020-01-01T12:34:00") == 12


def test_days_between():
    assert days_between("2020-01-01", "2020-01-10") == 9
    assert days_between("2020-01-10", "2020-01-01") == 9


def test_is_null():
    assert is_null(None) is True
    assert is_null(float("nan")) is True
    assert is_null(0) is False


def test_strip():
    assert strip("  hello  ") == "hello"


def test_not_null():
    assert not_null("value") is True
    assert not_null(None) is False


def test_min():
    assert min(3, 5) == 3


def test_max():
    assert max(3, 5) == 5
