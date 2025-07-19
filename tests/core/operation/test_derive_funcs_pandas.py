# test_builtin_functions.py

import math
import datetime

from rotab.core.operation.derive_funcs_pandas import (
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
    format_timestamp,
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
)


def test_log():
    assert math.isclose(log(100, 10), 2.0)


def test_log1p():
    assert math.isclose(log1p(1), math.log(2))


def test_exp():
    assert math.isclose(exp(1), math.e)


def test_sqrt():
    assert math.isclose(sqrt(16), 4.0)


def test_clip():
    assert clip(5, 1, 10) == 5
    assert clip(-5, 1, 10) == 1
    assert clip(15, 1, 10) == 10


def test_round():
    assert round(3.14159, 2) == 3.14


def test_floor():
    assert floor(2.7) == 2


def test_ceil():
    assert ceil(2.1) == 3


def test_abs():
    assert abs(-5) == 5


def test_len():
    assert len("hello") == 5


def test_startswith():
    assert startswith("hello", "he")


def test_endswith():
    assert endswith("hello", "lo")


def test_lower():
    assert lower("ABC") == "abc"


def test_upper():
    assert upper("abc") == "ABC"


def test_replace_values():
    assert replace_values("a-b-c", "-", "_") == "a_b_c"


def test_format_datetime():
    dt = datetime.datetime(2023, 6, 1, 12, 0)
    assert format_timestamp(dt, "%Y-%m-%d") == "2023-06-01"
    assert format_timestamp("2023-06-01T12:00:00", "%Y-%m-%d") == "2023-06-01"


def test_year():
    assert year("2023-06-01T12:00:00") == 2023


def test_month():
    assert month("2023-06-01T12:00:00") == 6


def test_day():
    assert day("2023-06-01T12:00:00") == 1


def test_weekday():
    assert weekday("2023-06-01T12:00:00") == 3  # Thursday


def test_hour():
    assert hour("2023-06-01T12:34:00") == 12


def test_days_between():
    assert days_between("2023-06-01", "2023-06-10") == 9
    assert days_between("2023-06-10", "2023-06-01") == 9


def test_is_null():
    assert is_null(None)
    assert is_null(float("nan"))
    assert not is_null(0)


def test_strip():
    assert strip("  abc  ") == "abc"


def test_not_null():
    assert not_null("a")
    assert not not_null(None)


def test_min():
    assert min(1, 2) == 1


def test_max():
    assert max(1, 2) == 2
