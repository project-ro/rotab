import polars as pl
from typing import Union
from datetime import datetime

ExprOrStr = Union[str, pl.Expr]


def _col(x: ExprOrStr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def log(x: ExprOrStr, base: float = 10) -> pl.Expr:
    return _col(x).log(base)


def log1p(x: ExprOrStr) -> pl.Expr:
    return _col(x).log1p().cast(pl.Float64)


def exp(x: ExprOrStr) -> pl.Expr:
    return _col(x).exp()


def sqrt(x: ExprOrStr) -> pl.Expr:
    return _col(x).sqrt()


def clip(x: ExprOrStr, min_val: float, max_val: float) -> pl.Expr:
    return _col(x).clip(min_val, max_val)


def round(x: ExprOrStr, decimals: int = 0) -> pl.Expr:
    return _col(x).round(decimals)


def floor(x: ExprOrStr) -> pl.Expr:
    return _col(x).floor()


def ceil(x: ExprOrStr) -> pl.Expr:
    return _col(x).ceil()


def abs(x: ExprOrStr) -> pl.Expr:
    return _col(x).abs()


def startswith(x: ExprOrStr, prefix: str) -> pl.Expr:
    return _col(x).str.starts_with(prefix)


def endswith(x: ExprOrStr, suffix: str) -> pl.Expr:
    return _col(x).str.ends_with(suffix)


def lower(x: ExprOrStr) -> pl.Expr:
    return _col(x).str.to_lowercase()


def upper(x: ExprOrStr) -> pl.Expr:
    return _col(x).str.to_uppercase()


def replace_values(x: ExprOrStr, old: str, new: str) -> pl.Expr:
    return _col(x).str.replace(old, new)


def strip(x: ExprOrStr) -> pl.Expr:
    return _col(x).str.strip_chars()


def year(x: ExprOrStr) -> pl.Expr:
    return _col(x).dt.year()


def month(x: ExprOrStr) -> pl.Expr:
    return _col(x).dt.month()


def day(x: ExprOrStr) -> pl.Expr:
    return _col(x).dt.day()


def hour(x: ExprOrStr) -> pl.Expr:
    return _col(x).dt.hour()


def weekday(x: ExprOrStr) -> pl.Expr:
    return _col(x).dt.weekday()


def days_between(x: ExprOrStr, y: ExprOrStr) -> pl.Expr:
    return (_col(y).cast(pl.Datetime) - _col(x).cast(pl.Datetime)).dt.total_days()


def is_null(x: ExprOrStr) -> pl.Expr:
    return _col(x).is_null()


def not_null(x: ExprOrStr) -> pl.Expr:
    return _col(x).is_not_null()


def min(x: ExprOrStr, y: ExprOrStr) -> pl.Expr:
    return pl.min_horizontal([_col(x), _col(y)])


def max(x: ExprOrStr, y: ExprOrStr) -> pl.Expr:
    return pl.max_horizontal([_col(x), _col(y)])


def len(x: ExprOrStr) -> pl.Expr:
    return _col(x).str.len_chars()


def str_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Utf8)


def int_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Int64)


def float_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64)


def bool_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Boolean)


def substr(x: ExprOrStr, start: int, length: int) -> pl.Expr:
    return _col(x).str.slice(start, length)


def left(x: ExprOrStr, n: int) -> pl.Expr:
    return _col(x).str.slice(0, n)


def right(x: ExprOrStr, n: int) -> pl.Expr:
    return _col(x).str.slice(_col(x).str.len_chars() - n, n)


def contains(x: ExprOrStr, substring: str) -> pl.Expr:
    return _col(x).str.contains(substring)


def days_since_last_birthday(x: ExprOrStr, ref_date: str = None) -> pl.Expr:
    if ref_date is None:
        ref_date = datetime.today().strftime("%Y-%m-%d")

    ref_dt = datetime.strptime(ref_date, "%Y-%m-%d")
    ref_year = ref_dt.year

    ref_expr = pl.lit(ref_date).str.strptime(pl.Date, "%Y-%m-%d")

    this_year_birthday_str = _col(x).str.strptime(pl.Date, "%Y-%m-%d").dt.strftime(f"{ref_year}-%m-%d")
    this_year_birthday = this_year_birthday_str.str.strptime(pl.Date, "%Y-%m-%d")

    last_birthday = (
        pl.when(this_year_birthday > ref_expr)
        .then(this_year_birthday - pl.duration(days=365))
        .otherwise(this_year_birthday)
    )

    return (ref_expr - last_birthday).dt.total_days()


def format_timestamp(
    x: ExprOrStr, parse_fmt: str, output_format: str, input_tz: str = None, output_tz: str = None
) -> pl.Expr:
    expr = _col(x)
    offset_expr = expr.str.extract(r"([\+\-]\d{2}:\d{2})$", 0)
    expr_cleaned = expr.str.replace(r"([\+\-]\d{2}:\d{2})$", "")

    is_date_only = not any(token in parse_fmt for token in ("%H", "%M", "%S"))
    fmt = parse_fmt
    if "%d" not in fmt:
        if "%m" in parse_fmt:
            separator = "-" if "-" in parse_fmt else "/" if "/" in parse_fmt else ""
            expr_cleaned = expr_cleaned + f"{separator}01"
            fmt += f"{separator}%d"

    dt_expr = expr_cleaned.str.strptime(pl.Datetime, fmt, strict=False)

    offset_hour = expr.str.extract(r"([\+\-]\d{2}):\d{2}$", 1).cast(pl.Int32)
    offset_minute = expr.str.extract(r":(\d{2})$", 1).cast(pl.Int32)
    offset_duration = pl.duration(minutes=(offset_hour.sign() * (offset_hour.abs() * 60 + offset_minute)))

    utc_expr = (
        pl.when(offset_expr.is_not_null())
        .then((dt_expr - offset_duration).dt.replace_time_zone("UTC"))
        .otherwise(
            pl.when(pl.lit(input_tz).is_not_null())
            .then(dt_expr.dt.replace_time_zone(input_tz).dt.convert_time_zone("UTC"))
            .otherwise(dt_expr.dt.replace_time_zone("UTC"))
        )
    )

    if output_tz and not is_date_only:
        final_expr = utc_expr.dt.convert_time_zone(output_tz)
    else:
        if input_tz:
            final_expr = (
                pl.when(offset_expr.is_not_null())
                .then(utc_expr + offset_duration)
                .otherwise(
                    utc_expr.dt.convert_time_zone(input_tz).dt.replace_time_zone(None).dt.replace_time_zone("UTC")
                )
            )
        else:
            final_expr = pl.when(offset_expr.is_not_null()).then(utc_expr + offset_duration).otherwise(utc_expr)

    return final_expr.dt.strftime(output_format)


FUNC_NAMESPACE = {
    "log": log,
    "log1p": log1p,
    "exp": exp,
    "sqrt": sqrt,
    "clip": clip,
    "round": round,
    "floor": floor,
    "ceil": ceil,
    "abs": abs,
    "startswith": startswith,
    "endswith": endswith,
    "lower": lower,
    "upper": upper,
    "replace_values": replace_values,
    "strip": strip,
    "year": year,
    "month": month,
    "day": day,
    "hour": hour,
    "weekday": weekday,
    "days_between": days_between,
    "is_null": is_null,
    "not_null": not_null,
    "min": min,
    "max": max,
    "len": len,
    "format_timestamp": format_timestamp,
    "str": str_,
    "int": int_,
    "float": float_,
    "bool": bool_,
    "substr": substr,
    "left": left,
    "right": right,
    "days_since_last_birthday": days_since_last_birthday,
    "contains": contains,
}
