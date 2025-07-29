import polars as pl
from typing import Union
from datetime import datetime

ExprOrStr = Union[str, pl.Expr]


def _col(x: ExprOrStr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def _to_date_expr(x: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    if isinstance(x, str):
        return pl.col(x).str.strptime(pl.Date, fmt, strict=False)
    return x.cast(pl.Date)


def _to_datetime_expr(x: ExprOrStr, fmt: str = "%Y-%m-%d %H:%M:%S") -> pl.Expr:
    if isinstance(x, str):
        return pl.col(x).str.strptime(pl.Datetime, fmt, strict=False)
    return x.cast(pl.Datetime)


def log(x: ExprOrStr, base: float = 10) -> pl.Expr:
    return _col(x).cast(pl.Float64).log(base)


def log1p(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).log1p()


def exp(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).exp()


def sqrt(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).sqrt()


def clip(x: ExprOrStr, min_val: float, max_val: float) -> pl.Expr:
    return _col(x).cast(pl.Float64).clip(min_val, max_val)


def round(x: ExprOrStr, decimals: int = 0) -> pl.Expr:
    return _col(x).cast(pl.Float64).round(decimals)


def floor(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).floor()


def ceil(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).ceil()


def abs(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64).abs()


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


def year(x: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    return _to_date_expr(x, fmt).dt.year()


def month(x: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    return _to_date_expr(x, fmt).dt.month()


def day(x: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    return _to_date_expr(x, fmt).dt.day()


def hour(x: ExprOrStr, fmt: str = "%Y-%m-%d %H:%M:%S") -> pl.Expr:
    return _to_datetime_expr(x, fmt).dt.hour()


def weekday(x: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    return _to_date_expr(x, fmt).dt.weekday()


def _to_date_expr_safe(x: ExprOrStr, fmt: str) -> pl.Expr:
    if isinstance(x, pl.Expr):
        return x.cast(pl.Date)
    elif isinstance(x, str):
        try:
            datetime.strptime(x, fmt)
            return pl.lit(x).str.strptime(pl.Date, fmt, strict=False)
        except ValueError:
            return pl.col(x).cast(pl.Utf8).str.strptime(pl.Date, fmt, strict=False)
    else:
        raise TypeError(f"Unsupported type for date conversion: {type(x)}")


def days_between(x: ExprOrStr, y: ExprOrStr, fmt: str = "%Y-%m-%d") -> pl.Expr:
    x_expr = _to_date_expr_safe(x, fmt)
    y_expr = _to_date_expr_safe(y, fmt)
    return (y_expr - x_expr).dt.total_days()


def is_null(x: ExprOrStr) -> pl.Expr:
    return _col(x).is_null()


def not_null(x: ExprOrStr) -> pl.Expr:
    return _col(x).is_not_null()


def min(x: ExprOrStr, y: ExprOrStr) -> pl.Expr:
    return pl.min_horizontal([_col(x).cast(pl.Float64), _col(y).cast(pl.Float64)])


def max(x: ExprOrStr, y: ExprOrStr) -> pl.Expr:
    return pl.max_horizontal([_col(x).cast(pl.Float64), _col(y).cast(pl.Float64)])


def len(x: ExprOrStr) -> pl.Expr:
    return _col(x).str.len_chars()


def int_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def float_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Float64, strict=False)


def bool_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Boolean, strict=False)


def str_(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Utf8, strict=False)


def substr(x: ExprOrStr, start: int, length: int) -> pl.Expr:
    return _col(x).str.slice(start, length)


def left(x: ExprOrStr, n: int) -> pl.Expr:
    return _col(x).str.slice(0, n)


def right(x: ExprOrStr, n: int) -> pl.Expr:
    return _col(x).str.slice(_col(x).str.len_chars() - n, n)


def contains(x: ExprOrStr, substring: str) -> pl.Expr:
    return _col(x).str.contains(substring)


def days_since_last_birthday(
    x: ExprOrStr,
    ref_date: Union[str, pl.Expr, None] = None,
    fmt: str = "%Y-%m-%d",
) -> pl.Expr:
    birthday = _to_date_expr(x, fmt)

    if ref_date is None:
        ref_expr = _to_date_expr(pl.lit(datetime.today().strftime(fmt)), fmt)
    elif isinstance(ref_date, str):
        try:
            datetime.strptime(ref_date, fmt)
            ref_expr = _to_date_expr(pl.lit(ref_date), fmt)
        except ValueError:
            ref_expr = _to_date_expr(_col(ref_date), fmt)
    else:
        ref_expr = _to_date_expr(ref_date, fmt)

    ref_year = ref_expr.dt.year()
    this_year_birthday_str = birthday.dt.strftime("%m-%d")
    this_year_birthday = pl.concat_str([ref_year.cast(str), pl.lit("-"), this_year_birthday_str]).str.strptime(
        pl.Date, "%Y-%m-%d", strict=False
    )

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
            if "%Y%m" in parse_fmt:
                expr_cleaned = expr_cleaned + "01"
                fmt += "%d"
            elif "%Y-%m" in parse_fmt or "%Y/%m" in parse_fmt:
                separator = "-" if "-" in parse_fmt else "/"
                expr_cleaned = expr_cleaned + f"{separator}01"
                fmt += f"{separator}%d"
            else:
                expr_cleaned = expr_cleaned + "01"
                fmt += "%d"

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


def zfill(x: ExprOrStr, width: int, normalize: bool = False) -> pl.Expr:
    col = pl.col(x) if isinstance(x, str) else x

    if normalize:

        def norm(v):
            try:
                return str(int(float(v)))
            except:
                return None

        return col.map_elements(norm, return_dtype=pl.Utf8).str.zfill(width)
    else:
        return col.cast(pl.Utf8).str.zfill(width)


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
    "zfill": zfill,
}
