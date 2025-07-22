import polars as pl
from typing import Union


def log(col: str, base: float = 10) -> pl.Expr:
    return pl.col(col).log(base)


def log1p(col: str) -> pl.Expr:
    return pl.col(col).log1p().cast(pl.Float64)


def exp(col: str) -> pl.Expr:
    return pl.col(col).exp()


def sqrt(col: str) -> pl.Expr:
    return pl.col(col).sqrt()


def clip(col: str, min_val: float, max_val: float) -> pl.Expr:
    return pl.col(col).clip(min_val, max_val)


def round(col: str, decimals: int = 0) -> pl.Expr:
    return pl.col(col).round(decimals)


def floor(col: str) -> pl.Expr:
    return pl.col(col).floor()


def ceil(col: str) -> pl.Expr:
    return pl.col(col).ceil()


def abs(col: Union[str, pl.Expr]) -> pl.Expr:
    expr = pl.col(col) if isinstance(col, str) else col
    return expr.abs()


def startswith(col: str, prefix: str) -> pl.Expr:
    return pl.col(col).str.starts_with(prefix)


def endswith(col: str, suffix: str) -> pl.Expr:
    return pl.col(col).str.ends_with(suffix)


def lower(col: str) -> pl.Expr:
    return pl.col(col).str.to_lowercase()


def upper(col: str) -> pl.Expr:
    return pl.col(col).str.to_uppercase()


def replace_values(col: str, old: str, new: str) -> pl.Expr:
    return pl.col(col).str.replace(old, new)


def strip(col: str) -> pl.Expr:
    return pl.col(col).str.strip_chars()


def year(col: str) -> pl.Expr:
    return pl.col(col).dt.year()


def month(col: str) -> pl.Expr:
    return pl.col(col).dt.month()


def day(col: str) -> pl.Expr:
    return pl.col(col).dt.day()


def hour(col: str) -> pl.Expr:
    return pl.col(col).dt.hour()


def weekday(col: str) -> pl.Expr:
    return pl.col(col).dt.weekday()


def days_between(col1: str, col2: str) -> pl.Expr:
    return (pl.col(col2).cast(pl.Datetime) - pl.col(col1).cast(pl.Datetime)).dt.total_days()


def is_null(col: str) -> pl.Expr:
    return pl.col(col).is_null()


def not_null(col: str) -> pl.Expr:
    return pl.col(col).is_not_null()


def min(col1: str, col2: str) -> pl.Expr:
    return pl.min_horizontal([pl.col(col1), pl.col(col2)])


def max(col1: str, col2: str) -> pl.Expr:
    return pl.max_horizontal([pl.col(col1), pl.col(col2)])


def len(col: str) -> pl.Expr:
    return pl.col(col).str.len_chars()


def format_timestamp(
    col: str, parse_fmt: str, output_format: str, input_tz: str = None, output_tz: str = None
) -> pl.Expr:
    """Formats a timestamp string into a datetime expression, handling timezones and parsing formats."""
    # 1. Prepare for parsing
    expr = pl.col(col)
    offset_expr = expr.str.extract(r"([\+\-]\d{2}:\d{2})$", 0)
    expr_cleaned = expr.str.replace(r"([\+\-]\d{2}:\d{2})$", "")

    # 2. Determine and complete the date/time format
    is_date_only = not any(token in parse_fmt for token in ("%H", "%M", "%S"))
    fmt = parse_fmt
    if "%d" not in fmt:
        if "%m" in parse_fmt:
            separator = "-" if "-" in parse_fmt else "/" if "/" in parse_fmt else ""
            expr_cleaned = expr_cleaned + f"{separator}01"
            fmt += f"{separator}%d"

    dt_expr = expr_cleaned.str.strptime(pl.Datetime, fmt, strict=False)

    # 3. Always calculate the absolute time in the UTC timezone
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

    # 4. Branch the logic using a Python if-statement to construct the final expression
    final_expr: pl.Expr

    # Case 1: The timezone conversion is straightforward
    if output_tz and not is_date_only:
        final_expr = utc_expr.dt.convert_time_zone(output_tz)
    # Case 2: All other cases (date-only, or no output_tz specified) -> display the original local time
    else:
        # If input_tz is available, use it as a fallback
        if input_tz:
            final_expr = (
                pl.when(offset_expr.is_not_null())
                .then(utc_expr + offset_duration)
                .otherwise(
                    # Convert utc_expr to input_tz to get the "wall clock" time,
                    # then strip the timezone info, and re-label it as UTC.
                    utc_expr.dt.convert_time_zone(input_tz)
                    .dt.replace_time_zone(None)
                    .dt.replace_time_zone("UTC")
                )
            )
        # If input_tz is not available, the fallback remains UTC
        else:
            final_expr = pl.when(offset_expr.is_not_null()).then(utc_expr + offset_duration).otherwise(utc_expr)

    # 5. Finally, format to a string
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
}
