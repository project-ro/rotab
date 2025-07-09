import polars as pl


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


def abs(col: str) -> pl.Expr:
    return pl.col(col).abs()


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


def format_datetime(col: str, fmt: str) -> pl.Expr:
    return pl.col(col).dt.strftime(fmt)


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
