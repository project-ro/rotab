import polars as pl


def str_custom(x: str) -> pl.Expr:
    return pl.col(x).cast(pl.Utf8)
