import polars as pl
from typing import Union
from datetime import datetime

ExprOrStr = Union[str, pl.Expr]


def _col(x: ExprOrStr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def str_custom(x: ExprOrStr) -> pl.Expr:
    return _col(x).cast(pl.Utf8)
