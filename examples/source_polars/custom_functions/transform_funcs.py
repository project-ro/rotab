import polars as pl
from typing import Dict


def groupby_agg_custom(table: pl.DataFrame, by: str, aggregations: Dict[str, str]) -> pl.DataFrame:
    aggs = [getattr(pl.col(col), agg)().alias(col) for col, agg in aggregations.items()]
    return table.group_by(by).agg(aggs)
