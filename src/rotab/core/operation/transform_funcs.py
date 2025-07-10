import polars as pl
from typing import List, Dict, Any


def normalize_dtype(dtype: str):
    mapping = {
        "int": pl.Int64,
        "float": pl.Float64,
        "str": pl.String,
        "string": pl.String,
        "bool": pl.Boolean,
        "datetime": pl.Datetime,
        "datetime[ns]": pl.Datetime,
        "object": pl.String,
    }
    return mapping.get(dtype, dtype)


def validate_table_schema(df: pl.DataFrame, columns: List[dict]) -> bool:
    supported_dtypes = {pl.Int64, pl.Float64, pl.String, pl.Boolean, pl.Datetime}

    if not isinstance(columns, list) or not all(isinstance(c, dict) for c in columns):
        raise ValueError("Invalid schema: 'columns' must be a list of dicts")

    for col_def in columns:
        col_name = col_def["name"]
        expected = normalize_dtype(col_def["dtype"])

        if expected not in supported_dtypes:
            raise ValueError(f"Unsupported type in schema: {col_def['dtype']} for column: {col_name}")

        if col_name not in df.columns:
            raise ValueError(f"Missing required column: {col_name}")

        actual = df.schema[col_name]

        if actual != expected:
            raise ValueError(
                f"Type mismatch for column '{col_name}': expected {expected.__class__.__name__}, got {actual.__class__.__name__}"
            )

    return True


def sort_by(table: pl.DataFrame, column: str, ascending: bool = True) -> pl.DataFrame:
    return table.sort(by=column, descending=not ascending)


def groupby_agg(table: pl.DataFrame, by: str, aggregations: Dict[str, str]) -> pl.DataFrame:
    aggs = [getattr(pl.col(col), agg)().alias(col) for col, agg in aggregations.items()]
    return table.group_by(by).agg(aggs)


def drop_duplicates(table: pl.DataFrame, subset: List[str] = None) -> pl.DataFrame:
    return table.unique(subset=subset)


def merge(left: pl.DataFrame, right: pl.DataFrame, on: str, how: str = "inner") -> pl.DataFrame:
    return left.join(right, on=on, how=how)


def reshape(
    table: pl.DataFrame,
    column_to: str = None,
    columns_from: List[str] = None,
    column_value: str = None,
    agg: str = None,
) -> pl.DataFrame:
    if agg:
        if columns_from is None or column_value is None:
            raise ValueError("columns_from and column_value must be specified for pivot with aggregation.")
        pivoted = table.pivot(values=column_value, index=column_to, columns=columns_from[0], aggregate_function=agg)
        return pivoted

    elif column_value and columns_from:
        pivoted = table.pivot(values=column_value, index=column_to, columns=columns_from[0])
        return pivoted

    elif column_value and not columns_from:
        melted = table.melt(
            id_vars=[column_to], value_vars=[column_value], variable_name="variable", value_name="melted_value"
        )
        return melted

    else:
        raise ValueError("Invalid combination of parameters for reshape.")


def fillna(table: pl.DataFrame, mapping: Dict[str, Any]) -> pl.DataFrame:
    return table.with_columns([pl.col(k).fill_null(v) for k, v in mapping.items()])


def sample(table: pl.DataFrame, frac: float) -> pl.DataFrame:
    return table.sample(fraction=frac)


def concat(tables: List[pl.DataFrame]) -> pl.DataFrame:
    return pl.concat(tables, how="vertical")


def drop_na(table: pl.DataFrame, subset: List[str] = None) -> pl.DataFrame:
    return table.drop_nulls(subset=subset)


def replace(table: pl.DataFrame, columns: List[str], old: Any, new: Any) -> pl.DataFrame:
    table_copy = table.clone()
    for col in columns:
        table_copy = table_copy.with_columns(pl.col(col).replace(old, new).alias(col))
    return table_copy


TRANSFORM_NAMESPACE = {
    "normalize_dtype": normalize_dtype,
    "validate_table_schema": validate_table_schema,
    "sort_by": sort_by,
    "groupby_agg": groupby_agg,
    "drop_duplicates": drop_duplicates,
    "merge": merge,
    "reshape": reshape,
    "fillna": fillna,
    "sample": sample,
    "concat": concat,
    "drop_na": drop_na,
    "replace": replace,
}
