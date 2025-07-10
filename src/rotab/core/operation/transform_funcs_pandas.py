import pandas as pd
from typing import List, Dict, Any


def normalize_dtype(dtype: str) -> str:
    """
    user_specified dtype to pandas/polars dtype mapping.
    """
    mapping = {
        "int": "int64",
        "float": "float64",
        "str": "object",
        "string": "object",
        "datetime": "datetime64[ns]",
        "bool": "bool",
        "object": "object",
    }
    return mapping.get(dtype, dtype)


def validate_table_schema(df: pd.DataFrame, columns: List[dict]) -> bool:
    supported_dtypes = {"int64", "float64", "object", "bool", "datetime64[ns]"}

    # 最初に構文チェック（これがないと TypeError になる）
    if not isinstance(columns, list) or not all(isinstance(c, dict) for c in columns):
        raise ValueError("Invalid schema: 'columns' must be a list of dicts")

    for col_def in columns:
        col_name = col_def["name"]
        schema_dtype_raw = col_def["dtype"]
        expected = normalize_dtype(schema_dtype_raw)

        if expected not in supported_dtypes:
            raise ValueError(f"Unsupported type in schema: {schema_dtype_raw} for column: {col_name}")

        if col_name not in df.columns:
            raise ValueError(f"Missing required column: {col_name}")

        actual = str(df[col_name].dtype)
        if actual != expected:
            raise ValueError(f"Type mismatch for column '{col_name}': expected {expected}, got {actual}")

    return True


def sort_by(table: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """
    Sort the table by a specific column.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - column (str): Column to sort by.
    - ascending (bool): Sort order. Default is True (ascending).

    Returns:
    - pd.DataFrame: Sorted dataframe.
    """
    return table.sort_values(by=column, ascending=ascending)


def groupby_agg(table: pd.DataFrame, by: str, aggregations: Dict[str, str]) -> pd.DataFrame:
    """
    Group the table by a column and apply aggregation functions.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - by (str): Column to group by.
    - aggregations (Dict[str, str]): Dictionary mapping column names to aggregation functions.

    Returns:
    - pd.DataFrame: Aggregated dataframe.
    """
    return table.groupby(by).agg(aggregations).reset_index()


def drop_duplicates(table: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the table.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - subset (List[str], optional): Columns to consider for identifying duplicates.

    Returns:
    - pd.DataFrame: DataFrame without duplicates.
    """
    return table.drop_duplicates(subset=subset)


def merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    """
    Merge two dataframes on a common column.

    Parameters:
    - left (pd.DataFrame): Left dataframe.
    - right (pd.DataFrame): Right dataframe.
    - on (str): Column name to join on.
    - how (str): Type of join. Options are 'left', 'right', 'outer', 'inner'. Default is 'inner'.

    Returns:
    - pd.DataFrame: Merged dataframe.
    """
    return pd.merge(left, right, on=on, how=how)


def reshape(
    table: pd.DataFrame,
    column_to: str = None,
    columns_from: List[str] = None,
    column_value: str = None,
    agg: str = None,
) -> pd.DataFrame:
    """
    Reshape a DataFrame using pivot, pivot_table, or melt depending on parameters.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - column_to (str): For pivot/pivot_table: index or id_vars. For melt: id_vars.
    - columns_from (List[str]): For pivot/pivot_table: columns. Not used in melt.
    - column_value (str): For pivot/pivot_table: values. For melt: value_vars.
    - agg (str): If specified, uses pivot_table with aggregation. If None, uses pivot or melt.

    Returns:
    - pd.DataFrame: Reshaped dataframe.
    """
    if agg:
        pivot = table.pivot_table(index=column_to, columns=columns_from, values=column_value, aggfunc=agg)
        if isinstance(pivot.columns, pd.MultiIndex):
            pivot.columns = ["_".join(map(str, col)).strip() for col in pivot.columns.values]
        else:
            pivot.columns = pivot.columns.astype(str)
        pivot.columns.name = None
        return pivot.reset_index()

    elif column_value and columns_from:
        pivot = table.pivot(index=column_to, columns=columns_from[0], values=column_value)
        pivot.columns.name = None
        return pivot.reset_index()

    elif column_value and not columns_from:
        return table.melt(id_vars=column_to, value_vars=[column_value], var_name="variable", value_name="melted_value")

    else:
        raise ValueError("Invalid combination of parameters for reshape.")


def fillna(table: pd.DataFrame, mapping: Dict[str, Any]) -> pd.DataFrame:
    """
    Fill missing values in specified columns with given values.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - mapping (Dict[str, Any]): Dictionary mapping columns to fill values.

    Returns:
    - pd.DataFrame: DataFrame with filled values.
    """
    return table.fillna(value=mapping)


def sample(table: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    Return a random sample of the table.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - frac (float): Fraction of rows to return (e.g., 0.1 for 10%).

    Returns:
    - pd.DataFrame: Sampled dataframe.
    """
    return table.sample(frac=frac)


def concat(tables: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of dataframes vertically.

    Parameters:
    - tables (List[pd.DataFrame]): List of dataframes to concatenate.

    Returns:
    - pd.DataFrame: Concatenated dataframe.
    """
    return pd.concat(tables, ignore_index=True)


def drop_na(table: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Drop rows with missing values.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - subset (List[str], optional): Columns to check for missing values.

    Returns:
    - pd.DataFrame: DataFrame without missing values.
    """
    return table.dropna(subset=subset)


def replace(table: pd.DataFrame, columns: List[str], old: Any, new: Any) -> pd.DataFrame:
    """
    Replace specific values in selected columns.

    Parameters:
    - table (pd.DataFrame): Input dataframe.
    - columns (List[str]): Columns to apply replacement on.
    - old (Any): Value to be replaced.
    - new (Any): Replacement value.

    Returns:
    - pd.DataFrame: Modified dataframe.
    """
    table_copy = table.copy()
    for col in columns:
        table_copy[col] = table_copy[col].replace(old, new)
    return table_copy
