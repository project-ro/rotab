import polars as pl
from typing import List, Dict, Any, Optional
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple
import polars as pl
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
import shap
import joblib
from pathlib import Path
from typing import Union
from sklearn.metrics import mean_squared_error, r2_score


def print_all_null_columns(df: pl.DataFrame) -> None:
    for col in df.columns:
        if df.select(pl.col(col).is_null().all()).item():
            print(f"[Warning] Column '{col}' is entirely null.")


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


from typing import Dict, List, Union

import polars as pl


def groupby_agg(
    table: pl.DataFrame,
    by: Union[str, List[str]],
    aggregations: Dict[str, str],
) -> pl.DataFrame:
    aggs = [getattr(pl.col(col), agg)().alias(col) for col, agg in aggregations.items()]
    return table.group_by(by).agg(aggs)


def drop_duplicates(table: pl.DataFrame, subset: List[str] = None) -> pl.DataFrame:
    return table.unique(subset=subset)


def merge(left: pl.DataFrame, right: pl.DataFrame, on: Union[str, List[str]], how: str = "inner") -> pl.DataFrame:
    if isinstance(on, str):
        on = [on]
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


def fillna(
    table: pl.DataFrame,
    values: Optional[Dict[str, Any]] = None,
    strategies: Optional[Dict[str, str]] = None,
) -> pl.DataFrame:
    """
    Fill null values in the given DataFrame using either fixed values or predefined strategies.

    Args:
        table (pl.DataFrame): The input Polars DataFrame.
        values (Optional[Dict[str, Any]]): A dictionary mapping column names to fixed fill values.
            These take precedence over strategies if both are provided for the same column.
        strategies (Optional[Dict[str, str]]): A dictionary mapping column names to fill strategies.
            Supported strategies:
                - "mean": fill with the mean of the column
                - "median": fill with the median of the column
                - "mode": fill with the most frequent value
                - "min": fill with the minimum value
                - "max": fill with the maximum value
                - "zero": fill with 0
                - "forward": forward fill
                - "backward": backward fill

    Returns:
        pl.DataFrame: A new DataFrame with null values filled accordingly.

    Raises:
        ValueError: If an unknown strategy is provided.
    """
    values = values or {}
    strategies = strategies or {}

    def get_fill_value(col: pl.Series, method: str):
        method = method.lower()
        if method == "mean":
            return col.mean()
        elif method == "median":
            return col.median()
        elif method == "mode":
            mode_vals = col.mode()
            return mode_vals[0] if len(mode_vals) > 0 else None
        elif method == "min":
            return col.min()
        elif method == "max":
            return col.max()
        elif method == "zero":
            return 0
        else:
            raise ValueError(f"Unknown strategy: {method}")

    filled_columns = []

    for col_name in table.columns:
        if col_name in values:
            val = values[col_name]
            filled_columns.append(pl.col(col_name).fill_null(val))
        elif col_name in strategies:
            method = strategies[col_name].lower()

            if method in {"forward", "backward"}:
                filled_columns.append(pl.col(col_name).fill_null(strategy=method))
            else:
                val = get_fill_value(table[col_name], method)
                filled_columns.append(pl.col(col_name).fill_null(val))
        else:
            filled_columns.append(pl.col(col_name))

    return table.with_columns(filled_columns)


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


def unique(df: pl.DataFrame, group_keys: list[str], sort_by: str, ascending: bool = True) -> pl.DataFrame:
    return df.sort(group_keys + [sort_by], descending=not ascending).group_by(group_keys, maintain_order=True).first()


def _parse_date_column(column: pl.Expr, fmt: str) -> pl.Expr:
    if "%d" not in fmt:
        last_char = fmt[-1]
        is_delimited = not last_char.isalpha()

        if is_delimited:
            column = column + f"{last_char}01"
            fmt += f"{last_char}%d"
        else:
            column = column + "01"
            fmt += "%d"

    return column.str.strptime(pl.Datetime, fmt)


import polars as pl
from typing import Union, List, Optional


def month_window(
    df_base: Union[pl.LazyFrame, pl.DataFrame],
    df_data: Union[pl.LazyFrame, pl.DataFrame],
    date_col: str,
    date_format: str,
    value_cols: List[str],
    months_list: List[int],
    new_col_name_prefix: str = "future_value",
    metrics: Optional[List[str]] = None,
    keys: Optional[List[str]] = None,
) -> Union[pl.LazyFrame, pl.DataFrame]:
    metrics = metrics or ["mean", "sum", "max"]
    keys = keys or []

    base_was_lazy = isinstance(df_base, pl.LazyFrame)
    data_was_lazy = isinstance(df_data, pl.LazyFrame)
    return_lazy = base_was_lazy if (base_was_lazy != data_was_lazy) else base_was_lazy
    df_base = df_base.lazy() if not base_was_lazy else df_base
    df_data = df_data.lazy() if not data_was_lazy else df_data

    # scratch names（外部へ絶対に漏らさない）
    _base_date = "__mw_base_date__"
    _anchor = "__mw_anchor__"
    _data_date_raw = "__mw_data_date_raw__"
    _data_date = "__mw_data_date__"
    _window_start = "__mw_window_start__"
    _window_end = "__mw_window_end__"
    _join_key = "__mw_join_key__"

    # base: parse + key + anchor
    df_base_prep = (
        df_base.with_columns(
            [
                pl.col(date_col)
                .str.strptime(pl.Datetime, date_format, strict=False)
                .cast(pl.Datetime("us"))
                .alias(_base_date),
                (pl.concat_str([pl.col(k) for k in keys], separator="|") if keys else pl.lit("")).alias(_join_key),
            ]
        )
        .with_columns(pl.col(_base_date).alias(_anchor))
        .sort([_join_key, _base_date])
    )
    df_base_for_join = df_base_prep.select([_join_key, _base_date, _anchor])

    # data: parse + key
    df_data_parsed = df_data.with_columns(
        [
            pl.col(date_col)
            .str.strptime(pl.Datetime, date_format, strict=False)
            .cast(pl.Datetime("us"))
            .alias(_data_date_raw),
            (pl.concat_str([pl.col(k) for k in keys], separator="|") if keys else pl.lit("")).alias(_join_key),
        ]
    )

    results = []
    expected_cols = []  # ★追加：期待列の収集
    for m in months_list:
        suffix = f"_{abs(m)}m"
        direction = "past" if m < 0 else "future"
        join_strategy = "forward" if m < 0 else "backward"

        df_data_corrected = df_data_parsed.with_columns(
            (pl.col(_data_date_raw) + (pl.duration(microseconds=1) if m < 0 else pl.duration(microseconds=0))).alias(
                _data_date
            )
        ).sort([_join_key, _data_date])

        df_joined = df_data_corrected.join_asof(
            df_base_for_join,
            left_on=_data_date,
            right_on=_base_date,
            by=_join_key,
            strategy=join_strategy,
            suffix="_r",
        )

        df_joined = df_joined.with_columns(
            [
                pl.when(m < 0)
                .then(pl.col(_anchor).dt.offset_by(f"{m}mo"))
                .otherwise(pl.col(_anchor))
                .alias(_window_start),
                pl.when(m < 0)
                .then(pl.col(_anchor))
                .otherwise(pl.col(_anchor).dt.offset_by(f"{m}mo"))
                .alias(_window_end),
            ]
        )

        df_filtered = df_joined.filter(
            (pl.col(_data_date) >= pl.col(_window_start)) & (pl.col(_data_date) < pl.col(_window_end))
        )

        aggs = []
        for col in value_cols:
            for metric in metrics:
                cname = f"{new_col_name_prefix}_{col}_{metric}_{direction}{suffix}"
                aggs.append(getattr(pl.col(col), metric)().alias(cname))
                expected_cols.append(cname)  # ★収集

        group_keys = (keys + [_anchor]) if keys else [_anchor]
        results.append(df_filtered.group_by(group_keys).agg(aggs))

    # join
    final_lf = df_base_prep
    join_keys = (keys + [_anchor]) if keys else [_anchor]
    for res in results:
        final_lf = final_lf.join(res, on=join_keys, how="left")

    # 足りない列を None で補完
    for c in expected_cols:
        if c not in final_lf.columns:
            final_lf = final_lf.with_columns(pl.lit(None).alias(c))

    # 掃除
    final_lf = final_lf.drop([_base_date, _join_key, _anchor])

    return final_lf if return_lazy else final_lf.collect()


def is_date_column(series: pl.Series, fmt: str = "%Y-%m-%d") -> bool:
    if series.is_empty() or series.null_count() == len(series):
        return True

    non_null_str_series = series.drop_nulls().cast(str)

    if non_null_str_series.is_empty():
        return True

    try:
        parsed = non_null_str_series.str.strptime(pl.Date, fmt=fmt, strict=False)
    except TypeError:
        try:
            parsed = non_null_str_series.str.strptime(pl.Date, format=fmt, strict=False)
        except Exception:
            return False
    except Exception:
        return False

    return parsed.drop_nulls().len() == len(non_null_str_series)


def is_float_column(series: pl.Series) -> bool:
    try:
        temp_series = series.drop_nulls().cast(str)
        if temp_series.is_empty():
            return True
        return temp_series.cast(pl.Float64, strict=False).null_count() == 0
    except Exception:
        return False


def is_int_column(series: pl.Series) -> bool:
    try:
        temp_series = series.drop_nulls().cast(str)
        if temp_series.is_empty():
            return True
        return temp_series.cast(pl.Int64, strict=False).null_count() == 0
    except Exception:
        return False


def _get_item_or_scalar(polars_result: Any) -> Any:
    """Safely extracts item from Polars Series or returns scalar directly."""
    if isinstance(polars_result, pl.Series):
        if polars_result.len() == 1 and not polars_result.is_null()[0]:
            return polars_result.item()
        return None  # Series is empty or contains null, return None
    return polars_result  # Already a scalar


def describe(df: pl.DataFrame, date_format: str = None) -> pl.DataFrame:
    summaries = []

    full_summary_keys = {
        "column",
        "dtype",
        "mean",
        "std",
        "min",
        "max",
        "Q1",
        "median",
        "Q3",
        "zeros",
        "infinite",
        "top",
        "top_freq",
        "top_ratio",
        "min_cat",
        "min_freq",
        "min_ratio",
        "avg_length",
        "min_length",
        "max_length",
        "n_unique",
        "n_nulls",
        "min_year",
        "max_year",
        "min_month",
        "max_month",
        "mode",
        "mode_freq",
        "mode_ratio",
        "range_days",
    }

    for col in df.columns:
        series = df[col]
        nulls = series.null_count()
        n_unique = series.n_unique()

        current_summary_data = {"column": col, "n_unique": n_unique, "n_nulls": nulls}
        for key in full_summary_keys:
            if key not in current_summary_data:
                current_summary_data[key] = None

        summarized = False

        if is_date_column(series, date_format):
            parsed_series_dt = None

            formats_to_try = []
            if date_format:
                formats_to_try.append(date_format)
            formats_to_try.extend(["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m", "%Y"])

            seen = set()
            unique_formats = []
            for fmt in formats_to_try:
                if fmt not in seen:
                    unique_formats.append(fmt)
                    seen.add(fmt)

            for fmt in unique_formats:
                temp_series_str = series.cast(pl.String).str.strip_chars()
                temp_parsed = _parse_date_column(temp_series_str, fmt)

                if (series.len() - nulls) > 0 and temp_parsed.drop_nulls().len() / (series.len() - nulls) >= 0.8:
                    parsed_series_dt = temp_parsed
                    break

            if parsed_series_dt is not None:
                # min/max/modeの値をPythonのdateオブジェクトとして取得し、strftimeでフォーマット
                min_date_obj = _get_item_or_scalar(parsed_series_dt.min())
                max_date_obj = _get_item_or_scalar(parsed_series_dt.max())

                current_summary_data.update(
                    {
                        "dtype": "date",
                        "min": min_date_obj.strftime("%Y-%m-%d") if min_date_obj is not None else None,
                        "max": max_date_obj.strftime("%Y-%m-%d") if max_date_obj is not None else None,
                        "min_year": (
                            str(_get_item_or_scalar(parsed_series_dt.dt.year().min()))
                            if not parsed_series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "max_year": (
                            str(_get_item_or_scalar(parsed_series_dt.dt.year().max()))
                            if not parsed_series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "min_month": (
                            str(_get_item_or_scalar(parsed_series_dt.dt.month().min()))
                            if not parsed_series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "max_month": (
                            str(_get_item_or_scalar(parsed_series_dt.dt.month().max()))
                            if not parsed_series_dt.drop_nulls().is_empty()
                            else None
                        ),
                    }
                )

                date_only_series = parsed_series_dt.drop_nulls()
                mode_counts = date_only_series.value_counts().sort("count", descending=True)
                if not mode_counts.is_empty():
                    mode_val_date_obj = _get_item_or_scalar(mode_counts[0, date_only_series.name])
                    mode_freq = _get_item_or_scalar(mode_counts[0, "count"])

                    mode_ratio = float(mode_freq) / len(date_only_series) if len(date_only_series) > 0 else 0.0

                    current_summary_data.update(
                        {
                            "mode": mode_val_date_obj.strftime("%Y-%m-%d") if mode_val_date_obj is not None else None,
                            "mode_freq": str(mode_freq),
                            "mode_ratio": str(mode_ratio),
                        }
                    )
                else:
                    current_summary_data.update({"mode": None, "mode_freq": None, "mode_ratio": None})

                if current_summary_data["min"] is not None and current_summary_data["max"] is not None:
                    try:
                        # ここではすでにYYYY-MM-DD形式の文字列になっているので、date.fromisoformatで安全に変換できる
                        temp_min_date = date.fromisoformat(current_summary_data["min"])
                        temp_max_date = date.fromisoformat(current_summary_data["max"])
                        range_days_val = (temp_max_date - temp_min_date).days
                        current_summary_data["range_days"] = str(range_days_val)
                    except ValueError:
                        current_summary_data["range_days"] = None
                else:
                    current_summary_data["range_days"] = None

                summarized = True
            else:
                current_summary_data["dtype"] = "error"
                summarized = True

            if summarized:
                summaries.append(current_summary_data)
                continue

        elif is_float_column(series):
            series_float = series.cast(pl.String).cast(pl.Float64, strict=False)

            current_summary_data.update(
                {
                    "mean": str(_get_item_or_scalar(series_float.mean())),
                    "std": str(_get_item_or_scalar(series_float.std())),
                    "min": str(_get_item_or_scalar(series_float.min())),
                    "max": str(_get_item_or_scalar(series_float.max())),
                    "Q1": str(_get_item_or_scalar(series_float.quantile(0.25, "nearest"))),
                    "median": str(_get_item_or_scalar(series_float.median())),
                    "Q3": str(_get_item_or_scalar(series_float.quantile(0.75, "nearest"))),
                    "zeros": str(_get_item_or_scalar((series_float == 0).sum())),
                    "infinite": str(_get_item_or_scalar(series_float.is_infinite().sum())),
                }
            )

            if is_int_column(series):
                current_summary_data["dtype"] = "int"
            else:
                current_summary_data["dtype"] = "float"
            summarized = True
            if summarized:
                summaries.append(current_summary_data)
                continue

        else:
            series_str = series.drop_nulls().cast(pl.String)
            vc = series_str.value_counts().sort("count", descending=True)

            top = top_freq = min_cat = min_freq = None
            if not vc.is_empty():
                top = str(_get_item_or_scalar(vc[0, series_str.name]))
                top_freq = str(_get_item_or_scalar(vc[0, "count"]))

                if vc.height > 1:
                    min_cat = str(_get_item_or_scalar(vc[-1, series_str.name]))
                    min_freq = str(_get_item_or_scalar(vc[-1, "count"]))
                elif vc.height == 1:
                    min_cat = top
                    min_freq = top_freq

            lengths = series_str.str.len_chars()

            current_summary_data.update(
                {
                    "dtype": "string",
                    "top": top,
                    "top_freq": top_freq,
                    "top_ratio": (
                        str(float(top_freq) / len(series)) if top_freq is not None and len(series) > 0 else None
                    ),
                    "min_cat": min_cat,
                    "min_freq": min_freq,
                    "min_ratio": (
                        str(float(min_freq) / len(series)) if min_freq is not None and len(series) > 0 else None
                    ),
                    "avg_length": str(_get_item_or_scalar(lengths.mean())) if lengths.len() > 0 else None,
                    "min_length": str(_get_item_or_scalar(lengths.min())) if lengths.len() > 0 else None,
                    "max_length": str(_get_item_or_scalar(lengths.max())) if lengths.len() > 0 else None,
                }
            )
            summaries.append(current_summary_data)

    final_schema = {k: pl.String for k in full_summary_keys}
    final_summary_df = pl.DataFrame(summaries, schema=final_schema)

    return final_summary_df


def get_categorical_counts_table(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    if column_name not in df.columns:
        raise ValueError(f"Error: The specified column '{column_name}' does not exist in the DataFrame.")

    series = df[column_name]
    non_null_series = series.drop_nulls()

    if non_null_series.is_empty():
        print(
            f"Warning: Column '{column_name}' contains only null values or is empty, so aggregation cannot be performed."
        )
        return pl.DataFrame()

    # Sort by 'count' descending, then by the original column_name (categories) ascending for stable order
    counts_df = non_null_series.value_counts().sort(
        ["count", column_name], descending=[True, False]  # Added secondary sort
    )
    return counts_df


def plot_categorical_bar_chart(
    categories: np.ndarray, counts: np.ndarray, column_name: str, output_filename: str = None
):
    if categories.size == 0 or counts.size == 0:
        print(f"Warning: No data available for column '{column_name}' to create a chart.")
        return

    data_list = sorted(zip(categories, counts), key=lambda x: (-x[1], x[0]))
    sorted_categories = [item[0] for item in data_list]
    sorted_counts = [item[1] for item in data_list]

    bar_trace = go.Bar(y=sorted_categories, x=sorted_counts, orientation="h", marker_color="steelblue")

    fig = go.Figure(data=[bar_trace])

    fig.update_layout(
        title={
            "text": f"Frequency of Categories for Column: {column_name}",
            "font_size": 24,
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title={"text": "Count", "font_size": 18},
        yaxis_title={"text": "Category", "font_size": 18},
        xaxis=dict(tickfont=dict(size=16)),
        yaxis=dict(tickfont=dict(size=18), automargin=True),
        width=1200,
        height=800,
    )

    try:
        fig.write_html(output_filename, auto_open=False)
        print(f"Horizontal bar chart for '{column_name}' saved as '{output_filename}'.")
        print(f"Please open '{output_filename}' in your web browser to view the chart.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the HTML file: {e}")


def plot_numerical_distribution(data: np.ndarray, column_name: str, output_filename: str = None):
    # Check for empty data
    if data.size == 0:
        print(f"Warning: No data available for column '{column_name}' to create a chart.")
        return

    # Create subplots: 2 rows, 1 column for histogram and boxplot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,  # Share the X-axis for better comparison
        vertical_spacing=0.13,  # Space between subplots
        subplot_titles=(f"Histogram of {column_name}", f"Boxplot of {column_name}"),
    )

    # Add Histogram trace to the first subplot
    fig.add_trace(
        go.Histogram(
            x=data,
            name="Count",  # This name appears in the legend if multiple traces are used
            marker_color="steelblue",
            xbins=dict(size=None),  # Auto binning for histogram
        ),
        row=1,
        col=1,
    )

    # Add Boxplot trace to the second subplot
    fig.add_trace(
        go.Box(
            x=data,
            name="Distribution",  # This name appears in the legend
            marker_color="steelblue",
            boxpoints="outliers",  # Show all points and outliers
            jitter=0.3,  # Spread out points if boxpoints is set
            pointpos=-1.8,  # Position of the points
            line_width=2,
            orientation="h",  # Horizontal boxplot
        ),
        row=2,
        col=1,
    )

    # Customize overall layout
    fig.update_layout(
        title={
            "text": f"Distribution of Numerical Data for Column: {column_name}",
            "font_size": 12,  # Main title font size
            "x": 0.5,  # Center the main title
            "xanchor": "center",
        },
        height=800,  # Total height of the figure
        width=1200,  # Total width of the figure
        showlegend=False,  # No need for legend as traces are clear by subplot titles
    )

    # Customize axis titles and tick fonts for each subplot
    # Row 1 (Histogram)
    fig.update_xaxes(title_text="Value", title_font_size=12, tickfont_size=10, row=1, col=1)
    fig.update_yaxes(title_text="Frequency", title_font_size=12, tickfont_size=10, row=1, col=1)

    # Row 2 (Boxplot)
    fig.update_xaxes(title_text="Value", title_font_size=12, tickfont_size=10, row=2, col=1)
    # For a horizontal boxplot, y-axis is categorical (implicitly), no title needed
    fig.update_yaxes(visible=False, row=2, col=1)  # Hide y-axis for cleaner boxplot

    try:
        fig.write_html(output_filename, auto_open=False)
        print(f"Numerical distribution chart for '{column_name}' saved as '{output_filename}'.")
        print(f"Please open '{output_filename}' in your web browser to view the chart.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the HTML file: {e}")


def plot_timeseries_histogram(dates: np.ndarray, column_name: str):
    # Ensure dates are in datetime format for proper Plotly handling
    # Convert various input types to datetime, handling potential errors
    try:
        # Attempt to convert to pandas Series of datetime, then to numpy array of datetimes
        # This handles datetime objects, strings, timestamps etc.
        processed_dates = pd.to_datetime(dates).to_numpy()
    except Exception as e:
        print(
            f"Error: Could not convert input 'dates' to datetime format. Please ensure data is convertible. Error: {e}"
        )
        return

    # Check for empty data after potential conversion
    if processed_dates.size == 0:
        print(f"Warning: No valid date data available for column '{column_name}' to create a time-series histogram.")
        return

    # Create histogram trace
    # Plotly automatically handles binning for date axes
    fig = go.Figure(data=[go.Histogram(x=processed_dates, marker_color="steelblue")])

    # Customize layout for a time-series histogram
    fig.update_layout(
        title={
            "text": f"Time-Series Histogram of {column_name}",
            "font_size": 24,
            "x": 0.5,  # Center the title
            "xanchor": "center",
        },
        xaxis_title_text="Date",
        yaxis_title_text="Frequency",
        xaxis=dict(type="date", tickfont_size=10, title_font_size=12),  # Ensure x-axis is treated as a date axis
        yaxis=dict(tickfont_size=10, title_font_size=12),
        height=600,  # Height of the figure
        width=1000,  # Width of the figure
        bargap=0.1,  # Gap between bars for better visualization
        showlegend=False,
    )

    # Define output directory and ensure it exists
    output_filename = f"./samples/{column_name}_timeseries_histogram.html"

    try:
        fig.write_html(output_filename, auto_open=False)
        print(f"Time-series histogram for '{column_name}' saved as '{output_filename}'.")
        print(f"Please open '{output_filename}' in your web browser to view the chart.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the HTML file: {e}")


import polars as pl
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from typing import Optional, List


def profile(
    df: pl.DataFrame,
    output_filename: str = "./samples/all_columns_charts.html",
    date_format: str = "%Y-%m-%d %H:%M",
    exclude_columns_for_plot: Optional[List[str]] = None,
    display_inline: bool = False,
    open_in_browser: bool = False,
):
    if df.is_empty():
        print("Warning: Input DataFrame is empty. No charts will be generated.")
        return

    exclude_columns_for_plot = exclude_columns_for_plot or []
    plot_info = []
    MAX_CATEGORIES = 50

    for col_name in df.columns:
        if col_name in exclude_columns_for_plot:
            print(f"Skipping column '{col_name}' (excluded by user).")
            continue

        series = df.get_column(col_name)
        non_null_series = series.drop_nulls()

        if non_null_series.is_empty():
            plot_info.append(
                {
                    "name": col_name,
                    "type": "null_only",
                    "rows": 1,
                    "data": None,
                }
            )
            continue

        is_handled = False
        if non_null_series.dtype == pl.String:
            try:
                parsed_datetime_series = _parse_date_column(non_null_series, date_format)
                if parsed_datetime_series.dtype == pl.Datetime and parsed_datetime_series.drop_nulls().len() > 0:
                    plot_info.append(
                        {
                            "name": col_name,
                            "type": "datetime",
                            "rows": 1,
                            "data": parsed_datetime_series.drop_nulls().to_numpy(),
                        }
                    )
                    is_handled = True
            except Exception:
                pass  # skip silently

        if is_handled:
            continue

        if non_null_series.dtype.is_numeric():
            plot_info.append(
                {
                    "name": col_name,
                    "type": "numerical",
                    "rows": 2,
                    "data": non_null_series.to_numpy(),
                }
            )
        elif non_null_series.dtype == pl.Datetime:
            plot_info.append(
                {
                    "name": col_name,
                    "type": "datetime",
                    "rows": 1,
                    "data": non_null_series.to_numpy(),
                }
            )
        elif non_null_series.dtype == pl.String or non_null_series.dtype == pl.Categorical:
            counts_df = non_null_series.value_counts()
            if counts_df.shape[0] > MAX_CATEGORIES:
                print(f"Skipping column '{col_name}' (too many categories: {counts_df.shape[0]}).")
                continue
            sorted_df = counts_df.sort(["count", col_name], descending=[True, False])
            plot_info.append(
                {
                    "name": col_name,
                    "type": "categorical",
                    "rows": 1,
                    "data": {
                        "categories": sorted_df[col_name].to_numpy(),
                        "counts": sorted_df["count"].to_numpy(),
                    },
                }
            )
        elif non_null_series.dtype == pl.Boolean:
            counts_df = non_null_series.value_counts()
            sorted_df = counts_df.sort(["count", col_name], descending=[True, False])
            plot_info.append(
                {
                    "name": col_name,
                    "type": "categorical",
                    "rows": 1,
                    "data": {
                        "categories": sorted_df[col_name].to_numpy(),
                        "counts": sorted_df["count"].to_numpy(),
                    },
                }
            )
        else:
            print(f"Warning: Column '{col_name}' has unhandled dtype ({non_null_series.dtype}). Skipping.")
            continue

    if not plot_info:
        print("No suitable columns found for plotting.")
        return

    total_rows = sum(item["rows"] for item in plot_info)
    subplot_titles = []
    for item in plot_info:
        if item["type"] == "numerical":
            subplot_titles.append(f"Histogram of {item['name']}")
            subplot_titles.append(f"Boxplot of {item['name']}")
        elif item["type"] == "categorical":
            subplot_titles.append(f"Frequency of {item['name']}")
        elif item["type"] == "datetime":
            subplot_titles.append(f"Time-Series Histogram of {item['name']}")
        elif item["type"] == "null_only":
            subplot_titles.append(f"{item['name']} (all null)")

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=min(0.01, 1 / max(1, total_rows - 1) * 0.8),
        subplot_titles=subplot_titles,
    )

    current_row = 1
    for item in plot_info:
        col_name = item["name"]
        col_type = item["type"]

        if col_type == "null_only":
            # 空の dummy trace を置く
            fig.add_trace(
                go.Scatter(x=[None], y=[None], showlegend=False),
                row=current_row,
                col=1,
            )
            # アノテーションでメッセージを表示
            fig.add_annotation(
                text=f"Column '{col_name}' contains only null values.",
                xref="x",
                yref="y",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
                row=current_row,
                col=1,
            )
            fig.update_xaxes(visible=False, row=current_row, col=1)
            fig.update_yaxes(visible=False, row=current_row, col=1)
            current_row += 1
            continue

        if col_type == "categorical":
            cats = item["data"]["categories"]
            counts = item["data"]["counts"]
            fig.add_trace(
                go.Bar(
                    y=cats,
                    x=counts,
                    orientation="h",
                    marker_color="steelblue",
                    name=col_name,
                ),
                row=current_row,
                col=1,
            )
            fig.update_xaxes(title_text="Count", row=current_row, col=1)
            fig.update_yaxes(title_text="Category", row=current_row, col=1)
            current_row += 1

        elif col_type == "numerical":
            col_data = item["data"]
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            x_range = [min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05]

            fig.add_trace(
                go.Histogram(x=col_data, name=col_name, marker_color="steelblue"),
                row=current_row,
                col=1,
            )
            fig.update_yaxes(title_text="Frequency", row=current_row, col=1)
            fig.update_xaxes(title_text="Value", range=x_range, row=current_row, col=1)
            current_row += 1

            fig.add_trace(
                go.Box(
                    x=col_data,
                    name=col_name,
                    marker_color="steelblue",
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=-1.8,
                    line_width=2,
                    orientation="h",
                ),
                row=current_row,
                col=1,
            )
            fig.update_yaxes(visible=False, row=current_row, col=1)
            fig.update_xaxes(title_text="Value", range=x_range, row=current_row, col=1)
            current_row += 1

        elif col_type == "datetime":
            col_data = item["data"]
            fig.add_trace(
                go.Histogram(x=col_data, name=col_name, marker_color="steelblue"),
                row=current_row,
                col=1,
            )
            fig.update_xaxes(title_text="Date", type="date", row=current_row, col=1)
            fig.update_yaxes(title_text="Frequency", row=current_row, col=1)
            current_row += 1

    fig.update_layout(
        title={"text": "Comprehensive Data Distribution Analysis", "font_size": 28, "x": 0.5, "xanchor": "center"},
        height=800 * total_rows,
        width=1200,
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    if display_inline:
        fig.show()
    else:
        try:
            fig.write_html(output_filename, auto_open=open_in_browser)
            print(f"All charts saved to: '{output_filename}'.")
        except Exception as e:
            print(f"An unexpected error occurred while saving the HTML file: {e}")


def profile_bivariate(
    df: pl.DataFrame,
    column_pairs: List[Tuple[str, str]],
    output_filename: str = "./samples/bivariate_report.html",
    date_format: str = "%Y-%m-%d %H:%M",
    display_inline: bool = False,
    open_in_browser: bool = False,
):
    if df.is_empty():
        print("Warning: Input DataFrame is empty. No bivariate charts will be generated.")
        return

    processed_df = df.clone()
    for col_name in df.columns:
        series = df.get_column(col_name)
        if series.dtype == pl.String:
            try:
                parsed = _parse_date_column(pl.lit(series), date_format).to_series()
                if parsed.dtype == pl.Datetime and parsed.drop_nulls().len() > 0:
                    processed_df = processed_df.with_columns(parsed.alias(col_name))
            except Exception:
                pass

    plots_to_add = []
    subplot_titles = []

    for col1_name, col2_name in column_pairs:
        if col1_name not in processed_df.columns or col2_name not in processed_df.columns:
            print(f"Warning: One or both columns '{col1_name}', '{col2_name}' not found in DataFrame. Skipping pair.")
            continue

        # Boolean列をCategoricalとして扱う
        for col in [col1_name, col2_name]:
            if processed_df[col].dtype == pl.Boolean:
                processed_df = processed_df.with_columns(
                    processed_df[col].cast(pl.Utf8).cast(pl.Categorical).alias(col)
                )
        paired_df_cleaned = processed_df.select([col1_name, col2_name]).drop_nulls()
        if paired_df_cleaned.is_empty():
            print(f"Warning: Pair ({col1_name}, {col2_name}) is empty after dropping nulls. Skipping plot.")
            continue

        s1_cleaned = paired_df_cleaned.get_column(col1_name)
        s2_cleaned = paired_df_cleaned.get_column(col2_name)

        if s1_cleaned.null_count() == s1_cleaned.len():
            print(f"Error: Column '{col1_name}' is entirely null. Skipping pair.")
            continue
        if s2_cleaned.null_count() == s2_cleaned.len():
            print(f"Error: Column '{col2_name}' is entirely null. Skipping pair.")
            continue

        type1 = s1_cleaned.dtype
        type2 = s2_cleaned.dtype

        trace = None
        plot_type_key = ""

        if type1.is_numeric() and type2.is_numeric():
            trace = go.Scatter(
                x=s1_cleaned.to_numpy(),
                y=s2_cleaned.to_numpy(),
                mode="markers",
                marker=dict(color="steelblue", opacity=0.7, size=8),
                name=f"{col2_name} vs {col1_name}",
                showlegend=False,
            )
            plot_type_key = "num_num_scatter"
            subplot_titles.append(f"Scatter Plot: {col1_name} vs {col2_name}")

        elif (type1.is_numeric() and (type2 == pl.String or type2 == pl.Categorical)) or (
            (type1 == pl.String or type1 == pl.Categorical) and type2.is_numeric()
        ):
            num_s_cleaned = s1_cleaned if type1.is_numeric() else s2_cleaned
            cat_s_cleaned = s2_cleaned if type1.is_numeric() else s1_cleaned
            trace = go.Box(
                x=num_s_cleaned.to_numpy(),
                y=cat_s_cleaned.to_numpy(),
                orientation="h",
                marker_color="steelblue",
                name=f"{num_s_cleaned.name} by {cat_s_cleaned.name}",
                showlegend=False,
            )
            plot_type_key = "num_cat_box"
            subplot_titles.append(f"Box Plot: {num_s_cleaned.name} by {cat_s_cleaned.name}")

        elif (type1 == pl.String or type1 == pl.Categorical) and (type2 == pl.String or type2 == pl.Categorical):
            counts_df = processed_df.group_by(col1_name, col2_name).len().rename({"len": "count"})
            all_cat1 = processed_df.get_column(col1_name).unique().sort().to_numpy()
            all_cat2 = processed_df.get_column(col2_name).unique().sort().to_numpy()
            traces = []
            for cat2_val in all_cat2:
                subset = counts_df.filter(pl.col(col2_name) == cat2_val)
                full_cat1_df = pl.DataFrame({col1_name: all_cat1})
                merged = full_cat1_df.join(subset, on=col1_name, how="left").fill_null(0).sort(col1_name)
                traces.append(
                    go.Bar(
                        x=merged["count"].to_numpy(),
                        y=merged[col1_name].to_numpy(),
                        orientation="h",
                        name=str(cat2_val),
                        hoverinfo="x+y+name+text",
                        text=np.where(
                            merged["count"].to_numpy() > 0,
                            np.full(merged.height, str(cat2_val)),
                            "",
                        ),
                        textposition="auto",
                    )
                )
            trace = traces
            plot_type_key = "cat_cat_stacked_bar"
            subplot_titles.append(f"Stacked Bar Plot: {col1_name} by {col2_name} (Counts)")

        elif (type1 == pl.Datetime and type2.is_numeric()) or (type1.is_numeric() and type2 == pl.Datetime):
            dt_s = s1_cleaned if type1 == pl.Datetime else s2_cleaned
            num_s = s2_cleaned if type1 == pl.Datetime else s1_cleaned
            trace = go.Scatter(
                x=dt_s.to_numpy(),
                y=num_s.to_numpy(),
                mode="lines+markers",
                name=f"{num_s.name} over {dt_s.name}",
                marker_color="steelblue",
                showlegend=False,
            )
            plot_type_key = "dt_num_line"
            subplot_titles.append(f"Time Series: {num_s.name} over {dt_s.name}")

        elif (type1 == pl.Datetime and (type2 == pl.String or type2 == pl.Categorical)) or (
            (type1 == pl.String or type1 == pl.Categorical) and type2 == pl.Datetime
        ):
            dt_s = s1_cleaned if type1 == pl.Datetime else s2_cleaned
            cat_s = s2_cleaned if type1 == pl.Datetime else s1_cleaned
            trace = go.Box(
                x=dt_s.to_numpy(),
                y=cat_s.to_numpy(),
                orientation="h",
                name=f"Date Distribution by {cat_s.name}",
                marker_color="steelblue",
                showlegend=False,
            )
            plot_type_key = "dt_cat_box"
            subplot_titles.append(f"Date Distribution: {dt_s.name} by {cat_s.name}")

        elif type1 == pl.Datetime and type2 == pl.Datetime:
            trace = go.Scatter(
                x=s1_cleaned.to_numpy(),
                y=s2_cleaned.to_numpy(),
                mode="markers",
                marker=dict(color="steelblue", opacity=0.7, size=8),
                name=f"{col2_name} vs {col1_name}",
                showlegend=False,
            )
            plot_type_key = "dt_dt_scatter"
            subplot_titles.append(f"Scatter Plot: {col1_name} vs {col2_name}")

        else:
            print(f"Warning: Unhandled data type combination for pair ({col1_name}, {col2_name}). Skipping plot.")
            continue

        if trace:
            plots_to_add.append(
                {
                    "trace": trace,
                    "col1_name": col1_name,
                    "col2_name": col2_name,
                    "plot_type_key": plot_type_key,
                    "types": (type1, type2),
                }
            )

    if not plots_to_add:
        print("No suitable column pairs found for plotting. No chart will be generated.")
        return

    fig = make_subplots(
        rows=len(plots_to_add),
        cols=1,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    for i, plot_info in enumerate(plots_to_add):
        trace_data = plot_info["trace"]
        col1_name = plot_info["col1_name"]
        col2_name = plot_info["col2_name"]
        plot_type_key = plot_info["plot_type_key"]
        type1, type2 = plot_info["types"]

        if isinstance(trace_data, list):
            for single_trace in trace_data:
                fig.add_trace(single_trace, row=i + 1, col=1)
        else:
            fig.add_trace(trace_data, row=i + 1, col=1)

        xaxis_title = col1_name
        yaxis_title = col2_name
        xaxis_type = None
        yaxis_type = None

        if plot_type_key == "num_cat_box":
            num_s_name = col1_name if type1.is_numeric() else col2_name
            cat_s_name = col2_name if type1.is_numeric() else col1_name
            xaxis_title = num_s_name
            yaxis_title = cat_s_name
            yaxis_type = "category"
        elif plot_type_key == "cat_cat_stacked_bar":
            xaxis_title = "Count"
            yaxis_title = col1_name
            xaxis_type = "linear"
            yaxis_type = "category"
        elif plot_type_key == "dt_num_line":
            dt_s_name = col1_name if type1 == pl.Datetime else col2_name
            num_s_name = col2_name if type1 == pl.Datetime else col1_name
            xaxis_title = dt_s_name
            yaxis_title = num_s_name
            xaxis_type = "date"
        elif plot_type_key == "dt_cat_box":
            dt_s_name = col1_name if type1 == pl.Datetime else col2_name
            cat_s_name = col2_name if type1 == pl.Datetime else col1_name
            xaxis_title = dt_s_name
            yaxis_title = cat_s_name
            xaxis_type = "date"
            yaxis_type = "category"
        elif plot_type_key == "dt_dt_scatter":
            xaxis_title = col1_name
            yaxis_title = col2_name
            xaxis_type = "date"
            yaxis_type = "date"

        fig.update_xaxes(
            title_text=xaxis_title, title_font_size=12, tickfont_size=10, type=xaxis_type, row=i + 1, col=1
        )
        fig.update_yaxes(
            title_text=yaxis_title,
            title_font_size=12,
            tickfont_size=10,
            automargin=True,
            type=yaxis_type,
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        title={"text": "Bivariate Data Analysis Report", "font_size": 28, "x": 0.5, "xanchor": "center"},
        height=500 * len(plots_to_add),
        width=1200,
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),
        barmode="stack",
    )

    if display_inline:
        fig.show()
    else:
        try:
            fig.write_html(output_filename, auto_open=open_in_browser)
            print(f"All charts saved to: '{output_filename}'.")
        except Exception as e:
            print(f"An unexpected error occurred while saving the HTML file: {e}")


import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set, Callable, Any

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
import shap
from polars.exceptions import PolarsError
from sklearn.metrics import mean_squared_error, r2_score

# ========= helpers =========

NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _is_numeric_dtype(dt) -> bool:
    return any(dt == t for t in NUMERIC_DTYPES)


def _collect_onehot_mapping(df_with_dummies: pl.DataFrame, oh_cols: List[str]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    cols = df_with_dummies.columns
    for c in oh_cols:
        pref = f"{c}_"
        mapping[c] = [name for name in cols if name.startswith(pref)]
    return mapping


def _apply_onehot_and_align(
    df: pl.DataFrame,
    oh_cols: List[str],
    drop_first: bool,
    required_map: Dict[str, List[str]],
) -> pl.DataFrame:
    dfd = df.to_dummies(columns=oh_cols, drop_first=drop_first) if oh_cols else df
    required_dummies = [col for cols in required_map.values() for col in cols]
    missing = [c for c in required_dummies if c not in dfd.columns]
    if missing:
        dfd = dfd.with_columns([pl.lit(0.0).alias(c) for c in missing])
    if oh_cols:
        extra = [c for c in dfd.columns if any(c.startswith(f"{o}_") for o in oh_cols) and c not in required_dummies]
        if extra:
            dfd = dfd.drop(extra)
    return dfd


_BOOL_TOKENS_TRUE = {"true", "t", "yes", "y"}
_BOOL_TOKENS_FALSE = {"false", "f", "no", "n"}


def _looks_boolean_series(s: pl.Series) -> bool:
    def ok(v) -> bool:
        if v is None:
            return True
        v = str(v).strip().lower()
        if v == "":
            return True
        return (v in _BOOL_TOKENS_TRUE) or (v in _BOOL_TOKENS_FALSE)

    return all(ok(v) for v in s.unique().to_list())


def _infer_boolean_columns_from_sample(
    data_path: str,
    candidate_cols: List[str],
    sample_rows: int = 50000,
) -> Set[str]:
    if not candidate_cols:
        return set()
    dtypes = {c: pl.Utf8 for c in candidate_cols}
    lf = pl.scan_csv(data_path, dtypes=dtypes)
    sample = lf.select(candidate_cols).head(sample_rows).collect()
    bool_cols: Set[str] = set()
    for c in candidate_cols:
        if c in sample.columns and _looks_boolean_series(sample[c]):
            bool_cols.add(c)
    return bool_cols


def train_lgbm_with_optuna_multi_target(
    data_path: str,
    features: List[str],
    targets: List[str],
    split_by: str = "timestamp",  # "timestamp" | "random"
    split_by_column: str = "yyyymm",
    timestamp_format: str = "%Y%m",
    test_size: float = 0.2,
    validate_size: float = 0.2,
    n_trials: int = 50,
    model_path: str = "./models",
    log_period_optuna: int = 10,
    log_period_final: int = 10,
    run_id: Optional[str] = None,
    # One-Hot
    one_hot_cols: Optional[List[str]] = None,
    one_hot_drop_first: bool = False,
    # SHAP
    shap_sample_size: int = 10000,
    # 探索空間の外だし
    search_space: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
    # 簡易：各パラメータの(min,max)を渡す。未指定はデフォルト範囲
    param_ranges: Optional[Dict[str, tuple]] = None,
    # LightGBMの固定パラメータ（探索外）を外部から注入
    lgb_static_params: Optional[Dict[str, Any]] = None,
    # early stopping
    early_stopping_rounds: Optional[int] = None,
    # Optuna設定
    optuna_direction: str = "minimize",
    optuna_sampler: Optional[optuna.samplers.BaseSampler] = None,
    # サンプルスキャン行数（Bool推定）
    bool_infer_sample_rows: int = 50000,
) -> Dict[str, Tuple[lgb.Booster, pl.DataFrame]]:
    """
    戻り値: {target: (best_model, shap_importance_df)}
    主要成果物は model_path 配下へ保存:
      - optuna_trials.csv, metrics.csv
      - lgbm_reg_{target}_model.joblib
      - shap_values_{target}.csv
      - onehot_columns.json（One-Hot使用時）
    """

    run_started_at = datetime.now(timezone.utc).isoformat()
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    print(f"[INFO] run_id={run_id} started_at(UTC)={run_started_at}")

    # ---- IO（サンプルスキャンで Bool 列推定 → 型固定で全量読み込み）----
    try:
        print(f"[INFO] Loading data (with schema inference) from: {data_path}")

        one_hot_set = set(one_hot_cols or [])
        ft_cols = set(features + targets)

        candidate_bool = sorted(list(ft_cols - {split_by_column} - one_hot_set))
        inferred_bools = _infer_boolean_columns_from_sample(
            data_path, candidate_bool, sample_rows=bool_infer_sample_rows
        )
        print(f"[INFO] Inferred boolean columns from sample: {sorted(inferred_bools)}")

        schema_overrides: Dict[str, pl.DataType] = {}
        schema_overrides[split_by_column] = pl.Utf8
        for c in one_hot_set:
            schema_overrides[c] = pl.Utf8
        for c in ft_cols:
            if c in schema_overrides:
                continue
            schema_overrides[c] = pl.Boolean if c in inferred_bools else pl.Float64

        df = pl.read_csv(data_path, schema_overrides=schema_overrides)
        print(f"[INFO] Data loaded: rows={df.shape[0]}, cols={df.shape[1]}")
    except (PolarsError, FileNotFoundError) as e:
        print(f"[ERROR] Error occurred while loading data: {e}")
        return {}

    Path(model_path).mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Model output directory: {model_path}")

    optuna_log_path = Path(model_path) / "optuna_trials.csv"
    metrics_path = Path(model_path) / "metrics.csv"
    onehot_meta_path = Path(model_path) / "onehot_columns.json"

    # ---- Split ----
    if split_by == "timestamp":
        if split_by_column not in df.columns:
            raise ValueError(f"The specified split column '{split_by_column}' does not exist in the data.")

        print(f"[INFO] Converting '{split_by_column}' to date using format '{timestamp_format}'")
        try:
            df = df.with_columns(pl.col(split_by_column).str.to_date(timestamp_format).alias("_sort_by_date"))
        except pl.InvalidOperationError:
            raise ValueError(
                f"Failed to convert column '{split_by_column}' to a date type using format '{timestamp_format}'."
            )
        df = df.sort(by="_sort_by_date")
        features_for_model = [f for f in features if f != split_by_column]

        n_total = len(df)
        n_test = int(n_total * test_size)
        n_validate = int(n_total * validate_size)
        n_train = n_total - n_test - n_validate
        if n_train <= 0 or n_validate <= 0 or n_test <= 0:
            raise ValueError(f"Split sizes invalid (train={n_train}, validate={n_validate}, test={n_test}).")
        print(f"[INFO] Split by timestamp: total={n_total}, train={n_train}, validate={n_validate}, test={n_test}")

        df_train = df.head(n_train)
        df_validate = df.slice(n_train, n_validate)
        df_test = df.tail(n_test)

    elif split_by == "random":
        features_for_model = features[:]
        print("[INFO] Splitting data randomly (polars-based shuffle)")
        df_shuf = df.sample(
            fraction=1.0, with_replacement=False, shuffle=True, seed=(lgb_static_params or {}).get("seed", 42)
        )
        n_total = len(df_shuf)
        n_test = int(n_total * test_size)
        n_validate = int(n_total * validate_size)
        n_train = n_total - n_test - n_validate
        if n_train <= 0 or n_validate <= 0 or n_test <= 0:
            raise ValueError(f"Split sizes invalid (train={n_train}, validate={n_validate}, test={n_test}).")
        print(f"[INFO] Split random: total={n_total}, train={n_train}, validate={n_validate}, test={n_test}")

        df_train = df_shuf.head(n_train)
        df_validate = df_shuf.slice(n_train, n_validate)
        df_test = df_shuf.tail(n_test)
    else:
        raise ValueError("split_by must be 'timestamp' or 'random'.")

    # ---- One-Hot strict ----
    oh_cols = [c for c in (one_hot_cols or []) if c in df.columns and c != split_by_column]
    dropped_oh = sorted(set(one_hot_cols or []) - set(oh_cols))
    if dropped_oh:
        print(f"[INFO] Skipped one-hot for columns (not found or split_by_column): {dropped_oh}")

    onehot_required_map: Dict[str, List[str]] = {}
    if oh_cols:
        print(f"[INFO] One-hot strict (drop_first={one_hot_drop_first}) fit on train+validate")
        fit_df = pl.concat([df_train, df_validate], how="vertical_relaxed")
        fit_dummies = fit_df.to_dummies(columns=oh_cols, drop_first=one_hot_drop_first)
        onehot_required_map = _collect_onehot_mapping(fit_dummies, oh_cols)

        try:
            meta = {"run_id": run_id, "drop_first": one_hot_drop_first, "mapping": onehot_required_map}
            onehot_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
            print(f"[INFO] One-hot metadata saved to: {onehot_meta_path}")
        except Exception as e:
            print(f"[WARN] Failed to save one-hot metadata: {e}")

        df_train = _apply_onehot_and_align(df_train, oh_cols, one_hot_drop_first, onehot_required_map)
        df_validate = _apply_onehot_and_align(df_validate, oh_cols, one_hot_drop_first, onehot_required_map)
        df_test = _apply_onehot_and_align(df_test, oh_cols, one_hot_drop_first, onehot_required_map)

        expanded = []
        for f in features_for_model:
            if f in onehot_required_map:
                expanded.extend(onehot_required_map[f])
            else:
                expanded.append(f)
        features_for_model = expanded
        print(f"[INFO] features_for_model expanded to {len(features_for_model)} columns after one-hot")

    # ---- DType safety ----
    bad_cols: List[Tuple[str, str]] = []
    casts = []
    for c in features_for_model + targets:
        if c not in df_train.columns:
            raise ValueError(f"Column '{c}' not found in the split data.")
        dt = df_train.schema[c]
        if dt == pl.Boolean:
            casts.append(pl.col(c).cast(pl.Float64))
        elif _is_numeric_dtype(dt):
            casts.append(pl.col(c).cast(pl.Float64))
        else:
            bad_cols.append((c, str(dt)))
    if bad_cols:
        raise ValueError(f"Non-numeric columns detected among features/targets: {bad_cols}")

    df_train = df_train.with_columns(casts)
    df_validate = df_validate.with_columns(casts)
    df_test = df_test.with_columns(casts)

    # ---- To numpy ----
    X_train_np = df_train[features_for_model].to_numpy(writable=True).astype(np.float32)
    X_validate_np = df_validate[features_for_model].to_numpy(writable=True).astype(np.float32)
    X_test_np = df_test[features_for_model].to_numpy(writable=True).astype(np.float32)

    # ---- Params base（外から全部上書き可能）----
    base_params = {
        "objective": "regression",
        "metric": "rmse",
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if lgb_static_params:
        base_params.update(lgb_static_params)

    # ---- Optuna ----
    results: Dict[str, Tuple[lgb.Booster, pl.DataFrame]] = {}
    study = optuna.create_study(direction=optuna_direction, sampler=optuna_sampler)

    for target in targets:
        print(f"\n[INFO] ===== Target: {target} =====")
        y_train_np = df_train[target].to_numpy(writable=True).astype(np.float32)
        y_validate_np = df_validate[target].to_numpy(writable=True).astype(np.float32)

        lgb_train = lgb.Dataset(X_train_np, y_train_np, free_raw_data=False)
        lgb_validate = lgb.Dataset(X_validate_np, y_validate_np, free_raw_data=False)
        lgb_train.construct()
        lgb_validate.construct()

        trial_logs = []

        def objective(trial: optuna.trial.Trial) -> float:
            if search_space is not None:
                tuned = search_space(trial)
            else:
                R = param_ranges or {}
                tuned = {
                    "n_estimators": trial.suggest_int("n_estimators", *R.get("n_estimators", (100, 1000))),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", *R.get("learning_rate", (0.01, 0.1)), log=True
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", *R.get("num_leaves", (2, 256))),
                    "max_depth": trial.suggest_int("max_depth", *R.get("max_depth", (3, 15))),
                    "subsample": trial.suggest_float("subsample", *R.get("subsample", (0.5, 1.0))),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", *R.get("colsample_bytree", (0.5, 1.0))),
                    "reg_alpha": trial.suggest_float("reg_alpha", *R.get("reg_alpha", (1e-8, 10.0)), log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", *R.get("reg_lambda", (1e-8, 10.0)), log=True),
                }

            params = {**base_params, **tuned}
            print(f"[INFO] Trial {trial.number}: n_estimators={params.get('n_estimators')}")

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_validate],
                num_boost_round=params.get("n_estimators"),
                callbacks=[
                    optuna.integration.LightGBMPruningCallback(trial, base_params.get("metric", "rmse")),
                    *([lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds else []),
                    lgb.log_evaluation(period=log_period_optuna),
                ],
            )
            rmse = model.best_score["valid_0"][base_params.get("metric", "rmse")]
            print(f"[TRIAL] run_id={run_id} target={target} #{trial.number:03d} rmse={rmse:.5f}")
            trial_logs.append(
                {
                    "run_id": run_id,
                    "run_started_at": run_started_at,
                    "target": target,
                    "trial": trial.number,
                    "rmse": float(rmse),
                    "params": json.dumps(params, sort_keys=True),
                }
            )
            return rmse

        study.optimize(objective, n_trials=n_trials)

        # trials CSV
        trial_df = pl.DataFrame(trial_logs)
        if Path(optuna_log_path).exists():
            with open(optuna_log_path, "ab") as f:
                trial_df.write_csv(f, include_header=False)
        else:
            trial_df.write_csv(optuna_log_path, include_header=True)
        print(f"[INFO] Appended trial logs to: {optuna_log_path}")

        print(f"[INFO] Best params for {target}: {study.best_params}")
        print(f"[INFO] Best value: {study.best_value:.5f}")

        # ---- Final training on train+validate ----
        X_train_full_np = (
            pl.concat([df_train[features_for_model], df_validate[features_for_model]])
            .to_numpy(writable=True)
            .astype(np.float32)
        )
        y_train_full_np = pl.concat([df_train[target], df_validate[target]]).to_numpy(writable=True).astype(np.float32)
        lgb_train_full = lgb.Dataset(X_train_full_np, y_train_full_np, free_raw_data=False)
        lgb_train_full.construct()

        final_params = {**base_params, **study.best_params}
        best_model = lgb.train(
            final_params,
            lgb_train_full,
            valid_sets=[lgb_train_full],
            num_boost_round=final_params.get("n_estimators"),
            callbacks=[
                *([lgb.early_stopping(early_stopping_rounds)] if early_stopping_rounds else []),
                lgb.log_evaluation(period=log_period_final),
            ],
        )

        model_filename = Path(model_path) / f"lgbm_reg_{target}_model.joblib"
        joblib.dump(best_model, model_filename)
        print(f"[INFO] Model saved to: {model_filename}")

        # ---- SHAP（サンプリング）----
        print(f"[INFO] Calculating SHAP values for {target}")
        sample_n = min(shap_sample_size, X_test_np.shape[0])
        if sample_n < X_test_np.shape[0]:
            rng = np.random.default_rng(seed=base_params.get("seed", 42))
            idx = rng.choice(X_test_np.shape[0], size=sample_n, replace=False)
            X_shap = X_test_np[idx]
        else:
            X_shap = X_test_np

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_shap)
        shap_df = pl.DataFrame(
            {"feature": features_for_model, "importance": [abs(v).mean() for v in shap_values.T]}
        ).sort("importance", descending=True)

        shap_df_path = Path(model_path) / f"shap_values_{target}.csv"
        shap_df.write_csv(shap_df_path)
        print(f"[INFO] SHAP values saved to: {shap_df_path}")

        # ---- Metrics on test ----
        y_test_np = df_test[target].to_numpy(writable=True).astype(np.float32)
        y_pred = best_model.predict(X_test_np)
        rmse_test = float(np.sqrt(mean_squared_error(y_test_np, y_pred)))
        r2_test = float(r2_score(y_test_np, y_pred))
        print(f"[INFO] Test RMSE for {target}: {rmse_test:.5f}")
        print(f"[INFO] Test R²   for {target}: {r2_test:.5f}")

        metrics_df = pl.DataFrame(
            {
                "run_id": [run_id],
                "run_started_at": [run_started_at],
                "target": [target],
                "rmse_test": [rmse_test],
                "r2_test": [r2_test],
            }
        )
        if Path(metrics_path).exists():
            with open(metrics_path, "ab") as f:
                metrics_df.write_csv(f, include_header=False)
        else:
            metrics_df.write_csv(metrics_path, include_header=True)
        print(f"[INFO] Metrics appended to: {metrics_path}")

        results[target] = (best_model, shap_df)

    print("\n[INFO] All targets processed successfully.")
    return results
