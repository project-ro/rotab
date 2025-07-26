import polars as pl
from typing import List, Dict, Any
from datetime import date
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def unique(df: pl.DataFrame, group_keys: list[str], sort_by: str, ascending: bool = True) -> pl.DataFrame:
    return df.sort(group_keys + [sort_by], descending=not ascending).group_by(group_keys, maintain_order=True).first()


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


# is_date_column 関数はコメントなし版が最新と仮定
# is_float_column, is_int_column 関数も変更なしで、ここに存在すると仮定
# 仮の is_float_column と is_int_column の定義（実際のコードに合わせてください）
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


def describe(df: pl.DataFrame) -> pl.DataFrame:
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

        if is_date_column(series):
            try:
                series_dt = series.cast(str).str.strptime(pl.Date, format="%Y-%m-%d", strict=False)

                min_date_val = _get_item_or_scalar(series_dt.min())
                max_date_val = _get_item_or_scalar(series_dt.max())

                current_summary_data.update(
                    {
                        "dtype": "date",
                        "min": str(min_date_val) if min_date_val is not None else None,
                        "max": str(max_date_val) if max_date_val is not None else None,
                        "min_year": (
                            str(_get_item_or_scalar(series_dt.dt.year().min()))
                            if not series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "max_year": (
                            str(_get_item_or_scalar(series_dt.dt.year().max()))
                            if not series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "min_month": (
                            str(_get_item_or_scalar(series_dt.dt.month().min()))
                            if not series_dt.drop_nulls().is_empty()
                            else None
                        ),
                        "max_month": (
                            str(_get_item_or_scalar(series_dt.dt.month().max()))
                            if not series_dt.drop_nulls().is_empty()
                            else None
                        ),
                    }
                )

                date_only_series = series_dt.drop_nulls()
                mode_counts = date_only_series.value_counts().sort("count", descending=True)
                if not mode_counts.is_empty():
                    mode_val = _get_item_or_scalar(mode_counts[0, date_only_series.name])
                    mode_freq = _get_item_or_scalar(mode_counts[0, "count"])
                    mode_ratio = float(mode_freq) / len(date_only_series)

                    current_summary_data.update(
                        {
                            "mode": str(mode_val),
                            "mode_freq": str(mode_freq),
                            "mode_ratio": str(mode_ratio),
                        }
                    )
                else:
                    current_summary_data.update({"mode": None, "mode_freq": None, "mode_ratio": None})

                if current_summary_data["min"] is not None and current_summary_data["max"] is not None:
                    try:
                        temp_min_date = date.fromisoformat(current_summary_data["min"])
                        temp_max_date = date.fromisoformat(current_summary_data["max"])
                        range_days_val = (temp_max_date - temp_min_date).days
                        current_summary_data["range_days"] = str(range_days_val)
                    except ValueError:
                        current_summary_data["range_days"] = None
                else:
                    current_summary_data["range_days"] = None

                summarized = True
            except Exception:
                current_summary_data["dtype"] = "error"
                summarized = True
            finally:
                if summarized:
                    summaries.append(current_summary_data)
                    continue

        elif not summarized and is_float_column(series):
            series_float = series.cast(str).cast(pl.Float64, strict=False)

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
            series_str = series.drop_nulls().cast(str)
            vc = series_str.value_counts().sort("count", descending=True)

            top = top_freq = min_cat = min_freq = None
            if not vc.is_empty():
                top = str(_get_item_or_scalar(vc[0, series_str.name]))
                top_freq = str(_get_item_or_scalar(vc[0, "count"]))
                if vc.height > 1:
                    min_cat = str(_get_item_or_scalar(vc[-1, series_str.name]))
                    min_freq = str(_get_item_or_scalar(vc[-1, "count"]))

            lengths = series_str.str.len_chars()

            current_summary_data.update(
                {
                    "dtype": "string",
                    "top": top,
                    "top_freq": top_freq,
                    "top_ratio": str(float(top_freq) / len(series)) if top_freq is not None else None,
                    "min_cat": min_cat,
                    "min_freq": min_freq,
                    "min_ratio": str(float(min_freq) / len(series)) if min_freq is not None else None,
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
        vertical_spacing=0.1,  # Space between subplots
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
            "font_size": 24,  # Main title font size
            "x": 0.5,  # Center the main title
            "xanchor": "center",
        },
        height=800,  # Total height of the figure
        width=1200,  # Total width of the figure
        showlegend=False,  # No need for legend as traces are clear by subplot titles
    )

    # Customize axis titles and tick fonts for each subplot
    # Row 1 (Histogram)
    fig.update_xaxes(title_text="Value", title_font_size=18, tickfont_size=16, row=1, col=1)
    fig.update_yaxes(title_text="Frequency", title_font_size=18, tickfont_size=16, row=1, col=1)

    # Row 2 (Boxplot)
    fig.update_xaxes(title_text="Value", title_font_size=18, tickfont_size=16, row=2, col=1)
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
        xaxis=dict(type="date", tickfont_size=16, title_font_size=18),  # Ensure x-axis is treated as a date axis
        yaxis=dict(tickfont_size=16, title_font_size=18),
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


def profile(  # Renamed function to profile
    df: pl.DataFrame,
    output_filename: str = "./samples/all_columns_charts.html",
    date_format: str = "%Y-%m-%d %H:%M",  # Added date_format argument
):
    if df.is_empty():
        print("Warning: Input DataFrame is empty. No charts will be generated.")
        return

    plot_info = []
    for col_name in df.columns:
        series = df.get_column(col_name)
        non_null_series = series.drop_nulls()

        if non_null_series.is_empty():
            print(
                f"Warning: Column '{col_name}' is empty after dropping nulls. Skipping chart generation for this column."
            )
            continue

        # --- START: Modified logic to handle string dates ---
        is_handled = False
        if non_null_series.dtype == pl.String:
            # Attempt to parse string as datetime with the specified format.
            parsed_datetime_series = non_null_series.str.to_datetime(format=date_format, strict=False)

            # Check if it successfully parsed into a datetime type AND has non-null datetime values
            if (
                parsed_datetime_series.dtype == pl.Datetime and parsed_datetime_series.drop_nulls().len() > 0
            ):  # Changed .is_datetime() to == pl.Datetime
                plot_info.append(
                    {
                        "name": col_name,
                        "type": "datetime",
                        "rows": 1,
                        "data": parsed_datetime_series.drop_nulls().to_numpy(),
                    }
                )
                is_handled = True  # Mark as handled and move to next column

        if is_handled:
            continue
        # --- END: Modified logic to handle string dates ---

        if non_null_series.dtype.is_numeric():
            plot_info.append({"name": col_name, "type": "numerical", "rows": 2, "data": non_null_series.to_numpy()})
        elif non_null_series.dtype == pl.Datetime:  # Changed .is_datetime() to == pl.Datetime
            plot_info.append({"name": col_name, "type": "datetime", "rows": 1, "data": non_null_series.to_numpy()})
        elif (
            non_null_series.dtype == pl.String or non_null_series.dtype == pl.Categorical
        ):  # Using direct comparison for string/categorical
            counts_pl_df = non_null_series.value_counts()
            category_col_name = col_name
            count_col_name = "count"

            sorted_counts_pl_df = counts_pl_df.sort([count_col_name, category_col_name], descending=[True, False])

            sorted_categories = sorted_counts_pl_df[category_col_name].to_numpy()
            sorted_counts = sorted_counts_pl_df[count_col_name].to_numpy()

            plot_info.append(
                {
                    "name": col_name,
                    "type": "categorical",
                    "rows": 1,
                    "data": {"categories": sorted_categories, "counts": sorted_counts},
                }
            )
        else:
            print(
                f"Warning: Column '{col_name}' has an unhandled data type ({non_null_series.dtype}). Skipping chart generation."
            )
            continue

    if not plot_info:
        print("No suitable columns found for plotting. No chart will be generated.")
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

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=False,  # Shared_xaxes is controlled per-plot below for numerical type
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    current_row = 1
    for item in plot_info:
        col_name = item["name"]
        col_type = item["type"]

        if col_type == "categorical":
            sorted_categories = item["data"]["categories"]
            sorted_counts = item["data"]["counts"]

            fig.add_trace(
                go.Bar(y=sorted_categories, x=sorted_counts, orientation="h", marker_color="steelblue", name=col_name),
                row=current_row,
                col=1,
            )
            fig.update_xaxes(title_text="Count", title_font_size=18, tickfont_size=16, row=current_row, col=1)
            fig.update_yaxes(
                title_text="Category", title_font_size=18, tickfont_size=16, automargin=True, row=current_row, col=1
            )
            current_row += 1

        elif col_type == "numerical":
            col_data = item["data"]

            # Calculate common x-axis range for histogram and boxplot for alignment
            min_val = np.min(col_data)
            max_val = np.max(col_data)
            # Add a small padding to the range for better visualization
            x_range = [min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05]

            fig.add_trace(
                go.Histogram(
                    x=col_data,
                    name=col_name,
                    marker_color="steelblue",
                    xbins=dict(size=None),
                ),
                row=current_row,
                col=1,
            )
            fig.update_yaxes(title_text="Frequency", title_font_size=18, tickfont_size=16, row=current_row, col=1)
            # Apply the calculated common x_range
            fig.update_xaxes(
                title_text="Value", title_font_size=18, tickfont_size=16, range=x_range, row=current_row, col=1
            )

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
            # Apply the same common x_range to the boxplot's x-axis
            fig.update_xaxes(
                title_text="Value", title_font_size=18, tickfont_size=16, range=x_range, row=current_row, col=1
            )
            current_row += 1

        elif col_type == "datetime":
            col_data = item["data"]
            fig.add_trace(go.Histogram(x=col_data, name=col_name, marker_color="steelblue"), row=current_row, col=1)
            fig.update_xaxes(
                title_text="Date", title_font_size=18, tickfont_size=16, type="date", row=current_row, col=1
            )
            fig.update_yaxes(title_text="Frequency", title_font_size=18, tickfont_size=16, row=current_row, col=1)
            current_row += 1

    fig.update_layout(
        title={
            "text": "Comprehensive Data Distribution Analysis",
            "font_size": 28,
            "x": 0.5,
            "xanchor": "center",
        },
        height=400 * total_rows,
        width=1200,
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=50),
    )

    try:
        fig.write_html(output_filename, auto_open=False)
        print(f"All charts saved to: '{output_filename}'.")
        print(f"Please open '{output_filename}' in your web browser to view the combined report.")
    except Exception as e:
        print(f"An unexpected error occurred while saving the HTML file: {e}")
