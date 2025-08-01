import os
import numpy as np
from datetime import date
from math import isclose
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from rotab.core.operation.transform_funcs_polars import (
    normalize_dtype,
    validate_table_schema,
    sort_by,
    groupby_agg,
    drop_duplicates,
    merge,
    reshape,
    fillna,
    sample,
    concat,
    drop_na,
    replace,
    unique,
    describe,
    get_categorical_counts_table,
    plot_categorical_bar_chart,
    plot_numerical_distribution,
    plot_timeseries_histogram,
    profile,
    profile_bivariate,
    month_window,
)


def test_normalize_dtype():
    assert normalize_dtype("int") == pl.Int64
    assert normalize_dtype("float") == pl.Float64
    assert normalize_dtype("string") == pl.String
    assert normalize_dtype("unknown") == "unknown"


def test_validate_table_schema_ok():
    df = pl.DataFrame({"a": [1], "b": ["x"]})
    schema = [{"name": "a", "dtype": "int"}, {"name": "b", "dtype": "str"}]
    assert validate_table_schema(df, schema)


def test_validate_table_schema_error_missing():
    df = pl.DataFrame({"a": [1]})
    schema = [{"name": "a", "dtype": "int"}, {"name": "b", "dtype": "str"}]
    with pytest.raises(ValueError):
        validate_table_schema(df, schema)


def test_sort_by():
    df = pl.DataFrame({"a": [3, 1, 2]})
    sorted_df = sort_by(df, "a")
    assert sorted_df["a"].to_list() == [1, 2, 3]


def test_groupby_agg():
    df = pl.DataFrame({"group": ["x", "x", "y"], "val": [1, 2, 3]})
    out = groupby_agg(df, by="group", aggregations={"val": "sum"})
    assert out.filter(pl.col("group") == "x")["val"].to_list()[0] == 3


def test_groupby_agg_multiple_keys():
    df = pl.DataFrame(
        {
            "cat": ["A", "A", "B", "B", "A"],
            "sub": [1, 1, 1, 2, 2],
            "val": [10, 20, 30, 40, 50],
        }
    )

    result = groupby_agg(
        table=df,
        by=["cat", "sub"],
        aggregations={"val": "sum"},
    ).sort(["cat", "sub"])

    expected = pl.DataFrame(
        {
            "cat": ["A", "A", "B", "B"],
            "sub": [1, 2, 1, 2],
            "val": [30, 50, 30, 40],
        }
    ).sort(["cat", "sub"])

    assert_frame_equal(result, expected)


def test_drop_duplicates():
    df = pl.DataFrame({"a": [1, 1, 2]})
    out = drop_duplicates(df, subset=["a"])
    assert out["a"].n_unique() == 2


def test_merge_inner():
    df1 = pl.DataFrame({"id": [1, 2], "val1": ["a", "b"]})
    df2 = pl.DataFrame({"id": [2, 3], "val2": ["c", "d"]})
    out = merge(df1, df2, on="id", how="inner")
    assert out["id"].to_list() == [2]


def test_merge_left():
    df1 = pl.DataFrame({"id": [1, 2], "val1": ["a", "b"]})
    df2 = pl.DataFrame({"id": [2, 3], "val2": ["c", "d"]})
    out = merge(df1, df2, on=["id"], how="left")
    assert out["id"].to_list() == [1, 2]


def test_reshape_pivot():
    df = pl.DataFrame({"id": [1, 1, 2], "col": ["a", "b", "a"], "val": [10, 20, 30]})
    out = reshape(df, column_to="id", columns_from=["col"], column_value="val")
    assert "a" in out.columns and "b" in out.columns


def test_reshape_pivot_agg():
    df = pl.DataFrame({"id": [1, 1, 2], "col": ["a", "a", "a"], "val": [10, 20, 30]})
    out = reshape(df, column_to="id", columns_from=["col"], column_value="val", agg="sum")
    assert out["a"].to_list() == [30, 30]


def test_reshape_melt():
    df = pl.DataFrame({"id": [1], "a": [10]})
    out = reshape(df, column_to="id", column_value="a")
    assert out["melted_value"].to_list() == [10]


@pytest.mark.parametrize(
    "values, strategies, expected_a, expected_b, expected_c, expected_d",
    [
        ({"a": 99}, None, [1, 2, 99, 4, 99], None, None, None),
        (None, {"b": "mean"}, None, [10.0, 30.0, 30.0, 30.0, 50.0], None, None),
        (None, {"c": "mode"}, None, None, ["x", "y", "x", "x", "z"], None),
        (None, {"d": "forward"}, None, None, None, [None, 100, 200, 100, 300]),
        (
            {"a": 99},
            {"b": "mean"},
            [1, 2, 99, 4, 99],
            [10.0, 30.0, 30.0, 30.0, 50.0],
            None,
            None,
        ),
        (
            None,
            {"a": "min", "d": "forward"},
            [1, 2, 1, 4, 1],
            None,
            None,
            [None, 100, 200, 100, 300],
        ),
        (
            None,
            {"b": "median", "c": "mode"},
            None,
            [10.0, 30.0, 30.0, 30.0, 50.0],
            ["x", "y", "x", "x", "z"],
            None,
        ),
        (
            {"a": 99},
            {"a": "mean", "b": "median"},
            [1, 2, 99, 4, 99],
            [10.0, 30.0, 30.0, 30.0, 50.0],
            None,
            None,
        ),
    ],
)
def test_fillna_multiple_columns(sample_dataframe, values, strategies, expected_a, expected_b, expected_c, expected_d):
    result = fillna(sample_dataframe, values, strategies)

    if expected_a is not None:
        assert list(result["a"]) == expected_a
    if expected_b is not None:
        assert list(result["b"]) == expected_b
    if expected_c is not None:
        assert list(result["c"]) == expected_c
    if expected_d is not None:
        assert list(result["d"]) == expected_d


def test_fillna_no_nulls(sample_dataframe):
    result = fillna(sample_dataframe, values={"e": 999})
    assert result["e"].equals(sample_dataframe["e"])

    result = fillna(sample_dataframe, strategies={"e": "mean"})
    assert result["e"].equals(sample_dataframe["e"])


def test_fillna_no_ops(sample_dataframe):
    result = fillna(sample_dataframe, values=None, strategies=None)
    assert result.equals(sample_dataframe)


@pytest.mark.parametrize(
    "strategies, error_msg",
    [
        ({"a": "unknown"}, "Unknown strategy: unknown"),
        ({"b": "invalid"}, "Unknown strategy: invalid"),
    ],
)
def test_fillna_invalid_strategy(sample_dataframe, strategies, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        fillna(sample_dataframe, strategies=strategies)


def test_sample():
    df = pl.DataFrame({"a": list(range(100))})
    out = sample(df, frac=0.1)
    assert 5 <= out.height <= 15


def test_concat():
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [2]})
    out = concat([df1, df2])
    assert out["a"].to_list() == [1, 2]


def test_drop_na():
    df = pl.DataFrame({"a": [1, None]})
    out = drop_na(df, subset=["a"])
    assert out.height == 1


def test_replace():
    df = pl.DataFrame({"a": ["x", "y"]})
    out = replace(df, columns=["a"], old="x", new="z")
    assert out["a"].to_list()[0] == "z"


def test_unique_min_numeric():
    df = pl.DataFrame(
        {"user": ["A", "A", "B", "B", "B"], "score": [50, 30, 70, 60, 80], "info": ["x1", "x2", "x3", "x4", "x5"]}
    )
    out = unique(df, group_keys=["user"], sort_by="score", ascending=True)

    expected = pl.DataFrame({"user": ["A", "B"], "score": [30, 60], "info": ["x2", "x4"]})

    assert out.sort(by=out.columns).rows() == expected.sort(by=expected.columns).rows()


def test_unique_max_numeric():
    df = pl.DataFrame(
        {"user": ["A", "A", "B", "B", "B"], "score": [50, 30, 70, 60, 80], "info": ["x1", "x2", "x3", "x4", "x5"]}
    )
    out = unique(df, group_keys=["user"], sort_by="score", ascending=False)

    expected = pl.DataFrame({"user": ["A", "B"], "score": [50, 80], "info": ["x1", "x5"]})

    assert out.sort(by=out.columns).rows() == expected.sort(by=expected.columns).rows()


def test_unique_string_sort():
    df = pl.DataFrame(
        {"category": ["x", "x", "y", "y"], "value": ["banana", "apple", "cherry", "apricot"], "weight": [1, 2, 3, 4]}
    )
    out = unique(df, group_keys=["category"], sort_by="value", ascending=True)

    expected = pl.DataFrame({"category": ["x", "y"], "value": ["apple", "apricot"], "weight": [2, 4]})

    assert out.sort(by=out.columns).rows() == expected.sort(by=expected.columns).rows()


def test_describe():
    df = pl.DataFrame(
        {
            "age": ["10", "20", "30", "0.0", None],  # force float
            "name": ["A", "B", "A", "C", None],
            "date": ["2020-01-01", "2020-01-01", "2020-05-01", "2021-01-01", None],
        }
    )

    summary = describe(df, date_format="%Y-%m-%d")

    # --- float column ---
    row = summary.filter(pl.col("column") == "age").row(0)
    idx = lambda key: summary.columns.index(key)

    assert row[idx("dtype")] == "float"
    assert isclose(float(row[idx("mean")]), 15.0)
    assert isclose(float(row[idx("std")]), 12.909944, rel_tol=1e-3)
    assert float(row[idx("min")]) == 0.0
    assert float(row[idx("max")]) == 30.0
    assert float(row[idx("Q1")]) == 10.0
    assert float(row[idx("median")]) == 15.0
    assert float(row[idx("Q3")]) == 20.0
    assert float(row[idx("zeros")]) == 1.0
    assert float(row[idx("infinite")]) == 0.0

    # --- string column ---
    row = summary.filter(pl.col("column") == "name").row(0)
    assert row[idx("dtype")] == "string"
    assert row[idx("top")] == "A"
    assert row[idx("top_freq")] == "2"
    assert isclose(float(row[idx("top_ratio")]), 2 / 5)
    assert row[idx("min_cat")] in {"B", "C"}
    assert row[idx("min_freq")] == "1"
    assert isclose(float(row[idx("min_ratio")]), 1 / 5)
    assert isclose(float(row[idx("avg_length")]), 1.0)
    assert float(row[idx("min_length")]) == 1.0
    assert float(row[idx("max_length")]) == 1.0

    # --- date column ---
    row = summary.filter(pl.col("column") == "date").row(0)
    assert row[idx("dtype")] == "date"

    assert row[idx("min")] == "2020-01-01"
    assert row[idx("max")] == "2021-01-01"

    assert row[idx("range_days")] == "366"
    assert row[idx("min_year")] == "2020"
    assert row[idx("max_year")] == "2021"
    assert row[idx("min_month")] == "1"
    assert row[idx("max_month")] == "5"
    assert row[idx("mode")] == "2020-01-01"
    assert row[idx("mode_freq")] == "2"
    assert isclose(float(row[idx("mode_ratio")]), 0.5)


def test_describe_supports_yyyymm_format():
    df = pl.DataFrame(
        {
            "month_str": ["202301", "202301", "202302", "202303", None],
            "value": ["1", "2", "3", "4", "5"],
        }
    )

    summary = describe(df, date_format="%Y%m")

    row = summary.filter(pl.col("column") == "month_str").row(0)
    idx = lambda key: summary.columns.index(key)

    assert row[idx("dtype")] == "date"
    assert row[idx("min")] == "2023-01-01"
    assert row[idx("max")] == "2023-03-01"
    assert row[idx("range_days")] == "59"
    assert row[idx("min_year")] == "2023"
    assert row[idx("max_year")] == "2023"
    assert row[idx("min_month")] == "1"
    assert row[idx("max_month")] == "3"
    assert row[idx("mode")] == "2023-01-01"
    assert row[idx("mode_freq")] == "2"
    assert isclose(float(row[idx("mode_ratio")]), 0.5)


def test_get_categorical_counts_table_valid_column():
    df = pl.DataFrame({"category": ["A", "B", "A", "C", "B", "A", "D", "C", "A"], "value": [1, 2, 3, 4, 5, 6, 7, 8, 9]})

    result_df = get_categorical_counts_table(df, "category")

    expected_data = {"category": ["A", "B", "C", "D"], "count": [4, 2, 2, 1]}
    expected_df = (
        pl.DataFrame(expected_data).with_columns(pl.col("count").cast(pl.UInt32)).sort("count", descending=True)
    )

    assert result_df.equals(expected_df)


def test_get_categorical_counts_table_with_nulls():
    df = pl.DataFrame({"color": ["Red", "Blue", None, "Red", "Green", None, "Red"], "value": [1, 2, 3, 4, 5, 6, 7]})

    result_df = get_categorical_counts_table(df, "color")

    expected_data = {"color": ["Red", "Blue", "Green"], "count": [3, 1, 1]}
    expected_df = (
        pl.DataFrame(expected_data).with_columns(pl.col("count").cast(pl.UInt32)).sort("count", descending=True)
    )

    assert result_df.equals(expected_df)


def test_get_categorical_counts_table_empty_column_after_dropping_nulls():
    df = pl.DataFrame({"empty_category": [None, None, None], "value": [1, 2, 3]})

    result_df = get_categorical_counts_table(df, "empty_category")

    assert result_df.is_empty()


def test_get_categorical_counts_table_non_existent_column():
    df = pl.DataFrame({"id": [1, 2, 3], "name": ["X", "Y", "Z"]})

    with pytest.raises(ValueError) as excinfo:
        get_categorical_counts_table(df, "non_existent_col")

    assert "Error: The specified column 'non_existent_col' does not exist in the DataFrame." in str(excinfo.value)


def test_get_categorical_counts_table_empty_dataframe():
    df = pl.DataFrame({"category": pl.Series(dtype=pl.String), "count": pl.Series(dtype=pl.Int64)})

    result_df = get_categorical_counts_table(df, "category")

    assert result_df.is_empty()


def test_plot_categorical_bar_chart_saves_image_file():
    sample_categories = np.array(["ひらがな", "カタカナ", "漢字", "1"])
    sample_counts = np.array([25, 18, 12, 7])
    sample_column_name = "Experiment_Results"  # This will be part of the filename

    expected_output_filename = f"./samples/{sample_column_name}_categorical_bar_chart.html"

    if os.path.exists(expected_output_filename):
        os.remove(expected_output_filename)

    plot_categorical_bar_chart(sample_categories, sample_counts, sample_column_name, expected_output_filename)

    assert os.path.exists(
        expected_output_filename
    ), f"Error: The chart file '{expected_output_filename}' was not created."


def test_plot_numerical_distribution_saves_html_file():
    sample_data = np.random.rand(100) * 100  # Random data between 0 and 100
    sample_column_name = "Sample_Numerical_Data"
    expected_output_filename = f"./samples/{sample_column_name}_distribution_chart.html"

    # Clean up any previously created file to ensure a clean test run
    if os.path.exists(expected_output_filename):
        os.remove(expected_output_filename)

    # Call the function to create and save the chart
    plot_numerical_distribution(sample_data, sample_column_name, expected_output_filename)

    # Assert that the HTML file was successfully created
    assert os.path.exists(
        expected_output_filename
    ), f"Error: The chart file '{expected_output_filename}' was not created."


def test_plot_timeseries_histogram_saves_html_file():
    # Sample data for testing (strings will be converted to datetime by the function)
    sample_dates = np.array(
        [
            "2023-01-01",
            "2023-01-05",
            "2023-01-10",
            "2023-01-15",
            "2023-01-20",
            "2023-02-01",
            "2023-02-05",
            "2023-02-10",
            "2023-02-15",
            "2023-02-20",
            "2023-03-01",
            "2023-03-05",
            "2023-03-10",
        ]
    )
    sample_column_name = "EventDates"
    expected_output_filename = f"./samples/{sample_column_name}_timeseries_histogram.html"

    # Clean up any previously created file before running the test
    if os.path.exists(expected_output_filename):
        os.remove(expected_output_filename)

    # Call the function to generate the chart
    plot_timeseries_histogram(sample_dates, sample_column_name)

    # Assert that the HTML file was created successfully
    assert os.path.exists(
        expected_output_filename
    ), f"Error: The chart file '{expected_output_filename}' was not created."


def test_profile():
    sample_pl_df = pl.DataFrame(
        {
            "商品カテゴリ": ["野菜", "果物", "肉", "野菜", "果物", "魚", "野菜", "肉", "果物"],
            "売上高": np.random.normal(loc=10000, scale=3000, size=9).astype(int),
            "顧客満足度": np.random.randint(1, 6, size=9),
            "購入日時": [
                "2023-10-01 10:00",
                "2023-10-05 14:30",
                "2023-10-08 11:00",
                "2023-10-10 16:00",
                "2023-10-12 09:00",
                "2023-10-15 18:00",
                "2023-11-01 13:00",
                "2023-11-05 10:00",
                "2023-11-08 17:00",
            ],
            "店舗ID": ["A", "B", "A", "C", "B", "A", "C", "B", "A"],
            "顧客年齢": np.random.normal(loc=35, scale=10, size=9).astype(int),
        }
    )

    output_file = "./samples/test_combined_report_valid_data.html"
    profile(sample_pl_df, output_file)

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_profile_supports_yyyymm_format():
    sample_pl_df = pl.DataFrame(
        {
            "年月": [
                "202310",
                "202311",
                "202312",
                "202401",
                "202402",
                "202403",
                "202404",
                "202405",
                "202406",
            ],
            "売上高": np.random.normal(loc=12000, scale=2500, size=9).astype(int),
            "店舗ID": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
        }
    )

    output_file = "./samples/test_combined_report_yyyymm_format.html"
    profile(sample_pl_df, output_file, date_format="%Y%m")

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_profile_bivariate_valid_data():
    sample_pl_df = pl.DataFrame(
        {
            "商品カテゴリ": ["野菜", "果物", "肉", "野菜", "果物", "魚", "野菜", "肉", "果物", "野菜"],
            "売上高": np.random.normal(loc=10000, scale=3000, size=10).astype(int),
            "顧客満足度": np.random.randint(1, 6, size=10),
            "購入日時": [
                "2023-10-01 10:00",
                "2023-10-05 14:30",
                "2023-10-08 11:00",
                "2023-10-10 16:00",
                "2023-10-12 09:00",
                "2023-10-15 18:00",
                "2023-11-01 13:00",
                "2023-11-05 10:00",
                "2023-11-08 17:00",
                "2023-11-10 10:00",
            ],
            "店舗ID": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "A"],
            "顧客年齢": np.random.normal(loc=35, scale=10, size=10).astype(int),
            "イベント日": [
                "2023-10-02 11:00",
                "2023-10-06 15:00",
                "2023-10-09 12:00",
                "2023-10-11 17:00",
                "2023-10-13 10:00",
                "2023-10-16 19:00",
                "2023-11-02 14:00",
                "2023-11-06 11:00",
                "2023-11-09 18:00",
                "2023-11-11 11:00",
            ],
        }
    )

    column_pairs = [
        ("売上高", "顧客年齢"),  # Numerical x Numerical
        ("売上高", "商品カテゴリ"),  # Numerical x Categorical
        ("商品カテゴリ", "店舗ID"),  # Categorical x Categorical
        ("購入日時", "売上高"),  # Datetime x Numerical
        ("購入日時", "商品カテゴリ"),  # Datetime x Categorical
        ("購入日時", "イベント日"),  # Datetime x Datetime
        ("店舗ID", "商品カテゴリ"),  # Another Categorical x Categorical
    ]

    output_file = "./samples/bivariate_report_valid_data.html"
    profile_bivariate(sample_pl_df, column_pairs, output_file, date_format="%Y-%m-%d %H:%M")

    assert os.path.exists(output_file)
    # Check if the file is not empty
    assert os.path.getsize(output_file) > 0


def test_profile_bivariate_supports_yyyymm_format():
    sample_pl_df = pl.DataFrame(
        {
            "年月": [
                "202310",
                "202310",
                "202311",
                "202311",
                "202312",
                "202312",
                "202401",
                "202401",
                "202402",
                "202402",
            ],
            "売上高": np.random.normal(loc=12000, scale=2500, size=10).astype(int),
            "商品カテゴリ": ["野菜", "果物", "肉", "魚", "野菜", "果物", "肉", "魚", "野菜", "果物"],
        }
    )

    column_pairs = [
        ("年月", "売上高"),  # Datetime (from string) × Numerical
        ("年月", "商品カテゴリ"),  # Datetime (from string) × Categorical
        ("売上高", "商品カテゴリ"),  # Numerical × Categorical
    ]

    output_file = "./samples/bivariate_report_yyyymm_format.html"
    profile_bivariate(sample_pl_df, column_pairs, output_file, date_format="%Y%m")

    assert os.path.exists(output_file)
    assert os.path.getsize(output_file) > 0


def test_month_window():
    data_a = {
        "date_str": [
            "2023-10-15",
            "2023-11-01",
            "2023-11-15",
            "2023-12-01",
            "2023-12-15",
            "2024-01-01",
            "2024-01-15",
            "2024-02-01",
            "2024-02-15",
            "2024-03-01",
            "2024-03-15",
            "2024-04-01",
        ],
        "value_a": [5, 6, 7, 8, 9, 10, 11, 20, 21, 30, 31, 40],
        "base_id": ["B1"] * 12,
    }
    df_data_lazy = pl.DataFrame(data_a).lazy()

    data_b = {
        "date_str": ["2024-01-01", "2024-02-01", "2024-03-01"],  # 同じカラム名・同じフォーマット
        "base_id": ["B1", "B1", "B1"],
    }
    df_base_lazy = pl.DataFrame(data_b).lazy()

    result_df_mixed = month_window(
        df_base=df_base_lazy,
        df_data=df_data_lazy,
        date_col="date_str",
        date_format="%Y-%m-%d",  # 両方で共通
        value_cols=["value_a"],
        months_list=[1, -1],
        new_col_name_prefix="test_metric",
        metrics=["mean", "sum"],
        keys=["base_id"],
    )

    expected_data_mixed = pl.DataFrame(
        {
            "base_id": ["B1", "B1", "B1"],
            "test_metric_value_a_mean_future_1m": [10.5, 20.5, 30.5],
            "test_metric_value_a_sum_future_1m": [21.0, 41.0, 61.0],
            "test_metric_value_a_mean_past_1m": [8.5, 10.5, 20.5],
            "test_metric_value_a_sum_past_1m": [17.0, 21.0, 41.0],
        }
    )

    result_mixed_sorted = result_df_mixed.collect().select(expected_data_mixed.columns).sort("base_id")
    expected_mixed_sorted = expected_data_mixed.sort("base_id")

    assert_frame_equal(
        result_mixed_sorted,
        expected_mixed_sorted,
        check_dtypes=False,
        check_column_order=False,
    )


def test_month_window_yyyymm_format():
    df_data = pl.DataFrame(
        {
            "date_str": ["202302", "202303", "202304"],
            "value": [10, 20, 30],
            "base_id": ["B1"] * 3,
        }
    ).lazy()

    df_base = pl.DataFrame(
        {
            "date_str": ["202303"],  # 同一カラム名
            "base_id": ["B1"],
        }
    ).lazy()

    result = month_window(
        df_base=df_base,
        df_data=df_data,
        date_col="date_str",
        date_format="%Y%m",
        value_cols=["value"],
        months_list=[1, -1],
        new_col_name_prefix="metric",
        metrics=["mean", "sum"],
        keys=["base_id"],
    )

    expected = pl.DataFrame(
        {
            "base_id": ["B1"],
            "metric_value_mean_future_1m": [20.0],
            "metric_value_sum_future_1m": [20],
            "metric_value_mean_past_1m": [10.0],
            "metric_value_sum_past_1m": [10],
        }
    )

    result_sorted = result.collect().select(expected.columns)

    assert_frame_equal(result_sorted, expected, check_column_order=False, check_dtypes=False)


def test_month_window_no_data_in_window():
    df_data = pl.DataFrame(
        {
            "date_str": ["202201", "202212"],  # 完全にウィンドウ外
            "value": [100, 200],
            "base_id": ["B1", "B1"],
        }
    ).lazy()

    df_base = pl.DataFrame(
        {
            "date_str": ["202303"],
            "base_id": ["B1"],
        }
    ).lazy()

    result = month_window(
        df_base=df_base,
        df_data=df_data,
        date_col="date_str",
        date_format="%Y%m",
        value_cols=["value"],
        months_list=[1, -1],
        new_col_name_prefix="metric",
        metrics=["mean", "sum"],
        keys=["base_id"],
    )

    expected = pl.DataFrame(
        {
            "base_id": ["B1"],
            "metric_value_mean_future_1m": [None],
            "metric_value_sum_future_1m": [None],
            "metric_value_mean_past_1m": [None],
            "metric_value_sum_past_1m": [None],
        }
    )

    result_sorted = result.collect().select(expected.columns)

    assert_frame_equal(result_sorted, expected, check_column_order=False, check_dtypes=False)
