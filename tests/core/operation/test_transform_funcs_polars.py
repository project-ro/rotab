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
    summarize_columns,
    is_date_column,
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
    out = merge(df1, df2, on="id", how="left")
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


def test_fillna():
    df = pl.DataFrame({"a": [None, 2]})
    out = fillna(df, {"a": 0})
    assert out["a"].to_list()[0] == 0


def test_sample():
    df = pl.DataFrame({"a": list(range(100))})
    out = sample(df, frac=0.1)
    assert 5 <= out.height <= 15  # 許容範囲チェック


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


def test_summarize_columns_all_string_full_metrics():
    df = pl.DataFrame(
        {
            "age": ["10", "20", "30", "0.0", None],  # force float
            "name": ["A", "B", "A", "C", None],
            "date": ["2020-01-01", "2020-01-01", "2020-05-01", "2021-01-01", None],
        }
    )

    summary = summarize_columns(df)

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
