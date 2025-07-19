import polars as pl
import pytest
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
