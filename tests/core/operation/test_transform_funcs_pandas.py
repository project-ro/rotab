import pandas as pd
from pandas.testing import assert_frame_equal

from rotab.core.operation.transform_funcs_pandas import (
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


def test_validate_table_schema_pass():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    columns = [{"name": "a", "dtype": "int64"}, {"name": "b", "dtype": "str"}]
    assert validate_table_schema(df, columns)


def test_sort_by():
    df = pd.DataFrame({"a": [2, 1]})
    result = sort_by(df, "a")
    expected = pd.DataFrame({"a": [1, 2]})
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_groupby_agg():
    df = pd.DataFrame({"group": ["x", "x", "y"], "val": [1, 2, 3]})
    result = groupby_agg(df, "group", {"val": "sum"})
    expected = pd.DataFrame({"group": ["x", "y"], "val": [3, 3]})
    assert_frame_equal(result, expected)


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    result = drop_duplicates(df)
    expected = pd.DataFrame({"a": [1, 2]})
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_merge():
    left = pd.DataFrame({"id": [1, 2], "val1": ["a", "b"]})
    right = pd.DataFrame({"id": [2, 3], "val2": ["c", "d"]})
    result = merge(left, right, on="id", how="inner")
    expected = pd.DataFrame({"id": [2], "val1": ["b"], "val2": ["c"]})
    assert_frame_equal(result, expected)


def test_reshape_pivot_table():
    df = pd.DataFrame({"id": [1, 1, 2], "key": ["x", "y", "x"], "val": [1, 2, 3]})
    result = reshape(df, column_to="id", columns_from=["key"], column_value="val", agg="sum")
    expected = pd.DataFrame({"id": [1, 2], "x": [1, 3], "y": [2, None]})
    assert_frame_equal(result, expected, check_dtype=False)


def test_reshape_pivot():
    df = pd.DataFrame({"id": [1, 1, 2], "key": ["x", "y", "x"], "val": [1, 2, 3]})
    result = reshape(df, column_to="id", columns_from=["key"], column_value="val")
    expected = pd.DataFrame({"id": [1, 1, 2], "x": [1.0, None, 3.0], "y": [None, 2.0, None]}).dropna(axis=1, how="all")
    assert sorted(result.columns) == sorted(expected.columns)


def test_reshape_melt():
    df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
    result = reshape(df, column_to="id", column_value="val")
    expected = pd.DataFrame({"id": [1, 2], "variable": ["val", "val"], "melted_value": [10, 20]})
    assert_frame_equal(result, expected)


def test_fillna():
    df = pd.DataFrame({"a": [1, None]})
    result = fillna(df, {"a": 0})
    expected = pd.DataFrame({"a": [1.0, 0.0]})
    assert_frame_equal(result, expected)


def test_sample():
    df = pd.DataFrame({"a": range(100)})
    result = sample(df, frac=0.1)
    assert len(result) == 10


def test_concat():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})
    result = concat([df1, df2])
    expected = pd.DataFrame({"a": [1, 2]})
    assert_frame_equal(result, expected)


def test_drop_na():
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    result = drop_na(df, subset=["a"])
    expected = pd.DataFrame({"a": [1.0], "b": [2]})
    assert_frame_equal(result.reset_index(drop=True), expected)


def test_replace():
    df = pd.DataFrame({"a": ["x", "y", "x"]})
    result = replace(df, ["a"], "x", "z")
    expected = pd.DataFrame({"a": ["z", "y", "z"]})
    assert_frame_equal(result, expected)
