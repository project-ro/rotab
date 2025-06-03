import pandas as pd
from rotab.core.operation.transform_funcs import *
from pandas.testing import assert_frame_equal


def test_sort_by():
    df = pd.DataFrame({"id": [3, 1, 2]})
    result = sort_by(df, "id")
    expected = pd.DataFrame({"id": [1, 2, 3]}, index=[1, 2, 0])
    assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test_groupby_agg():
    df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1, 2, 3]})
    result = groupby_agg(df, "key", {"val": "sum"})
    expected = pd.DataFrame({"key": ["a", "b"], "val": [3, 3]})
    assert_frame_equal(result, expected)


def test_drop_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2], "y": [3, 3, 4]})
    result = drop_duplicates(df)
    expected = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test_merge():
    df1 = pd.DataFrame({"id": [1, 2], "val1": ["a", "b"]})
    df2 = pd.DataFrame({"id": [2, 3], "val2": ["c", "d"]})
    result = merge(df1, df2, on="id", how="inner")
    expected = pd.DataFrame({"id": [2], "val1": ["b"], "val2": ["c"]})
    assert_frame_equal(result, expected)


def test_reshape_pivot_table():
    df = pd.DataFrame({"date": ["2021-01", "2021-01", "2021-02"], "type": ["A", "B", "A"], "value": [10, 20, 30]})
    result = reshape(df, column_to="date", columns_from=["type"], column_value="value", agg="sum")
    expected = pd.DataFrame({"date": ["2021-01", "2021-02"], "A": [10.0, 30.0], "B": [20.0, None]})
    assert_frame_equal(
        result.set_index("date").sort_index(axis=1), expected.set_index("date").sort_index(axis=1), check_dtype=False
    )


def test_reshape_pivot():
    df = pd.DataFrame({"date": ["2021-01", "2021-01"], "type": ["A", "B"], "value": [10, 20]})
    result = reshape(df, column_to="date", columns_from=["type"], column_value="value")
    expected = pd.DataFrame({"date": ["2021-01"], "A": [10], "B": [20]})
    assert_frame_equal(result, expected)


def test_reshape_melt():
    df = pd.DataFrame({"date": ["2021-01", "2021-02"], "value": [10, 20]})
    result = reshape(df, column_to="date", column_value="value")
    expected = pd.DataFrame({"date": ["2021-01", "2021-02"], "variable": ["value", "value"], "melted_value": [10, 20]})
    assert_frame_equal(result, expected)


def test_fillna():
    df = pd.DataFrame({"a": [1, None], "b": [None, "x"]})
    result = fillna(df, {"a": 0, "b": "y"})
    expected = pd.DataFrame({"a": [1.0, 0.0], "b": ["y", "x"]})
    assert_frame_equal(result, expected)


def test_sample():
    df = pd.DataFrame({"x": range(100)})
    result = sample(df, 0.1)
    assert len(result) == 10
    assert set(result.columns) == {"x"}


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
    assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))


def test_replace():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["x", "z"]})
    result = replace(df, ["a", "b"], "x", "replaced")
    expected = pd.DataFrame({"a": ["replaced", "y"], "b": ["replaced", "z"]})
    assert_frame_equal(result, expected)
