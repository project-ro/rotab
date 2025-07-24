import pytest
import polars as pl

from rotab.core.parse.parse import parse
from rotab.core.operation.derive_funcs_polars import _col


# -------------------------------
# derive 系テスト
# -------------------------------


def test_parse_derive_basic():
    derive_str = """
        new_col = a + b
        double_col = a * 2
    """

    res = parse(derive_str)
    assert isinstance(res, list)
    assert all(isinstance(e, pl.Expr) for e in res)

    aliases = [e.meta.output_name() for e in res]
    assert "new_col" in aliases
    assert "double_col" in aliases


def test_parse_derive_with_functions(monkeypatch):
    from rotab.core.operation.derive_funcs_polars import FUNC_NAMESPACE

    FUNC_NAMESPACE["func"] = lambda x: _col(x) * 10
    FUNC_NAMESPACE["funv"] = lambda x: _col(x) + 100

    derive_str = """
        new_col = func(a) + funv(b)
    """
    res = parse(derive_str)
    assert isinstance(res, list)
    aliases = [e.meta.output_name() for e in res]
    assert "new_col" in aliases

    FUNC_NAMESPACE.pop("func")
    FUNC_NAMESPACE.pop("funv")


def test_parse_derive_complex_expression():
    derive_str = """
        complex_col = (a + b) * (c - d) / (e + 1)
    """
    res = parse(derive_str)
    assert isinstance(res, list)
    aliases = [e.meta.output_name() for e in res]
    assert "complex_col" in aliases


def test_parse_derive_invalid_line_fallback_to_filter():
    bad_derive = """
        invalid_line a + b
    """
    with pytest.raises(ValueError, match=r"Invalid syntax in filter expression"):
        parse(bad_derive)


# -------------------------------
# filter 系テスト
# -------------------------------


def test_parse_filter_basic():
    filter_str = "age > 20 and income < 3000"
    res = parse(filter_str)
    assert isinstance(res, pl.Expr)
    assert set(res.meta.root_names()) >= {"age", "income"}


def test_parse_filter_complex_logic():
    filter_str = "(score > 80) or (score < 20 and passed)"
    res = parse(filter_str)
    assert isinstance(res, pl.Expr)
    assert set(res.meta.root_names()) >= {"score", "passed"}


def test_parse_filter_not_in_list():
    filter_str = "city not in ['Tokyo', 'Osaka']"
    res = parse(filter_str)
    assert isinstance(res, pl.Expr)
    assert "city" in res.meta.root_names()


def test_parse_filter_with_unary_op():
    filter_str = "not is_active"
    res = parse(filter_str)
    assert isinstance(res, pl.Expr)
    assert "is_active" in res.meta.root_names()


def test_parse_filter_invalid_syntax():
    bad_str = "this is not an expression"
    with pytest.raises(ValueError, match="Invalid syntax"):
        parse(bad_str)


def test_parse_filter_unsupported_ast_type():
    bad_str = "{'a': 1}"  # dict literal, ast.Dict
    with pytest.raises(ValueError, match="Unsupported expression type"):
        parse(bad_str)


def test_parse_filter_with_is_operator():
    bad_str = "flag is True"
    with pytest.raises(ValueError):
        parse(bad_str)


# -------------------------------
# select 系テスト
# -------------------------------


def test_parse_select_basic():
    cols = ["col1", "col2", "col3"]
    res = parse(cols)
    assert isinstance(res, list)
    assert res == cols


def test_parse_select_empty_list():
    res = parse([])
    assert isinstance(res, list)
    assert res == []


def test_parse_select_non_string_elements():
    with pytest.raises(ValueError, match="List elements must be strings"):
        parse([1, 2, 3])


# -------------------------------
# 型不正・汎用エラー
# -------------------------------


def test_parse_invalid_type_int():
    with pytest.raises(ValueError, match="Unsupported expression format"):
        parse(123)


def test_parse_invalid_type_none():
    with pytest.raises(ValueError, match="Unsupported expression format"):
        parse(None)
