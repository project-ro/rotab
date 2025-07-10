import pytest
import polars as pl

from rotab.core.parse.expr import expr


# -------------------------------
# derive 系テスト
# -------------------------------


def test_expr_derive_basic():
    derive_str = """
        new_col = a + b
        double_col = a * 2
    """
    res = expr(derive_str)
    assert isinstance(res, list)
    assert all(isinstance(e, pl.Expr) for e in res)

    aliases = [e.meta.output_name() for e in res]
    assert "new_col" in aliases
    assert "double_col" in aliases


def test_expr_derive_with_functions(monkeypatch):
    from rotab.core.operation.derive_funcs import FUNC_NAMESPACE

    FUNC_NAMESPACE["func"] = lambda x: pl.col(x) * 10
    FUNC_NAMESPACE["funv"] = lambda x: pl.col(x) + 100

    derive_str = """
        new_col = func(a) + funv(b)
    """
    res = expr(derive_str)
    assert isinstance(res, list)
    aliases = [e.meta.output_name() for e in res]
    assert "new_col" in aliases

    FUNC_NAMESPACE.pop("func")
    FUNC_NAMESPACE.pop("funv")


def test_expr_derive_complex_expression():
    derive_str = """
        complex_col = (a + b) * (c - d) / (e + 1)
    """
    res = expr(derive_str)
    assert isinstance(res, list)
    aliases = [e.meta.output_name() for e in res]
    assert "complex_col" in aliases


def test_expr_derive_invalid_line_fallback_to_filter():
    bad_derive = """
        invalid_line a + b
    """
    with pytest.raises(ValueError, match=r"Invalid syntax in filter expression"):
        expr(bad_derive)


# -------------------------------
# filter 系テスト
# -------------------------------


def test_expr_filter_basic():
    filter_str = "age > 20 and income < 3000"
    res = expr(filter_str)
    assert isinstance(res, pl.Expr)
    assert set(res.meta.root_names()) >= {"age", "income"}


def test_expr_filter_complex_logic():
    filter_str = "(score > 80) or (score < 20 and passed)"
    res = expr(filter_str)
    assert isinstance(res, pl.Expr)
    assert set(res.meta.root_names()) >= {"score", "passed"}


def test_expr_filter_not_in_list():
    filter_str = "city not in ['Tokyo', 'Osaka']"
    res = expr(filter_str)
    assert isinstance(res, pl.Expr)
    assert "city" in res.meta.root_names()


def test_expr_filter_with_unary_op():
    filter_str = "not is_active"
    res = expr(filter_str)
    assert isinstance(res, pl.Expr)
    assert "is_active" in res.meta.root_names()


def test_expr_filter_invalid_syntax():
    bad_str = "this is not an expression"
    with pytest.raises(ValueError, match="Invalid syntax"):
        expr(bad_str)


def test_expr_filter_unsupported_ast_type():
    bad_str = "{'a': 1}"  # dict literal, ast.Dict
    with pytest.raises(ValueError, match="Unsupported expression type"):
        expr(bad_str)


def test_expr_filter_with_is_operator():
    bad_str = "flag is True"
    with pytest.raises(ValueError):
        expr(bad_str)


# -------------------------------
# select 系テスト
# -------------------------------


def test_expr_select_basic():
    cols = ["col1", "col2", "col3"]
    res = expr(cols)
    assert isinstance(res, list)
    assert res == cols


def test_expr_select_empty_list():
    res = expr([])
    assert isinstance(res, list)
    assert res == []


def test_expr_select_non_string_elements():
    with pytest.raises(ValueError, match="List elements must be strings"):
        expr([1, 2, 3])


# -------------------------------
# 型不正・汎用エラー
# -------------------------------


def test_expr_invalid_type_int():
    with pytest.raises(ValueError, match="Unsupported expression format"):
        expr(123)


def test_expr_invalid_type_none():
    with pytest.raises(ValueError, match="Unsupported expression format"):
        expr(None)
