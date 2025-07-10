import polars as pl
from rotab.core.parse.expr import parse_filter_expr


def test_parse_filter_expr():
    df = pl.DataFrame(
        {
            "age": [15, 20, 25, 30],
            "income": [3000, 4000, 6000, 7000],
            "status": ["single", "married", "single", "married"],
            "remarks": [None, "ok", None, "ok"],
        }
    )

    # in 条件
    expr1 = parse_filter_expr("status in ['single']")
    assert df.filter(expr1)["age"].to_list() == [15, 25]

    # not in 条件
    expr2 = parse_filter_expr("status not in ['single']")
    assert df.filter(expr2)["age"].to_list() == [20, 30]

    # is None 条件
    expr3 = parse_filter_expr("remarks is None")
    assert df.filter(expr3)["age"].to_list() == [15, 25]

    # is not None 条件
    expr4 = parse_filter_expr("remarks is not None")
    assert df.filter(expr4)["age"].to_list() == [20, 30]

    # 複合条件
    expr5 = parse_filter_expr("age >= 20 and income < 7000")
    assert df.filter(expr5)["age"].to_list() == [20, 25]

    # or 条件
    expr6 = parse_filter_expr("age < 18 or income > 6500")
    assert df.filter(expr6)["age"].to_list() == [15, 30]

    # not 条件
    expr7 = parse_filter_expr("not age < 25")
    assert df.filter(expr7)["age"].to_list() == [25, 30]

    print("All extended filter expression tests passed.")
