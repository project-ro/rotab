import polars as pl
import ast
from rotab.core.parse.expr import parse_derive_expr


def test_parse_derive_expr():
    df = pl.DataFrame(
        {
            "age": [15, 20, 25],
            "income": [3000, 4000, 5000],
            "bonus": [1000, 1500, 2000],
            "name": ["Alice", "Bob", "Charlie"],
        }
    )

    derive_str = """
    add_col = age + bonus
    sub_col = income - bonus
    mul_col = age * 2
    div_col = income / age
    floordiv_col = income // age
    mod_col = income % age
    gt_col = income > 3500
    gte_col = income >= 4000
    lt_col = age < 25
    lte_col = age <= 20
    eq_col = age == 20
    neq_col = age != 15
    and_col = (income > 3000) & (age < 25)
    or_col = (income < 3500) | (age >= 25)
    not_col = not (age < 20)
    complex_score = (log(age) + clip(income, 3000, 5000) / 2) * sqrt(bonus) - abs(age - bonus)
    name_upper = upper(name)
    """

    exprs = parse_derive_expr(derive_str)
    df2 = df.with_columns(exprs)

    import math

    # add_col
    expected_add = [a + b for a, b in zip(df["age"], df["bonus"])]
    assert df2["add_col"].to_list() == expected_add

    # sub_col
    expected_sub = [i - b for i, b in zip(df["income"], df["bonus"])]
    assert df2["sub_col"].to_list() == expected_sub

    # mul_col
    expected_mul = [a * 2 for a in df["age"]]
    assert df2["mul_col"].to_list() == expected_mul

    # div_col
    expected_div = [i / a for i, a in zip(df["income"], df["age"])]
    assert all(abs(a - b) < 1e-6 for a, b in zip(df2["div_col"], expected_div))

    # floordiv_col
    expected_fdiv = [i // a for i, a in zip(df["income"], df["age"])]
    assert df2["floordiv_col"].to_list() == expected_fdiv

    # mod_col
    expected_mod = [i % a for i, a in zip(df["income"], df["age"])]
    assert df2["mod_col"].to_list() == expected_mod

    # gt_col
    expected_gt = [i > 3500 for i in df["income"]]
    assert df2["gt_col"].to_list() == expected_gt

    # gte_col
    expected_gte = [i >= 4000 for i in df["income"]]
    assert df2["gte_col"].to_list() == expected_gte

    # lt_col
    expected_lt = [a < 25 for a in df["age"]]
    assert df2["lt_col"].to_list() == expected_lt

    # lte_col
    expected_lte = [a <= 20 for a in df["age"]]
    assert df2["lte_col"].to_list() == expected_lte

    # eq_col
    expected_eq = [a == 20 for a in df["age"]]
    assert df2["eq_col"].to_list() == expected_eq

    # neq_col
    expected_neq = [a != 15 for a in df["age"]]
    assert df2["neq_col"].to_list() == expected_neq

    # and_col
    expected_and = [(i > 3000) and (a < 25) for i, a in zip(df["income"], df["age"])]
    assert df2["and_col"].to_list() == expected_and

    # or_col
    expected_or = [(i < 3500) or (a >= 25) for i, a in zip(df["income"], df["age"])]
    assert df2["or_col"].to_list() == expected_or

    # not_col
    expected_not = [not (a < 20) for a in df["age"]]
    assert df2["not_col"].to_list() == expected_not

    # complex_score
    expected_score = []
    for a, i, b in zip(df["age"], df["income"], df["bonus"]):
        v = (math.log(a, 10) + min(max(i, 3000), 5000) / 2) * math.sqrt(b) - abs(a - b)
        expected_score.append(v)
    assert all(abs(a - b) < 1e-6 for a, b in zip(df2["complex_score"], expected_score))

    # name_upper
    expected_upper = [x.upper() for x in df["name"]]
    assert df2["name_upper"].to_list() == expected_upper

    print("All extended derive expression tests passed (full operators).")
