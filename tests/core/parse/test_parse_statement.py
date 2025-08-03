from rotab.core.parse.parse_statement import parse_statement
import pytest


@pytest.mark.parametrize(
    "input_expr,expected",
    [
        # Standard cases with various spacing
        (r"merge left hoge, right hage, on 'key'", r"merge(left=hoge,right=hage,on='key')"),
        (r"merge  left   hoge, right    hage,   on'key'", r"merge(left=hoge,right=hage,on='key')"),
        # General function with list and dict arguments
        (r"groupby table users, by age and gender", r"groupby(table=users,by=[age,gender])"),
        (
            r"aggregate table enriched, with amount mean and high_value sum",
            r"aggregate(table=enriched,with={amount:mean,high_value:sum})",
        ),
        (r"join left x and y, right z", r"join(left=[x,y],right=z)"),
        # 'add' statements with various expressions and spacing
        (r"add new_col / int(old_col) / 30 + 5", r"new_col=int(old_col)/30+5"),
        (r"add new_col/int(old_col)/30+5", r"new_col=int(old_col)/30+5"),
        (r"add new_col/(int(old_col)/30)+5", r"new_col=(int(old_col)/30)+5"),  # with parenthesis
        # 'add' with nested function calls
        (
            r"add base_date / formatting for yyyymm from '%Y%m' to '%Y-%m'",
            r"base_date=formatting(for=yyyymm,from='%Y%m',to='%Y-%m')",
        ),
        (r"add total_price / price * quantity", r"total_price=price*quantity"),
        # 'select' statements
        (r"select col1 and col2 and col3", r"[col1,col2,col3]"),
        (r"select  col_a, col_b, col_c", r"[col_a,col_b,col_c]"),  # handles commas in select
        # 'filter' statements
        (r"filter / col1 > 10", r"col1>10"),
        (r"filter / valid_date for col1 or valid_name for col2", r"valid_date(for=col1) or valid_name(for=col2)"),
        (
            r"filter/price>100 and status== 'completed' ",
            r"price>100 and status=='completed'",
        ),  # with quotes and operators
        # Edge cases and single arguments
        (r"transform", r"transform()"),  # no arguments
        (r"transform  ", r"transform()"),  # no arguments with spaces
        (r"run", r"run()"),
        (r"sort by col1", r"sort(by=col1)"),
    ],
)
def test_parse_statement_success(input_expr, expected):
    assert parse_statement(input_expr) == expected


# @pytest.mark.parametrize(
#     "bad_input",
#     [
#         r"",
#         r"merge left",  # incomplete key-value pair
#         r"groupby table",  # missing 'by' argument
#         r"add new_col",  # missing '/'
#         r"select",  # no columns
#         r"filter /",  # empty condition
#         r"aggregate table, with amount",  # incomplete dict syntax
#         r"add new_col/",  # empty right-hand side
#         r"filter / is valid date",  # missing column name
#         r"invalid_func arg1",  # unknown argument separator
#         r"merge left=hoge, right",  # missing value for 'right'
#     ],
# )
# def test_parse_statement_fail(bad_input):
#     with pytest.raises(ValueError):
#         parse_statement(bad_input)
