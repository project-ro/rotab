import polars as pl
import ast


def parse_filter_expr(expr_str: str) -> pl.Expr:
    """
    ユーザーからの文字列条件式を pl.Expr に変換する関数
    例: "age > 18 and income < 5000" → pl.col("age") > 18 & pl.col("income") < 5000
    """
    tree = ast.parse(expr_str, mode="eval")

    def _convert(node):
        if isinstance(node, ast.BoolOp):
            ops = [_convert(v) for v in node.values]
            if isinstance(node.op, ast.And):
                expr = ops[0]
                for op in ops[1:]:
                    expr = expr & op
                return expr
            elif isinstance(node.op, ast.Or):
                expr = ops[0]
                for op in ops[1:]:
                    expr = expr | op
                return expr
            else:
                raise ValueError("Unsupported boolean operator")

        elif isinstance(node, ast.Compare):
            left = _convert(node.left)
            right = _convert(node.comparators[0])
            op = node.ops[0]

            if isinstance(op, ast.Eq):
                return left == right
            elif isinstance(op, ast.NotEq):
                return left != right
            elif isinstance(op, ast.Gt):
                return left > right
            elif isinstance(op, ast.GtE):
                return left >= right
            elif isinstance(op, ast.Lt):
                return left < right
            elif isinstance(op, ast.LtE):
                return left <= right
            elif isinstance(op, ast.In):
                return left.is_in(right)
            elif isinstance(op, ast.NotIn):
                return ~left.is_in(right)
            elif isinstance(op, ast.Is):
                if right is None:
                    return left.is_null()
                else:
                    raise ValueError("Unsupported 'is' comparison with non-None")
            elif isinstance(op, ast.IsNot):
                if right is None:
                    return left.is_not_null()
                else:
                    raise ValueError("Unsupported 'is not' comparison with non-None")
            else:
                raise ValueError("Unsupported comparison operator")

        elif isinstance(node, ast.Name):
            return pl.col(node.id)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.List):
            return [_convert(elt) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            return tuple(_convert(elt) for elt in node.elts)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return ~_convert(node.operand)

        else:
            raise ValueError(f"Unsupported node: {ast.dump(node)}")

    return _convert(tree.body)


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
