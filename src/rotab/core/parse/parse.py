import polars as pl
import ast
from rotab.core.operation.derive_funcs_polars import FUNC_NAMESPACE


def parse_derive_expr(derive_str: str) -> list[pl.Expr]:
    exprs = []
    lines = [line.strip() for line in derive_str.strip().splitlines() if line.strip()]

    for line in lines:
        if "=" not in line:
            raise ValueError(f"Invalid derive expression line: {line}")

        target, expr_str = [part.strip() for part in line.split("=", 1)]
        tree = ast.parse(expr_str, mode="eval")

        def _convert(node):
            if isinstance(node, ast.Name):
                return pl.col(node.id)
            elif isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = _convert(node.left)
                right = _convert(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                elif isinstance(node.op, ast.Sub):
                    return left - right
                elif isinstance(node.op, ast.Mult):
                    return left * right
                elif isinstance(node.op, ast.Div):
                    return left / right
                elif isinstance(node.op, ast.FloorDiv):
                    return left // right
                elif isinstance(node.op, ast.Mod):
                    return left % right
                elif isinstance(node.op, ast.Pow):
                    return left**right
                elif isinstance(node.op, ast.BitAnd):
                    return left & right
                elif isinstance(node.op, ast.BitOr):
                    return left | right
                elif isinstance(node.op, ast.BitXor):
                    return left ^ right
                else:
                    raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
            elif isinstance(node, ast.BoolOp):
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
                    raise ValueError(f"Unsupported boolean operator: {ast.dump(node.op)}")
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
                else:
                    raise ValueError(f"Unsupported comparison operator: {ast.dump(op)}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in FUNC_NAMESPACE:
                        args = []
                        for arg in node.args:
                            if isinstance(arg, ast.Name):
                                args.append(arg.id)
                            elif isinstance(arg, ast.Constant):
                                args.append(arg.value)
                            elif isinstance(arg, ast.Str):  # Python <3.8
                                args.append(arg.s)
                            else:
                                args.append(_convert(arg))
                        return FUNC_NAMESPACE[func_name](*args)
                    else:
                        raise ValueError(f"Unsupported function: {func_name}")
                else:
                    raise ValueError(f"Unsupported function structure: {ast.dump(node.func)}")
            elif isinstance(node, ast.UnaryOp):
                operand = _convert(node.operand)
                if isinstance(node.op, ast.USub):
                    return -operand
                elif isinstance(node.op, ast.UAdd):
                    return +operand
                elif isinstance(node.op, ast.Not):
                    return ~operand
                else:
                    raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
            else:
                raise ValueError(f"Unsupported node: {ast.dump(node)}")

        expr = _convert(tree.body).alias(target)
        exprs.append(expr)

    return exprs


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


def parse(value):
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return value
        else:
            raise ValueError(f"List elements must be strings for select mode: {value}")

    if isinstance(value, str):
        v = value.strip()

        if "=" in v:
            try:
                res = parse_derive_expr(v)
                return res
            except ValueError as e:
                pass

        try:
            tree = ast.parse(v, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid syntax in filter expression: {v}") from e

        if isinstance(tree.body, (ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
            res = parse_filter_expr(v)
            return res
        else:
            raise ValueError(f"Unsupported expression type for filter: {ast.dump(tree.body)}")

    raise ValueError(f"Unsupported expression format for expr(): {value}")
