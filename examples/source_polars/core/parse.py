import polars as pl
import ast
from rotab.core.operation.derive_funcs_polars import FUNC_NAMESPACE
import inspect
from typing import List, Dict, Any, Union, Tuple, Callable


def parse_derive_expr(derive_str: str) -> List[pl.Expr]:
    derive_str = inspect.cleandoc(derive_str)

    try:
        tree = ast.parse(derive_str, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in derive expression: {derive_str}") from e
    except Exception as e:
        raise

    exprs = []

    def _convert(node):
        if isinstance(node, ast.Name):
            return pl.col(node.id)
        elif isinstance(node, ast.Constant):
            return pl.lit(node.value)
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
                    func = FUNC_NAMESPACE[func_name]
                elif func_name in globals():
                    func = globals()[func_name]
                else:
                    raise ValueError(f"Unsupported function: {func_name}")

                args = []
                for arg_node in node.args:
                    if isinstance(arg_node, ast.Name):
                        args.append(pl.col(arg_node.id))
                    elif isinstance(arg_node, ast.Constant):
                        args.append(arg_node.value)
                    else:
                        args.append(_convert(arg_node))
                return func(*args)
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

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            expr = _convert(stmt.value).alias(target)
            exprs.append(expr)
        else:
            raise ValueError(f"Only simple assignments are allowed: {ast.dump(stmt)}")

    return exprs


def parse_filter_expr(expr_str: str) -> pl.Expr:
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid syntax in filter expression: {expr_str}") from e
    except Exception as e:
        raise

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

    result = _convert(tree.body)
    return result


def parse(value: Union[str, List[str]]) -> Union[List[str], List[pl.Expr], pl.Expr]:

    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return value
        else:
            raise ValueError(f"List elements must be strings for select mode: {value}")

    if isinstance(value, str):
        v = value.strip()

        if "\n" in v or "\r" in v:
            return parse_derive_expr(v)

        if "=" in v:
            try:
                return parse_derive_expr(v)
            except Exception as e:
                raise ValueError(f"Invalid syntax in derive expression: {v}") from e
        else:
            try:
                tree = ast.parse(v, mode="eval")
            except SyntaxError as e:
                raise ValueError(f"Invalid syntax in filter expression: {v}") from e

            if isinstance(tree.body, (ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
                return parse_filter_expr(v)
            else:
                raise ValueError(f"Unsupported expression type for filter: {ast.dump(tree.body)}")

    raise ValueError(f"Unsupported expression format: {type(value)}")
