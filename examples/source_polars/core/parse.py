import polars as pl
import ast
from rotab.core.operation.derive_funcs_polars import FUNC_NAMESPACE  # このパスはあなたの環境に合わせてください
import inspect
from typing import List, Dict, Any, Union, Tuple, Callable  # 型ヒントを追加


def parse_derive_expr(derive_str: str) -> List[pl.Expr]:
    derive_str = inspect.cleandoc(derive_str)
    print(f"\n--- Entering parse_derive_expr ---")
    print(f"DEBUG: Input derive_str: {derive_str!r}")

    try:
        tree = ast.parse(derive_str, mode="exec")
        print(f"DEBUG: Parsed AST for derive_str (mode='exec'): {ast.dump(tree, indent=2)}")
    except SyntaxError as e:
        print(f"ERROR: SyntaxError during AST parsing in parse_derive_expr: {e}")
        raise ValueError(f"Invalid syntax in derive expression: {derive_str}") from e
    except Exception as e:
        print(f"ERROR: Unexpected error during AST parsing in parse_derive_expr: {e}")
        raise  # その他の予期せぬエラー

    exprs = []

    def _convert(node):
        print(f"DEBUG: _convert processing node type: {type(node).__name__}")
        if isinstance(node, ast.Name):
            print(f"DEBUG: _convert -> ast.Name: id={node.id!r}")
            return pl.col(node.id)
        elif isinstance(node, ast.Constant):
            print(f"DEBUG: _convert -> ast.Constant: value={node.value!r}")
            return pl.lit(node.value)
        elif isinstance(node, ast.BinOp):
            print(f"DEBUG: _convert -> ast.BinOp: op={type(node.op).__name__}")
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
            print(f"DEBUG: _convert -> ast.BoolOp: op={type(node.op).__name__}")
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
            print(f"DEBUG: _convert -> ast.Compare: op={type(node.ops[0]).__name__}")
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
            print(f"DEBUG: _convert -> ast.Call (Function call detected)")
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                print(f"DEBUG: Function name identified: {func_name!r}")
                if func_name in FUNC_NAMESPACE:
                    print(f"DEBUG: Function '{func_name}' found in FUNC_NAMESPACE.")
                    func = FUNC_NAMESPACE[func_name]
                elif func_name in globals():
                    print(f"DEBUG: Function '{func_name}' found in globals().")
                    func = globals()[func_name]
                else:
                    print(f"ERROR: Function '{func_name}' not found in FUNC_NAMESPACE or globals().")
                    raise ValueError(f"Unsupported function: {func_name}")

                args = []
                for arg_node in node.args:  # 変数名を arg から arg_node に変更して競合を避ける
                    if isinstance(arg_node, ast.Name):
                        print(f"DEBUG: Arg (ast.Name): {arg_node.id!r}")
                        args.append(pl.col(arg_node.id))
                    elif isinstance(arg_node, ast.Constant):
                        print(f"DEBUG: Arg (ast.Constant): {arg_node.value!r}")
                        args.append(arg_node.value)
                    else:
                        print(f"DEBUG: Arg (other type): {type(arg_node).__name__}")
                        args.append(_convert(arg_node))
                return func(*args)
            else:
                print(f"ERROR: Unsupported function structure (not ast.Name): {ast.dump(node.func)}")
                raise ValueError(f"Unsupported function structure: {ast.dump(node.func)}")
        elif isinstance(node, ast.UnaryOp):
            print(f"DEBUG: _convert -> ast.UnaryOp: op={type(node.op).__name__}")
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
            print(f"ERROR: Unsupported node type in _convert: {ast.dump(node)}")
            raise ValueError(f"Unsupported node: {ast.dump(node)}")

    for stmt in tree.body:
        print(f"DEBUG: Processing statement type: {type(stmt).__name__}")
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            print(f"DEBUG: Assignment found. Target: {target!r}")
            expr = _convert(stmt.value).alias(target)
            exprs.append(expr)
        else:
            print(f"ERROR: Only simple assignments are allowed (unsupported statement): {ast.dump(stmt)}")
            raise ValueError(f"Only simple assignments are allowed: {ast.dump(stmt)}")

    print(f"--- Exiting parse_derive_expr ---")
    return exprs


def parse_filter_expr(expr_str: str) -> pl.Expr:
    print(f"\n--- Entering parse_filter_expr ---")
    print(f"DEBUG: Input expr_str: {expr_str!r}")
    try:
        tree = ast.parse(expr_str, mode="eval")
        print(f"DEBUG: Parsed AST for filter_expr (mode='eval'): {ast.dump(tree, indent=2)}")
    except SyntaxError as e:
        print(f"ERROR: SyntaxError during AST parsing in parse_filter_expr: {e}")
        raise ValueError(f"Invalid syntax in filter expression: {expr_str}") from e
    except Exception as e:
        print(f"ERROR: Unexpected error during AST parsing in parse_filter_expr: {e}")
        raise

    def _convert(node):
        print(f"DEBUG: _convert processing node type for filter: {type(node).__name__}")
        if isinstance(node, ast.BoolOp):
            print(f"DEBUG: _convert (filter) -> ast.BoolOp: op={type(node.op).__name__}")
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
            print(f"DEBUG: _convert (filter) -> ast.Compare: op={type(node.ops[0]).__name__}")
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
            print(f"DEBUG: _convert (filter) -> ast.Name: id={node.id!r}")
            return pl.col(node.id)

        elif isinstance(node, ast.Constant):
            print(f"DEBUG: _convert (filter) -> ast.Constant: value={node.value!r}")
            return node.value

        elif isinstance(node, ast.List):
            print(f"DEBUG: _convert (filter) -> ast.List")
            return [_convert(elt) for elt in node.elts]

        elif isinstance(node, ast.Tuple):
            print(f"DEBUG: _convert (filter) -> ast.Tuple")
            return tuple(_convert(elt) for elt in node.elts)

        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            print(f"DEBUG: _convert (filter) -> ast.UnaryOp (Not)")
            return ~_convert(node.operand)

        else:
            print(f"ERROR: Unsupported node type in _convert (filter): {ast.dump(node)}")
            raise ValueError(f"Unsupported node: {ast.dump(node)}")

    result = _convert(tree.body)
    print(f"--- Exiting parse_filter_expr ---")
    return result


def parse(value: Union[str, List[str]]) -> Union[List[str], List[pl.Expr], pl.Expr]:
    print(f"\n--- Entering parse function ---")
    print(f"DEBUG: Input value type: {type(value).__name__}, value: {value!r}")

    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            print(f"DEBUG: parse -> List of strings detected (select mode).")
            return value
        else:
            print(f"ERROR: List elements must be strings for select mode: {value}")
            raise ValueError(f"List elements must be strings for select mode: {value}")

    if isinstance(value, str):
        v = value.strip()
        print(f"DEBUG: parse -> Stripped string value: {v!r}")

        if "\n" in v or "\r" in v:
            print(f"DEBUG: parse -> Newline detected, calling parse_derive_expr.")
            return parse_derive_expr(v)

        if "=" in v:
            print(f"DEBUG: parse -> '=' detected, attempting to parse as derive expression.")
            try:
                return parse_derive_expr(v)
            except Exception as e:
                print(f"ERROR: Failed to parse as derive expression: {v}, Error: {e}")
                raise ValueError(f"Invalid syntax in derive expression: {v}") from e
        else:
            print(f"DEBUG: parse -> No '=' and no newline, attempting to parse as filter expression.")
            try:
                tree = ast.parse(v, mode="eval")
                print(f"DEBUG: AST parsed successfully for filter_expr (mode='eval'): {ast.dump(tree.body, indent=2)}")
            except SyntaxError as e:
                print(f"ERROR: SyntaxError in filter expression: {v}, Error: {e}")
                raise ValueError(f"Invalid syntax in filter expression: {v}") from e

            if isinstance(tree.body, (ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name)):
                print(f"DEBUG: parse -> Recognized as valid filter expression type, calling parse_filter_expr.")
                return parse_filter_expr(v)
            else:
                print(f"ERROR: Unsupported expression type for filter: {ast.dump(tree.body)}")
                raise ValueError(f"Unsupported expression type for filter: {ast.dump(tree.body)}")

    print(f"ERROR: Unsupported expression format: type={type(value).__name__}, value={value!r}")
    raise ValueError(f"Unsupported expression format: {type(value)}")
