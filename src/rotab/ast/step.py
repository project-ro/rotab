from typing import List, Optional, Any, Dict, Union, Callable
from pydantic import BaseModel, Field, root_validator
import ast
import re
import textwrap
from rotab.ast.node import Node
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


class StepNode(Node, BaseModel):
    name: str
    input_vars: List[str] = Field(..., alias="input_vars")
    output_var: Optional[str] = None
    lineno: Optional[int] = None

    def to_dict(self) -> dict:
        return self.dict(by_alias=True, exclude_none=True)


class MutateStep(StepNode):
    operations: List[Dict[str, Any]]
    when: Optional[str] = None

    def validate(self, context: ValidationContext) -> None:
        available_vars = context.available_vars
        schemas = context.schemas

        input_var = self.input_vars[0]
        if input_var not in schemas:
            raise ValueError(f"[{self.name}] `{input_var}` is not defined in schemas.")

        var_info = schemas[input_var]
        if var_info.type != "dataframe":
            raise ValueError(f"[{self.name}] `{input_var}` must be a dataframe.")

        df_columns = var_info.columns

        for i, op in enumerate(self.operations):
            if not isinstance(op, dict) or len(op) != 1:
                raise ValueError(f"[{self.name}] Operation #{i} must be a single-key dict.")
            key, value = next(iter(op.items()))
            if key == "filter":
                try:
                    tree = ast.parse(value, mode="eval")
                    if not isinstance(tree.body, (ast.Compare, ast.BoolOp, ast.Call, ast.Name, ast.UnaryOp, ast.BinOp)):
                        raise ValueError
                except Exception:
                    raise ValueError(f"[{self.name}] Invalid filter expression: {value!r}")
            elif key == "derive":
                for lineno, line in enumerate(value.splitlines(), 1):
                    if not line.strip():
                        continue
                    if "=" not in line or re.match(r"^[^=]+==[^=]+$", line):
                        raise ValueError(f"[{self.name}] derive line {lineno}: malformed '=' in {line!r}")
                    lhs, rhs = map(str.strip, line.split("=", 1))
                    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", lhs):
                        raise ValueError(f"[{self.name}] Invalid LHS in derive: {lhs!r}")
                    try:
                        ast.parse(rhs, mode="eval")
                    except Exception:
                        raise ValueError(f"[{self.name}] Syntax error in RHS: {rhs!r}")
                    available_vars.add(lhs)

            elif key == "select":
                if not isinstance(value, list) or not all(isinstance(col, str) for col in value):
                    raise ValueError(f"[{self.name}] select must be a list of strings.")
                if not df_columns:  # スキーマ未定義なら検証スキップ
                    continue
                for col in value:
                    if col not in df_columns:
                        raise ValueError(f"[{self.name}] select references undefined column: {col}")

            else:
                raise ValueError(f"[{self.name}] Unknown mutate operation: {key}")

    def _rewrite_rhs_with_row(self, rhs: str) -> str:
        try:
            tree = ast.parse(rhs, mode="eval")

            class InjectRowTransformer(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.Subscript(
                        value=ast.Name(id="row", ctx=ast.Load()), slice=ast.Constant(value=node.id), ctx=ast.Load()
                    )

            class ScopeLimiter(ast.NodeTransformer):
                def visit_Call(self, node):
                    node.args = [InjectRowTransformer().visit(arg) for arg in node.args]
                    return node

                def visit_BinOp(self, node):
                    node.left = InjectRowTransformer().visit(node.left)
                    node.right = InjectRowTransformer().visit(node.right)
                    return node

                def visit_Compare(self, node):
                    node.left = InjectRowTransformer().visit(node.left)
                    node.comparators = [InjectRowTransformer().visit(c) for c in node.comparators]
                    return node

            transformed = ScopeLimiter().visit(tree)
            ast.fix_missing_locations(transformed)
            unparsed_code = ast.unparse(transformed)

            # ここで正規表現によってシングルクオートをダブルクオートに
            double_quoted = re.sub(r"'([a-zA-Z0-9_]+)'", r'"\1"', unparsed_code)

            return double_quoted

        except Exception as e:
            raise ValueError(f"[{self.name}] Failed to transform RHS '{rhs}': {e}")

    def generate_script(self, context: ValidationContext = None) -> List[str]:
        var = self.input_vars[0]
        var_result = self.output_var or var
        lines = [f"{var_result} = {var}.copy()"]
        for op in self.operations:
            for key, value in op.items():
                if key == "filter":
                    lines.append(f"{var_result} = {var_result}.query('{value}')")
                elif key == "derive":
                    for line in value.split("\n"):
                        lhs, rhs = map(str.strip, line.split("=", 1))
                        transformed_rhs = self._rewrite_rhs_with_row(rhs)
                        lines.append(
                            f'{var_result}["{lhs}"] = {var_result}.apply(lambda row: {transformed_rhs}, axis=1)'
                        )
                elif key == "select":
                    cols = ", ".join([f'"{col}"' for col in value])
                    lines.append(f"{var_result} = {var_result}[[{cols}]]")

        if self.when:
            return [f"if {self.when}:"] + [textwrap.indent(line, INDENT) for line in lines]
        return lines


class TransformStep(StepNode):
    expr: str
    when: Optional[str] = None

    def validate(self, context: ValidationContext) -> None:
        def is_unsupported_syntax(expr: str) -> bool:
            return re.search(r"\)\s*\(", expr) is not None or re.match(r"\(\s*\w+\s*\)\s*\(", expr) is not None

        if is_unsupported_syntax(self.expr):
            raise ValueError(f"[{self.name}] Unsupported function syntax in expression.")

        available_vars = context.available_vars
        eval_scope = context.eval_scope
        schemas = context.schemas

        for var in self.input_vars:
            if var not in available_vars:
                raise ValueError(f"[{self.name}] `{var}` is not defined.")

        try:
            parsed = ast.parse(self.expr, mode="eval")
            call_node = parsed.body
            if not isinstance(call_node, ast.Call):
                raise ValueError(f"[{self.name}] Expression must be a function call.")
            if not isinstance(call_node.func, ast.Name):
                raise ValueError(f"[{self.name}] Unsupported function syntax in expression.")
            func_name = call_node.func.id
            if func_name not in eval_scope:
                raise ValueError(f"[{self.name}] Function `{func_name}` not found in eval_scope.")
        except SyntaxError as e:
            raise ValueError(f"[{self.name}] Invalid Python expression in `transform`: {e}")

        available_vars.add(self.output_var)
        schemas[self.output_var] = VariableInfo(type="dataframe", columns={})

    def generate_script(self, context: ValidationContext = None) -> List[str]:
        line = f"{self.output_var} = {self.expr}"
        if self.when:
            return [f"if {self.when}:", textwrap.indent(line, INDENT)]
        return [line]
