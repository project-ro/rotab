import sys
from typing import List, Optional, Any, Dict, Union, Callable
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import ast
import re
import unicodedata
import textwrap
from rotab.ast.node import Node
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


class StepNode(Node):
    name: str
    type: str
    input_vars: List[str] = Field(..., alias="input_vars")
    output_vars: List[str] = Field(..., alias="output_vars")
    lineno: Optional[int] = None

    def to_dict(self) -> dict:
        return self.dict(by_alias=True, exclude_none=True)

    def get_inputs(self) -> List[str]:
        return self.input_vars

    def get_outputs(self) -> List[str]:
        return self.output_vars


class MutateStep(StepNode):
    type: Literal["mutate"] = "mutate"
    operations: List[Dict[str, Any]]
    when: Optional[Union[str, bool]] = None

    @staticmethod
    def normalize_expr(expr: str) -> str:
        return unicodedata.normalize("NFKC", expr)

    @model_validator(mode="before")
    @classmethod
    def normalize_operations(cls, values):
        operations = values.get("operations", [])
        normalized = []
        for op in operations:
            if not isinstance(op, dict) or len(op) != 1:
                normalized.append(op)
                continue

            key, value = next(iter(op.items()))
            if isinstance(value, str) and key in {"filter", "derive"}:
                value = cls.normalize_expr(value)
            normalized.append({key: value})

        values["operations"] = normalized
        return values

    def validate(self, context: ValidationContext) -> None:

        available_vars = context.available_vars
        schemas = context.schemas

        input_var = self.input_vars[0]

        if input_var in schemas:
            var_info = schemas[input_var]
            if var_info.type != "dataframe":
                raise ValueError(f"[{self.name}] `{input_var}` must be a dataframe.")
            df_columns = var_info.columns.copy()
        else:
            df_columns = {}

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
                    parts = line.split("=", 1)
                    if len(parts) != 2:
                        raise ValueError(f"[{self.name}] Invalid assignment expression: `{line}`")
                    lhs, rhs = map(str.strip, parts)

                    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", lhs):
                        raise ValueError(f"[{self.name}] Invalid LHS in derive: {lhs!r}")
                    try:
                        ast.parse(rhs, mode="eval")
                    except Exception:
                        raise ValueError(f"[{self.name}] Syntax error in RHS: {rhs!r}")
                    available_vars.add(lhs)

                    if not df_columns:
                        continue

                    df_columns[lhs] = "str"  # 暫定スキーマ

            elif key == "select":
                if not isinstance(value, list) or not all(isinstance(col, str) for col in value):
                    raise ValueError(f"[{self.name}] select must be a list of strings.")

                if not df_columns or df_columns == {}:
                    # 推論不能なスキーマ（transform後など）であれば select チェックはスキップ
                    continue

                for col in value:
                    if col not in df_columns:
                        raise ValueError(f"[{self.name}] select references undefined column: {col}")

            else:
                raise ValueError(f"[{self.name}] Unknown mutate operation: {key}")

        for out in self.output_vars:
            available_vars.add(out)
            if out not in schemas:
                if df_columns:
                    schemas[out] = VariableInfo(type="dataframe", columns=df_columns.copy())
                else:
                    schemas[out] = VariableInfo(type="dataframe", columns={})

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
            return re.sub(r"'([a-zA-Z0-9_]+)'", r'"\1"', unparsed_code)
        except Exception as e:
            raise ValueError(f"[{self.name}] Failed to transform RHS '{rhs}': {e}")

    def generate_script(self, backend: str = "pandas", context: ValidationContext = None) -> List[str]:
        if backend == "pandas":
            return self.generate_script_pandas(context)
        elif backend == "polars":
            return self.generate_script_polars(context)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate_script_pandas(self, context: ValidationContext = None) -> List[str]:
        var = self.input_vars[0]
        var_result = self.output_vars[0] if self.output_vars else var
        lines = [f"{var_result} = {var}.copy()"]

        for op in self.operations:
            for key, value in op.items():
                if key == "filter":
                    lines.append(f"{var_result} = {var_result}.query('{value}').copy()")

                elif key == "derive":
                    for line in value.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split("=", 1)
                        if len(parts) != 2:
                            raise ValueError(f"[{self.name}] Invalid assignment expression: `{line}`")
                        lhs, rhs = map(str.strip, parts)
                        transformed_rhs = self._rewrite_rhs_with_row(rhs)
                        lines.append(
                            f'{var_result}["{lhs}"] = {var_result}.apply(lambda row: {transformed_rhs}, axis=1)'
                        )

                elif key == "select":
                    cols = ", ".join([f'"{col}"' for col in value])
                    lines.append(f"{var_result} = {var_result}[[{cols}]]")

        return [f"if {self.when}:", *[textwrap.indent(line, INDENT) for line in lines]] if self.when else lines

    def generate_script_polars(self, context: ValidationContext = None) -> List[str]:
        var = self.input_vars[0]
        var_result = self.output_vars[0] if self.output_vars else var
        lines = [f"{var_result} = {var}"]

        for op in self.operations:
            for key, value in op.items():
                # 改行含む場合
                if "\n" in value:
                    # 過剰インデントを削除し、単一の INDENT で統一
                    indented_body = textwrap.indent(value.strip(), INDENT)
                    formatted_value = f'"""\n{indented_body}\n{INDENT}"""'
                else:
                    formatted_value = repr(value)

                if key == "filter":
                    lines.append(f"{var_result} = {var_result}.filter(parse({formatted_value}))")

                elif key == "derive":
                    lines.append(f"{var_result} = {var_result}.with_columns(parse({formatted_value}))")

                elif key == "select":
                    cols_str = ", ".join([f"'{col}'" for col in value])
                    lines.append(f"{var_result} = {var_result}.select([{cols_str}])")

        final_lines = [f"if {self.when}:", *[textwrap.indent(line, INDENT) for line in lines]] if self.when else lines
        return final_lines


class TransformStep(StepNode):
    type: Literal["transform"] = "transform"
    expr: str
    when: Optional[Union[str, bool]] = None

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

        for out in self.output_vars:  # 修正③: 全出力を反映
            available_vars.add(out)
            if out not in schemas:
                schemas[out] = VariableInfo(type="dataframe", columns={})

    def generate_script(self, backend, context: ValidationContext = None) -> List[str]:
        line = f"{self.output_vars[0]} = {self.expr}"  # 修正④: 最初の出力に代入
        return [f"if {self.when}:", textwrap.indent(line, INDENT)] if self.when else [line]
