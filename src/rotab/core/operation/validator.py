from typing import List, Optional, Any, Dict, Union
import ast
import re
from rotab.ast.node import Node


class StepNode(Node):
    def __init__(
        self,
        name: str,
        input_vars: Union[str, List[str]],
        output_var: Optional[str] = None,
        lineno: Optional[int] = None,
    ):
        super().__init__(name, lineno)
        self.input_vars = [input_vars] if isinstance(input_vars, str) else input_vars
        self.output_var = output_var

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(
            {
                "input_vars": self.input_vars,
                "output_var": self.output_var,
            }
        )
        return base


class MutateStep(StepNode):
    def __init__(
        self,
        name: str,
        input_var: str,
        operations: List[Dict[str, Any]],
        output_var: Optional[str] = None,
        when: Optional[Union[str, bool]] = None,
        lineno: Optional[int] = None,
    ):
        super().__init__(name, input_var, output_var, lineno)
        self.operations = operations
        self.when = when

    def validate(self, context: Any) -> None:
        if self.input_vars[0] not in context.get("available_vars", []):
            raise ValueError(f"[{self.name}] `{self.input_vars[0]}` is not defined.")

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
            elif key == "select":
                if not isinstance(value, list) or not all(isinstance(col, str) for col in value):
                    raise ValueError(f"[{self.name}] select must be a list of strings.")
            else:
                raise ValueError(f"[{self.name}] Unknown mutate operation: {key}")

    def generate_script(self) -> List[str]:
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
                        lines.append(f'{var_result}["{lhs}"] = {var_result}.apply(lambda row: {rhs}, axis=1)')
                elif key == "select":
                    cols = ", ".join([f'"{col}"' for col in value])
                    lines.append(f"{var_result} = {var_result}[[{cols}]]")
        return lines


class TransformStep(StepNode):
    def __init__(
        self,
        name: str,
        input_vars: List[str],
        function: str,
        kwargs: Dict[str, Any],
        output_var: str,
        when: Optional[Union[str, bool]] = None,
        lineno: Optional[int] = None,
    ):
        super().__init__(name, input_vars, output_var, lineno)
        self.function = function
        self.kwargs = kwargs
        self.when = when

    def validate(self, context: Any) -> None:
        for var in self.input_vars:
            if var not in context.get("available_vars", []):
                raise ValueError(f"[{self.name}] `{var}` is not defined.")
        if self.function not in context.get("eval_scope", {}):
            raise ValueError(f"[{self.name}] Function `{self.function}` not found in eval_scope.")

    def generate_script(self) -> List[str]:
        args = ", ".join(self.input_vars)
        kwargs_str = ", ".join(f"{k}={repr(v)}" for k, v in self.kwargs.items())
        return [f"{self.output_var} = {self.function}({args}, {kwargs_str})"]


class UseMacro(StepNode):
    def __init__(self, name: str, macro_name: str, args: Dict[str, Any], lineno: Optional[int] = None):
        super().__init__(name, [], None, lineno)
        self.macro_name = macro_name
        self.args = args

    def validate(self, context: Any) -> None:
        if self.macro_name not in context.get("macros", {}):
            raise ValueError(f"[{self.name}] Macro `{self.macro_name}` not defined.")

    def generate_script(self) -> List[str]:
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in self.args.items())
        return [f"{self.macro_name}({args_str})"]
