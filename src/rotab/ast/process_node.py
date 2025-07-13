from typing import List, Optional
from pydantic import BaseModel, Field
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.step_node import StepNode
from rotab.ast.context.validation_context import ValidationContext
import textwrap
from pydantic import TypeAdapter
from rotab.ast.util import INDENT
from rotab.ast.node import Node
from rotab.ast.step_node import MutateStep, TransformStep
from typing import Union

try:
    from typing import Annotated  # Python 3.9+
except ImportError:
    from typing_extensions import Annotated  # Python 3.8

StepUnion = Annotated[Union[MutateStep, TransformStep], Field(discriminator="type")]


class ProcessNode(Node):
    name: str
    description: Optional[str] = None
    inputs: List[InputNode] = Field(default_factory=list)
    outputs: List[OutputNode] = Field(default_factory=list)
    steps: List[StepUnion]
    lineno: Optional[int] = None

    def validate(self, context: ValidationContext) -> None:
        defined_vars = set()

        # 1. 入力変数を available_vars と defined_vars に登録
        for inp in self.inputs:
            inp.validate(context)
            defined_vars.add(inp.name)
            context.available_vars.add(inp.name)

        # 2. ステップ出力名の重複チェックと available_vars への事前登録
        for step in self.steps:
            for var in step.output_vars:
                if var in defined_vars:
                    raise ValueError(f"[{step.name}] Variable '{var}' already defined.")
            defined_vars.update(step.output_vars)
            context.available_vars.update(step.output_vars)

        # 3. ステップバリデーション（input_vars が available_vars に含まれるかなどをチェック）
        for step in self.steps:
            step.validate(context)

        # 4. 出力変数の整合性確認
        for out in self.outputs:
            out.validate(context)
            if out.name not in defined_vars:
                raise ValueError(f"[{self.name}] Output variable '{out.name}' is not defined in steps or inputs.")

    def generate_script(self, backend: str, context: ValidationContext) -> List[str]:
        if backend not in {"pandas", "polars"}:
            raise ValueError(f"Unsupported backend: {backend}")

        # === Import section ===
        imports = ["import os"]

        if backend == "pandas":
            imports.append("import pandas as pd")
        elif backend == "polars":
            imports.append("import polars as pl")
            imports.append("import fsspec")
            imports.append("from rotab.core.parse.parse import parse")

        if backend == "pandas":
            imports.extend(
                [
                    "from rotab.core.operation.derive_funcs_pandas import *",
                    "from rotab.core.operation.transform_funcs_pandas import *",
                ]
            )
        elif backend == "polars":
            imports.extend(
                [
                    "from rotab.core.operation.derive_funcs_polars import *",
                    "from rotab.core.operation.transform_funcs_polars import *",
                ]
            )

        if context.derive_func_path is not None:
            imports.append("from custom_functions.derive_funcs import *")
        if context.transform_func_path is not None:
            imports.append("from custom_functions.transform_funcs import *")

        imports.extend(["", ""])

        # === Step functions ===
        step_funcs: List[str] = []
        for step in self.steps:
            func_name = f"step_{step.name}_{self.name}"
            args = ", ".join(step.input_vars)
            func_lines = [f"def {func_name}({args}):"]
            inner = step.generate_script(backend, context)

            return_line = (
                f"return {step.output_vars[0]}" if len(step.output_vars) == 1 else f"return {tuple(step.output_vars)}"
            )
            inner.append(return_line)

            func_lines += [textwrap.indent(line, INDENT) for line in inner]
            step_funcs.extend(func_lines)
            step_funcs.extend(["", ""])

        if step_funcs and step_funcs[-1] != "":
            step_funcs.extend(["", ""])

        # === Main function ===
        main_lines = []
        func_header = f"def {self.name}():"
        main_lines.append(func_header)
        if self.description:
            main_lines.append(textwrap.indent(f'"""{self.description.strip()}"""', INDENT))

        # Input nodes
        for inp in self.inputs:
            main_lines += [textwrap.indent(line, INDENT) for line in inp.generate_script(backend, context)]

        # Steps
        for step in self.steps:
            args = ", ".join(step.input_vars)
            assign_lhs = step.output_vars[0] if len(step.output_vars) == 1 else f"{tuple(step.output_vars)}"
            call_line = f"{assign_lhs} = step_{step.name}_{self.name}({args})"
            main_lines.append(textwrap.indent(call_line, INDENT))

        # Output nodes
        for out in self.outputs:
            main_lines += [textwrap.indent(line, INDENT) for line in out.generate_script(backend, context)]

        if self.outputs:
            return_vars = (
                self.outputs[0].name if len(self.outputs) == 1 else ", ".join(out.name for out in self.outputs)
            )
            main_lines.append(textwrap.indent(f"return {return_vars}", INDENT))

        main_lines.extend(["", ""])

        # === Main entry point ===
        main_wrapper = ['if __name__ == "__main__":', textwrap.indent(f"{self.name}()", INDENT), ""]

        # === Final script ===
        return imports + step_funcs + main_lines + main_wrapper

    def get_children(self) -> List[Node]:
        return self.inputs + self.steps + self.outputs

    def get_inputs(self) -> List[str]:
        return [inp.path for inp in self.inputs]

    def get_outputs(self) -> List[str]:
        return [out.path for out in self.outputs]

    def to_dict(self) -> dict:
        return {
            "type": "ProcessNode",
            "name": self.name,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "outputs": [o.to_dict() for o in self.outputs],
        }
