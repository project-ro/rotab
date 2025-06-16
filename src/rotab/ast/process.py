from typing import List, Optional
from pydantic import BaseModel, Field
from rotab.ast.io import InputNode, OutputNode
from rotab.ast.step import StepNode
from rotab.ast.context.validation_context import ValidationContext
import textwrap
from pydantic import TypeAdapter
from rotab.ast.util import INDENT
from rotab.ast.node import Node
from rotab.ast.step import MutateStep, TransformStep
from typing import Union

STEP_TYPE_MAP = {
    "mutate": MutateStep,
    "transform": TransformStep,
}


class ProcessNode(Node):
    name: str
    description: Optional[str] = None
    inputs: List[InputNode] = Field(default_factory=list)
    outputs: List[OutputNode] = Field(default_factory=list)
    steps: List[Union[MutateStep, TransformStep]] = Field(default_factory=list)
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
            if step.output_var in defined_vars:
                raise ValueError(f"[{step.name}] Variable '{step.output_var}' already defined.")
            defined_vars.add(step.output_var)
            context.available_vars.add(step.output_var)  # ← ここが必須

        # 3. ステップバリデーション
        for step in self.steps:
            step.validate(context)

        # 4. 出力変数の整合性確認
        for out in self.outputs:
            out.validate(context)
            if out.name not in defined_vars:
                raise ValueError(f"[{self.name}] Output variable '{out.name}' is not defined in steps or inputs.")

    def generate_script(self, context: ValidationContext) -> List[str]:
        # === Import section ===
        imports = [
            "import os",
            "import pandas as pd",
            "from rotab.core.operation.derive_funcs import *",
            "from rotab.core.operation.transform_funcs import *",
            "from custom_functions import derive_funcs, transform_funcs",
            "",
            "",
        ]

        # === Step functions ===
        step_funcs: List[str] = []
        for step in self.steps:
            func_name = f"step_{step.name}_{self.name}"
            args = ", ".join(step.input_vars)
            func_lines = [f"def {func_name}({args}):"]
            inner = step.generate_script(context)
            inner.append(f"return {step.output_var}")
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

        for inp in self.inputs:
            main_lines += [textwrap.indent(line, INDENT) for line in inp.generate_script(context)]

        for step in self.steps:
            args = ", ".join(step.input_vars)
            call_line = f"{step.output_var} = step_{step.name}_{self.name}({args})"
            main_lines.append(textwrap.indent(call_line, INDENT))

        for out in self.outputs:
            main_lines += [textwrap.indent(line, INDENT) for line in out.generate_script(context)]

        if self.outputs:
            return_vars = ", ".join([out.name for out in self.outputs])
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

    @classmethod
    def from_dict(cls, data: dict, schema_manager=None):
        if "steps" in data:
            steps = []
            for step_dict in data["steps"]:
                step_type = step_dict.get("type")
                concrete_cls = STEP_TYPE_MAP.get(step_type)
                if not concrete_cls:
                    raise ValueError(f"Unknown or missing step type: {step_type}")
                steps.append(concrete_cls.from_dict(step_dict, schema_manager))
            data = dict(data)
            data["steps"] = steps
        return TypeAdapter(cls).validate_python(data, by_alias=True)
