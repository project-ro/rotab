from typing import List, Optional
from pydantic import BaseModel, Field
from rotab.ast.io import InputNode, OutputNode
from rotab.ast.step import StepNode
from rotab.ast.context.validation_context import ValidationContext
import textwrap
from rotab.ast.util import INDENT


class ProcessNode(BaseModel):
    name: str
    description: Optional[str] = None
    inputs: List[InputNode] = Field(default_factory=list)
    outputs: List[OutputNode] = Field(default_factory=list)
    steps: List[StepNode] = Field(default_factory=list)
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
        step_funcs: List[str] = []
        body: List[str] = []

        # Step functions first
        for step in self.steps:
            func_name = f"step_{step.name}_{self.name}"
            args = ", ".join(step.input_vars)
            func_lines = [f"def {func_name}({args}):"]
            inner = step.generate_script(context)
            inner.append(f"return {step.output_var}")
            func_lines += [textwrap.indent(line, INDENT) for line in inner]
            step_funcs.extend(func_lines)
            step_funcs.append("")

        # Main process function
        func_header = f"def {self.name}():"
        lines = [func_header]
        if self.description:
            lines.append(textwrap.indent(f'"""{self.description.strip()}"""', INDENT))

        # Load inputs
        for inp in self.inputs:
            lines += [textwrap.indent(line, INDENT) for line in inp.generate_script(context)]

        # Call steps
        for step in self.steps:
            args = ", ".join(step.input_vars)
            call_line = f"{step.output_var} = step_{step.name}_{self.name}({args})"
            lines.append(textwrap.indent(call_line, INDENT))

        # Output
        for out in self.outputs:
            lines += [textwrap.indent(line, INDENT) for line in out.generate_script(context)]

        if self.outputs:
            return_vars = ", ".join([out.name for out in self.outputs])
            lines.append(textwrap.indent(f"return {return_vars}", INDENT))

        # Wrap everything in main
        full_script = step_funcs + lines
        main_wrapper = ['if __name__ == "__main__":']
        main_wrapper.append(textwrap.indent(f"{self.name}()", INDENT))

        return full_script + ["", *main_wrapper]

    def to_dict(self) -> dict:
        return {
            "type": "ProcessNode",
            "name": self.name,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "outputs": [o.to_dict() for o in self.outputs],
        }
