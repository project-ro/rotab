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

        # validate inputs
        for inp in self.inputs:
            inp.validate(context)
            defined_vars.add(inp.name)

        # Check for output_var duplication before actual validation
        for step in self.steps:
            if step.output_var in defined_vars:
                raise ValueError(f"[{step.name}] Variable '{step.output_var}' already defined.")
            defined_vars.add(step.output_var)

        # Now validate steps
        for step in self.steps:
            step.validate(context)

        # validate outputs
        for out in self.outputs:
            out.validate(context)
            if out.name not in defined_vars:
                raise ValueError(f"[{self.name}] Output variable '{out.name}' is not defined in steps or inputs.")

    def generate_script(self, context: ValidationContext) -> List[str]:
        lines: List[str] = []
        body: List[str] = []

        for inp in self.inputs:
            body += inp.generate_script(context)

        for step in self.steps:
            body += step.generate_script(context)

        for out in self.outputs:
            body += out.generate_script(context)

        if self.outputs:
            return_vars = ", ".join([out.name for out in self.outputs])
            body.append(f"return {return_vars}")

        lines.append(f"def {self.name}():")
        if self.description:
            docstring = textwrap.indent(f'"""{self.description.strip()}"""', INDENT)
            lines.append(docstring)
        lines.extend(textwrap.indent(line, INDENT) for line in body)

        return lines

    def to_dict(self) -> dict:
        return {
            "type": "ProcessNode",
            "name": self.name,
            "description": self.description,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "outputs": [o.to_dict() for o in self.outputs],
        }
