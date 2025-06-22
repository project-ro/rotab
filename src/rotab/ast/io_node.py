from pydantic import BaseModel
from rotab.ast.node import Node
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from typing import Optional, List, Literal


class IOBaseNode(Node):
    name: str
    io_type: str
    path: str
    schema_name: Optional[str] = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update(
            {
                "io_type": self.io_type,
                "path": self.path,
                "schema_name": self.schema_name,
            }
        )
        return base


class InputNode(IOBaseNode):
    type: Literal["input"] = "input"

    def validate(self, context: ValidationContext) -> None:
        if self.io_type != "csv":
            raise ValueError(f"[{self.name}] Only 'csv' type is supported, got: {self.io_type}")

        context.available_vars.add(self.name)

        if self.schema_name:
            if self.schema_name not in context.schemas:
                raise ValueError(f"[{self.name}] Schema '{self.schema_name}' not found in scope.")
            schema_info = context.schemas[self.schema_name]
            context.schemas[self.name] = VariableInfo(type="dataframe", columns=schema_info.columns.copy())
        else:
            context.schemas[self.name] = VariableInfo(type="dataframe", columns={})

    def generate_script(self, context: ValidationContext) -> List[str]:
        var_info = context.schemas.get(self.name)
        if not isinstance(var_info, VariableInfo):
            raise ValueError(f"[{self.name}] VariableInfo not found for input.")

        if var_info.columns:
            return [f'{self.name} = pd.read_csv("{self.path}", dtype={repr(var_info.columns)})']
        return [f'{self.name} = pd.read_csv("{self.path}")']

    def get_outputs(self) -> List[str]:
        return [self.name]


class OutputNode(IOBaseNode):
    type: Literal["output"] = "output"

    def validate(self, context: ValidationContext) -> None:
        if self.io_type != "csv":
            raise ValueError(f"[{self.name}] Only 'csv' type is supported, got: {self.io_type}")

        if self.name not in context.available_vars:
            raise ValueError(f"[{self.name}] Output variable '{self.name}' is not defined in scope.")

        if self.schema_name and self.schema_name not in context.schemas:
            raise ValueError(f"[{self.name}] Schema '{self.schema_name}' not found in scope.")

    def generate_script(self, context: ValidationContext) -> List[str]:
        scripts = []

        schema_key = self.schema_name or self.name
        var_info = context.schemas.get(schema_key)

        if isinstance(var_info, VariableInfo) and var_info.columns:
            for col, dtype in var_info.columns.items():
                scripts.append(f'{self.name}["{col}"] = {self.name}["{col}"].astype("{dtype}")')
            scripts.append(f'{self.name}.to_csv("{self.path}", index=False, columns={list(var_info.columns.keys())})')
        else:
            scripts.append(f'{self.name}.to_csv("{self.path}", index=False)')
        return scripts

    def get_inputs(self) -> List[str]:
        return [self.name]
