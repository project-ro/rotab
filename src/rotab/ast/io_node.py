import os
import re
import polars as pl
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
    wildcard_column: Optional[str] = None

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

    def generate_script(self, backend: str = "pandas", context: ValidationContext = None) -> List[str]:
        if context is None:
            raise ValueError("context must be provided.")

        var_info = context.schemas.get(self.name)
        if not isinstance(var_info, VariableInfo):
            raise ValueError(f"[{self.name}] VariableInfo not found for input.")

        # 型変換マップ（Polars用）
        polars_type_map = {"int": "Int64", "float": "Float64", "str": "Utf8", "bool": "Boolean"}

        if backend == "pandas":
            dtype_arg = f", dtype={repr(var_info.columns)}" if var_info.columns else ""
        elif backend == "polars":
            if var_info.columns:
                dtype_dict = {
                    col: f"pl.{polars_type_map.get(dtype, 'Utf8')}" for col, dtype in var_info.columns.items()
                }
                dtype_items = ", ".join([f'"{k}": {v}' for k, v in dtype_dict.items()])
                dtype_arg = f", dtypes={{{dtype_items}}}"
            else:
                dtype_arg = ""
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        if "*" in self.path:
            if not self.wildcard_column:
                raise ValueError(f"[{self.name}] 'wildcard_column' must be specified for wildcard path.")

            basename_pattern = os.path.basename(self.path)
            regex_pattern = re.escape(basename_pattern).replace("\\*", "(.+)")

            if backend == "pandas":
                lines = [
                    "import glob, os, re",
                    f"{self.name}_files = glob.glob('{self.path}')",
                    f"{self.name}_df_list = []",
                    f"_regex = re.compile(r'{regex_pattern}')",
                    f"for _file in {self.name}_files:",
                    f"    _basename = os.path.basename(_file)",
                    f"    _match = _regex.match(_basename)",
                    f"    if not _match: raise ValueError(f'Unexpected filename: {{_basename}}')",
                    f"    _val = _match.group(1)",
                    f"    _df = pd.read_csv(_file{dtype_arg})",
                    f"    _df['{self.wildcard_column}'] = _val",
                    f"    _df['{self.wildcard_column}'] = _df['{self.wildcard_column}'].astype(str)",
                    f"    {self.name}_df_list.append(_df)",
                    f"{self.name} = pd.concat({self.name}_df_list, ignore_index=True)",
                ]
            elif backend == "polars":
                lines = [
                    "import glob, os, re, polars as pl",
                    f"{self.name}_files = glob.glob('{self.path}')",
                    f"{self.name}_df_list = []",
                    f"_regex = re.compile(r'{regex_pattern}')",
                    f"for _file in {self.name}_files:",
                    f"    _basename = os.path.basename(_file)",
                    f"    _match = _regex.match(_basename)",
                    f"    if not _match: raise ValueError(f'Unexpected filename: {{_basename}}')",
                    f"    _val = _match.group(1)",
                    f"    _df = pl.scan_csv(_file{dtype_arg})",
                    f"    _df = _df.with_columns(pl.lit(_val).cast(pl.Utf8).alias('{self.wildcard_column}'))",
                    f"    {self.name}_df_list.append(_df)",
                    f"{self.name} = pl.concat({self.name}_df_list, how='vertical')",
                ]
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        else:
            if backend == "pandas":
                lines = [f'{self.name} = pd.read_csv("{self.path}"{dtype_arg})']
            elif backend == "polars":
                lines = [f'{self.name} = pl.scan_csv("{self.path}"{dtype_arg})']
            else:
                raise ValueError(f"Unsupported backend: {backend}")

        return lines

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

    def generate_script(self, backend: str = "pandas", context: ValidationContext = None) -> List[str]:
        if context is None:
            raise ValueError("context must be provided.")

        scripts = []

        schema_key = self.schema_name or self.name
        var_info = context.schemas.get(schema_key)

        # Polars用型マップ
        polars_type_map = {"int": "Int64", "float": "Float64", "str": "Utf8", "bool": "Boolean"}

        if isinstance(var_info, VariableInfo) and var_info.columns:
            for col, dtype in var_info.columns.items():
                if backend == "pandas":
                    scripts.append(f'{self.name}["{col}"] = {self.name}["{col}"].astype("{dtype}")')
                elif backend == "polars":
                    polars_dtype = polars_type_map.get(dtype, "Utf8")
                    scripts.append(f'{self.name} = {self.name}.with_columns(pl.col("{col}").cast(pl.{polars_dtype}))')
                else:
                    raise ValueError(f"Unsupported backend: {backend}")

            if backend == "pandas":
                scripts.append(
                    f'{self.name}.to_csv("{self.path}", index=False, columns={list(var_info.columns.keys())})'
                )
            elif backend == "polars":
                scripts.append(f'with fsspec.open("{self.path}", "w") as f:')
                scripts.append(f"    {self.name}.collect(streaming=True).write_csv(f)")
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        else:
            if backend == "pandas":
                scripts.append(f'{self.name}.to_csv("{self.path}", index=False)')
            elif backend == "polars":
                scripts.append(f'with fsspec.open("{self.path}", "w") as f:')
                scripts.append(f"    {self.name}.collect(streaming=True).write_csv(f)")
            else:
                raise ValueError(f"Unsupported backend: {backend}")
        return scripts

    def get_inputs(self) -> List[str]:
        return [self.name]
