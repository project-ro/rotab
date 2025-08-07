import os
import re
from rotab.ast.node import Node
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from typing import Optional, List, Literal, Dict


class IOBaseNode(Node):
    name: str
    io_type: str
    path: str
    schema_name: Optional[str] = None
    lazy: Optional[bool] = False

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
        if self.io_type not in ["csv", "parquet"]:
            raise ValueError(f"[{self.name}] Only 'csv' and 'parquet' types are supported, got: {self.io_type}")

        context.available_vars.add(self.name)

        schema_key_to_use = self.schema_name if self.schema_name else self.name

        if schema_key_to_use in context.schemas:
            schema_info = context.schemas[schema_key_to_use]
            context.schemas[self.name] = VariableInfo(type="dataframe", columns=schema_info.columns.copy())
        else:
            context.schemas[self.name] = VariableInfo(type="dataframe", columns={})

    def generate_script(self, backend: str = "pandas", context: ValidationContext = None) -> List[str]:
        if context is None:
            raise ValueError("context must be provided.")

        var_info = context.schemas.get(self.name)
        if not isinstance(var_info, VariableInfo):
            raise ValueError(f"[{self.name}] VariableInfo not found for input.")

        # Set path if missing
        if not self.path:
            self.path = context.schemas.get(self.schema_name, {}).path if self.schema_name else ""
        if not self.path:
            raise ValueError(f"[{self.name}] 'path' must be specified for input node.")

        # Build read function string constructor
        read_func = self._build_read_func(backend, var_info)

        # Dispatch to wildcard or single-path handler
        if "*" in self.path:
            return self._generate_wildcard_read_lines(backend, read_func)
        return self._generate_single_file_read_lines(backend, read_func)

    def _build_read_func(self, backend: str, var_info: VariableInfo):
        polars_type_map = {"int": "Int64", "float": "Float64", "str": "Utf8", "bool": "Boolean"}

        def pandas_csv(path_expr):
            dtype_arg = f", dtype={repr(var_info.columns)}" if var_info.columns else ""
            return f"pd.read_csv({path_expr}{dtype_arg})"

        def pandas_parquet(path_expr):
            return f"pd.read_parquet({path_expr})"

        def polars_csv(path_expr):
            dtype_arg = ""
            if var_info.columns:
                dtype_dict = {
                    col: f"pl.{polars_type_map.get(dtype, 'Utf8')}" for col, dtype in var_info.columns.items()
                }
                dtype_items = ", ".join([f'"{k}": {v}' for k, v in dtype_dict.items()])
                dtype_arg = f", dtypes={{{dtype_items}}}"
            if self.lazy:
                return f"pl.scan_csv({path_expr}{dtype_arg})"
            else:
                return f"pl.read_csv({path_expr}{dtype_arg})"

        def polars_parquet(path_expr):
            if self.lazy:
                return f"pl.scan_parquet({path_expr})"
            else:
                return f"pl.read_parquet({path_expr})"

        io_map = {
            ("csv", "pandas"): pandas_csv,
            ("csv", "polars"): polars_csv,
            ("parquet", "pandas"): pandas_parquet,
            ("parquet", "polars"): polars_parquet,
        }

        key = (self.io_type, backend)
        if key not in io_map:
            raise ValueError(f"Unsupported io_type/backend combination: {key}")
        return io_map[key]

    def _generate_wildcard_read_lines(self, backend: str, read_func) -> List[str]:
        if not self.wildcard_column:
            raise ValueError(f"[{self.name}] 'wildcard_column' must be specified for wildcard path.")

        basename_pattern = os.path.basename(self.path)
        regex_pattern = re.escape(basename_pattern).replace("\\*", "(.+)")

        imports = ["import glob", "os", "re"]
        if backend == "polars":
            imports.append("polars as pl")

        common = [
            f"{self.name}_files = glob.glob('{self.path}')",
            f"{self.name}_df_list = []",
            f"_regex = re.compile(r'{regex_pattern}')",
            f"for _file in {self.name}_files:",
            f"    _basename = os.path.basename(_file)",
            f"    _match = _regex.match(_basename)",
            f"    if not _match: raise ValueError(f'Unexpected filename: {{_basename}}')",
            f"    _val = _match.group(1)",
        ]

        if backend == "pandas":
            body = [
                f"    _df = {read_func('_file')}",
                f"    _df['{self.wildcard_column}'] = _val",
                f"    _df['{self.wildcard_column}'] = _df['{self.wildcard_column}'].astype(str)",
                f"    {self.name}_df_list.append(_df)",
                f"{self.name} = pd.concat({self.name}_df_list, ignore_index=True)",
            ]
        elif backend == "polars":
            body = [
                f"    _df = {read_func('_file')}",
                f"    _df = _df.with_columns(pl.lit(_val).cast(pl.Utf8).alias('{self.wildcard_column}'))",
                f"    {self.name}_df_list.append(_df)",
                f"{self.name} = pl.concat({self.name}_df_list, how='vertical')",
            ]
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return [", ".join(imports)] + common + body

    def _generate_single_file_read_lines(self, backend: str, read_func) -> List[str]:
        if backend == "pandas" or backend == "polars":
            path_literal = f'"{self.path}"'
            return [f"{self.name} = {read_func(path_literal)}"]

        raise ValueError(f"Unsupported backend: {backend}")

    def get_outputs(self) -> List[str]:
        return [self.name]


class OutputNode(IOBaseNode):
    type: Literal["output"] = "output"

    def validate(self, context: ValidationContext) -> None:
        if self.io_type not in ["csv", "parquet"]:
            raise ValueError(f"[{self.name}] Only 'csv' and 'parquet' type are supported, got: {self.io_type}")

        if self.name not in context.available_vars:
            raise ValueError(f"[{self.name}] Output variable '{self.name}' is not defined in scope.")

        schema_key = self.schema_name if self.schema_name else self.name

        if schema_key not in context.schemas:
            raise ValueError(f"[{self.name}] Schema '{schema_key}' not found in scope.")

    def generate_script(self, backend: str = "pandas", context: ValidationContext = None) -> List[str]:
        if context is None:
            raise ValueError("context must be provided.")

        schema_key = self.schema_name if self.schema_name else self.name
        var_info = context.schemas.get(schema_key)
        columns = var_info.columns if isinstance(var_info, VariableInfo) and var_info.columns else None

        if backend == "pandas":
            return self._generate_pandas_script(columns)
        if backend == "polars":
            return self._generate_polars_script(columns)
        raise ValueError(f"Unsupported backend: {backend}")

    def _generate_pandas_script(self, columns: Optional[Dict[str, str]]) -> List[str]:
        scripts = []

        if columns:
            for col, dtype in columns.items():
                scripts.append(f'{self.name}["{col}"] = {self.name}["{col}"].astype("{dtype}")')

        write_args = "index=False"
        if columns:
            write_args += f", columns={list(columns.keys())}"

        if self.io_type == "csv":
            scripts.append(f'{self.name}.to_csv("{self.path}", {write_args})')
        elif self.io_type == "parquet":
            scripts.append(f'{self.name}.to_parquet("{self.path}", {write_args})')
        else:
            raise ValueError(f"Unsupported io_type: {self.io_type}")

        return scripts

    def _generate_polars_script(self, columns: Optional[Dict[str, str]]) -> List[str]:
        scripts = []

        if columns:
            pl_type_map = {"int": "Int64", "float": "Float64", "str": "Utf8", "bool": "Boolean"}
            for col, dtype in columns.items():
                pl_dtype = pl_type_map.get(dtype, "Utf8")
                scripts.append(f'{self.name} = {self.name}.with_columns(pl.col("{col}").cast(pl.{pl_dtype}))')

        if self.lazy:
            # with fsspec.open("data/outputs/filtered_users.csv", "w") as f:
            #     _collected = filtered_users.collect(streaming=True)
            #     _collected .write_csv(f)
            #     print("result shape:", _collected.shape)
            # return filtered_users

            scripts.append(f'with fsspec.open("{self.path}", "wb") as f:')
            scripts.append(f"    _collected = {self.name}.collect(streaming=True)")

            if self.io_type == "csv":
                scripts.append(f"    _collected.write_csv(f)")
            elif self.io_type == "parquet":
                scripts.append(f"    _collected.write_parquet(f)")
            else:
                raise ValueError(f"Unsupported io_type: {self.io_type}")

            scripts.append(f"    print('result shape:', _collected.shape)")
            scripts.append(f"    print_all_null_columns(_collected)")

        else:
            # EagerFrame: no collect, direct write
            if self.io_type == "csv":
                scripts.append(f'{self.name}.write_csv("{self.path}")')
            elif self.io_type == "parquet":
                scripts.append(f'{self.name}.write_parquet("{self.path}")')
            else:
                raise ValueError(f"Unsupported io_type: {self.io_type}")

            scripts.append(f'print("result shape:", {self.name}.shape)')
            scripts.append(f"print_all_null_columns({self.name})")

        return scripts

    def get_inputs(self) -> List[str]:
        return [self.name]
