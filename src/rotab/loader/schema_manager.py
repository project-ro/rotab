import os
import yaml
from typing import Dict
from rotab.ast.context.validation_context import VariableInfo


class SchemaManager:
    def __init__(self, schema_dir: str):
        self.schema_dir = schema_dir

    def get_schema(self, name: str) -> VariableInfo:
        yaml_path = os.path.join(self.schema_dir, f"{name}.yaml")
        yml_path = os.path.join(self.schema_dir, f"{name}.yml")

        if os.path.exists(yaml_path):
            schema_path = yaml_path
        elif os.path.exists(yml_path):
            schema_path = yml_path
        else:
            raise FileNotFoundError(f"Schema file not found: {yaml_path} or {yml_path}")

        with open(schema_path, "r") as f:
            raw_schema = yaml.safe_load(f)

        if "columns" in raw_schema:
            return VariableInfo(type="dataframe", columns={col["name"]: col["dtype"] for col in raw_schema["columns"]})
        elif isinstance(raw_schema, dict):
            return VariableInfo(type="dataframe", columns=raw_schema)
        else:
            raise ValueError(f"Invalid schema format in {schema_path}")
