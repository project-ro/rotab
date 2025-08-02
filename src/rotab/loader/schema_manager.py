import os
import yaml
from typing import Dict, Optional
from rotab.ast.context.validation_context import VariableInfo
from rotab.utils.logger import get_logger

logger = get_logger()


class SchemaManager:
    def __init__(self, schema_dir: str):
        self.schema_dir = schema_dir
        self._cache: Dict[str, VariableInfo] = {}

    def get_schema_dir(self) -> str:
        return self.schema_dir

    def get_schema(self, schema_name: str, raise_error: bool = True) -> Optional[VariableInfo]:
        if schema_name in self._cache:
            return self._cache[schema_name]

        yaml_path = os.path.join(self.schema_dir, f"{schema_name}.yaml")
        yml_path = os.path.join(self.schema_dir, f"{schema_name}.yml")
        schema_path = None
        if os.path.exists(yaml_path):
            schema_path = yaml_path
        elif os.path.exists(yml_path):
            schema_path = yml_path
        if schema_path is None:
            if raise_error:
                raise FileNotFoundError(f"Schema file not found: {yaml_path} or {yml_path}")
            else:
                return None

        with open(schema_path, "r") as f:
            raw_schema = yaml.safe_load(f)
        logger.info(f"Loaded schema: {schema_path}")

        if "columns" not in raw_schema:
            raise ValueError(f"'columns' key not found in {schema_path}")
        columns = raw_schema["columns"]
        path = raw_schema.get("path", None)

        if isinstance(columns, dict):
            schema = VariableInfo(type="dataframe", columns=columns, path=path)
        elif isinstance(columns, list):
            schema = VariableInfo(type="dataframe", columns={col["name"]: col["dtype"] for col in columns}, path=path)
        else:
            raise ValueError(f"Invalid columns format in {schema_path}")

        self._cache[schema_name] = schema
        return schema
