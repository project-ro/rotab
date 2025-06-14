import os
import re
import yaml
from typing import Any, Dict


class ParameterResolver:
    PARAM_PATTERN = re.compile(r"\$\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}")

    def __init__(self, param_dir: str):
        self.param_dir = param_dir
        self.params = self._load_params()

    def _load_params(self) -> Dict[str, Any]:
        combined: Dict[str, Any] = {}
        for filename in os.listdir(self.param_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                path = os.path.join(self.param_dir, filename)
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Parameter file {filename} must contain a dictionary at the top level.")
                    for key in data:
                        if key in combined:
                            raise ValueError(f"Duplicate parameter key '{key}' found in {filename}")
                        combined[key] = data[key]
        return combined

    def resolve(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self._resolve_string(obj)
        elif isinstance(obj, list):
            return [self.resolve(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.resolve(v) for k, v in obj.items()}
        else:
            return obj

    def _resolve_string(self, s: str) -> Any:
        match = self.PARAM_PATTERN.fullmatch(s)
        if match:
            key_path = match.group(1).split(".")
            value = self.params
            for key in key_path:
                if not isinstance(value, dict) or key not in value:
                    raise KeyError(f"Parameter '{match.group(1)}' not found in parameter files.")
                value = value[key]
            return value  # 型そのまま返す（リストならリスト）
        else:

            def replace(m):
                key_path = m.group(1).split(".")
                value = self.params
                for key in key_path:
                    if not isinstance(value, dict) or key not in value:
                        raise KeyError(f"Parameter '{m.group(1)}' not found in parameter files.")
                    value = value[key]
                return str(value)  # 部分展開はstrで埋め込む

            return self.PARAM_PATTERN.sub(replace, s)
