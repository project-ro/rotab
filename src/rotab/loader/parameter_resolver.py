import os
import re
import yaml
from typing import Any, Dict
from rotab.utils.logger import get_logger

logger = get_logger()


class ParameterResolver:
    PARAM_PATTERN = re.compile(r"\$\{([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)\}")

    def __init__(self, param_dir: str):
        self.param_dir = param_dir
        self.params = self._load_params()

    def _load_params(self) -> Dict[str, Any]:
        combined: Dict[str, Any] = {}
        for filename in os.listdir(self.param_dir):
            if filename.endswith((".yaml", ".yml")):
                path = os.path.join(self.param_dir, filename)
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    if not isinstance(data, dict):
                        raise ValueError(f"Parameter file {filename} must contain a dictionary at the top level.")
                    for key in data:
                        if key in combined:
                            raise ValueError(f"Duplicate parameter key '{key}' found in {filename}")
                        combined[key] = data[key]
                logger.info(f"Loaded parameters from {filename}")
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
            return self._lookup_param(match.group(1))
        else:
            return self.PARAM_PATTERN.sub(lambda m: str(self._lookup_param(m.group(1))), s)

    def _lookup_param(self, path: str) -> Any:
        keys = path.split(".")
        value = self.params
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                logger.warning(f"Failed to resolve parameter: {path}")
                raise KeyError(f"Parameter '{path}' not found in parameter files.")
            value = value[key]
        return value
