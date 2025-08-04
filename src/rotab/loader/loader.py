import os
import yaml
from typing import List, Dict, Any
from pathlib import Path
from copy import deepcopy

from rotab.loader.parameter_resolver import ParameterResolver
from rotab.loader.macro_expander import MacroExpander
from rotab.loader.schema_manager import SchemaManager
from rotab.ast.template_node import TemplateNode
from rotab.utils.logger import get_logger

logger = get_logger()


class Loader:
    def __init__(self, template_dir: str, param_dir: str, schema_manager: SchemaManager):
        self.template_dir = Path(template_dir).resolve()
        self.param_resolver = ParameterResolver(param_dir)
        self.schema_manager = schema_manager
        self.schema_dir = Path(self.schema_manager.get_schema_dir()).resolve()

    def load(self) -> List[TemplateNode]:
        logger.info("Loading templates...")
        templates = self._load_all_templates()
        templates = self._resolve_dependencies(templates)
        templates = self._resolve_io_definitions(templates)
        templates = self._resolve_steps(templates)
        return [self._to_node(t) for t in templates]

    def _is_remote_path(self, path: str) -> bool:
        path = str(path)
        return path.startswith(("s3://", "s3a://", "gs://", "http://", "https://"))

    def _load_all_templates(self) -> List[dict]:
        templates = []
        for filename in os.listdir(self.template_dir):
            if not filename.endswith((".yaml", ".yml")):
                continue

            path = self.template_dir / filename
            logger.info(f"Parsing template file: {filename}")
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid YAML format in {filename}")

                global_macros = raw.get("macros", {})

                for process in raw.get("processes", []):
                    for io_section in ("inputs", "outputs"):
                        for io_def in process.get("io", {}).get(io_section, []):
                            if "schema" in io_def:
                                self.schema_manager.get_schema(io_def["schema"])

                for process in raw.get("processes", []):
                    macros = process.get("macros", global_macros)
                    expander = MacroExpander(macros)
                    if "steps" in process:
                        process["steps"] = expander.expand(process["steps"])
                    process.pop("macros", None)

                raw.pop("macros", None)
                resolved = self.param_resolver.resolve(raw)
                resolved["__filename__"] = filename
                templates.append(resolved)

        logger.info(f"Loaded {len(templates)} templates.")
        return templates

    def _resolve_io_definitions(self, templates: List[dict]) -> List[dict]:
        for t in templates:
            for process in t.get("processes", []):
                io = process.pop("io", {"inputs": [], "outputs": []})
                global_lazy = io.get("lazy", False)

                for io_key in ("inputs", "outputs"):
                    resolved_io = []
                    for io_def in io.get(io_key, []):
                        if "schema" in io_def:
                            io_def["schema_name"] = io_def.pop("schema")
                        io_def.setdefault("schema_name", io_def.get("name", ""))
                        io_def.setdefault("lazy", global_lazy)

                        original_path = io_def.get("path")
                        schema_name = io_def.get("schema_name")

                        if original_path:
                            if not self._is_remote_path(original_path):
                                abs_path = str((self.template_dir / original_path).resolve())
                                io_def["path"] = abs_path
                            else:
                                io_def["path"] = original_path
                        elif schema_name:
                            var_info = self.schema_manager.get_schema(schema_name, raise_error=False)
                            if var_info and var_info.path:
                                abs_path = str((self.schema_dir / var_info.path).resolve())
                                io_def["path"] = abs_path

                        resolved_io.append(io_def)
                    process[io_key] = resolved_io

        return templates

    def _resolve_steps(self, templates: List[dict]) -> List[dict]:
        for t in templates:
            for process in t.get("processes", []):
                if "steps" in process:
                    process["steps"] = self._infer_step_io(process["steps"])
                    process["steps"] = [self._preprocess_step_dict(s) for s in process["steps"]]
        return templates

    def _infer_step_io(self, steps: List[dict]) -> List[dict]:
        prev_output_var = None
        resolved_steps = []
        for i, step in enumerate(steps):
            step_copy = deepcopy(step)

            if "with" not in step_copy:
                if i == 0:
                    raise ValueError(
                        f"Step '{step_copy.get('name', 'unnamed')}' is the first step and must explicitly define 'with'."
                    )
                if prev_output_var is None:
                    raise ValueError(
                        f"Previous step did not define an 'as' variable, so 'with' cannot be inferred for step '{step_copy.get('name', 'unnamed')}'."
                    )
                step_copy["with"] = prev_output_var

            if "as" not in step_copy:
                if i == len(steps) - 1:
                    raise ValueError(
                        f"Step '{step_copy.get('name', 'unnamed')}' is the last step and must explicitly define 'as'."
                    )

                input_var = step_copy["with"]
                if isinstance(input_var, str):
                    new_as = f"tmp_{i}_{input_var}"
                    step_copy["as"] = new_as
                else:
                    raise ValueError(
                        f"Cannot infer 'as' for step '{step_copy.get('name', 'unnamed')}' because 'with' is a list. Please define 'as' explicitly."
                    )

            prev_output_var = step_copy["as"]
            resolved_steps.append(step_copy)

        return resolved_steps

    def _preprocess_step_dict(self, step: dict) -> dict:
        if "type" in step:
            return step
        if "mutate" in step:
            step["type"] = "mutate"
            step["operations"] = step.pop("mutate")
        elif "transform" in step:
            step["type"] = "transform"
            step["expr"] = step.pop("transform")
        else:
            raise ValueError(f"Step `{step.get('name', '<unnamed>')}` must contain either 'mutate' or 'transform'.")

        if "with" in step:
            v = step.pop("with")
            step["input_vars"] = [v] if isinstance(v, str) else v
        if "as" in step:
            v = step.pop("as")
            step["output_vars"] = [v] if isinstance(v, str) else v

        return step

    def _replace_with_key(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k == "with":
                    new_dict["input_vars"] = [v] if isinstance(v, str) else v
                elif k == "as":
                    new_dict["output_vars"] = [v] if isinstance(v, str) else v
                else:
                    new_dict[k] = self._replace_with_key(v)
            return new_dict
        elif isinstance(obj, list):
            return [self._replace_with_key(item) for item in obj]
        else:
            return obj

    def _resolve_dependencies(self, templates: List[dict]) -> List[dict]:
        name_to_template = {t["name"]: t for t in templates}
        visited = set()
        result = []

        def visit(t):
            tname = t["name"]
            if tname in visited:
                return
            for dep in t.get("depends", []):
                if dep not in name_to_template:
                    raise ValueError(f"Missing dependency: {dep}")
                visit(name_to_template[dep])
            visited.add(tname)
            result.append(t)

        for t in templates:
            visit(t)

        return result

    def _to_node(self, template: dict) -> TemplateNode:
        template_copy = deepcopy(template)
        for process in template_copy.get("processes", []):
            for io_section in ("inputs", "outputs"):
                for io_def in process.get(io_section, []):
                    if "path" in io_def and "io_type" not in io_def:
                        ext = Path(io_def["path"]).suffix.lower()
                        if ext == ".csv":
                            io_def["io_type"] = "csv"
                        elif ext == ".json":
                            io_def["io_type"] = "json"
                        elif ext == ".parquet":
                            io_def["io_type"] = "parquet"
                        else:
                            raise ValueError(f"Could not infer `io_type` from path: {io_def['path']}")

        return TemplateNode.from_dict(template_copy, self.schema_manager)
