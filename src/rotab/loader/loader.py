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
        sorted_templates = self._resolve_dependencies(templates)
        resolved_templates = self._resolve_paths(sorted_templates)

        return [self._to_node(t) for t in resolved_templates]

    def _load_all_templates(self) -> List[dict]:
        templates = []
        for filename in os.listdir(self.template_dir):
            if not (filename.endswith(".yaml") or filename.endswith(".yml")):
                continue

            path = self.template_dir / filename
            logger.info(f"Parsing template file: {filename}")
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    raise ValueError(f"Invalid YAML format in {filename}")

                global_macros = raw.get("macros", {})
                if "processes" in raw:
                    for process in raw["processes"]:
                        io_defs = process.get("io", {})
                        for io_section in ("inputs", "outputs"):
                            for io_def in io_defs.get(io_section, []):
                                name = io_def.get("name")
                                schema_name = io_def.get("schema")
                                if schema_name:
                                    self.schema_manager.get_schema(schema_name)
                                else:
                                    self.schema_manager.get_schema(name, raise_error=False)

                        macro_definitions = process.get("macros", global_macros)
                        expander = MacroExpander(macro_definitions)
                        if "steps" in process:
                            process["steps"] = expander.expand(process["steps"])
                        if "macros" in process:
                            del process["macros"]
                if "macros" in raw:
                    del raw["macros"]

                normalized = self._replace_with_key(raw)
                for process in normalized.get("processes", []):
                    original_io = deepcopy(process.get("io", {}))
                    process = self._preprocess_io_dict(process)
                    process["__original_io__"] = original_io

                    if "steps" in process:
                        process["steps"] = [self._preprocess_step_dict(step) for step in process["steps"]]

                resolved = self.param_resolver.resolve(normalized)
                resolved["__filename__"] = filename
                templates.append(resolved)
        logger.info(f"Loaded {len(templates)} templates.")
        return templates

    def _resolve_paths(self, templates: List[dict]) -> List[dict]:
        resolved_templates = []
        for t in templates:
            t_copy = deepcopy(t)
            if "processes" in t_copy:
                for process in t_copy["processes"]:
                    original_io = process.pop("__original_io__", {})
                    for io_section in ("inputs", "outputs"):
                        io_definitions = original_io.get(io_section, [])
                        for i, io_def in enumerate(io_definitions):
                            current_io_def = process.get(io_section, [])[i]
                            original_path = io_def.get("path")

                            if original_path:
                                resolved_path = str((self.template_dir / original_path).resolve())
                                current_io_def["path"] = resolved_path
                                continue

                            schema_name = current_io_def.get("schema_name", current_io_def.get("name"))
                            if schema_name:
                                var_info = self.schema_manager.get_schema(schema_name)
                                if var_info and var_info.path:
                                    resolved_path = str((self.schema_dir / var_info.path).resolve())
                                    current_io_def["path"] = resolved_path
                                else:
                                    current_io_def["path"] = ""
                            else:
                                current_io_def["path"] = ""
            resolved_templates.append(t_copy)
        return resolved_templates

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
        return step

    def _preprocess_io_dict(self, process: dict) -> dict:
        io = process.pop("io", {"inputs": [], "outputs": []})

        for io_key in ("inputs", "outputs"):
            io_list = io.get(io_key, [])
            for io_def in io_list:
                io_def.setdefault("type", "input" if io_key == "inputs" else "output")
                if "schema" in io_def:
                    io_def["schema_name"] = io_def.pop("schema")
                io_def.setdefault("schema_name", "")
                if "name" not in io_def:
                    raise ValueError(
                        f"Missing `name` in {io_key} definition of process `{process.get('name', '<unnamed>')}`"
                    )
            process[io_key] = io_list

        return process

    def _replace_with_key(self, obj):
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
        return TemplateNode.from_dict(template, self.schema_manager)
