import os
import yaml
from typing import List
from copy import deepcopy

from rotab.loader.parameter_resolver import ParameterResolver
from rotab.loader.macro_expander import MacroExpander
from rotab.loader.schema_manager import SchemaManager

from rotab.ast.template import TemplateNode


class Loader:
    def __init__(self, template_dir: str, param_dir: str, schema_dir: str):
        self.template_dir = template_dir
        self.param_resolver = ParameterResolver(param_dir)
        self.schema_manager = SchemaManager(schema_dir)

    def load(self) -> List[TemplateNode]:
        templates = self._load_all_templates()
        sorted_templates = self._resolve_dependencies(templates)
        return [self._to_node(t) for t in sorted_templates]

    def _load_all_templates(self) -> List[dict]:
        templates = []
        for filename in os.listdir(self.template_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                path = os.path.join(self.template_dir, filename)
                with open(path, "r") as f:
                    raw = yaml.safe_load(f)
                    if not isinstance(raw, dict):
                        raise ValueError(f"Invalid YAML format in {filename}")

                    macro_applied = self._replace_with_key(deepcopy(raw))

                    global_macros = macro_applied.get("macros", {})

                    if "processes" in macro_applied:
                        for process in macro_applied["processes"]:
                            # --- ここでスキーマ存在チェック ---
                            for io_section in ("inputs", "outputs"):
                                for io_def in process.get("io", {}).get(io_section, []):
                                    schema_name = io_def.get("schema")
                                    if schema_name:
                                        self.schema_manager.get_schema(schema_name)
                            # --------------------------------

                            macro_definitions = process.get("macros", global_macros)
                            expander = MacroExpander(macro_definitions)
                            if "steps" in process:
                                process["steps"] = expander.expand(process["steps"])
                                # type付与＋フィールド名リネーム
                                process["steps"] = [self._preprocess_step_dict(step) for step in process["steps"]]
                            if "macros" in process:
                                del process["macros"]
                        if "macros" in macro_applied:
                            del macro_applied["macros"]

                    resolved = self.param_resolver.resolve(macro_applied)
                    resolved["__filename__"] = filename
                    templates.append(resolved)
        return templates

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

    def _replace_with_key(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if k == "with":
                    if isinstance(v, str):
                        new_dict["input_vars"] = [v]
                    else:
                        new_dict["input_vars"] = v
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
