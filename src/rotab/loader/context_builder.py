import importlib.util
import importlib
from typing import List, Optional
import types
import uuid
from rotab.ast.template_node import TemplateNode
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.loader.schema_manager import SchemaManager


class ContextBuilder:
    def __init__(self, derive_func_path: str, transform_func_path: str, schema_manager: SchemaManager, backend: str):
        self.derive_func_path = derive_func_path
        self.transform_func_path = transform_func_path
        self.schema_manager = schema_manager
        self.backend = backend

    def _load_module_functions(self, module_name: str) -> dict:
        module = importlib.import_module(module_name)
        return {k: v for k, v in vars(module).items() if isinstance(v, types.FunctionType) and not k.startswith("_")}

    def _load_file_functions(self, path: Optional[str]) -> dict:
        if not path:
            return {}

        module_name = f"custom_module_{uuid.uuid4().hex}"  # 一意な名前で競合防止
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Invalid Python file path or spec could not be loaded: {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return {name: obj for name, obj in vars(module).items() if callable(obj) and not name.startswith("_")}

    def _get_eval_scope(self) -> dict:
        if self.backend == "pandas":
            core_derive = self._load_module_functions("rotab.core.operation.derive_funcs_pandas")
            core_transform = self._load_module_functions("rotab.core.operation.transform_funcs_pandas")
        elif self.backend == "polars":
            core_derive = self._load_module_functions("rotab.core.operation.derive_funcs_polars")
            core_transform = self._load_module_functions("rotab.core.operation.transform_funcs_polars")
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        custom_derive = self._load_file_functions(self.derive_func_path)
        custom_transform = self._load_file_functions(self.transform_func_path)

        sources = {
            "core.derive": core_derive,
            "core.transform": core_transform,
            "custom.derive": custom_derive,
            "custom.transform": custom_transform,
        }

        name_to_sources = {}
        for source_label, funcs in sources.items():
            for name in funcs:
                name_to_sources.setdefault(name, []).append(source_label)

        conflicts = {name: srcs for name, srcs in name_to_sources.items() if len(srcs) > 1}
        if conflicts:
            conflict_messages = [f"Function `{name}` defined in: {', '.join(srcs)}" for name, srcs in conflicts.items()]
            raise ValueError("Function name conflicts detected:\n" + "\n".join(conflict_messages))

        merged_scope = {}
        merged_scope.update(core_derive)
        merged_scope.update(core_transform)
        merged_scope.update(custom_derive)
        merged_scope.update(custom_transform)
        return merged_scope

    def build(self, templates: List[TemplateNode]) -> ValidationContext:
        eval_scope = self._get_eval_scope()

        available_vars = set()
        schemas = {}

        for tpl in templates:
            for proc in tpl.processes:
                for inp in proc.inputs:
                    available_vars.add(inp.name)
                    if inp.schema_name:
                        schemas[inp.name] = self.schema_manager.get_schema(inp.schema_name)
                for out in proc.outputs:
                    available_vars.add(out.name)
                    if out.schema_name:
                        schemas[out.name] = self.schema_manager.get_schema(out.schema_name)

        return ValidationContext(
            derive_func_path=self.derive_func_path,
            transform_func_path=self.transform_func_path,
            available_vars=available_vars,
            eval_scope=eval_scope,
            schemas=schemas,
        )
