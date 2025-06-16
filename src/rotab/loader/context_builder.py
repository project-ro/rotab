import importlib.util
from typing import List
from rotab.ast.template import TemplateNode
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.loader.schema_manager import SchemaManager


class ContextBuilder:
    def __init__(self, derive_func_path: str, transform_func_path: str, schema_manager: SchemaManager):

        self.derive_func_path = derive_func_path
        self.transform_func_path = transform_func_path
        self.schema_manager = schema_manager

    def _load_functions(self, path: str) -> dict:
        spec = importlib.util.spec_from_file_location("custom_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {k: v for k, v in vars(module).items() if callable(v) and not k.startswith("_")}

    def build(self, templates: List[TemplateNode]) -> ValidationContext:
        eval_scope = {}
        eval_scope.update(self._load_functions(self.derive_func_path))
        eval_scope.update(self._load_functions(self.transform_func_path))

        available_vars = set()
        schemas = {}

        for tpl in templates:
            for proc in tpl.processes:
                for inp in proc.inputs:
                    available_vars.add(inp.name)
                    if inp.schema and inp.schema not in schemas:
                        schemas[inp.schema] = self.schema_manager.get_schema(inp.schema)
                for out in proc.outputs:
                    available_vars.add(out.name)
                    if out.schema and out.schema not in schemas:
                        schemas[out.schema] = self.schema_manager.get_schema(out.schema)

        return ValidationContext(
            available_vars=available_vars,
            eval_scope=eval_scope,
            schemas=schemas,
        )
