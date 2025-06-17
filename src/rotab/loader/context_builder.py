import importlib.util
from typing import List
from rotab.ast.template_node import TemplateNode
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
            print(f"DEBUG: template = {tpl.name}")
            for proc in tpl.processes:
                print(f"DEBUG: process = {proc.name}, inputs = {proc.inputs}")
                for inp in proc.inputs:
                    print(f"DEBUG: input = {inp.name}, schema = {inp.schema}")

                    print(f"DEBUG: requesting schema for name={inp.schema} → key={inp.name}")
                    print(f"DEBUG: schema content = {self.schema_manager.get_schema(inp.schema)}")
                    available_vars.add(inp.name)
                    if inp.schema:
                        schemas[inp.name] = self.schema_manager.get_schema(inp.schema)
                for out in proc.outputs:
                    available_vars.add(out.name)
                    if out.schema:
                        schemas[out.name] = self.schema_manager.get_schema(out.schema)

        return ValidationContext(
            derive_func_path=self.derive_func_path,
            transform_func_path=self.transform_func_path,
            available_vars=available_vars,
            eval_scope=eval_scope,
            schemas=schemas,
        )
