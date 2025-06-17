import os
import subprocess
from typing import Optional
import shutil
from copy import deepcopy
from rotab.loader.loader import Loader
from rotab.ast.context.validation_context import VariableInfo
from rotab.loader.schema_manager import SchemaManager
from rotab.runtime.code_generator import CodeGenerator
from rotab.runtime.dag_generator import DagGenerator
from rotab.loader.context_builder import ContextBuilder


class Pipeline:
    def __init__(self, templates, context):
        self.templates = templates
        self.context = context

    @classmethod
    def from_setting(
        cls,
        template_dir: str,
        param_dir: str,
        schema_dir: str,
        derive_func_path: Optional[str] = None,
        transform_func_path: Optional[str] = None,
    ):
        schema_manager = SchemaManager(schema_dir)

        loader = Loader(template_dir, param_dir, schema_manager)
        templates = loader.load()

        context_builder = ContextBuilder(
            derive_func_path=derive_func_path,
            transform_func_path=transform_func_path,
            schema_manager=schema_manager,
        )
        context = context_builder.build(templates)

        print("DEBUG: initial context = ", context)

        return cls(templates, context)

    def run(self, execute: bool = True, dag: bool = False, output_dir: str = "generated") -> None:
        """
        - execute=True: Pythonスクリプト(main.py)をその場で実行
        - dag=True: Mermaid DAGファイルを生成
        """

        # create directories
        os.makedirs(output_dir, exist_ok=True)
        cf_dir = os.path.join(output_dir, "custom_functions")
        os.makedirs(cf_dir, exist_ok=True)
        if self.context.derive_func_path:
            shutil.copy(self.context.derive_func_path, os.path.join(cf_dir, "derive_funcs.py"))
        if self.context.transform_func_path:
            shutil.copy(self.context.transform_func_path, os.path.join(cf_dir, "transform_funcs.py"))

        # generate dag
        if dag:
            dag_gen = DagGenerator(self.templates)
            # dag_gen.write_mermaid(os.path.join(output_dir, "dag.mmd"))

        # validate only once
        validate_context = deepcopy(self.context)
        for template in self.templates:
            template.validate(validate_context)
        print("DEBUG: validate_context = ", validate_context)

        # generate code
        codegen = CodeGenerator(self.templates, self.context)
        codegen.write_all(output_dir)

        # execute scripts
        if execute:
            try:
                subprocess.run(["python", "main.py"], cwd=output_dir, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print("STDOUT:\n", e.stdout)
                print("STDERR:\n", e.stderr)
                raise
