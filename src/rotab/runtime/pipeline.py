import os
import subprocess
import shutil
from copy import deepcopy
from typing import Optional

from rotab.loader.loader import Loader
from rotab.loader.schema_manager import SchemaManager
from rotab.loader.context_builder import ContextBuilder
from rotab.runtime.code_generator import CodeGenerator
from rotab.runtime.dag_generator import DagGenerator


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

    def copy_custom_functions(self, output_dir: str) -> None:
        cf_dir = os.path.join(output_dir, "custom_functions")
        os.makedirs(cf_dir, exist_ok=True)
        if self.context.derive_func_path:
            shutil.copy(self.context.derive_func_path, os.path.join(cf_dir, "derive_funcs.py"))
        if self.context.transform_func_path:
            shutil.copy(self.context.transform_func_path, os.path.join(cf_dir, "transform_funcs.py"))

    def validate_all(self) -> None:
        validate_context = deepcopy(self.context)
        for template in self.templates:
            template.validate(validate_context)
        print("DEBUG: validate_context = ", validate_context)

    def generate_code(self, output_dir: str) -> None:
        codegen = CodeGenerator(self.templates, self.context)
        codegen.write_all(output_dir)
        print("Code generated at:", output_dir)

    def generate_dag(self, output_dir: str) -> None:
        dag_gen = DagGenerator(self.templates)
        mermeid = dag_gen.generate_mermaid()
        with open(os.path.join(output_dir, "mermaid.mmd"), "w") as f:
            f.write(mermeid)
        print("Mermaid DAG generated at:", os.path.join(output_dir, "mermaid.mmd"))

    def execute_script(self, output_dir: str) -> None:
        try:
            subprocess.run(["python", os.path.join(output_dir, "main.py")], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

    def run(self, execute: bool = True, dag: bool = False, output_dir: str = ".generated") -> None:
        """
        - execute=True: Pythonスクリプト(main.py)をその場で実行
        - dag=True: Mermaid DAGファイルを生成
        """
        os.makedirs(output_dir, exist_ok=True)
        self.copy_custom_functions(output_dir)
        if dag:
            self.generate_dag(output_dir)
        self.validate_all()
        self.generate_code(output_dir)
        if execute:
            self.execute_script(output_dir)
