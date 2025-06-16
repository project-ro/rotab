import os
import subprocess
from typing import Optional
from rotab.loader.loader import Loader
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

        return cls(templates, context)

    def run(self, execute: bool = True, dag: bool = False, output_dir: str = "generated") -> None:
        """
        - execute=True: Pythonスクリプト(main.py)をその場で実行
        - dag=True: Mermaid DAGファイルを生成
        """
        os.makedirs(output_dir, exist_ok=True)

        if dag:
            dag_gen = DagGenerator(self.templates)
            # dag_gen.write_mermaid(os.path.join(output_dir, "dag.mmd"))

        codegen = CodeGenerator(self.templates, self.context)
        codegen.write_all(output_dir)

        if execute:
            subprocess.run(["python", "main.py"], check=True)
