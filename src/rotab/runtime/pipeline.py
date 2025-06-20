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
import shutil


class Pipeline:
    def __init__(self, template_dir, templates, context, source_dir=".generated"):
        self.template_dir = template_dir
        self.templates = templates
        self.context = context
        self.source_dir = source_dir

    @staticmethod
    def _clean_source_dir(source_dir: str):
        abs_source_dir = os.path.abspath(source_dir)
        abs_cwd = os.path.abspath(os.getcwd())

        print(f"Preparing source directory: {source_dir}")

        # source_dir が存在しない場合は作成
        os.makedirs(source_dir, exist_ok=True)

        # 危険な削除を避けるために明示的に削除対象を限定
        protected = abs_source_dir == abs_cwd

        def safe_remove(path):
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

        # 削除対象（必要に応じて拡張可能）
        targets = [
            os.path.join(source_dir, "main.py"),
            os.path.join(source_dir, "mermaid.mmd"),
            os.path.join(source_dir, "data"),
        ]

        for path in targets:
            if protected and not os.path.abspath(path).startswith(abs_source_dir + os.sep):
                raise RuntimeError(f"Unsafe path deletion attempted: {path}")
            print(f"Removing: {path}")
            safe_remove(path)

        print(f"Source directory ready: {source_dir}")

    @classmethod
    def from_setting(
        cls,
        template_dir: str,
        source_dir: str,
        param_dir: str,
        schema_dir: str,
        derive_func_path: Optional[str] = None,
        transform_func_path: Optional[str] = None,
    ):

        cls._clean_source_dir(source_dir)
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

        return cls(template_dir, templates, context, source_dir)

    def rewrite_template_paths_and_copy_data(self, source_dir: str, template_dir: str):
        input_dir = os.path.join(source_dir, "data", "inputs")
        output_dir = os.path.join(source_dir, "data", "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # 1. 出力ファイルの絶対パスを収集（テンプレート相対 → 絶対）
        output_paths = set()
        for template in self.templates:

            for proc in template.processes:
                for node in proc.outputs:
                    orig_abs = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))
                    output_paths.add(orig_abs)

        # 2. 入力ファイルのコピー処理（出力に含まれてないもののみ）
        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:
                    orig_abs = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))
                    fname = os.path.basename(node.path)
                    dst_path = os.path.join(input_dir, fname)

                    if orig_abs in output_paths:
                        print(f"SKIP COPY (is output): {orig_abs}")
                    else:
                        print(f"COPY INPUT: {orig_abs} → {dst_path}")
                        shutil.copyfile(orig_abs, dst_path)

        # 3. パス書き換え
        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:
                    fname = os.path.basename(node.path)

                    original_path = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))

                    if original_path in output_paths:
                        new_path = os.path.relpath(os.path.join(output_dir, fname), source_dir)
                    else:
                        new_path = os.path.relpath(os.path.join(input_dir, fname), source_dir)

                    print(f"REWRITE INPUT: {node.path} → {new_path}")
                    node.path = new_path

                for node in proc.outputs:
                    fname = os.path.basename(node.path)
                    new_path = os.path.relpath(os.path.join(output_dir, fname), source_dir)
                    print(f"REWRITE OUTPUT: {node.path} → {new_path}")
                    node.path = new_path

    def copy_custom_functions(self, source_dir: str) -> None:
        cf_dir = os.path.join(source_dir, "custom_functions")
        os.makedirs(cf_dir, exist_ok=True)
        if self.context.derive_func_path:
            shutil.copy(self.context.derive_func_path, os.path.join(cf_dir, "derive_funcs.py"))
        if self.context.transform_func_path:
            shutil.copy(self.context.transform_func_path, os.path.join(cf_dir, "transform_funcs.py"))

    def validate_all(self) -> None:
        # Create a deep copy to avoid modifying the original context while validating
        validate_context = deepcopy(self.context)

        for template in self.templates:
            template.validate(validate_context)
        print("DEBUG: validate_context = ", validate_context)

    def generate_code(self, source_dir: str) -> None:
        codegen = CodeGenerator(self.templates, self.context)
        codegen.write_all(source_dir)
        print("Code generated at:", source_dir)

    def generate_dag(self, source_dir: str) -> None:
        dag_gen = DagGenerator(self.templates)
        mermeid = dag_gen.generate_mermaid()
        with open(os.path.join(source_dir, "mermaid.mmd"), "w") as f:
            f.write(mermeid)
        print("Mermaid DAG generated at:", os.path.join(source_dir, "mermaid.mmd"))

    def execute_script(self, source_dir: str) -> None:
        try:
            subprocess.run(
                ["python", "main.py"],
                cwd=source_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
            raise

    def run(self, execute: bool = True, dag: bool = False) -> None:
        """
        - execute=True: Pythonスクリプト(main.py)をその場で実行
        - dag=True: Mermaid DAGファイルを生成
        """
        os.makedirs(self.source_dir, exist_ok=True)
        self.copy_custom_functions(self.source_dir)
        self.rewrite_template_paths_and_copy_data(self.source_dir, self.template_dir)

        if dag:
            self.generate_dag(self.source_dir)
        self.validate_all()
        self.generate_code(self.source_dir)
        if execute:
            self.execute_script(self.source_dir)
