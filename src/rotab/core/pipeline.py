import os
import subprocess
import shutil
import glob
from copy import deepcopy
from typing import Optional, List

from rotab.loader.loader import Loader
from rotab.loader.schema_manager import SchemaManager
from rotab.loader.context_builder import ContextBuilder
from rotab.runtime.code_generator import CodeGenerator
from rotab.runtime.dag_generator import DagGenerator
from rotab.utils.logger import get_logger

logger = get_logger()


class Pipeline:
    def __init__(self, template_dir, templates, backend, context, source_dir=".generated"):
        self.template_dir = template_dir
        self.templates = templates
        self.backend = backend
        self.context = context
        self.source_dir = source_dir

    @staticmethod
    def _clean_source_dir(source_dir: str):
        abs_source_dir = os.path.abspath(source_dir)
        abs_cwd = os.path.abspath(os.getcwd())

        logger.info(f"Preparing source directory: {source_dir}")
        os.makedirs(source_dir, exist_ok=True)

        protected = abs_source_dir == abs_cwd

        def safe_remove(path):
            if os.path.exists(path):
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)

        targets = [
            os.path.join(source_dir, "main.py"),
            os.path.join(source_dir, "mermaid.mmd"),
            os.path.join(source_dir, "data"),
        ]

        for path in targets:
            if protected and not os.path.abspath(path).startswith(abs_source_dir + os.sep):
                raise RuntimeError(f"Unsafe path deletion attempted: {path}")
            safe_remove(path)

        logger.info(f"Source directory ready: {source_dir}")

    @classmethod
    def from_setting(
        cls,
        template_dir: str,
        source_dir: str,
        param_dir: str,
        schema_dir: str,
        derive_func_path: Optional[str] = None,
        transform_func_path: Optional[str] = None,
        backend: str = "pandas",
    ):
        cls._clean_source_dir(source_dir)
        schema_manager = SchemaManager(schema_dir)
        loader = Loader(template_dir, param_dir, schema_manager)
        templates = loader.load()

        context_builder = ContextBuilder(
            derive_func_path=derive_func_path,
            transform_func_path=transform_func_path,
            schema_manager=schema_manager,
            backend=backend,
        )
        context = context_builder.build(templates)

        return cls(template_dir, templates, backend, context, source_dir)

    def is_remote_path(self, path: str) -> bool:
        return path.startswith("s3://") or "://" in path

    def rewrite_template_paths_and_copy_data(self, source_dir: str, template_dir: str):
        input_dir = os.path.join(source_dir, "data", "inputs")
        output_dir = os.path.join(source_dir, "data", "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        output_paths = set()
        for template in self.templates:
            for proc in template.processes:
                for node in proc.outputs:
                    orig_abs = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))
                    output_paths.add(orig_abs)

        # 入力ファイルコピー
        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:
                    if self.is_remote_path(node.path):
                        continue

                    if "*" in node.path:
                        pattern = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))
                        for matched_file in glob.glob(pattern):
                            fname = os.path.basename(matched_file)
                            dst_path = os.path.join(input_dir, fname)
                            shutil.copyfile(matched_file, dst_path)
                    else:
                        orig_abs = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))
                        fname = os.path.basename(node.path)
                        dst_path = os.path.join(input_dir, fname)

                        if orig_abs not in output_paths:
                            shutil.copyfile(orig_abs, dst_path)

        # パスの書き換え
        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:
                    if self.is_remote_path(node.path):
                        continue

                    fname = os.path.basename(node.path)
                    original_path = os.path.normpath(os.path.abspath(os.path.join(template_dir, node.path)))

                    if "*" in node.path:
                        new_path = os.path.relpath(os.path.join(input_dir, fname), source_dir)
                    elif original_path in output_paths:
                        new_path = os.path.relpath(os.path.join(output_dir, fname), source_dir)
                    else:
                        new_path = os.path.relpath(os.path.join(input_dir, fname), source_dir)

                    node.path = new_path

                for node in proc.outputs:
                    fname = os.path.basename(node.path)
                    new_path = os.path.relpath(os.path.join(output_dir, fname), source_dir)
                    node.path = new_path

    def copy_custom_functions(self, source_dir: str) -> None:
        cf_dir = os.path.join(source_dir, "custom_functions")
        os.makedirs(cf_dir, exist_ok=True)
        if self.context.derive_func_path:
            shutil.copy(self.context.derive_func_path, os.path.join(cf_dir, "derive_funcs.py"))
        if self.context.transform_func_path:
            shutil.copy(self.context.transform_func_path, os.path.join(cf_dir, "transform_funcs.py"))

    def validate_all(self) -> None:
        validate_context = deepcopy(self.context)
        for template in self.templates:
            template.validate(validate_context)

    def generate_code(self, source_dir: str, selected_processes: Optional[List[str]] = None) -> None:
        codegen = CodeGenerator(self.templates, self.backend, self.context)
        codegen.write_all(source_dir, selected_processes=selected_processes)
        logger.info(f"Code generated at: {source_dir}")

    def generate_dag(self, source_dir: str) -> None:
        dag_gen = DagGenerator(self.templates)
        mermeid = dag_gen.generate_mermaid()
        path = os.path.join(source_dir, "mermaid.mmd")
        with open(path, "w") as f:
            f.write(mermeid)
        logger.info(f"Mermaid DAG generated at: {path}")

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
            logger.error("Script execution failed.")
            logger.error(f"STDOUT:\n{e.stdout}")
            logger.error(f"STDERR:\n{e.stderr}")
            raise

    def run(self, execute: bool = True, dag: bool = False, selected_processes: Optional[List[str]] = None) -> None:
        logger.info("Pipeline run started.")
        os.makedirs(self.source_dir, exist_ok=True)
        self.copy_custom_functions(self.source_dir)
        self.rewrite_template_paths_and_copy_data(self.source_dir, self.template_dir)

        if dag:
            self.generate_dag(self.source_dir)
        self.validate_all()
        self.generate_code(self.source_dir, selected_processes=selected_processes)
        if execute:
            self.execute_script(self.source_dir)
        logger.info("Pipeline run completed.")
