import os
import subprocess
import shutil
import glob
from copy import deepcopy
from typing import Optional, List
import tempfile
from rotab.loader.loader import Loader
from rotab.loader.schema_manager import SchemaManager
from rotab.loader.context_builder import ContextBuilder
from rotab.runtime.code_generator import CodeGenerator
from rotab.runtime.dag_generator import DagGenerator
from rotab.utils.logger import get_logger

logger = get_logger()


class Pipeline:
    def __init__(self, templates, backend, context, source_dir=".generated"):
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

        return cls(templates, backend, context, source_dir)

    def is_remote_path(self, path: str) -> bool:
        return path.startswith("s3://") or "://" in path

    def _copy_data(self, source_dir: str):
        input_dir = os.path.join(source_dir, "data", "inputs")
        output_dir = os.path.join(source_dir, "data", "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        output_paths = set()
        for template in self.templates:
            for proc in template.processes:
                for node in proc.outputs:
                    if self.is_remote_path(node.path):
                        continue
                    output_paths.add(node.path)

        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:

                    if self.is_remote_path(node.path):
                        continue

                    if "*" in node.path:
                        pattern = node.path
                        for matched in glob.glob(pattern):
                            dst = os.path.join(input_dir, os.path.basename(matched))
                            shutil.copyfile(matched, dst)
                    else:
                        abs_src = node.path
                        dst = os.path.join(input_dir, os.path.basename(abs_src))
                        if abs_src not in output_paths:
                            shutil.copyfile(abs_src, dst)

        for template in self.templates:
            for proc in template.processes:
                for node in proc.inputs:
                    if self.is_remote_path(node.path):
                        continue
                    fname = os.path.basename(node.path)
                    abs_src = node.path

                    if "*" in node.path:
                        new_path = os.path.relpath(os.path.join(input_dir, fname), source_dir)
                    elif abs_src in output_paths:
                        new_path = os.path.relpath(os.path.join(output_dir, fname), source_dir)
                    else:
                        new_path = os.path.relpath(os.path.join(input_dir, fname), source_dir)

                    node.path = new_path

                for node in proc.outputs:
                    if self.is_remote_path(node.path):
                        continue

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

    def copy_core_modules(self, source_dir: str) -> None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if self.backend == "pandas":
            files_to_copy = [
                ("operation/derive_funcs_pandas.py", "core/operation/derive_funcs_pandas.py"),
                ("operation/transform_funcs_pandas.py", "core/operation/transform_funcs_pandas.py"),
                ("parse/parse.py", "core/parse.py"),
            ]
        elif self.backend == "polars":
            files_to_copy = [
                ("operation/derive_funcs_polars.py", "core/operation/derive_funcs_polars.py"),
                ("operation/transform_funcs_polars.py", "core/operation/transform_funcs_polars.py"),
                ("parse/parse.py", "core/parse.py"),
            ]

        for rel_src, rel_dst in files_to_copy:
            src_path = os.path.join(base_dir, rel_src)
            dst_path = os.path.join(source_dir, rel_dst)
            if not os.path.exists(src_path):
                continue
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)

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
        mermaid = dag_gen.generate_mermaid()
        path = os.path.join(source_dir, "mermaid.mmd")
        with open(path, "w") as f:
            f.write(mermaid)
        logger.info(f"Mermaid DAG generated at: {path}")

    def execute_script(self, source_dir: str) -> None:
        # SageMaker でもローカルでも動く一時ディレクトリ生成
        tmpdir_base = "/opt/ml/tmp" if os.path.exists("/opt/ml/tmp") else None
        tmpdir = tempfile.mkdtemp(prefix="rotab_tmp_", dir=tmpdir_base)

        env = os.environ.copy()
        env["TMPDIR"] = tmpdir

        try:
            proc = subprocess.Popen(
                ["python", "main.py"],
                cwd=source_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in proc.stdout:
                logger.info(line.rstrip())  # printなし、logger.infoのみ

            proc.wait()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, proc.args)

        except subprocess.CalledProcessError as e:
            logger.error("Script execution failed.")
            raise

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def run(self, execute: bool = True, dag: bool = False, selected_processes: Optional[List[str]] = None) -> None:
        logger.info("Pipeline run started.")
        os.makedirs(self.source_dir, exist_ok=True)
        self.copy_core_modules(self.source_dir)
        self.copy_custom_functions(self.source_dir)

        self._copy_data(self.source_dir)

        if dag:
            self.generate_dag(self.source_dir)
        self.validate_all()
        self.generate_code(self.source_dir, selected_processes=selected_processes)
        if execute:
            self.execute_script(self.source_dir)
        logger.info("Pipeline run completed.")
