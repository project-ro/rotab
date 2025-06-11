import os
import subprocess
import ast
import glob
import yaml
import textwrap
import re
from pathlib import Path
from typing import Dict, Any, List
from rotab.core.operation.script_generator import ScriptGenerator
from rotab.core.operation import derive_funcs, transform_funcs

indent = "  "


class RowExprTransformer(ast.NodeTransformer):
    def __init__(self, external_names: set):
        self.external_names = external_names

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.external_names:
            return ast.Subscript(
                value=ast.Name(id="row", ctx=ast.Load()),
                slice=ast.Index(value=ast.Constant(value=node.id)),
                ctx=node.ctx,
            )
        return node


class Pipeline:
    def __init__(
        self,
        template: Dict[str, Any],
        schemas: Dict[str, Any],
        template_dir: str,
        derive_func_path: str,
        transform_func_path: str,
    ):
        self.template = template
        self.schemas = schemas
        self.template_dir = template_dir
        self.derive_func_path = derive_func_path
        self.transform_func_path = transform_func_path
        self.eval_scope = {}
        for module in [derive_funcs, transform_funcs]:
            self.eval_scope.update({k: v for k, v in module.__dict__.items() if not k.startswith("__") and callable(v)})
        self.eval_scope.update(__builtins__)

    def _to_lambda_expr(self, rhs: str) -> str:
        tree = ast.parse(rhs, mode="eval")
        func_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func_names.add(node.func.id)

        preserved = {name for name in func_names if name in self.eval_scope and callable(self.eval_scope[name])}
        preserved.update(dir(__builtins__))

        transformer = RowExprTransformer(external_names=preserved)
        new_tree = transformer.visit(tree)
        ast.fix_missing_locations(new_tree)

        expr = ast.unparse(new_tree)
        return f"lambda row: {expr}"

    def _get_common_import_lines(self) -> list[str]:
        lines = [
            "import pandas as pd",
            "from rotab.core.operation.derive_funcs import *",
            "from rotab.core.operation.transform_funcs import *",
            "import importlib.util",
            "import os",
        ]
        for path in [self.derive_func_path] + [self.transform_func_path]:
            abs_path = os.path.abspath(path)
            module_name = os.path.splitext(os.path.basename(path))[0]
            lines.extend(
                [
                    f"spec = importlib.util.spec_from_file_location('{module_name}', r'{abs_path}')",
                    f"{module_name} = importlib.util.module_from_spec(spec)",
                    f"spec.loader.exec_module({module_name})",
                    f"globals().update({{k: v for k, v in {module_name}.__dict__.items() if callable(v) and not k.startswith('__')}})",
                ]
            )
        return lines

    def write_script(self, path: str):
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        script_generator = ScriptGenerator(self.template, self.schemas)
        with open(abs_path, "w") as f:
            f.write(script_generator.generate_script())

    def execute_script(self, path: str):
        result = subprocess.run(["python", path], cwd=self.template_dir, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Script failed:\n{result.stderr}")

    def _add_successor(self, graph: Dict[str, List[str]], src: str, dst: str):
        graph.setdefault(src, [])
        if dst not in graph[src]:
            graph[src].append(dst)

    def _get_template_successors(self, cfg: Dict[str, Any], template_key: str) -> Dict[str, List[str]]:
        result = {}
        for dep in cfg.get("depends", []):
            dep_key = dep.replace(".yaml", "")
            self._add_successor(result, dep_key, template_key)
        return result

    def _get_process_successors(self, processes: List[Dict[str, Any]], template_key: str):
        successors = {}
        proc_names = []
        proc_to_step = {}

        for i, proc in enumerate(processes):
            proc_name = proc.get("name", f"process_{i}_{template_key}")
            proc_names.append(proc_name)

            # プロセス同士の依存（直列）
            if i > 0:
                prev_proc = processes[i - 1]
                prev_proc_name = prev_proc.get("name", f"process_{i - 1}_{template_key}")
                self._add_successor(successors, prev_proc_name, proc_name)

            # === プロセス内のステップ列挙 ===
            step_list = []

            # 新構文: io.inputs → 入力ファイルパスを追加
            io_config = proc.get("io", {})
            for inp in io_config.get("inputs", []):
                if "path" in inp:
                    step_list.append(inp["path"])

            # steps → ステップ名を追加
            for j, step in enumerate(proc.get("steps", [])):
                step_name = step.get("name", f"step_{j}_{proc_name}")
                step_list.append(step_name)

            proc_to_step[proc_name] = step_list

        return successors, proc_names, proc_to_step

    def _get_step_successors(self, processes: List[Dict[str, Any]], template_key: str) -> Dict[str, List[str]]:
        result = {}

        for i, proc in enumerate(processes):
            proc_name = proc.get("name", f"process_{i}_{template_key}")
            steps = proc.get("steps", [])
            io_config = proc.get("io", {})
            inputs = io_config.get("inputs", [])

            # === ステップ間および入力との依存構築 ===
            for step_index, step in enumerate(steps):
                step_name = step.get("name", f"step_{step_index}_{proc_name}")

                vars_used = []

                # transform に含まれる変数抽出
                if "transform" in step:
                    rhs = step["transform"].split("=", 1)[-1].strip() if "=" in step["transform"] else ""
                    vars_used += self._extract_variable_names(rhs)

                # with に含まれる変数抽出
                if "with" in step:
                    with_val = step["with"]
                    if isinstance(with_val, list):
                        vars_used += with_val
                    elif isinstance(with_val, str):
                        vars_used.append(with_val)

                # 抽出された変数に対応する prior ステップ or 入力ファイルに依存を張る
                for var in vars_used:
                    prev = self._find_prior_step_using_var(steps, step_index, var, proc_name, inputs)
                    if prev:
                        self._add_successor(result, prev, step_name)

            # === ステップ → ダンプファイル の依存構築 ===
            for out in io_config.get("outputs", []):
                return_var = out.get("name")
                output_path = out.get("path")
                if return_var and output_path:
                    source_step = self._find_prior_step_using_var(steps, len(steps), return_var, proc_name, inputs)
                    if source_step:
                        self._add_successor(result, source_step, output_path)

        return result

    def _find_prior_step_using_var(
        self, steps: List[Dict[str, Any]], step_index: int, var: str, proc_name: str, inputs: List[Dict[str, Any]]
    ) -> str:
        # steps内を逆順に検索
        for k in range(step_index - 1, -1, -1):
            prev = steps[k]
            name = prev.get("name", f"step_{k}_{proc_name}")
            if "as" in prev and prev["as"] == var:
                return name

        # 見つからなければ inputs から探す
        for inp in inputs:
            if inp.get("name") == var:
                return inp.get("path")

        return None

    def _extract_variable_names(self, expr: str) -> List[str]:
        # 形式: key=var → var だけを抽出
        return re.findall(r"\b\w+\s*=\s*(\w+)", expr)

    def _get_dependencies(self) -> Dict[str, Any]:
        template_successors = {}
        process_successors = {}
        step_successors = {}
        template_to_process = {}
        process_to_step = {}

        print("template_dir:", self.template_dir)

        templates = sorted(Path(self.template_dir).glob("*.yaml"))

        assert templates, "No YAML templates found in the specified directory."

        for path in templates:
            template_name = path.name
            template_key = template_name.replace(".yaml", "")
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

            tmpl_succ = self._get_template_successors(cfg, template_key)
            for k, v_list in tmpl_succ.items():
                for v in v_list:
                    self._add_successor(template_successors, k, v)

            processes = cfg.get("processes", [])
            proc_succ, proc_names, proc_to_step_map = self._get_process_successors(processes, template_key)

            for k, v_list in proc_succ.items():
                for v in v_list:
                    self._add_successor(process_successors, k, v)

            template_to_process[template_key] = proc_names
            for proc, steps in proc_to_step_map.items():
                process_to_step[proc] = steps

            step_succ = self._get_step_successors(processes, template_key)
            for k, v_list in step_succ.items():
                for v in v_list:
                    self._add_successor(step_successors, k, v)

        result = {
            "template_successors": template_successors,
            "process_successors": process_successors,
            "step_successors": step_successors,
            "template_to_process": template_to_process,
            "process_to_step": process_to_step,
        }

        return result

    def _get_dag_one_process(self, template_name, process_name, steps, step_successors):

        print(f"Generating DAG for process: {process_name} in template: {template_name}")

        lines_one_process = [textwrap.indent(f"""subgraph P_{process_name} ["{process_name}"]""", indent * 1)]

        lines_nodes = [textwrap.indent(f"""S_{step}(["{step}"])""", indent * 2) for step in steps]
        lines_edges = [
            textwrap.indent(
                f"""S_{step} --> S_{successor}""",
                indent * 2,
            )
            for step, successors in step_successors.items()
            for successor in successors
        ]

        print("lines_nodes:", lines_nodes)
        print("lines_edges:", lines_edges)

        lines_one_process.extend(lines_nodes)
        lines_one_process.extend(lines_edges)
        lines_one_process.append(textwrap.indent("end", indent * 1))

        return lines_one_process

    def generate_dag(self) -> str:

        dependencies = self._get_dependencies()

        print(dependencies)

        template_successors = dependencies["template_successors"]
        step_successors = dependencies["step_successors"]
        template_to_process = dependencies["template_to_process"]
        process_to_step = dependencies["process_to_step"]

        # textwrap.indent("\n".join(lines), indent * level)

        dag_lines = ["graph TB"]

        # Template dependency
        dag_lines.extend(["", """%% ==== Template dependencies ===="""])
        lines_template_dependency = [
            f"T_{template_name} --> T_{successor}"
            for template_name, successors in template_successors.items()
            for successor in successors
        ]
        dag_lines.extend(lines_template_dependency)

        # Processes and steps dependency for each template
        for template_name, process_names in template_to_process.items():
            print("")
            print(f"Processing template: {template_name}")
            dag_lines.append("")
            dag_lines.append(f"""%% ==== Processes in {template_name} ====""")
            dag_lines.append(f"""subgraph T_{template_name.split(".")[0]} ["{template_name}"]""")
            for process_name in process_names:
                steps_one_process = process_to_step.get(process_name, [])

                # BUG これだと、dumpも入ってしまう
                step_successors_one_process = {step: step_successors.get(step, []) for step in steps_one_process}
                print(f"Processing process: {process_name}")
                print("steps_one_process:", steps_one_process)
                print("step_successors_one_process:", step_successors_one_process)

                if not steps_one_process:
                    continue
                dag_lines.extend(
                    self._get_dag_one_process(
                        template_name, process_name, steps_one_process, step_successors_one_process
                    )
                )

            dag_lines.append("end")

        return "\n".join(dag_lines)

    def run(self, execute: bool, dag: bool = False):
        base_dir = os.path.join(self.template_dir, ".generated")
        os.makedirs(base_dir, exist_ok=True)
        abs_path = os.path.join(base_dir, "generated_script.py")

        if dag:
            mermaid_path = os.path.splitext(abs_path)[0] + ".mmd"
            mermaid_content = self.generate_dag()
            with open(mermaid_path, "w", encoding="utf-8") as f:
                f.write(mermaid_content)

        self.write_script(abs_path)

        if execute:
            self.execute_script(abs_path)

    @classmethod
    def from_setting(
        cls,
        template_dir: str,
        param_dir: str,
        schema_dir: str,
        derive_func_path: str,
        transform_func_path: str,
    ):
        params = cls._load_params(param_dir)
        schemas = cls._load_schemas(schema_dir)
        templates = cls._load_templates_with_render(template_dir, params, schemas)
        resolved_order = cls._toposort_templates(templates)

        merged_template = {"processes": []}
        for name in resolved_order:
            merged_template["processes"].extend(templates[name].get("processes", []))

        return cls(
            template=merged_template,
            schemas=schemas,
            template_dir=template_dir,
            derive_func_path=derive_func_path,
            transform_func_path=transform_func_path,
        )

    @staticmethod
    def _load_params(dirpath: str) -> dict:
        settings = {}
        for filename in os.listdir(dirpath):
            if filename.endswith((".yaml", ".yml")):
                full_path = os.path.join(dirpath, filename)
                with open(full_path, "r") as f:
                    content = yaml.safe_load(f) or {}
                    if not isinstance(content, dict):
                        raise ValueError(f"{filename} does not contain a dictionary at the top level.")
                    settings.update(content)
        return settings

    @staticmethod
    def _render_placeholders(template_str: str, params: dict) -> str:
        def resolve(expr: str):
            keys = expr.split(".")
            val = params
            for key in keys:
                if not isinstance(val, dict) or key not in val:
                    raise KeyError(f"Parameter `{expr}` not found.")
                val = val[key]
            return str(val)

        pattern = r"\$\{([a-zA-Z0-9_.]+)\}"  # Match ${param} or ${nested.param}
        return re.sub(pattern, lambda m: resolve(m.group(1)), template_str)

    @staticmethod
    def _load_templates_with_render(dirpath: str, params: dict, schemas: dict) -> dict:
        templates = {}
        for filepath in glob.glob(os.path.join(dirpath, "*.yaml")):
            with open(filepath, "r") as f:
                content = f.read()
                rendered = Pipeline._render_placeholders(content, params)
                cfg = yaml.safe_load(rendered) or {}
                Pipeline._inject_schema_columns(cfg, schemas)
            name = os.path.basename(filepath)
            templates[name] = cfg
        return templates

    @staticmethod
    def _inject_schema_columns(cfg: dict, schemas: dict) -> None:
        io_config = cfg.get("io", {})
        for inp in io_config.get("inputs", []):
            schema_name = inp.pop("schema", None)
            if schema_name and schema_name in schemas:
                inp["columns"] = schemas[schema_name]["columns"]

    @staticmethod
    def _load_schemas(schema_dir: str) -> dict:
        schemas = {}
        for filepath in glob.glob(os.path.join(schema_dir, "*.yaml")):
            with open(filepath, "r") as f:
                cfg = yaml.safe_load(f)
                name = cfg.get("name") or os.path.splitext(os.path.basename(filepath))[0]
                schemas[name] = cfg
        return schemas

    @staticmethod
    def _toposort_templates(templates: dict[str, dict]) -> list[str]:
        graph = {name: cfg.get("depends", []) for name, cfg in templates.items()}
        resolved = []
        visited = set()
        temp = set()

        def visit(name):
            if name in visited:
                return
            if name in temp:
                raise ValueError(f"Circular dependency detected at {name}")
            temp.add(name)
            for dep in graph.get(name, []):
                visit(dep)
            temp.remove(name)
            visited.add(name)
            resolved.append(name)

        for name in templates:
            visit(name)

        return resolved
