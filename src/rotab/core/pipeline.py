import os
import subprocess
import ast
import glob
import yaml
import textwrap
import re
from pathlib import Path
from typing import Dict, Any, List
from rotab.core.operation import define_funcs, transform_funcs

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
        self, config: Dict[str, Any], base_path: str, define_func_paths: list[str], transform_func_paths: list[str]
    ):
        self.config = config
        self.base_path = base_path
        self.define_func_paths = define_func_paths
        self.transform_func_paths = transform_func_paths
        self.eval_scope = {}
        for module in [define_funcs, transform_funcs]:
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
            "from rotab.core.operation.define_funcs import *",
            "from rotab.core.operation.transform_funcs import *",
            "import importlib.util",
            "import os",
        ]
        for path in self.define_func_paths + self.transform_func_paths:
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

    def generate_script(self) -> str:
        import_lines = self._get_common_import_lines()
        step_funcs = ["# STEPS FUNCTIONS:", ""]
        process_funcs = ["# PROCESSES FUNCTIONS:"]
        main_calls = []

        for i, process in enumerate(self.config.get("processes", [])):
            process_name = process.get("name", f"block_{i}")
            description = process.get("description", "")
            step_calls = []
            local_vars = set()

            for step in process.get("steps", []):
                step_name = step.get("name")
                if not step_name:
                    raise ValueError("Each step must have a 'name' field.")
                if "with" in step:
                    var = step["with"]
                    local_vars.add(var)
                    lines = [f"def step_{step_name}_{process_name}({var}):"]
                    lines.append(f'    """Step: {step_name} """')
                    if "filter" in step:
                        lines.append(f"    {var} = {var}.query('{step['filter']}').copy()")
                    if "define" in step:
                        for line in step["define"].split("\n"):
                            if line.strip():
                                lhs, rhs = map(str.strip, line.split("=", 1))
                                lambda_expr = self._to_lambda_expr(rhs)
                                lines.append(f"    {var}.loc[:, '{lhs}'] = {var}.apply({lambda_expr}, axis=1)")
                    if "select" in step:
                        cols = step["select"]
                        lines.append(f"    {var} = {var}[{cols}]")
                    lines.append(f"    return {var}")
                    step_funcs.append("\n".join(lines))
                    step_funcs.append("")
                    step_calls.append(f"    {var} = step_{step_name}_{process_name}({var})")

                elif "transform" in step:
                    assign = step["transform"]
                    lhs, rhs = map(str.strip, assign.split("=", 1))
                    try:
                        parsed = ast.parse(rhs, mode="eval")
                        arg_names = sorted({node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)})
                    except Exception:
                        raise ValueError(f"Cannot parse transform expression: {assign}")

                    step_calls.append(f"    {lhs} = step_{step_name}_{process_name}({', '.join(arg_names)})")
                    lines = [
                        f"def step_{step_name}_{process_name}({', '.join(arg_names)}):",
                        f'    """Step: {step_name}"""',
                        f"    return {rhs}",
                    ]
                    step_funcs.append("\n".join(lines))
                    step_funcs.append("")

            # create process function
            func_lines = [""]
            func_lines.append(f"def process_{process_name}():")
            if description:
                func_lines.append(f'    """{description}"""')

            func_lines.append("    # load tables")
            for table in process.get("tables", []):
                name, rel_path = table["name"], table["path"]
                abs_path = os.path.abspath(os.path.join(self.base_path, rel_path))
                func_lines.append(f"    {name} = pd.read_csv(r'{abs_path}')")

            func_lines.append("\n    # process steps")
            func_lines.extend(step_calls)

            func_lines.append("\n    # dump output")
            for dump in process.get("dumps", []):
                dfname, path = dump["return"], dump["path"]
                func_lines.extend(
                    [
                        f"    path = os.path.abspath(r'{path}')",
                        "    os.makedirs(os.path.dirname(path), exist_ok=True)",
                        f"    {dfname}.to_csv(path, index=False)",
                    ]
                )

            process_funcs.append("\n".join(func_lines))
            main_calls.append(f"process_{process_name}()")

        script_lines = (
            import_lines
            + ["", ""]
            + step_funcs
            + process_funcs
            + ["", "", "if __name__ == '__main__':"]
            + [f"    {call}" for call in main_calls]
        )
        return "\n".join(script_lines)

    def write_script(self, path: str):
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(self.generate_script())

    def execute_script(self, path: str):
        result = subprocess.run(["python", path], cwd=self.base_path, capture_output=True, text=True)
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

            if i > 0:
                prev_proc = processes[i - 1]
                prev_proc_name = prev_proc.get("name", f"process_{i - 1}_{template_key}")
                self._add_successor(successors, prev_proc_name, proc_name)

            proc_to_step.setdefault(proc_name, [])
            for tbl in proc.get("tables", []):
                proc_to_step[proc_name].append(tbl["path"])
            # for dump in proc.get("dumps", []):
            #     return_var = dump.get("return")
            #     output_path = dump.get("path")
            #     if return_var and output_path:
            #         proc_to_step[proc_name].append(output_path)
            for j, step in enumerate(proc.get("steps", [])):
                step_name = step.get("name", f"step_{j}_{proc_name}")
                proc_to_step[proc_name].append(step_name)

        return successors, proc_names, proc_to_step

    def _get_step_successors(self, processes: List[Dict[str, Any]], template_key: str) -> Dict[str, List[str]]:
        result = {}

        for i, proc in enumerate(processes):
            proc_name = proc.get("name", f"process_{i}_{template_key}")
            steps = proc.get("steps", [])
            tables = proc.get("tables", [])

            for step_index, step in enumerate(steps):
                step_name = step.get("name", f"step_{step_index}_{proc_name}")

                if "transform" in step:
                    rhs = step["transform"].split("=", 1)[-1].strip() if "=" in step["transform"] else ""
                    used_vars = self._extract_variable_names(rhs)
                    for var in used_vars:
                        prev = self._find_prior_step_using_var(steps, step_index, var, proc_name, tables)
                        if prev:
                            self._add_successor(result, prev, step_name)

                elif "with" in step:
                    with_var = step["with"]
                    prev = self._find_prior_step_using_var(steps, step_index, with_var, proc_name, tables)
                    if prev:
                        self._add_successor(result, prev, step_name)

            for dump in proc.get("dumps", []):
                return_var = dump.get("return")
                output_path = dump.get("path")
                if return_var and output_path:
                    step_name = self._find_prior_step_using_var(steps, len(steps), return_var, proc_name, tables)
                    if step_name:
                        self._add_successor(result, step_name, output_path)

        return result

    def _find_prior_step_using_var(
        self, steps: List[Dict[str, Any]], step_index: int, var: str, proc_name: str, tables: List[Dict[str, Any]]
    ) -> str:
        # steps内を逆順に検索
        for k in range(step_index - 1, -1, -1):
            prev = steps[k]
            name = prev.get("name", f"step_{k}_{proc_name}")
            if "with" in prev and prev["with"] == var:
                return name
            if "transform" in prev:
                lhs = prev["transform"].split("=", 1)[0].strip()
                if lhs == var:
                    return name
        # 見つからなければtablesから探す
        for tbl in tables:
            if tbl.get("name") == var:
                return tbl.get("path")
        return None

    def _extract_variable_names(self, expr: str) -> List[str]:
        return re.findall(r"\b\w+\b", expr)

    def get_dependencies(self) -> Dict[str, Any]:
        template_successors = {}
        process_successors = {}
        step_successors = {}
        template_to_process = {}
        process_to_step = {}

        print("base_path:", self.base_path)

        templates = sorted(Path(self.base_path).glob("*.yaml"))

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

        dependencies = self.get_dependencies()

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

    def run(self, script_path: str, execute: bool, dag: bool = False):
        abs_path = (
            script_path if os.path.isabs(script_path) else os.path.abspath(os.path.join(self.base_path, script_path))
        )

        if dag:
            base, _ = os.path.splitext(abs_path)
            mermaid_path = f"{base}.mmd"
            mermaid_content = self.generate_dag()
            with open(mermaid_path, "w", encoding="utf-8") as f:
                f.write(mermaid_content)

        self.write_script(abs_path)

        if execute:
            self.execute_script(abs_path)

    @classmethod
    def from_template_dir(cls, dirpath: str, define_func_paths: list[str], transform_func_paths: list[str]):
        templates = {}
        depends_graph = {}

        # read all YAML files in the directory
        for filepath in glob.glob(os.path.join(dirpath, "*.yaml")):
            with open(filepath, "r") as f:
                cfg = yaml.safe_load(f) or {}
            name = os.path.basename(filepath)
            templates[name] = cfg
            depends_graph[name] = cfg.get("depends", [])

        # topological sort to resolve dependencies
        resolved = []
        visited = set()
        temp = set()

        def visit(name):
            if name in visited:
                return
            if name in temp:
                raise ValueError(f"Circular dependency detected at {name}")
            temp.add(name)
            for dep in depends_graph.get(name, []):
                visit(dep)
            temp.remove(name)
            visited.add(name)
            resolved.append(name)

        for name in templates:
            visit(name)

        merged_config = {
            "processes": [],
        }

        for name in resolved:
            merged_config["processes"].extend(templates[name].get("processes", []))

        return cls(
            config=merged_config,
            base_path=dirpath,
            define_func_paths=define_func_paths,
            transform_func_paths=transform_func_paths,
        )
