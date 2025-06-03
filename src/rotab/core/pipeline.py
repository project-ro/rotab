import os
import subprocess
import ast
import glob
import yaml
from typing import Dict, Any
from rotab.core.operation import define_funcs, transform_funcs


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
    def __init__(self, config: Dict[str, Any], base_path: str, define_func_paths: list[str], transform_func_paths: list[str]):
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
            "import os"
        ]
        for path in self.define_func_paths + self.transform_func_paths:
            abs_path = os.path.abspath(path)
            module_name = os.path.splitext(os.path.basename(path))[0]
            lines.extend([
                f"spec = importlib.util.spec_from_file_location('{module_name}', r'{abs_path}')",
                f"{module_name} = importlib.util.module_from_spec(spec)",
                f"spec.loader.exec_module({module_name})",
                f"globals().update({{k: v for k, v in {module_name}.__dict__.items() if callable(v) and not k.startswith('__')}})"
            ])
        return lines

    def generate_script(self) -> str:
        import_lines = self._get_common_import_lines()
        function_blocks = []
        main_calls = []
        for i, process in enumerate(self.config.get("processes", [])):
            process_name = process.get("name", f"block_{i}")
            func_lines = [""]
            func_lines.append(f"def process_{process_name}():")
            description = process.get("description", "")
            if description:
                func_lines.append(f'    """{description}"""')

            func_lines.append("    # load tables")
            for table in process.get("tables", []):
                name, rel_path = table["name"], table["path"]
                abs_path = os.path.abspath(os.path.join(self.base_path, rel_path))
                func_lines.append(f"    {name} = pd.read_csv(r'{abs_path}')")

            func_lines.append("\n    # process steps")
            step_calls = []
            for step in process.get("steps", []):
                step_name = step.get("name")
                func_lines.append("")
                if not step_name:
                    raise ValueError("Each step must have a 'name' field.")
                if 'with' in step:
                    name = step['with']
                    func_lines.append(f"    def step_{step_name}({name}):")
                else:
                    func_lines.append(f"    def step_{step_name}():")
                func_lines.append(f'        """Step: {step_name}"""')
                if "with" in step:
                    name = step["with"]
                    if "filter" in step:
                        func_lines.append(f"        {name} = {name}.query('{step['filter']}').copy()")
                    if "define" in step:
                        for line in step["define"].split("\n"):
                            if line.strip():
                                lhs, rhs = map(str.strip, line.split("=", 1))
                                lambda_expr = self._to_lambda_expr(rhs)
                                func_lines.append(f"        {name}.loc[:, '{lhs}'] = {name}.apply({lambda_expr}, axis=1)")
                    if "select" in step:
                        cols = step["select"]
                        func_lines.append(f"        {name} = {name}[{cols}]")
                    func_lines.append(f"        return {name}")
                    if 'with' in step:
                        step_calls.append(f"    {name} = step_{step_name}({name})")
                elif "transform" in step:
                    lhs = step['transform'].split('=')[0].strip()
                    func_lines.append(f"        result = {step['transform']}")
                    func_lines.append(f"        return result")
                    step_calls.append(f"    {lhs} = step_{step_name}()")
            func_lines.append("")    
            func_lines.extend(step_calls)
            func_lines.append("\n    # dump output")
            for dump in process.get("dumps", []):
                dfname, path = dump["return"], dump["path"]
                func_lines.extend([
                    f"    path = os.path.abspath(r'{path}')",
                    "    os.makedirs(os.path.dirname(path), exist_ok=True)",
                    f"    {dfname}.to_csv(path, index=False)"
                ])
            function_blocks.append("\n".join(func_lines))
            main_calls.append(f"process_{process_name}()")

        script_lines = import_lines + [""] + function_blocks + ["", "if __name__ == '__main__':"] + [f"    {call}" for call in main_calls]
        return "\n".join(script_lines)

    def write_script(self, path: str):
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(self.generate_script())

    def execute_script(self, path: str):
        subprocess.run(["python", path], cwd=self.base_path)

    def run(self, script_path: str, execute: bool):
        abs_path = (
            script_path if os.path.isabs(script_path)
            else os.path.abspath(os.path.join(self.base_path, script_path))
        )
        self.write_script(abs_path)
        if execute:
            self.execute_script(abs_path)

    @classmethod
    def from_template_dir(
        cls,
        dirpath: str,
        define_func_paths: list[str],
        transform_func_paths: list[str]
        ):
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
            transform_func_paths=transform_func_paths
        )