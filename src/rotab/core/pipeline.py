import os
import ast
from typing import Dict, Any
import importlib.util
import subprocess
import rotab.core.operation.define_funcs as define_funcs
import rotab.core.operation.transform_funcs as transform_funcs
from rotab.core.operation.validator import TemplateValidator


eval_scope = {}
for module in [define_funcs, transform_funcs]:
    eval_scope.update({k: v for k, v in module.__dict__.items() if not k.startswith("__") and callable(v)})
eval_scope.update(__builtins__)


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
    def __init__(self, config: Dict[str, Any], base_path: str):
        self.config = config
        self.base_path = base_path
        self.eval_scope = dict(eval_scope)
        self._validate_config()

    def _validate_config(self):
        validator = TemplateValidator(self.config)
        validator.validate()
        if validator.errors:
            raise ValueError("Invalid config:\n" + "\n".join(str(e) for e in validator.errors))

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

    def _load_functions_from_paths(self, paths):
        for path in paths:
            module_name = os.path.splitext(os.path.basename(path))[0]
            abs_path = os.path.abspath(os.path.join(self.base_path, path))
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot import from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.eval_scope.update({k: v for k, v in module.__dict__.items() if callable(v) and not k.startswith("__")})

    def _generate_custom_function_imports(self, define_paths, transform_paths) -> list[str]:
        import_lines = ["import importlib.util"]
        all_paths = define_paths + transform_paths
        for path in all_paths:
            abs_path = os.path.abspath(os.path.join(self.base_path, path))
            module_name = os.path.splitext(os.path.basename(path))[0]
            import_lines.append(f"spec = importlib.util.spec_from_file_location('{module_name}', r'{abs_path}')")
            import_lines.append(f"{module_name} = importlib.util.module_from_spec(spec)")
            import_lines.append(f"spec.loader.exec_module({module_name})")
            import_lines.append(
                f"globals().update({{k: v for k, v in {module_name}.__dict__.items() if callable(v) and not k.startswith('__')}})"
            )
        return import_lines

    def run(self):
        code_lines = [
            "import pandas as pd",
            "from rotab.core.operation.define_funcs import *",
            "from rotab.core.operation.transform_funcs import *",
        ]

        custom_conf = self.config.get("custom_functions", {})
        define_paths = custom_conf.get("define_funcs", [])
        transform_paths = custom_conf.get("transform_funcs", [])

        code_lines.extend(self._generate_custom_function_imports(define_paths, transform_paths))
        self._load_functions_from_paths(define_paths)
        self._load_functions_from_paths(transform_paths)

        processes = self.config.get("processes") or [self.config]

        code_lines.append("")
        for i, process in enumerate(processes):
            description = process.get("description", "")
            if description:
                code_lines.append(f"# [PROCESS_{i+1}] {description}")
                code_lines.append("")

            code_lines.append("# load tables")
            tables = process.get("tables", [])
            for table in tables:
                name = table["name"]
                rel_path = table["path"]
                abs_path = os.path.abspath(os.path.join(self.base_path, rel_path))
                code_lines.append(f"{name} = pd.read_csv(r'{abs_path}')")

            code_lines.append("")
            code_lines.append("# process steps")
            for j, step in enumerate(process.get("steps", [])):
                if j > 0:
                    code_lines.append("")

                if "with" in step:
                    name = step["with"]
                    if "filter" in step:
                        code_lines.append(f"{name} = {name}.query('{step['filter']}')")

                    if "define" in step:
                        for line in step["define"].split("\n"):
                            if line.strip():
                                parts = line.strip().split("=", 1)
                                if len(parts) != 2:
                                    raise ValueError(f"Invalid define syntax: {line.strip()}")
                                lhs, rhs = parts[0].strip(), parts[1].strip()
                                lambda_expr = self._to_lambda_expr(rhs)
                                code_lines.append(f"{name}['{lhs}'] = {name}.apply({lambda_expr}, axis=1)")

                    if "select" in step:
                        cols = step["select"]
                        code_lines.append(f"{name} = {name}[{cols}]")

                elif "transform" in step:
                    code_lines.append(f"{step['transform']}")

            code_lines.append("")
            code_lines.append("# dump output")

            if any(p.get("dumps") for p in self.config.get("processes", [])):
                code_lines.append("import os")

            for dump in process.get("dumps", []):
                dfname = dump["return"]
                path = dump["path"]
                code_lines.append(f"path = os.path.abspath(r'{path}')")
                code_lines.append(f"os.makedirs(os.path.dirname(path), exist_ok=True)")
                code_lines.append(f"{dfname}.to_csv(path, index=False)")

        script_path = os.path.abspath(
            os.path.join(self.base_path, self.config.get("settings", {}).get("generate", {}).get("path", "script.py"))
        )
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as f:
            f.write("\n".join(code_lines))

        if self.config.get("settings", {}).get("generate", {}).get("execute", False):
            generate_path = self.config["settings"]["generate"]["path"]
            subprocess.run(["python", generate_path], cwd=self.base_path)

    @classmethod
    def from_template_file(cls, filepath: str):
        import yaml

        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        base_path = os.path.dirname(os.path.abspath(filepath))
        return cls(config=config, base_path=base_path)
