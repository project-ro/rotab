import os
import ast
import json
import textwrap

INDENT = "    "


import ast


class VariableToRowTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self._function_names = set()

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self._function_names.add(node.func.id)
        self.generic_visit(node)
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self._function_names:
            return ast.Subscript(
                value=ast.Name(id="row", ctx=ast.Load()), slice=ast.Constant(value=node.id), ctx=ast.Load()
            )
        return node

    @staticmethod
    def transform(rhs: str) -> str:
        tree = ast.parse(rhs, mode="eval")
        transformer = VariableToRowTransformer()
        tree = transformer.visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)


class ScriptGenerator:
    def __init__(self, template: dict, schemas: object):
        self.template = template
        self.schemas = schemas

    def _get_common_import_lines(self):
        return [
            "import pandas as pd",
            "import os",
            "import importlib.util",
            "from rotab.core.operation.derive_funcs import *",
            "from rotab.core.operation.transform_funcs import *",
        ]

    def generate_script(self) -> str:
        import_lines = self._get_common_import_lines()
        step_funcs = ["", "# STEPS FUNCTIONS:", "", ""]
        process_funcs = ["", "# PROCESSES FUNCTIONS:"]
        main_calls = []

        for i, process in enumerate(self.template.get("processes", [])):
            process_name = process.get("name", f"block_{i}")
            description = process.get("description", "")
            step_calls = []

            for step in process.get("steps", []):
                step_name = step.get("name")
                if not step_name:
                    raise ValueError("Each step must have a 'name' field.")
                if "with" not in step:
                    raise ValueError(f"Step '{step_name}' must have a 'with' field.")
                if "mutate" not in step and "transform" not in step:
                    raise ValueError(f"Step '{step_name}' must have a 'mutate' or 'transform' field.")
                if "as" not in step:
                    raise ValueError(f"Step '{step_name}' must have an 'as' field.")

                indent_depth = 1

                var = step["with"]
                if isinstance(var, str):
                    var_string = var.strip()
                elif isinstance(var, list):
                    var_string = " ,".join([v.strip() for v in var if v.strip()])

                lines = [f"def step_{step_name}_{process_name}({var_string}):"]
                body = [textwrap.indent(f'"""Step: {step_name} """', INDENT)]

                var_result = step["as"]
                body.append(textwrap.indent(f"{var_result} = {var_string}.copy()", INDENT * indent_depth))

                if "when" in step:
                    condition = step["when"]
                    body.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                    indent_depth += 1

                if "mutate" in step:
                    for sub in step["mutate"]:
                        if "filter" in sub:
                            body.append(
                                textwrap.indent(
                                    f"{var_result} = {var_result}.query('{sub['filter']}')",
                                    INDENT * indent_depth,
                                )
                            )
                        if "derive" in sub:
                            for line in sub["derive"].split("\n"):
                                if line.strip():
                                    lhs, rhs = map(str.strip, line.split("=", 1))
                                    try:
                                        rhs_transformed = VariableToRowTransformer.transform(rhs)
                                    except Exception as e:
                                        raise ValueError(f"Failed to parse RHS expression: {rhs!r}\n{e}")

                                    body.append(
                                        textwrap.indent(
                                            f'{var_result}["{lhs}"] = {var_result}.apply(lambda row: {rhs_transformed}, axis=1)',
                                            INDENT * indent_depth,
                                        )
                                    )
                        if "select" in sub:
                            cols = sub["select"]
                            body.append(textwrap.indent(f"{var_result} = {var_result}[{cols}]", INDENT * indent_depth))

                elif "transform" in step:
                    body.append(
                        textwrap.indent(
                            f"{var_result} = {step['transform']}",
                            INDENT * indent_depth,
                        )
                    )

                body.append(textwrap.indent(f"return {var_result}", INDENT * indent_depth))

                lines.extend([line for line in body])
                step_funcs.append("\n".join(lines))
                step_funcs.extend(["", ""])
                call_line = f"{var_result} = step_{step_name}_{process_name}({var_string})"
                step_calls.append(textwrap.indent(call_line, INDENT))

            func_lines = ["", "", f"def process_{process_name}():"]
            if description:
                func_lines.append(textwrap.indent(f'"""{description}"""', INDENT))

            # io

            func_lines.append(textwrap.indent("# load tables", INDENT))
            for table in process["io"]["inputs"]:
                indent_depth = 1
                if "when" in table:
                    condition = table["when"]
                    func_lines.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                    indent_depth += 1
                name = table["name"]
                path = table["path"]

                # Generate dtype mapping from schema
                dtype_map = {col["name"]: col.get("dtype") for col in self.schemas.get(name, {}).get("columns", [])}
                if dtype_map:
                    dtype_str = "{" + ", ".join(f"'{k}': {v}" for k, v in dtype_map.items()) + "}"
                    func_lines.append(
                        textwrap.indent(f"{name} = pd.read_csv(r'{path}', dtype={dtype_str})", INDENT * indent_depth)
                    )
                else:
                    func_lines.append(textwrap.indent(f"{name} = pd.read_csv(r'{path}')", INDENT * indent_depth))

            func_lines.append(textwrap.indent("# process steps", INDENT))
            func_lines.extend(step_calls)

            func_lines.append(textwrap.indent("# dump output", INDENT))

            for dump in process["io"]["outputs"]:
                indent_depth = 1
                if "when" in dump:
                    condition = dump["when"]
                    func_lines.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                    indent_depth += 1
                dfname, path = dump["name"], dump["path"]

                schema_name = dump.get("schema")
                schema = self.schemas.get(schema_name, {}).get("columns", []) if schema_name else None

                dump_code = [
                    textwrap.indent(f"path = os.path.abspath(r'{path}')", INDENT * indent_depth),
                    textwrap.indent(f"os.makedirs(os.path.dirname(path), exist_ok=True)", INDENT * indent_depth),
                ]

                if schema:
                    schema_str = json.dumps(schema, indent=4)
                    for col_def in schema:
                        col = col_def.get("name")
                        expected_type = col_def.get("dtype")
                        if expected_type == "string" or expected_type == "str":
                            dump_code.append(
                                textwrap.indent(
                                    f'{dfname}["{col}"] = {dfname}["{col}"].astype(str)', INDENT * indent_depth
                                )
                            )

                    dump_code.append(
                        textwrap.indent(f"validate_table_schema({dfname}, columns={schema_str})", INDENT * indent_depth)
                    )
                dump_code.append(textwrap.indent(f"{dfname}.to_csv(path, index=False)", INDENT * indent_depth))

                func_lines.append("\n".join(dump_code))
            process_funcs.append("\n".join(func_lines))
            main_calls.append(f"process_{process_name}()")

        custom_imports = [
            "spec = importlib.util.spec_from_file_location('derive_funcs', r'/home/yutaitatsu/rotab/custom_functions/derive_funcs.py')",
            "custom_derive_funcs = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(custom_derive_funcs)",
            "",
            "spec = importlib.util.spec_from_file_location('transform_funcs', r'/home/yutaitatsu/rotab/custom_functions/transform_funcs.py')",
            "custom_transform_funcs = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(custom_transform_funcs)",
        ]

        script_lines = (
            import_lines
            + ["", ""]
            + custom_imports
            + ["", ""]
            + step_funcs
            + process_funcs
            + ["", "", "if __name__ == '__main__':"]
            + [textwrap.indent(call, INDENT) for call in main_calls]
        )
        return "\n".join(script_lines)
