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
    def __init__(self, template: dict, params: object):
        self.template = template
        self.params = params

    def _get_common_import_lines(self):
        return [
            "import pandas as pd",
            "import os",
            "import importlib.util",
            "from rotab.core.operation.new_columns_funcs import *",
            "from rotab.core.operation.dataframes_funcs import *",
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
            local_vars = set()

            for step in process.get("steps", []):
                step_name = step.get("name")
                if not step_name:
                    raise ValueError("Each step must have a 'name' field.")

                indent_depth = 1
                if "with" in step:
                    var = step["with"]
                    local_vars.add(var)
                    lines = [f"def step_{step_name}_{process_name}({var}):"]
                    body = [textwrap.indent(f'"""Step: {step_name} """', INDENT)]

                    if "when" in step:
                        condition = step["when"]
                        body.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                        indent_depth += 1

                    if "as" in step:
                        var_result = step["as"]
                        body.append(textwrap.indent(f"{var_result} = {var}.copy()", INDENT * indent_depth))
                    else:
                        var_result = var

                    if "filter" in step:
                        body.append(
                            textwrap.indent(f"{var_result} = {var}.query('{step['filter']}')", INDENT * indent_depth)
                        )
                    if "new_columns" in step:
                        for line in step["new_columns"].split("\n"):
                            if line.strip():
                                lhs, rhs = map(str.strip, line.split("=", 1))
                                try:
                                    rhs_transformed = VariableToRowTransformer.transform(rhs)
                                except Exception as e:
                                    raise ValueError(f"Failed to parse RHS expression: {rhs!r}\n{e}")

                                body.append(
                                    textwrap.indent(
                                        f'{var_result}["{lhs}"] = {var}.apply(lambda row: {rhs_transformed}, axis=1)',
                                        INDENT * indent_depth,
                                    )
                                )

                    if "columns" in step and "new_columns" not in step:
                        cols = step["columns"]
                        body.append(textwrap.indent(f"{var_result} = {var}[{cols}]", INDENT * indent_depth))

                    body.append(textwrap.indent(f"return {var_result}", INDENT * indent_depth))

                    lines.extend([line for line in body])
                    step_funcs.append("\n".join(lines))
                    step_funcs.extend(["", ""])
                    call_line = f"{var_result} = step_{step_name}_{process_name}({var})"
                    step_calls.append(textwrap.indent(call_line, INDENT))

                elif "dataframes" in step:
                    assign = step["dataframes"]
                    lhs, rhs = map(str.strip, assign.split("=", 1))
                    try:
                        parsed = ast.parse(rhs, mode="eval")
                        arg_names = sorted({node.id for node in ast.walk(parsed) if isinstance(node, ast.Name)})
                    except Exception:
                        raise ValueError(f"Cannot parse dataframes expression: {assign}")

                    lines = [
                        f"def step_{step_name}_{process_name}({', '.join(arg_names)}):",
                        textwrap.indent(f'"""Step: {step_name}"""', INDENT),
                    ]

                    if "when" in step:
                        lines.append(textwrap.indent(f"if {step['when']}:", INDENT * indent_depth))
                        indent_depth += 1

                    lines.append(textwrap.indent(f"return {rhs}", INDENT * indent_depth))

                    step_funcs.append("\n".join(lines))
                    step_funcs.append("")
                    call_line = f"{lhs} = step_{step_name}_{process_name}({', '.join(arg_names)})"
                    step_calls.append(textwrap.indent(call_line, INDENT))

            func_lines = ["", "", f"def process_{process_name}():"]
            if description:
                func_lines.append(textwrap.indent(f'"""{description}"""', INDENT))

            func_lines.append(textwrap.indent("# load tables", INDENT))
            for table in process.get("tables", []):
                indent_depth = 1
                if "when" in table:
                    condition = table["when"]
                    func_lines.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                    indent_depth += 1
                name = table["name"]
                path = table["path"]
                # Generate dtype mapping from schema
                dtype_map = {}
                if self.params.get("schema", {}).get(name, {}).get("columns", None):
                    for col in self.params["schema"][name]["columns"]:
                        dtype = col.get("dtype")
                        dtype_map[col["name"]] = dtype
                        # Extend as needed for float, bool, etc.
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
            for dump in process.get("dumps", []):
                indent_depth = 1
                if "when" in dump:
                    condition = dump["when"]
                    func_lines.append(textwrap.indent(f"if {condition}:", INDENT * indent_depth))
                    indent_depth += 1
                dfname, path = dump["return"], dump["path"]
                schema = dump.get("schema")

                dump_code = [
                    textwrap.indent(f"path = os.path.abspath(r'{path}')", INDENT * indent_depth),
                    textwrap.indent(f"os.makedirs(os.path.dirname(path), exist_ok=True)", INDENT * indent_depth),
                ]

                if schema:
                    schema_str = json.dumps(schema, indent=4)
                    dump_code.append(
                        textwrap.indent(f"validate_table_schema({dfname}, columns={schema_str})", INDENT * indent_depth)
                    )
                dump_code.append(textwrap.indent(f"{dfname}.to_csv(path, index=False)", INDENT * indent_depth))

                func_lines.append("\n".join(dump_code))
            process_funcs.append("\n".join(func_lines))
            main_calls.append(f"process_{process_name}()")

        custom_imports = [
            "spec = importlib.util.spec_from_file_location('new_columns_funcs', r'/home/yutaitatsu/rotab/custom_functions/new_columns_funcs.py')",
            "custom_new_columns_funcs = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(custom_new_columns_funcs)",
            "",
            "spec = importlib.util.spec_from_file_location('dataframes_funcs', r'/home/yutaitatsu/rotab/custom_functions/dataframes_funcs.py')",
            "custom_dataframes_funcs = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(custom_dataframes_funcs)",
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
