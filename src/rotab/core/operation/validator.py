from typing import Any, Dict, List
import ast
import builtins
import re
import os
import inspect
import keyword
import importlib.util
import rotab.core.operation.derive_funcs as derive_funcs
import rotab.core.operation.transform_funcs as transform_funcs


class ValidationError:
    def __init__(self, path: str, message: str, suggestion: str = ""):
        self.path = path
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        return f"[{self.path}] {self.message}" + (f"\n  → Suggestion: {self.suggestion}" if self.suggestion else "")


class TemplateValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.errors: List[ValidationError] = []
        self.allowed_top_keys = {"processes", "depends"}
        self.required_keys = {"name", "tables", "steps", "dumps"}
        self.optional_keys = {"description"}
        self.eval_scope = self._build_eval_scope()
        self.seen_table_names = set()
        self.seen_dump_names = set()

        print(f"eval_scope: {self.eval_scope.keys()}")  # Debugging line to check eval_scope contents

    def _load_functions_from_paths(self, paths: List[str]) -> Dict[str, Any]:
        scope = {}
        for path in paths:
            if not os.path.isfile(path):
                continue
            module_name = os.path.splitext(os.path.basename(path))[0]
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    scope.update({k: v for k, v in mod.__dict__.items() if callable(v) and not k.startswith("__")})
                except Exception as e:
                    self.errors.append(ValidationError(f"custom_functions.{path}", f"Failed to load module: {e}"))
        return scope

    def _build_eval_scope(self) -> Dict[str, Any]:
        scope = {}
        for module in [derive_funcs, transform_funcs]:
            scope.update({k: v for k, v in module.__dict__.items() if not k.startswith("__") and callable(v)})
        scope.update({k: v for k, v in builtins.__dict__.items() if callable(v)})

        # Load custom functions from specified paths
        custom_conf = self.config.get("custom_functions", {})
        derive_paths = custom_conf.get("derive_funcs", [])
        transform_paths = custom_conf.get("transform_funcs", [])
        if derive_paths:
            scope.update(self._load_functions_from_paths(derive_paths))
        if transform_paths:
            scope.update(self._load_functions_from_paths(transform_paths))
        return scope

    def validate(self):
        print("Validateの最初の文字")  # Debugging line

        for key in self.config:
            if key not in self.allowed_top_keys:
                self.errors.append(
                    ValidationError(
                        f"config.{key}",
                        f"Unexpected top-level key: `{key}`. Only {sorted(self.allowed_top_keys)} are allowed.",
                    )
                )
        if "processes" not in self.config or not isinstance(self.config["processes"], list):
            self.errors.append(ValidationError("config", "`processes` must be a list."))
            return
        for i, proc in enumerate(self.config["processes"]):
            self._validate_process(proc, f"processes[{i}]")

    def validate_depends(self):
        print("Validating `depends`...")  # Debugging line
        if "depends" in self.config and not isinstance(self.config["depends"], list):
            self.errors.append(ValidationError("config.depends", "`depends` must be a list."))
            return
        if "depends" in self.config:
            for path in self.config["depends"]:
                if not isinstance(path, str):
                    self.errors.append(ValidationError(f"config.depends", "`depends` values must be a string."))
                    continue

    def _validate_process(self, proc: Dict[str, Any], path: str):
        print(f"Validating process at {path}...")  # Debugging line

        seen_table_names = set()

        missing_keys = self.required_keys - proc.keys()
        for key in missing_keys:
            self.errors.append(ValidationError(path, f"Missing required key: `{key}`"))

        unknown_keys = proc.keys() - (self.required_keys | self.optional_keys)
        for key in unknown_keys:
            self.errors.append(ValidationError(f"{path}.{key}", "Unexpected key."))

        if "name" in proc and not isinstance(proc["name"], str):
            self.errors.append(ValidationError(f"{path}.name", "`name` must be a string."))

        if "tables" in proc:
            self._validate_tables(proc["tables"], f"{path}.tables", seen_table_names)
        if "steps" in proc:
            table_names = {t["name"] for t in proc.get("tables", []) if isinstance(t, dict) and "name" in t}
            self._validate_steps(proc["steps"], table_names, f"{path}.steps")
        if "dumps" in proc:
            self._validate_dumps(proc["dumps"], proc, f"{path}.dumps")

    def _validate_tables(self, tables: Any, path: str, seen_names: set):
        if not isinstance(tables, list):
            self.errors.append(ValidationError(path, "`tables` must be a list."))
            return
        for i, table in enumerate(tables):
            if not isinstance(table, dict):
                self.errors.append(ValidationError(f"{path}[{i}]", "Each table must be a dict."))
                continue
            if set(table.keys()) != {"name", "path"}:
                self.errors.append(ValidationError(f"{path}[{i}]", "Each table must have `name` and `path`."))
                continue
            name, pth = table.get("name"), table.get("path")
            if not isinstance(name, str):
                self.errors.append(ValidationError(f"{path}[{i}].name", "`name` must be a string."))
            if not isinstance(pth, str):
                self.errors.append(ValidationError(f"{path}[{i}].path", "`path` must be a string."))
            elif re.search(r"[<>:\"|?*]", pth):
                self.errors.append(ValidationError(f"{path}[{i}].path", "Invalid characters in path."))
            if isinstance(name, str):
                if name in seen_names:
                    self.errors.append(ValidationError(f"{path}[{i}].name", f"Duplicate table name `{name}`"))
                else:
                    seen_names.add(name)

    def _validate_steps(self, steps: Any, table_names: set, path: str):
        print(f"Validating steps at {path}...")  # Debugging line
        if not isinstance(steps, list):
            self.errors.append(ValidationError(path, "`steps` must be a list."))
            return

        defined_vars = set(table_names)
        for i, step in enumerate(steps):
            print(f"Validating step {i} at {path}[{i}]...")  # Debugging line
            p = f"{path}[{i}]"
            if not isinstance(step, dict):
                self.errors.append(ValidationError(p, "Each step must be a dict."))
                continue

            self._validate_with(step, defined_vars, p)
            self._validate_as(step, defined_vars, p)
            self._validate_mutate_and_transform_exclusive(step, p)

            if "mutate" in step:
                self._validate_mutate(step["mutate"], defined_vars, p)

            if "transform" in step:
                self._validate_transform(step["transform"], defined_vars, p)

    def _validate_with(self, step: dict, defined_vars: set, path: str):
        if "with" not in step:
            self.errors.append(ValidationError(path, "`with` must be specified in each step."))
            return

        val = step["with"]
        if isinstance(val, list):
            for i, v in enumerate(val):
                if not isinstance(v, str):
                    self.errors.append(ValidationError(f"{path}.with[{i}]", "`with` list must contain strings only."))
                elif v not in defined_vars:
                    self.errors.append(ValidationError(f"{path}.with[{i}]", f"`{v}` is not a defined variable."))
        elif isinstance(val, str):
            if val not in defined_vars:
                self.errors.append(ValidationError(f"{path}.with", f"`{val}` is not a defined variable."))
        else:
            self.errors.append(ValidationError(f"{path}.with", "`with` must be a string or list of strings."))

    def _validate_as(self, step: dict, defined_vars: set, path: str):
        if "as" in step:
            if not isinstance(step["as"], str):
                self.errors.append(ValidationError(f"{path}.as", "`as` must be a string."))
            else:
                defined_vars.add(step["as"])

    def _validate_mutate_and_transform_exclusive(self, step: dict, path: str):
        if "mutate" in step and "transform" in step:
            self.errors.append(ValidationError(path, "Cannot use both `mutate` and `transform` in the same step."))

    def _validate_mutate(self, mutate: list, defined_vars: set, path: str):
        if not isinstance(mutate, list):
            self.errors.append(ValidationError(path, "`mutate` must be a list."))
            return

        for i, block in enumerate(mutate):
            if not isinstance(block, dict) or len(block) != 1:
                self.errors.append(ValidationError(f"{path}[{i}]", "Each mutate block must be a single-key dict."))
                continue

            key, value = next(iter(block.items()))
            sub_path = f"{path}[{i}].{key}"

            if key == "derive":
                self._validate_derive(value, defined_vars, sub_path)
            elif key == "filter":
                self._validate_filter(value, sub_path)
            elif key == "select":
                self._validate_select(value, defined_vars, sub_path)
            else:
                self.errors.append(ValidationError(sub_path, f"Unknown mutate operation: `{key}`"))

    def _validate_derive(self, derive_block: str, defined_vars: set, path: str):
        print(f"Validating derive at {path}...")  # Debugging line
        # まず構文エラーをチェック
        before_error_count = len(self.errors)
        self._validate_derive_syntax(derive_block, path)
        after_error_count = len(self.errors)

        # 構文エラーが発生していれば意味チェックはスキップ
        if after_error_count > before_error_count:
            print(f"構文エラーが発生しました: {after_error_count - before_error_count}件")
            return

        print("構文エラーは発生していない")

        # 構文が正しければ意味チェック（関数呼び出しの内容）を実施
        for line in derive_block.splitlines():
            line = line.strip()

            lhs, rhs = map(str.strip, line.split("=", 1))
            expr = ast.parse(rhs, mode="eval")

            if isinstance(expr.body, ast.Call):
                func = self._get_func_name(expr.body.func)
                if not func:
                    self.errors.append(ValidationError(f"{path}.{lhs}", "Unsupported function call format."))
                elif func in {"eval", "exec", "compile", "open", "__import__"}:
                    self.errors.append(ValidationError(f"{path}.{lhs}", f"Forbidden function `{func}` used."))
                elif func not in self.eval_scope:
                    self.errors.append(ValidationError(f"{path}.{lhs}", f"Function `{func}` not found."))
                else:
                    self._check_function_signature(func, expr.body.args, expr.body.keywords, f"{path}.{lhs}")

            # 意味検査通過後に定義変数へ追加
            defined_vars.add(lhs)

    def _validate_derive_syntax(self, derive: str, path: str):
        for lineno, line in enumerate(derive.splitlines(), 1):
            stripped = line.strip()
            if not stripped:
                continue

            # = が最低1個、かつ == などの比較演算子は対象外とする
            if stripped.count("=") < 1 or re.match(r"^[^=]+==[^=]+$", stripped):
                self.errors.append(
                    ValidationError(f"{path}.line{lineno}", f"Missing or malformed '=' in line: {stripped!r}")
                )
                continue

            try:
                lhs, rhs = map(str.strip, stripped.split("=", 1))
            except ValueError:
                self.errors.append(ValidationError(f"{path}.line{lineno}", f"Malformed assignment: {stripped!r}"))
                continue

            if not re.match(r"^[a-zA-Z_]\w*$", lhs):
                self.errors.append(ValidationError(f"{path}.{lhs}", f"Invalid variable name on LHS: {lhs!r}"))
                continue

            try:
                # RHS must be a valid expression
                tree = ast.parse(rhs, mode="eval")
                if not isinstance(tree.body, ast.expr):
                    self.errors.append(ValidationError(f"{path}.{lhs}", f"Invalid expression on RHS: {rhs!r}"))
            except Exception as e:
                self.errors.append(ValidationError(f"{path}.{lhs}", f"Syntax error in RHS: {rhs!r}"))

    def _validate_filter(self, expr: str, path: str):
        try:
            tree = ast.parse(expr, mode="eval")
            if not isinstance(tree.body, (ast.Compare, ast.BoolOp, ast.Call, ast.Name, ast.UnaryOp, ast.BinOp)):
                raise SyntaxError("filter expression must be a boolean condition.")
        except SyntaxError:
            self.errors.append(ValidationError(path, "Invalid filter expression."))

    def _validate_select(self, select: list, defined_vars: set, path: str):
        if not isinstance(select, list):
            self.errors.append(ValidationError(path, "`select` must be a list."))
            return
        for i, var in enumerate(select):
            if not isinstance(var, str):
                self.errors.append(ValidationError(f"{path}[{i}]", "Each select item must be a string."))
            elif var not in defined_vars:
                self.errors.append(ValidationError(f"{path}[{i}]", f"'{var}' is not a defined variable."))

    def _validate_transform(self, transform_expr: str, defined_vars: set, path: str):
        try:
            tree = ast.parse(transform_expr, mode="eval")

            if any(isinstance(node, ast.Assign) for node in ast.walk(tree)):
                self.errors.append(ValidationError(f"{path}.transform", "Transform must not contain assignment."))
                return

            self._validate_transform_syntax(transform_expr, allowed_args=defined_vars, path=path)

            expr = tree
            func = self._get_func_name(expr.body.func)
            if not func:
                self.errors.append(ValidationError(f"{path}.transform", "Unsupported function call format."))
            elif func in {"eval", "exec", "compile", "open", "__import__"}:
                self.errors.append(ValidationError(f"{path}.transform", f"Forbidden function `{func}` used."))
            elif func not in self.eval_scope:
                self.errors.append(ValidationError(f"{path}.transform", f"Function `{func}` not found."))
            else:
                self._check_function_signature(func, expr.body.args, expr.body.keywords, f"{path}.transform")

        except Exception:
            self.errors.append(ValidationError(f"{path}.transform", f"Invalid function call: {transform_expr}"))

    def _validate_transform_syntax(self, transform_expr: str, allowed_args: set, path: str):
        try:
            tree = ast.parse(transform_expr, mode="eval")
            if not isinstance(tree.body, ast.Call):
                raise SyntaxError("Transform must be a function call.")
            for kw in tree.body.keywords:
                if not isinstance(kw.value, ast.Name):
                    continue
                varname = kw.value.id
                if varname not in allowed_args:
                    self.errors.append(ValidationError(path, f"Transform uses undeclared variable: '{varname}'"))
        except SyntaxError:
            self.errors.append(ValidationError(path, "Invalid transform function call."))

    def _get_func_name(self, func_expr):
        if isinstance(func_expr, ast.Name):
            return func_expr.id
        return None

    def _check_function_signature(self, func_name, args, keywords, path):

        print(
            f"Checking function signature for: {func_name} with args: {args} and keywords: {keywords}"
        )  # Debugging line

        func = self.eval_scope[func_name]

        print(f"Function found: {func}")  # Debugging line

        try:
            sig = inspect.signature(func)
            mock_args = [self._mock_value(arg) for arg in args]
            mock_kwargs = {kw.arg: self._mock_value(kw.value) for kw in keywords if kw.arg}
            _ = sig.bind(*mock_args, **mock_kwargs)

        except TypeError as e:
            self.errors.append(
                ValidationError(
                    path=path,
                    message=f"Function `{func_name}` called with incorrect arguments: {str(e)}",
                    suggestion=f"Check the signature of `{func_name}`.",
                )
            )

    def _mock_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.List):
            return []
        elif isinstance(node, ast.Dict):
            return {}
        elif isinstance(node, ast.NameConstant):
            return node.value
        elif isinstance(node, ast.Name):
            return 0
        return 0

    def _validate_dumps(self, dumps: Any, proc: Dict[str, Any], path: str):
        if not isinstance(dumps, list):
            self.errors.append(ValidationError(path, "`dumps` must be a list."))
            return
        valid_returns = {t["name"] for t in proc.get("tables", []) if isinstance(t, dict)}
        for step in proc.get("steps", []):
            if "transform" in step:
                lhs = step["transform"].split("=", 1)[0].strip()
                valid_returns.add(lhs)
        for i, dump in enumerate(dumps):
            p = f"{path}[{i}]"
            if not isinstance(dump, dict) or set(dump.keys()) != {"output", "path"}:
                self.errors.append(ValidationError(p, "`dumps` must have `output` and `path`."))
                continue
            if dump["output"] not in valid_returns:
                self.errors.append(
                    ValidationError(
                        f"{p}.output", f"`output` must be in defined tables or transform results: `{dump['output']}`"
                    )
                )
            if not isinstance(dump["path"], str):
                self.errors.append(ValidationError(f"{p}.path", "`path` must be a string."))
            elif re.search(r"[<>:\"|?*]", dump["path"]):
                self.errors.append(ValidationError(f"{p}.path", "Invalid characters in path."))
            if dump["output"] in self.seen_dump_names:
                self.errors.append(ValidationError(f"{p}.output", f"Duplicate dump output `{dump['output']}`"))
            else:
                self.seen_dump_names.add(dump["output"])

    def report(self):
        if self.errors:
            error_string = "\n".join(str(error) for error in self.errors)
            raise ValueError(f"Invalid template. See validation errors below.\n{error_string}")
