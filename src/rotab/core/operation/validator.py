from typing import Any, Dict, List
import ast
import builtins
import re
import os
import inspect
import keyword
import importlib.util
import rotab.core.operation.define_funcs as define_funcs
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
        self.allowed_top_keys = {"processes", "custom_functions", "settings"}
        self.required_keys = {"process", "tables", "steps", "dumps"}
        self.optional_keys = {"description"}
        self.eval_scope = self._build_eval_scope()
        self.seen_table_names = set()
        self.seen_dump_names = set()

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
        for module in [define_funcs, transform_funcs]:
            scope.update({k: v for k, v in module.__dict__.items() if not k.startswith("__") and callable(v)})
        scope.update({k: v for k, v in builtins.__dict__.items() if callable(v)})

        # Load custom functions from specified paths
        custom_conf = self.config.get("custom_functions", {})
        define_paths = custom_conf.get("define_funcs", [])
        transform_paths = custom_conf.get("transform_funcs", [])
        if define_paths:
            scope.update(self._load_functions_from_paths(define_paths))
        if transform_paths:
            scope.update(self._load_functions_from_paths(transform_paths))
        return scope

    def validate(self):
        for key in self.config:
            if key not in self.allowed_top_keys:
                self.errors.append(
                    ValidationError(
                        f"config.{key}",
                        f"Unexpected top-level key: `{key}`. Only {sorted(self.allowed_top_keys)} are allowed.",
                    )
                )
        self._validate_custom_functions(self.config)
        if "processes" not in self.config or not isinstance(self.config["processes"], list):
            self.errors.append(ValidationError("config", "`processes` must be a list."))
            return
        for i, proc in enumerate(self.config["processes"]):
            self._validate_process(proc, f"processes[{i}]")

    def _validate_process(self, proc: Dict[str, Any], path: str):
        missing_keys = self.required_keys - proc.keys()
        for key in missing_keys:
            self.errors.append(ValidationError(path, f"Missing required key: `{key}`"))

        unknown_keys = proc.keys() - (self.required_keys | self.optional_keys)
        for key in unknown_keys:
            self.errors.append(ValidationError(f"{path}.{key}", "Unexpected key."))

        if "process" in proc and not isinstance(proc["process"], str):
            self.errors.append(ValidationError(f"{path}.process", "`process` must be a string."))

        if "tables" in proc:
            self._validate_tables(proc["tables"], f"{path}.tables")
        if "steps" in proc:
            table_names = {t["name"] for t in proc.get("tables", []) if isinstance(t, dict) and "name" in t}
            self._validate_steps(proc["steps"], table_names, f"{path}.steps")
        if "dumps" in proc:
            self._validate_dumps(proc["dumps"], proc, f"{path}.dumps")

    def _validate_custom_functions(self, config: dict, path: str = "custom_functions"):
        if "custom_functions" not in config:
            return  # optional, so silently skip if not present

        value = config["custom_functions"]
        if not isinstance(value, dict):
            self.errors.append(ValidationError(path, "`custom_functions` must be a dict."))
            return

        for key in ["define_funcs", "transform_funcs"]:
            if key not in value:
                continue  # optional keys
            funcs_path = f"{path}.{key}"
            funcs = value[key]
            if not isinstance(funcs, list):
                self.errors.append(ValidationError(funcs_path, f"`{key}` must be a list of file paths."))
                continue
            for i, item in enumerate(funcs):
                if not isinstance(item, str):
                    self.errors.append(ValidationError(f"{funcs_path}[{i}]", f"File path must be a string: {item!r}"))
                elif not item.endswith(".py"):
                    self.errors.append(ValidationError(f"{funcs_path}[{i}]", f"File path must end with `.py`: {item}"))

    def _validate_tables(self, tables: Any, path: str):
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
            if name in self.seen_table_names:
                self.errors.append(ValidationError(f"{path}[{i}].name", f"Duplicate table name `{name}`"))
            else:
                self.seen_table_names.add(name)

    def _validate_steps(self, steps: Any, table_names: set, path: str):
        if not isinstance(steps, list):
            self.errors.append(ValidationError(path, "`steps` must be a list."))
            return

        defined_vars = set(table_names)
        for i, step in enumerate(steps):
            p = f"{path}[{i}]"
            if not isinstance(step, dict):
                self.errors.append(ValidationError(p, "Each step must be a dict."))
                continue

            self._validate_with(step, defined_vars, p)
            self._validate_filter(step, p)
            self._validate_define_and_transform_exclusive(step, p)

            if "define" in step:
                self._validate_define(step["define"], defined_vars, p)

            if "transform" in step:
                self._validate_transform(step["transform"], defined_vars, p)

            if "select" in step and not isinstance(step["select"], list):
                self.errors.append(ValidationError(f"{p}.select", "`select` must be a list."))

    def _validate_with(self, step: dict, defined_vars: set, path: str):
        if "with" in step:
            if step["with"] not in defined_vars:
                self.errors.append(
                    ValidationError(
                        f"{path}.with", f"`with` must refer to a defined table or variable: `{step['with']}`"
                    )
                )

    def _validate_filter(self, step: dict, path: str):
        if "filter" in step:
            try:
                expr_ast = ast.parse(step["filter"], mode="eval")
                if not isinstance(expr_ast.body, (ast.Compare, ast.BoolOp, ast.Name, ast.Call, ast.BinOp)):
                    raise ValueError
            except Exception:
                self.errors.append(ValidationError(f"{path}.filter", "Invalid boolean expression."))

    def _validate_define_and_transform_exclusive(self, step: dict, path: str):
        if "define" in step and "transform" in step:
            self.errors.append(ValidationError(path, "Cannot use both `define` and `transform` in the same step."))

    def _validate_define(self, define_block: str, defined_vars: set, path: str):
        for line in define_block.split("\n"):
            if not line.strip():
                continue
            if line.count("=") != 1 or "==" in line:
                self.errors.append(ValidationError(f"{path}.define", f"Invalid assignment syntax: {line.strip()}"))
                continue
            lhs, rhs = map(str.strip, line.split("=", 1))
            if keyword.iskeyword(lhs):
                self.errors.append(ValidationError(f"{path}.define", f"`{lhs}` is a reserved keyword."))
            try:
                expr = ast.parse(rhs, mode="eval")
                if isinstance(expr.body, ast.Call):
                    func = self._get_func_name(expr.body.func)
                    if not func:
                        self.errors.append(ValidationError(f"{path}.define", "Unsupported function call format."))
                    elif func in {"eval", "exec", "compile", "open", "__import__"}:
                        self.errors.append(ValidationError(f"{path}.define", f"Forbidden function `{func}` used."))
                    elif func not in self.eval_scope:
                        self.errors.append(ValidationError(f"{path}.define", f"Function `{func}` not found."))
                    else:
                        self._check_function_signature(func, expr.body.args, expr.body.keywords, f"{path}.define")
                elif isinstance(expr.body, ast.Assign):
                    self.errors.append(ValidationError(f"{path}.define", "Multiple assignment not allowed."))
            except Exception:
                self.errors.append(ValidationError(f"{path}.define", f"Invalid expression on RHS: {rhs}"))
            defined_vars.add(lhs)

    def _validate_transform(self, transform_expr: str, defined_vars: set, path: str):
        if "=" not in transform_expr:
            self.errors.append(ValidationError(f"{path}.transform", "Transform must contain '='."))
            return
        lhs, rhs = map(str.strip, transform_expr.split("=", 1))
        if keyword.iskeyword(lhs):
            self.errors.append(ValidationError(f"{path}.transform", f"`{lhs}` is a reserved keyword."))
        try:
            expr = ast.parse(rhs, mode="eval")
            if not isinstance(expr.body, ast.Call):
                self.errors.append(ValidationError(f"{path}.transform", f"RHS must be a function call: {rhs}"))
                return
            func = self._get_func_name(expr.body.func)
            if not func:
                self.errors.append(ValidationError(f"{path}.transform", "Unsupported function call format."))
            elif func in {"eval", "exec", "compile", "open", "__import__"}:
                self.errors.append(ValidationError(f"{path}.transform", f"Forbidden function `{func}` used."))
            elif func not in self.eval_scope:
                self.errors.append(ValidationError(f"{path}.transform", f"Function `{func}` not found."))
            else:
                self._check_function_signature(func, expr.body.args, expr.body.keywords, f"{path}.transform")
            defined_vars.add(lhs)
        except Exception:
            self.errors.append(ValidationError(f"{path}.transform", f"Invalid function call: {rhs}"))

    def _get_func_name(self, func_expr):
        if isinstance(func_expr, ast.Name):
            return func_expr.id
        return None

    def _check_function_signature(self, func_name, args, keywords, path):
        func = self.eval_scope[func_name]
        keyword_names = [kw.arg for kw in keywords if kw.arg]
        if len(keyword_names) != len(set(keyword_names)):
            self.errors.append(ValidationError(path, "Duplicate keyword arguments in function call."))
            return
        try:
            sig = inspect.signature(func)
            mock_args = [self._mock_value(arg) for arg in args]
            mock_kwargs = {kw.arg: self._mock_value(kw.value) for kw in keywords if kw.arg}
            bound_args = sig.bind_partial(*mock_args, **mock_kwargs)
            for param in sig.parameters.values():
                if (
                    param.default is param.empty
                    and param.name not in bound_args.arguments
                    and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
                ):
                    self.errors.append(
                        ValidationError(
                            path=path,
                            message=f"Function `{func_name}` called with incorrect arguments: missing required argument `{param.name}`",
                            suggestion=f"Specify a value for `{param.name}`.",
                        )
                    )
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
            if not isinstance(dump, dict) or set(dump.keys()) != {"return", "path"}:
                self.errors.append(ValidationError(p, "`dumps` must have `return` and `path`."))
                continue
            if dump["return"] not in valid_returns:
                self.errors.append(
                    ValidationError(
                        f"{p}.return", f"`return` must be in defined tables or transform results: `{dump['return']}`"
                    )
                )
            if not isinstance(dump["path"], str):
                self.errors.append(ValidationError(f"{p}.path", "`path` must be a string."))
            elif re.search(r"[<>:\"|?*]", dump["path"]):
                self.errors.append(ValidationError(f"{p}.path", "Invalid characters in path."))
            if dump["return"] in self.seen_dump_names:
                self.errors.append(ValidationError(f"{p}.return", f"Duplicate dump return `{dump['return']}`"))
            else:
                self.seen_dump_names.add(dump["return"])

    def report(self):
        if self.errors:
            error_string = "\n".join(str(error) for error in self.errors)
            raise ValueError(f"Invalid template. See validation errors below.\n{error_string}")
