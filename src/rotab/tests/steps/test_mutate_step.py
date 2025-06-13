import pytest
import ast
from typing import List
from rotab.ast.step import MutateStep
from rotab.tests.conftest import INDENT


@pytest.mark.parametrize(
    "step, expected",
    [
        (
            MutateStep(
                name="mutate_valid",
                input_vars=["user"],
                operations=[
                    {"filter": "age > 20"},
                    {"derive": "log_age = log(age)\nage_bucket = age // 10 * 10"},
                    {"select": ["user_id", "log_age", "age_bucket"]},
                ],
                output_var="filtered_user",
            ),
            [
                "filtered_user = user.copy()",
                "filtered_user = filtered_user.query('age > 20')",
                'filtered_user["log_age"] = filtered_user.apply(lambda row: log(row["age"]), axis=1)',
                'filtered_user["age_bucket"] = filtered_user.apply(lambda row: row["age"] // 10 * 10, axis=1)',
                'filtered_user = filtered_user[["user_id", "log_age", "age_bucket"]]',
            ],
        ),
        (
            MutateStep(
                name="with_when",
                input_vars=["user"],
                operations=[{"filter": "age > 20"}],
                output_var="u2",
                when="params.test",
            ),
            [
                "if params.test:",
                f"{INDENT}u2 = user.copy()",
                f"{INDENT}u2 = u2.query('age > 20')",
            ],
        ),
    ],
)
def test_mutate_step_generate_script(base_context, step: MutateStep, expected: List[str]):
    step.validate(base_context)
    script = step.generate_script()

    def normalize_code(code_lines: List[str]) -> str:
        return "\n".join(code_lines)

    def ast_equal(code1: str, code2: str) -> bool:
        try:
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            return ast.dump(tree1) == ast.dump(tree2)
        except SyntaxError:
            return False

    assert ast_equal(normalize_code(script), normalize_code(expected))


@pytest.mark.parametrize(
    "filter_expr",
    ["", "age >>", "age = 20"],
)
def test_mutate_step_invalid_filter(base_context, filter_expr: str):
    step = MutateStep(name="bad_filter", input_vars=["user"], operations=[{"filter": filter_expr}])
    with pytest.raises(ValueError, match="Invalid filter expression"):
        step.validate(base_context)


@pytest.mark.parametrize(
    "derive_line, match",
    [
        ("= log(age)", "Invalid LHS"),
        ("log_age == log(age)", "malformed '='"),
        ("123abc = log(age)", "Invalid LHS"),
        ("log_age = log(", "Syntax error"),
    ],
)
def test_mutate_step_invalid_derive(base_context, derive_line: str, match: str):
    step = MutateStep(name="bad_derive", input_vars=["user"], operations=[{"derive": derive_line}])
    with pytest.raises(ValueError, match=match):
        step.validate(base_context)


@pytest.mark.parametrize(
    "select, match",
    [
        ("user_id", "select must be a list"),
        (["user_id", 123], "select must be a list of strings"),
        (["nonexistent_col"], "undefined column"),
    ],
)
def test_mutate_step_invalid_select(base_context, select, match: str):
    step = MutateStep(name="bad_select", input_vars=["user"], operations=[{"select": select}])
    with pytest.raises(ValueError, match=match):
        step.validate(base_context)


@pytest.mark.parametrize(
    "op, match",
    [
        ({"filter": "x", "derive": "y = z"}, "must be a single-key dict"),
        ({"unknown": "x"}, "Unknown mutate operation"),
    ],
)
def test_mutate_step_invalid_operation_structure(base_context, op, match: str):
    step = MutateStep(name="bad_op", input_vars=["user"], operations=[op])
    with pytest.raises(ValueError, match=match):
        step.validate(base_context)
