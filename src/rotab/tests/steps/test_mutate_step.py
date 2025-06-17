import pytest
import ast
from typing import List
from rotab.ast.step_node import MutateStep
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.tests.conftest import INDENT


def dummy_context():
    return ValidationContext(
        available_vars={"user"},
        schemas={
            "user": VariableInfo(
                type="dataframe",
                columns={"user_id": "str", "age": "int", "log_age": "float", "age_bucket": "int"},
            )
        },
        eval_scope={},
    )


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
                output_vars=["filtered_user"],
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
                output_vars=["u2"],
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
def test_mutate_step_generate_script(step: MutateStep, expected: List[str]):
    step.validate(dummy_context())
    script = step.generate_script()

    def normalize_code(code_lines: List[str]) -> str:
        return "\n".join(code_lines)

    def ast_equal(code1: str, code2: str) -> bool:
        try:
            return ast.dump(ast.parse(code1)) == ast.dump(ast.parse(code2))
        except SyntaxError:
            return False

    assert ast_equal(normalize_code(script), normalize_code(expected))


@pytest.mark.parametrize(
    "filter_expr",
    ["", "age >>", "age = 20"],
)
def test_mutate_step_invalid_filter(filter_expr: str):
    step = MutateStep(
        name="bad_filter", input_vars=["user"], operations=[{"filter": filter_expr}], output_vars=["dummy"]
    )
    with pytest.raises(ValueError, match=r"\[bad_filter\] Invalid filter expression:"):
        step.validate(dummy_context())


@pytest.mark.parametrize(
    "derive_line, match",
    [
        ("= log(age)", r"\[bad_derive\] Invalid LHS in derive:"),
        ("log_age == log(age)", r"\[bad_derive\] derive line \d+: malformed '='"),
        ("123abc = log(age)", r"\[bad_derive\] Invalid LHS in derive:"),
        ("log_age = log(", r"\[bad_derive\] Syntax error in RHS:"),
    ],
)
def test_mutate_step_invalid_derive(derive_line: str, match: str):
    step = MutateStep(
        name="bad_derive", input_vars=["user"], operations=[{"derive": derive_line}], output_vars=["dummy"]
    )
    with pytest.raises(ValueError, match=match):
        step.validate(dummy_context())


@pytest.mark.parametrize(
    "select, match",
    [
        ("user_id", r"\[bad_select\] select must be a list of strings"),
        (["user_id", 123], r"\[bad_select\] select must be a list of strings"),
        (["nonexistent_col"], r"\[bad_select\] select references undefined column:"),
    ],
)
def test_mutate_step_invalid_select(select, match: str):
    step = MutateStep(name="bad_select", input_vars=["user"], operations=[{"select": select}], output_vars=["dummy"])
    with pytest.raises(ValueError, match=match):
        step.validate(dummy_context())


@pytest.mark.parametrize(
    "op, match",
    [
        ({"filter": "x", "derive": "y = z"}, r"\[bad_op\] Operation #0 must be a single-key dict"),
        ({"unknown": "x"}, r"\[bad_op\] Unknown mutate operation:"),
    ],
)
def test_mutate_step_invalid_operation_structure(op, match: str):
    step = MutateStep(name="bad_op", input_vars=["user"], operations=[op], output_vars=["dummy"])
    with pytest.raises(ValueError, match=match):
        step.validate(dummy_context())
