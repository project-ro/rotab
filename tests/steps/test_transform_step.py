import pytest
import ast
from typing import List
from rotab.ast.step_node import TransformStep
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from conftest import INDENT


@pytest.mark.parametrize(
    "step, expected",
    [
        (
            TransformStep(
                name="transform_merge",
                input_vars=["df1", "df2"],
                expr="merge(df1, df2, on='id')",
                output_vars=["merged_df"],
            ),
            ["merged_df = merge(df1, df2, on='id')"],
        ),
        (
            TransformStep(
                name="transform_with_condition",
                input_vars=["df1", "df2"],
                expr="merge(df1, df2, on='id')",
                output_vars=["merged_df"],
                when="params.enabled",
            ),
            [
                "if params.enabled:",
                f"{INDENT}merged_df = merge(df1, df2, on='id')",
            ],
        ),
    ],
)
def test_transform_step_generate_script(base_context: ValidationContext, step, expected):
    # 属性アクセスに修正
    base_context.available_vars.update({"df1", "df2"})
    base_context.eval_scope["merge"] = lambda *args, **kwargs: None
    base_context.schemas["df1"] = VariableInfo(type="dataframe", columns={})
    base_context.schemas["df2"] = VariableInfo(type="dataframe", columns={})

    step.validate(base_context)
    script = step.generate_script(backend="pandas")

    def normalize_code(code_lines: List[str]) -> str:
        return "\n".join(code_lines)

    def ast_equal(code1: str, code2: str) -> bool:
        try:
            return ast.dump(ast.parse(code1)) == ast.dump(ast.parse(code2))
        except SyntaxError:
            return False

    assert ast_equal(normalize_code(script), normalize_code(expected))


@pytest.mark.parametrize(
    "input_vars, expr, error_message",
    [
        (["unknown_df"], "merge(df1, df2, on='id')", "`unknown_df` is not defined."),
        (["df1", "df2"], "merge(", "Invalid Python expression in `transform`"),
        (["df1", "df2"], "1 + 1", "Expression must be a function call."),
        (["df1", "df2"], "unknown_func(df1)", "Function `unknown_func` not found in eval_scope."),
        (["df1", "df2"], "(merge)(df1)", "Unsupported function syntax in expression."),
    ],
)
def test_transform_step_invalid_cases(input_vars, expr, error_message):
    context = ValidationContext(
        available_vars={"df1", "df2"},
        eval_scope={"merge": lambda *args, **kwargs: None},
        schemas={
            "df1": VariableInfo(type="dataframe", columns={}),
            "df2": VariableInfo(type="dataframe", columns={}),
        },
    )

    step = TransformStep(name="invalid_case", input_vars=input_vars, expr=expr, output_vars=["output"])

    with pytest.raises(ValueError, match=error_message):
        step.validate(context)
