from rotab.ast.step_node import MutateStep
from textwrap import dedent


class ValidationContext:
    def __init__(self):
        self.available_vars = set()
        self.schemas = {}


def test_generate_script_polars_basic():
    step = MutateStep(
        name="test_mutate",
        input_vars=["df"],
        output_vars=["df_result"],
        operations=[
            {"filter": "age > 18 and income < 5000"},
            {"derive": "new_col = a + b\nscore = a / b"},
            {"select": ["new_col", "score", "age"]},
        ],
        when="flag",
    )
    lines = step.generate_script_polars()

    expected = [
        "if flag:",
        "    df_result = df",
        "    df_result = df_result.filter(expr('age > 18 and income < 5000'))",
        '    df_result = df_result.with_columns(expr("""\n'
        "        new_col = a + b\n"
        "        score = a / b\n"
        '        """))',
        "    df_result = df_result.select(['new_col', 'score', 'age'])",
    ]
    assert lines == expected


def test_generate_script_polars_select_only():
    step = MutateStep(
        name="test_mutate",
        input_vars=["df"],
        output_vars=["df_res"],
        operations=[{"select": ["a", "b", "c"]}],
    )
    lines = step.generate_script_polars()

    expected = [
        "df_res = df",
        "df_res = df_res.select(['a', 'b', 'c'])",
    ]

    assert lines == expected


def test_generate_script_polars_without_when():
    step = MutateStep(
        name="test_mutate",
        input_vars=["df"],
        output_vars=["df_result"],
        operations=[{"filter": "flag == True"}],
    )
    lines = step.generate_script_polars()

    expected = [
        "df_result = df",
        "df_result = df_result.filter(expr('flag == True'))",
    ]

    assert lines == expected
