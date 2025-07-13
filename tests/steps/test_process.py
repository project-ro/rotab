import pytest
from rotab.ast.process_node import ProcessNode
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.step_node import TransformStep
from rotab.ast.context.validation_context import ValidationContext
from rotab.ast.util import INDENT


@pytest.mark.parametrize(
    "backend, node, expected_script",
    [
        (
            "pandas",
            ProcessNode(
                name="transaction_enrichment",
                inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema_name="user")],
                steps=[
                    TransformStep(
                        name="step_merge", input_vars=["user"], expr="transform_func(user)", output_vars=["result"]
                    )
                ],
                outputs=[OutputNode(name="result", io_type="csv", path="result.csv", schema_name=None)],
            ),
            [
                # === Import block ===
                "import os",
                "import pandas as pd",
                "from rotab.core.operation.derive_funcs_pandas import *",
                "from rotab.core.operation.transform_funcs_pandas import *",
                "",
                "",
                # === Step function ===
                "def step_step_merge_transaction_enrichment(user):",
                INDENT + "result = transform_func(user)",
                INDENT + "return result",
                "",
                "",
                # === Main function ===
                "def transaction_enrichment():",
                INDENT
                + "user = pd.read_csv(\"user.csv\", dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})",
                INDENT + "result = step_step_merge_transaction_enrichment(user)",
                INDENT + 'result.to_csv("result.csv", index=False)',
                INDENT + "return result",
                "",
                "",
                # === Entry point ===
                'if __name__ == "__main__":',
                INDENT + "transaction_enrichment()",
                "",
            ],
        ),
        (
            "polars",
            ProcessNode(
                name="transaction_enrichment",
                inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema_name="user")],
                steps=[
                    TransformStep(
                        name="step_merge", input_vars=["user"], expr="transform_func(user)", output_vars=["result"]
                    )
                ],
                outputs=[OutputNode(name="result", io_type="csv", path="result.csv", schema_name=None)],
            ),
            [
                # === Import block ===
                "import os",
                "import polars as pl",
                "from rotab.core.parse.parse import parse",
                "from rotab.core.operation.derive_funcs_polars import *",
                "from rotab.core.operation.transform_funcs_polars import *",
                "",
                "",
                # === Step function ===
                "def step_step_merge_transaction_enrichment(user):",
                INDENT + "result = transform_func(user)",
                INDENT + "return result",
                "",
                "",
                # === Main function ===
                "def transaction_enrichment():",
                INDENT
                + 'user = pl.scan_csv("user.csv", dtypes={"user_id": pl.Utf8, "age": pl.Int64, "log_age": pl.Float64, "age_bucket": pl.Int64})',
                INDENT + "result = step_step_merge_transaction_enrichment(user)",
                INDENT + 'result.collect().write_csv("result.csv")',
                INDENT + "return result",
                "",
                "",
                # === Entry point ===
                'if __name__ == "__main__":',
                INDENT + "transaction_enrichment()",
                "",
            ],
        ),
    ],
)
def test_process_node_generate_script(base_context: ValidationContext, backend, node, expected_script):
    base_context.eval_scope["transform_func"] = lambda df: df
    node.validate(base_context)
    script = node.generate_script(backend, base_context)
    assert script == expected_script


def test_process_node_duplicate_var_error(base_context: ValidationContext):
    node = ProcessNode(
        name="duplicate_case",
        inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema_name="user")],
        steps=[
            TransformStep(
                name="step_dup",
                input_vars=["user"],
                expr="transform_func(user)",
                output_vars=["user"],  # Duplicate
            )
        ],
        outputs=[],
    )

    with pytest.raises(ValueError, match=r"\[step_dup\] Variable 'user' already defined."):
        node.validate(base_context)


def test_process_node_output_undefined_error(base_context: ValidationContext):
    node = ProcessNode(
        name="undefined_output",
        inputs=[],
        steps=[],
        outputs=[OutputNode(name="ghost", io_type="csv", path="ghost.csv", schema_name=None)],
    )

    with pytest.raises(ValueError, match=r"\[ghost\] Output variable 'ghost' is not defined in scope."):
        node.validate(base_context)
