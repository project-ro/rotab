import pytest
from rotab.ast.template_node import TemplateNode
from rotab.ast.process_node import ProcessNode
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.step_node import MutateStep, TransformStep
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


@pytest.mark.parametrize(
    "process, expected_script",
    [
        (
            ProcessNode(
                name="transaction_enrichment",
                description="This process enriches user transactions by filtering users based on age and\ntransactions based on amount, then merging the two datasets.",
                inputs=[
                    InputNode(name="user", io_type="csv", path="../../output/filtered_users.csv", schema="user"),
                    InputNode(
                        name="trans", io_type="csv", path="../../output/filtered_transactions.csv", schema="trans"
                    ),
                ],
                steps=[
                    MutateStep(
                        name="filter_users_main",
                        input_vars=["user"],
                        operations=[
                            {"filter": "age > 18"},
                            {"derive": "log_age = log(age)\nage_bucket = age // 10 * 10"},
                            {"select": ["user_id", "log_age", "age_bucket"]},
                        ],
                        output_vars=["filtered_users"],
                    ),
                    MutateStep(
                        name="filter_transactions_main",
                        input_vars=["trans"],
                        operations=[
                            {"filter": "amount > 1000"},
                        ],
                        output_vars=["filtered_trans"],
                    ),
                    TransformStep(
                        name="merge_transactions",
                        input_vars=["filtered_users", "filtered_trans"],
                        expr="merge(left=filtered_users, right=filtered_trans, on='user_id')",
                        output_vars=["enriched"],
                    ),
                    MutateStep(
                        name="enrich_transactions",
                        input_vars=["enriched"],
                        operations=[
                            {"derive": "high_value = amount > 10000"},
                            {"select": ["user_id", "log_age", "age_bucket", "high_value"]},
                        ],
                        output_vars=["final_output"],
                    ),
                ],
                outputs=[
                    OutputNode(
                        name="final_output", io_type="csv", path="../../output/final_output.csv", schema="final_output"
                    ),
                ],
            ),
            [
                # === Imports ===
                "import os",
                "import pandas as pd",
                "from rotab.core.operation.derive_funcs import *",
                "from rotab.core.operation.transform_funcs import *",
                "from custom_functions import derive_funcs, transform_funcs",
                "",
                "",
                # === Step 1 ===
                "def step_filter_users_main_transaction_enrichment(user):",
                INDENT + "filtered_users = user.copy()",
                INDENT + "filtered_users = filtered_users.query('age > 18')",
                INDENT + 'filtered_users["log_age"] = filtered_users.apply(lambda row: log(row["age"]), axis=1)',
                INDENT
                + 'filtered_users["age_bucket"] = filtered_users.apply(lambda row: row["age"] // 10 * 10, axis=1)',
                INDENT + 'filtered_users = filtered_users[["user_id", "log_age", "age_bucket"]]',
                INDENT + "return filtered_users",
                "",
                "",
                # === Step 2 ===
                "def step_filter_transactions_main_transaction_enrichment(trans):",
                INDENT + "filtered_trans = trans.copy()",
                INDENT + "filtered_trans = filtered_trans.query('amount > 1000')",
                INDENT + "return filtered_trans",
                "",
                "",
                # === Step 3 ===
                "def step_merge_transactions_transaction_enrichment(filtered_users, filtered_trans):",
                INDENT + "enriched = merge(left=filtered_users, right=filtered_trans, on='user_id')",
                INDENT + "return enriched",
                "",
                "",
                # === Step 4 ===
                "def step_enrich_transactions_transaction_enrichment(enriched):",
                INDENT + "final_output = enriched.copy()",
                INDENT + 'final_output["high_value"] = final_output.apply(lambda row: row["amount"] > 10000, axis=1)',
                INDENT + 'final_output = final_output[["user_id", "log_age", "age_bucket", "high_value"]]',
                INDENT + "return final_output",
                "",
                "",
                # === Main function ===
                "def transaction_enrichment():",
                INDENT
                + '"""This process enriches user transactions by filtering users based on age and\n    transactions based on amount, then merging the two datasets."""',
                INDENT
                + "user = pd.read_csv(\"../../output/filtered_users.csv\", dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})",
                INDENT
                + "trans = pd.read_csv(\"../../output/filtered_transactions.csv\", dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})",
                INDENT + "filtered_users = step_filter_users_main_transaction_enrichment(user)",
                INDENT + "filtered_trans = step_filter_transactions_main_transaction_enrichment(trans)",
                INDENT + "enriched = step_merge_transactions_transaction_enrichment(filtered_users, filtered_trans)",
                INDENT + "final_output = step_enrich_transactions_transaction_enrichment(enriched)",
                INDENT + 'final_output["user_id"] = final_output["user_id"].astype("str")',
                INDENT + 'final_output["age"] = final_output["age"].astype("int")',
                INDENT + 'final_output["log_age"] = final_output["log_age"].astype("float")',
                INDENT + 'final_output["age_bucket"] = final_output["age_bucket"].astype("int")',
                INDENT + 'final_output["high_value"] = final_output["high_value"].astype("bool")',
                INDENT
                + "final_output.to_csv(\"../../output/final_output.csv\", index=False, columns=['user_id', 'age', 'log_age', 'age_bucket', 'high_value'])",
                INDENT + "return final_output",
                "",
                "",
                # === Entry point ===
                'if __name__ == "__main__":',
                INDENT + "transaction_enrichment()",
                "",
            ],
        )
    ],
)
def test_template_node_generation(process: ProcessNode, expected_script: list[str]):
    context = ValidationContext(
        available_vars={inp.name for inp in process.inputs},
        eval_scope={"merge": lambda left, right, on: None, "log": lambda x: None},
        schemas={
            "user": VariableInfo(
                type="dataframe", columns={"user_id": "str", "age": "int", "log_age": "float", "age_bucket": "int"}
            ),
            "trans": VariableInfo(
                type="dataframe", columns={"user_id": "str", "age": "int", "log_age": "float", "age_bucket": "int"}
            ),
            "enriched": VariableInfo(
                type="dataframe",
                columns={"user_id": "str", "log_age": "float", "age_bucket": "int", "amount": "float"},
            ),
            "final_output": VariableInfo(
                type="dataframe",
                columns={"user_id": "str", "age": "int", "log_age": "float", "age_bucket": "int", "high_value": "bool"},
            ),
        },
    )

    template = TemplateNode(
        name="test_template",
        depends=[],
        processes=[process],
    )

    template.validate(context)
    result = template.generate_script(context)

    assert result["transaction_enrichment"] == expected_script
