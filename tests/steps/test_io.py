import pytest
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.context.validation_context import ValidationContext, VariableInfo


def test_input_node_with_columns(base_context: ValidationContext):
    node = InputNode(name="user", io_type="csv", path="data/users.csv", schema_name="user")

    node.validate(base_context)
    script = node.generate_script(base_context)

    assert script == [
        "user = pd.read_csv(\"data/users.csv\", dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})"
    ]


def test_input_node_without_columns():
    context = ValidationContext(
        available_vars=set(),
        eval_scope={},
        schemas={},
    )

    node = InputNode(name="empty_df", io_type="csv", path="data/empty.csv", schema_name=None)

    node.validate(context)
    script = node.generate_script(context)
    assert script == ['empty_df = pd.read_csv("data/empty.csv")']


def test_output_node_with_columns(base_context: ValidationContext):
    node = OutputNode(name="user", io_type="csv", path="data/users_out.csv", schema_name="user")

    node.validate(base_context)
    script = node.generate_script(base_context)

    assert script[:4] == [
        'user["user_id"] = user["user_id"].astype("str")',
        'user["age"] = user["age"].astype("int")',
        'user["log_age"] = user["log_age"].astype("float")',
        'user["age_bucket"] = user["age_bucket"].astype("int")',
    ]
    assert (
        script[4]
        == "user.to_csv(\"data/users_out.csv\", index=False, columns=['user_id', 'age', 'log_age', 'age_bucket'])"
    )


def test_output_node_without_columns():
    context = ValidationContext(
        available_vars={"no_schema_df"},
        eval_scope={},
        schemas={"no_schema_df": VariableInfo(type="dataframe", columns={})},
    )

    node = OutputNode(name="no_schema_df", io_type="csv", path="data/no_schema.csv", schema_name=None)

    node.validate(context)
    script = node.generate_script(context)
    assert script == ['no_schema_df.to_csv("data/no_schema.csv", index=False)']
