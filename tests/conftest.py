import pytest
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


@pytest.fixture
def base_context() -> ValidationContext:
    return ValidationContext(
        available_vars={"user"},
        eval_scope={},
        schemas={
            "user": VariableInfo(
                type="dataframe",
                columns={
                    "user_id": "str",
                    "age": "int",
                    "log_age": "float",
                    "age_bucket": "int",
                },
            )
        },
    )
