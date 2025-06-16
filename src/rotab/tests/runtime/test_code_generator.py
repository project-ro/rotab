import os
import tempfile
import pytest
from rotab.runtime.code_generator import CodeGenerator
from rotab.ast.template import TemplateNode
from rotab.ast.process import ProcessNode
from rotab.ast.io import InputNode, OutputNode
from rotab.ast.step import MutateStep, TransformStep
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


@pytest.fixture
def template_and_context() -> tuple[TemplateNode, ValidationContext, list[str]]:
    process = ProcessNode(
        name="full_process",
        inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema="user")],
        steps=[
            MutateStep(
                name="mutate_step",
                input_vars=["user"],
                operations=[
                    {"filter": "age > 20"},
                    {"derive": "log_age = log(age)\nage_bucket = age // 10"},
                    {"select": ["user_id", "log_age", "age_bucket"]},
                ],
                output_var="mutated",
            ),
            TransformStep(
                name="transform_step", input_vars=["mutated"], expr="transform_func(mutated)", output_var="result"
            ),
        ],
        outputs=[OutputNode(name="result", io_type="csv", path="result.csv", schema="result")],
    )

    template = TemplateNode(name="test_template", depends=[], processes=[process])

    context = ValidationContext(
        available_vars=set(),
        eval_scope={"transform_func": lambda x: x, "log": lambda x: x},
        schemas={
            "user": VariableInfo(type="dataframe", columns={"user_id": "str", "age": "int"}),
            "result": VariableInfo(
                type="dataframe", columns={"user_id": "str", "log_age": "float", "age_bucket": "int"}
            ),
        },
    )

    expected = [
        "import os",
        "import pandas as pd",
        "from rotab.core.operation.derive_funcs import *",
        "from rotab.core.operation.transform_funcs import *",
        "from custom_functions import derive_funcs, transform_funcs",
        "",
        "",
        "def step_mutate_step_full_process(user):",
        INDENT + "mutated = user.copy()",
        INDENT + "mutated = mutated.query('age > 20')",
        INDENT + 'mutated["log_age"] = mutated.apply(lambda row: log(row["age"]), axis=1)',
        INDENT + 'mutated["age_bucket"] = mutated.apply(lambda row: row["age"] // 10, axis=1)',
        INDENT + 'mutated = mutated[["user_id", "log_age", "age_bucket"]]',
        INDENT + "return mutated",
        "",
        "",
        "def step_transform_step_full_process(mutated):",
        INDENT + "result = transform_func(mutated)",
        INDENT + "return result",
        "",
        "",
        "def full_process():",
        INDENT + "user = pd.read_csv(\"user.csv\", dtype={'user_id': 'str', 'age': 'int'})",
        INDENT + "mutated = step_mutate_step_full_process(user)",
        INDENT + "result = step_transform_step_full_process(mutated)",
        INDENT + 'result["user_id"] = result["user_id"].astype("str")',
        INDENT + 'result["log_age"] = result["log_age"].astype("float")',
        INDENT + 'result["age_bucket"] = result["age_bucket"].astype("int")',
        INDENT + "result.to_csv(\"result.csv\", index=False, columns=['user_id', 'log_age', 'age_bucket'])",
        INDENT + "return result",
        "",
        "",
        'if __name__ == "__main__":',
        INDENT + "full_process()",
        "",
    ]

    return template, context, expected


def test_code_generator_generate_exact(template_and_context):
    template, context, expected = template_and_context
    generator = CodeGenerator([template], context)

    template.validate(context)
    result = generator.generate()
    actual = result["test_template"]["full_process"]

    assert actual == expected


def test_code_generator_write_all_exact(template_and_context):
    template, context, expected = template_and_context
    generator = CodeGenerator([template], context)

    template.validate(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        generator.write_all(tmpdir)

        path = os.path.join(tmpdir, "test_template", "full_process.py")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().splitlines()

        assert content == expected
