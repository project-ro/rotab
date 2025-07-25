import os
import tempfile
import pytest
from typing import List
from rotab.runtime.code_generator import CodeGenerator
from rotab.ast.template_node import TemplateNode
from rotab.ast.process_node import ProcessNode
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.step_node import MutateStep, TransformStep
from rotab.ast.context.validation_context import ValidationContext, VariableInfo
from rotab.ast.util import INDENT


@pytest.fixture(params=["pandas", "polars"])
def template_and_context(request) -> tuple[TemplateNode, ValidationContext, List[str], List[str], str]:
    backend = request.param

    process = ProcessNode(
        name="full_process",
        inputs=[InputNode(name="user", io_type="csv", path="user.csv", schema_name="user")],
        steps=[
            MutateStep(
                name="mutate_step",
                input_vars=["user"],
                operations=[
                    {"filter": "age > 20"},
                    {"derive": "log_age = log(age)\nage_bucket = age // 10"},
                    {"select": ["user_id", "log_age", "age_bucket"]},
                ],
                output_vars=["mutated"],
            ),
            TransformStep(
                name="transform_step", input_vars=["mutated"], expr="transform_func(mutated)", output_vars=["result"]
            ),
        ],
        outputs=[OutputNode(name="result", io_type="csv", path="result.csv", schema_name="result")],
    )

    template = TemplateNode(name="test_template", depends=[], processes=[process])

    context = ValidationContext(
        derive_func_path="custom_functions/derive_funcs.py",
        transform_func_path="custom_functions/transform_funcs.py",
        available_vars=set(),
        eval_scope={"transform_func": lambda x: x, "log": lambda x: x},
        schemas={
            "user": VariableInfo(type="dataframe", columns={"user_id": "str", "age": "int"}),
            "result": VariableInfo(
                type="dataframe", columns={"user_id": "str", "log_age": "float", "age_bucket": "int"}
            ),
        },
    )

    if backend == "pandas":
        expected_process = [
            "import os",
            "import pandas as pd",
            "from rotab.core.operation.derive_funcs_pandas import *",
            "from rotab.core.operation.transform_funcs_pandas import *",
            "from custom_functions.derive_funcs import *",
            "from custom_functions.transform_funcs import *",
            "",
            "",
            "def step_mutate_step_full_process(user):",
            INDENT + "mutated = user.copy()",
            INDENT + "mutated = mutated.query('age > 20').copy()",
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
    else:  # polars
        expected_process = [
            "import os",
            "import polars as pl",
            "import fsspec",
            "from rotab.core.parse.parse import parse",
            "from rotab.core.operation.derive_funcs_polars import *",
            "from rotab.core.operation.transform_funcs_polars import *",
            "from custom_functions.derive_funcs import *",
            "from custom_functions.transform_funcs import *",
            "",
            "",
            "def step_mutate_step_full_process(user):",
            INDENT + "mutated = user",
            INDENT + "mutated = mutated.filter(parse('age > 20'))",
            INDENT + 'mutated = mutated.with_columns(parse("""\n'
            "        log_age = log(age)\n"
            "        age_bucket = age // 10\n"
            '        """))',
            INDENT + "mutated = mutated.select(['user_id', 'log_age', 'age_bucket'])",
            INDENT + "return mutated",
            "",
            "",
            "def step_transform_step_full_process(mutated):",
            INDENT + "result = transform_func(mutated)",
            INDENT + "return result",
            "",
            "",
            "def full_process():",
            INDENT + 'user = pl.scan_csv("user.csv", dtypes={"user_id": pl.Utf8, "age": pl.Int64})',
            INDENT + "mutated = step_mutate_step_full_process(user)",
            INDENT + "result = step_transform_step_full_process(mutated)",
            INDENT + 'result = result.with_columns(pl.col("user_id").cast(pl.Utf8))',
            INDENT + 'result = result.with_columns(pl.col("log_age").cast(pl.Float64))',
            INDENT + 'result = result.with_columns(pl.col("age_bucket").cast(pl.Int64))',
            INDENT + 'with fsspec.open("result.csv", "w") as f:',
            INDENT * 2 + "result.collect(streaming=True).write_csv(f)",
            INDENT + "return result",
            "",
            "",
            'if __name__ == "__main__":',
            INDENT + "full_process()",
            "",
        ]

    expected_main = [
        "import os",
        "import sys",
        "project_root = os.path.dirname(os.path.abspath(__file__))",
        "sys.path.insert(0, project_root)",
        "",
        "from test_template.full_process import full_process",
        "",
        "",
        "if __name__ == '__main__':",
        INDENT + "full_process()",
    ]
    return template, context, expected_process, expected_main, backend


def test_code_generator_generate_exact(template_and_context):
    template, context, expected_process, _, backend = template_and_context
    generator = CodeGenerator([template], backend, context)

    template.validate(context)
    result = generator.generate()
    actual = result["test_template"]["full_process"]

    assert actual == expected_process


def test_code_generator_write_all_exact(template_and_context):
    template, context, expected_process, expected_main, backend = template_and_context
    generator = CodeGenerator([template], backend, context)

    template.validate(context)

    with tempfile.TemporaryDirectory() as tmpdir:
        generator.write_all(tmpdir)

        # check full_process.py
        process_path = os.path.join(tmpdir, "test_template", "full_process.py")
        with open(process_path, "r", encoding="utf-8") as f:
            process_content = f.read()

        expected_str = "\n".join(expected_process) + "\n"
        assert process_content == expected_str

        # check main.py
        main_path = os.path.join(tmpdir, "main.py")
        with open(main_path, "r", encoding="utf-8") as f:
            main_content = f.read().splitlines()
        assert main_content == expected_main
