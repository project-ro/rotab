import tempfile
import os
import textwrap
import yaml

from rotab.loader.schema_manager import SchemaManager
from rotab.loader.context_builder import ContextBuilder
from rotab.ast.template_node import TemplateNode
from rotab.ast.context.validation_context import ValidationContext
from rotab.ast.process_node import ProcessNode
from rotab.ast.io_node import InputNode, OutputNode

derive_func_code = textwrap.dedent(
    """
def derive_func1(x):
    return x + 1

def _private_func():
    pass
"""
)

transform_func_code = textwrap.dedent(
    """
def transform_func1(df):
    return df[df["value"] > 0]
"""
)


# テスト関数
def test_context_builder():
    with tempfile.TemporaryDirectory() as tmpdir:
        derive_path = os.path.join(tmpdir, "derive_funcs.py")
        transform_path = os.path.join(tmpdir, "transform_funcs.py")
        schema_dir = os.path.join(tmpdir, "schemas")
        os.makedirs(schema_dir)

        # スキーマファイルをダミーで書き出し
        with open(os.path.join(schema_dir, "input_df.yaml"), "w") as f:
            yaml.dump({"columns": {"value": "int"}}, f)

        with open(os.path.join(schema_dir, "output_df.yaml"), "w") as f:
            yaml.dump({"columns": {"value": "int"}}, f)

        with open(derive_path, "w") as f:
            f.write(derive_func_code)
        with open(transform_path, "w") as f:
            f.write(transform_func_code)

        dummy_template = TemplateNode(
            name="dummy_template",
            processes=[
                ProcessNode(
                    name="dummy_process",
                    inputs=[InputNode(name="input_df", io_type="csv", path="dummy.csv", schema="input_df")],
                    outputs=[OutputNode(name="output_df", io_type="csv", path="out.csv", schema="output_df")],
                    steps=[],
                )
            ],
        )

        schema_manager = SchemaManager(schema_dir)

        builder = ContextBuilder(
            derive_func_path=derive_path, transform_func_path=transform_path, schema_manager=schema_manager
        )

        context = builder.build([dummy_template])

        assert isinstance(context, ValidationContext)
        assert "input_df" in context.available_vars
        assert "output_df" in context.available_vars
        assert "derive_func1" in context.eval_scope
        assert "transform_func1" in context.eval_scope
        assert "_private_func" not in context.eval_scope

        assert "input_df" in context.schemas
        assert "output_df" in context.schemas
        assert context.schemas["input_df"].columns == {"value": "int"}
        assert context.schemas["output_df"].columns == {"value": "int"}
