import os
import tempfile
import textwrap
import yaml
import subprocess
import pytest
from typing import Tuple

from rotab.core.pipeline import Pipeline


def setup_test_environment(tmpdir: str) -> Tuple[str, str, str, str, str]:
    template_dir = os.path.join(tmpdir, "templates")
    param_dir = os.path.join(tmpdir, "params")
    schema_dir = os.path.join(tmpdir, "schemas")
    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)

    # テンプレート定義
    template = {
        "name": "test_template",
        "processes": [
            {
                "name": "test_process",
                "io": {
                    "inputs": [{"name": "input_df", "io_type": "csv", "path": "input.csv", "schema": "input_df"}],
                    "outputs": [{"name": "output_df", "io_type": "csv", "path": "output.csv", "schema": "output_df"}],
                },
                "steps": [
                    {"name": "dummy_step", "with": "input_df", "mutate": [{"filter": "value > 0"}], "as": "output_df"}
                ],
            }
        ],
    }
    with open(os.path.join(template_dir, "template.yaml"), "w") as f:
        yaml.dump(template, f)

    # 入力CSV
    with open(os.path.join(template_dir, "input.csv"), "w") as f:
        f.write("value\n1\n-1\n2\n")

    # パラメータ定義
    with open(os.path.join(param_dir, "params.yaml"), "w") as f:
        yaml.dump({}, f)

    # スキーマ定義
    with open(os.path.join(schema_dir, "input_df.yaml"), "w") as f:
        yaml.dump({"columns": {"value": "int"}}, f)
    with open(os.path.join(schema_dir, "output_df.yaml"), "w") as f:
        yaml.dump({"columns": {"value": "int"}}, f)

    # 関数定義
    derive_path = os.path.join(tmpdir, "derive_funcs.py")
    transform_path = os.path.join(tmpdir, "transform_funcs.py")
    with open(derive_path, "w") as f:
        f.write("def derive_func1(x): return x + 1\n")
    with open(transform_path, "w") as f:
        f.write("def transform_func1(df): return df[df['value'] > 0]\n")

    return template_dir, param_dir, schema_dir, derive_path, transform_path


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pipeline_codegen_outputs(backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir, param_dir, schema_dir, derive_path, transform_path = setup_test_environment(tmpdir)
        source_dir = os.path.join(tmpdir, "out")

        # Pipeline 実行（コード生成のみ）
        pipeline = Pipeline.from_setting(
            template_dir=template_dir,
            source_dir=source_dir,
            param_dir=param_dir,
            schema_dir=schema_dir,
            derive_func_path=derive_path,
            transform_func_path=transform_path,
            backend=backend,
        )
        pipeline.run(execute=False, dag=False)

        # main.py の確認
        main_path = os.path.join(source_dir, "main.py")
        assert os.path.isfile(main_path), "main.py not found in source directory"

        # プロセスコードファイルの確認
        template_name = "test_template"
        process_file = "test_process.py"
        process_path = os.path.join(source_dir, template_name, process_file)
        assert os.path.isfile(process_path), f"{process_file} not found in {template_name} subdirectory"


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pipeline_codegen_outputs_with_process_selection(backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir, param_dir, schema_dir, derive_path, transform_path = setup_test_environment(tmpdir)
        source_dir = os.path.join(tmpdir, "out")

        pipeline = Pipeline.from_setting(
            template_dir=template_dir,
            source_dir=source_dir,
            param_dir=param_dir,
            schema_dir=schema_dir,
            derive_func_path=derive_path,
            transform_func_path=transform_path,
            backend=backend,
        )
        pipeline.run(execute=False, dag=False, selected_processes=["test_process"])

        main_path = os.path.join(source_dir, "main.py")
        assert os.path.isfile(main_path), "main.py not found"

        with open(main_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "from test_template.test_process import test_process" in content
        assert "test_process()" in content

        assert "dummy_process()" not in content
