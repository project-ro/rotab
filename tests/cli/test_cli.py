import os
import tempfile
import shutil
import yaml
import sys
import pytest
from unittest.mock import patch
from rotab.cli.cli import main


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def create_dummy_project(base_dir):
    template_dir = os.path.join(base_dir, "templates")
    param_dir = os.path.join(base_dir, "params")
    schema_dir = os.path.join(base_dir, "schemas")
    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)

    # 入力CSVを template_dir/input/ に配置
    input_dir = os.path.join(template_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, "dummy.csv"), "w") as f:
        f.write("user_id,age\nu1,30\n")

    # テンプレート（template_dir 起点の相対パスで記述）
    write_yaml(
        os.path.join(template_dir, "template.yaml"),
        {
            "name": "dummy_template",
            "processes": [
                {
                    "name": "p1",
                    "io": {
                        "inputs": [
                            {
                                "name": "input_df",
                                "io_type": "csv",
                                "path": "input/dummy.csv",  # template_dir からの相対パス
                                "schema": "input_df",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "out_df",
                                "io_type": "csv",
                                "path": "output/out.csv",  # 相対パス（run時に source_dir/output にコピーされる）
                                "schema": "input_df",
                            }
                        ],
                    },
                    "steps": [
                        {
                            "name": "step1",
                            "with": "input_df",
                            "mutate": [{"select": ["user_id", "age"]}],
                            "as": "out_df",
                        }
                    ],
                }
            ],
        },
    )

    # スキーマとパラメータ
    write_yaml(os.path.join(schema_dir, "input_df.yaml"), {"columns": {"user_id": "str", "age": "int"}})
    write_yaml(os.path.join(param_dir, "params.yaml"), {"params": {}})

    return template_dir, param_dir, schema_dir


@pytest.mark.parametrize(
    "execute,dag",
    [
        (False, False),
        (True, False),
        (True, True),
    ],
)
def test_cli_main_variants(execute, dag):
    tmpdir = tempfile.mkdtemp()
    try:
        template_dir, param_dir, schema_dir = create_dummy_project(tmpdir)
        source_dir = os.path.join(tmpdir, "runspace")
        os.makedirs(source_dir, exist_ok=True)

        cwd = os.getcwd()
        os.chdir(source_dir)
        try:
            argv = [
                "rotab",
                "--template-dir",
                os.path.relpath(template_dir, source_dir),
                "--param-dir",
                os.path.relpath(param_dir, source_dir),
                "--schema-dir",
                os.path.relpath(schema_dir, source_dir),
                "--source-dir",
                ".",
            ]
            if execute:
                argv.append("--execute")
            if dag:
                argv.append("--dag")

            with patch.object(sys, "argv", argv):
                main()

            assert os.path.exists("main.py"), "main.py not found"

            if dag:
                assert os.path.exists("mermaid.mmd"), "mermaid.mmd should be generated"
            else:
                assert not os.path.exists("mermaid.mmd"), "mermaid.mmd should not be generated"

        finally:
            os.chdir(cwd)

    finally:
        shutil.rmtree(tmpdir)
