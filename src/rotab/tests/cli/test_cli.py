import tempfile
import os
import shutil
import yaml
import sys
import pytest
from unittest.mock import patch
from rotab.core.cli import main


def write_yaml(path, data):
    with open(path, "w") as f:
        yaml.dump(data, f)


def create_dummy_project(base_dir):
    template_dir = os.path.join(base_dir, "templates")
    param_dir = os.path.join(base_dir, "params")
    schema_dir = os.path.join(base_dir, "schemas")
    input_dir = os.path.join(base_dir, "input")
    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)
    os.makedirs(input_dir)

    with open(os.path.join(input_dir, "dummy.csv"), "w") as f:
        f.write("user_id,age\nu1,30\n")

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
                                "path": "input/dummy.csv",  # 相対パス
                                "schema": "input_df",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "out_df",
                                "io_type": "csv",
                                "path": "output/out.csv",  # 相対パス
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

    write_yaml(os.path.join(schema_dir, "input_df.yaml"), {"columns": {"user_id": "str", "age": "int"}})
    write_yaml(os.path.join(param_dir, "params.yaml"), {"params": {}})

    return template_dir, param_dir, schema_dir, input_dir


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
        template_dir, param_dir, schema_dir, input_dir = create_dummy_project(tmpdir)
        current_dir = os.path.join(tmpdir, "current_project")
        os.makedirs(current_dir, exist_ok=True)

        input_dir_in_output = os.path.join(current_dir, "input")
        os.makedirs(input_dir_in_output, exist_ok=True)
        shutil.copy(os.path.join(input_dir, "dummy.csv"), os.path.join(input_dir_in_output, "dummy.csv"))

        cwd = os.getcwd()
        os.chdir(current_dir)
        try:
            os.makedirs("output", exist_ok=True)
            argv = [
                "rotab",
                "--template-dir",
                "../templates",
                "--param-dir",
                "../params",
                "--schema-dir",
                "../schemas",
                "--output-dir",
                ".",  # カレントディレクトリを出力先に
            ]
            if execute:
                argv.append("--execute")
            if dag:
                argv.append("--dag")

            with patch.object(sys, "argv", argv):
                main()

            assert os.path.exists("main.py")

            dag_path = "mermaid.mmd"
            if dag:
                assert os.path.exists(dag_path)
            else:
                assert not os.path.exists(dag_path)

        finally:
            os.chdir(cwd)

    finally:
        shutil.rmtree(tmpdir)
