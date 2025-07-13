import os
import tempfile
import shutil
import yaml
import sys
import pytest
import pandas as pd
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

    # 入力CSVを template_dir/input/ に複数配置
    input_dir = os.path.join(template_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, "dummy_202401.csv"), "w") as f:
        f.write("user_id,age\nu1,30\nu2,35\n")
    with open(os.path.join(input_dir, "dummy_202402.csv"), "w") as f:
        f.write("user_id,age\nu3,40\nu4,45\n")

    # テンプレート
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
                                "path": "input/dummy_*.csv",
                                "wildcard_column": "yyyymm",
                                "schema": "input_df",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "out_df",
                                "io_type": "csv",
                                "path": "output/out.csv",
                                "schema": "input_df",
                            }
                        ],
                    },
                    "steps": [
                        {
                            "name": "step1",
                            "with": "input_df",
                            "mutate": [{"select": ["user_id", "age", "yyyymm"]}],
                            "as": "out_df",
                        }
                    ],
                }
            ],
        },
    )

    # スキーマとパラメータ
    write_yaml(
        os.path.join(schema_dir, "input_df.yaml"), {"columns": {"user_id": "str", "age": "int", "yyyymm": "str"}}
    )
    write_yaml(os.path.join(param_dir, "params.yaml"), {"params": {}})

    return template_dir, param_dir, schema_dir


@pytest.mark.parametrize(
    "execute,dag,backend",
    [
        (False, False, "pandas"),
        (True, False, "pandas"),
        (True, True, "pandas"),
        (False, False, "polars"),
        (True, False, "polars"),
        (True, True, "polars"),
    ],
)
def test_cli_main_variants(execute, dag, backend):
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
                "--backend",
                backend,
            ]
            if execute:
                argv.append("--execute")
            if dag:
                argv.append("--dag")

            with patch.object(sys, "argv", argv):
                main()

            # main.py 確認
            assert os.path.exists("main.py"), "main.py not found"

            # DAG の出力有無確認
            if dag:
                assert os.path.exists("mermaid.mmd"), "mermaid.mmd should be generated"
            else:
                assert not os.path.exists("mermaid.mmd"), "mermaid.mmd should not be generated"

            # 実行モードであれば、出力CSV確認
            if execute:
                out_path = os.path.join(source_dir, "data", "outputs", "out.csv")
                assert os.path.exists(out_path), "data/outputs/out.csv should be generated"

                df = pd.read_csv(out_path)
                assert "yyyymm" in df.columns, "yyyymm column should exist in output"

                # polars backend でも出力は csv で統一されるので値を検証
                expected_vals = {"202401", "202402"}
                df_yyyymm = set(df["yyyymm"].astype(str))
                assert df_yyyymm == expected_vals, f"yyyymm values should match wildcard files: {df_yyyymm}"
                assert df.shape[0] == 4, "Unexpected number of rows in output"

        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmpdir)
