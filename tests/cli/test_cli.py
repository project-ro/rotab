import os
import tempfile
import shutil
import yaml
import sys
import pytest
import pandas as pd
from unittest.mock import patch
from rotab.cli.cli import main
from pathlib import Path


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

    input_dir = os.path.join(template_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    with open(os.path.join(input_dir, "dummy_202401.csv"), "w") as f:
        f.write("user_id,age\nu1,30\nu2,35\n")
    with open(os.path.join(input_dir, "dummy_202402.csv"), "w") as f:
        f.write("user_id,age\nu3,40\nu4,45\n")

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

            assert os.path.exists("main.py"), "main.py not found"

            if dag:
                assert os.path.exists("mermaid.mmd"), "mermaid.mmd should be generated"
            else:
                assert not os.path.exists("mermaid.mmd"), "mermaid.mmd should not be generated"

            if execute:
                out_path = os.path.join(source_dir, "data", "outputs", "out.csv")
                assert os.path.exists(out_path), "data/outputs/out.csv should be generated"

                df = pd.read_csv(out_path)
                assert "yyyymm" in df.columns, "yyyymm column should exist in output"

                expected_vals = {"202401", "202402"}
                df_yyyymm = set(df["yyyymm"].astype(str))
                assert df_yyyymm == expected_vals, f"yyyymm values should match wildcard files: {df_yyyymm}"
                assert df.shape[0] == 4, "Unexpected number of rows in output"

        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmpdir)


def test_cli_main_with_process_selection():
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
                "pandas",
                "--processes",
                "p1",
            ]

            with patch.object(sys, "argv", argv):
                main()

            main_path = os.path.join(source_dir, "main.py")
            assert os.path.exists(main_path), "main.py should be generated"

            with open(main_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "from dummy_template.p1 import p1" in content
            assert "p1()" in content

            assert "p2()" not in content
            assert "from dummy_template.p2 import p2" not in content

        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmpdir)


def test_cli_main_with_multiple_processes():
    tmpdir = tempfile.mkdtemp()
    try:
        template_dir, param_dir, schema_dir = create_dummy_project(tmpdir)

        template_path = os.path.join(template_dir, "template.yaml")
        with open(template_path, "r") as f:
            template_data = yaml.safe_load(f)
        template_data["processes"].append(
            {
                "name": "p2",
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
                            "path": "output/out2.csv",
                            "schema": "input_df",
                        }
                    ],
                },
                "steps": [
                    {
                        "name": "step2",
                        "with": "input_df",
                        "mutate": [{"select": ["user_id", "age"]}],
                        "as": "out_df",
                    }
                ],
            }
        )
        write_yaml(template_path, template_data)

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
                "pandas",
                "--processes",
                "p1",
                "p2",
            ]

            with patch.object(sys, "argv", argv):
                main()

            main_path = os.path.join(source_dir, "main.py")
            assert os.path.exists(main_path), "main.py should be generated"

            with open(main_path, "r", encoding="utf-8") as f:
                content = f.read()

            assert "from dummy_template.p1 import p1" in content
            assert "from dummy_template.p2 import p2" in content
            assert "p1()" in content
            assert "p2()" in content

        finally:
            os.chdir(cwd)
    finally:
        shutil.rmtree(tmpdir)


def test_cli_main_init_creates_expected_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        input_dir = Path(temp_dir) / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "input_data.csv").write_text("user_id,age\n1,25\n2,30\n", encoding="utf-8")

        project_name = "testproject"
        backend = "polars"
        project_path = Path(temp_dir) / project_name

        argv = ["rotab", "--init"]

        with patch.object(sys, "argv", argv), patch("questionary.text") as mock_text, patch(
            "questionary.path"
        ) as mock_path, patch("questionary.select") as mock_select:

            mock_text.return_value.ask.return_value = project_name
            mock_path.return_value.ask.return_value = str(input_dir)
            mock_select.return_value.ask.return_value = backend

            main()

        expected_files = [
            project_path / "config" / "schemas" / "input_data.yaml",
            project_path / "config" / "params" / "params.yaml",
            project_path / "config" / "templates" / "template.yaml",
            project_path / "custom_functions" / f"derive_funcs_{backend}.py",
            project_path / "custom_functions" / f"transform_funcs_{backend}.py",
        ]
        for path in expected_files:
            assert path.exists(), f"Missing file: {path}"

        input_yaml = yaml.safe_load((project_path / "config" / "schemas" / "input_data.yaml").read_text())
        assert input_yaml["name"] == "input_data"
        assert input_yaml["description"]
        assert len(input_yaml["columns"]) == 2
        assert input_yaml["columns"][0]["name"] == "user_id"
        assert input_yaml["columns"][0]["dtype"] == "int"
        assert input_yaml["columns"][1]["name"] == "age"
        assert input_yaml["columns"][1]["dtype"] == "int"

        tpl = yaml.safe_load((project_path / "config" / "templates" / "template.yaml").read_text())
        assert tpl["name"] == "main_template"
        proc = tpl["processes"][0]
        assert proc["name"] == "simple_process"
        assert proc["steps"][0]["name"] == "basic_filter"
        assert any("derive" in op and "age_group" in op["derive"] for op in proc["steps"][0]["mutate"])
        assert proc["steps"][1]["transform"] == "rename(col='user_id', to='id')"

        params = yaml.safe_load((project_path / "config" / "params" / "params.yaml").read_text())
        assert params["params"]["min_age"] == 20

        for kind in ["derive", "transform"]:
            fpath = project_path / "custom_functions" / f"{kind}_funcs_{backend}.py"
            content = fpath.read_text(encoding="utf-8")
            assert f"# {kind.capitalize()} functions for {backend}" in content
