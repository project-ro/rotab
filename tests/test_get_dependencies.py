import tempfile
import shutil
import os
import yaml
from pathlib import Path
import pytest
import pandas as pd
from rotab.core.pipeline import Pipeline


import pytest
import tempfile
from pathlib import Path
import pandas as pd


@pytest.fixture
def setup_virtual_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        tables_dir = data_dir / "tables"
        output_dir = data_dir / "output"
        template_dir = root / "template"
        param_dir = root / "params"

        data_dir.mkdir()
        tables_dir.mkdir()
        output_dir.mkdir()
        template_dir.mkdir()
        param_dir.mkdir()

        # CSVファイルを配置
        pd.DataFrame({"user_id": [1, 2], "age": [25, 15]}).to_csv(tables_dir / "table_a.csv", index=False)
        pd.DataFrame({"user_id": [1, 2], "amount": [1500, 500]}).to_csv(tables_dir / "table_b.csv", index=False)

        # Template A
        template_a_content = """\
processes:
  - name: process_a
    description: template A processes
    tables:
      - name: table_a
        path: ../data/tables/table_a.csv
    steps:
      - name: filter_users
        with: table_a
        mutate:
          filter: age > 18
          derive: |
            log_age = log(age)
            age_bucket = age // 10 * 10
          select: [user_id, log_age, age_bucket]
        as: table_a_filtered
    dumps:
      - return: table_a_filtered
        path: ../data/output/table_a_processed.csv
"""

        (template_dir / "template_a.yaml").write_text(template_a_content)

        # Template B
        template_b_content = """\
depends:
  - template_a.yaml
processes:
  - name: process_b
    description: template B processes
    tables:
      - name: table_a
        path: ../data/output/table_a_processed.csv
      - name: table_b
        path: ../data/tables/table_b.csv
    steps:
      - name: filter_transactions
        with: table_b
        mutate:
          filter: amount > ${ params.min_amount }
        as: table_b_filtered

      - name: merge_transactions
        with: [table_a, table_b_filtered]
        transform: merge(left=table_a, right=table_b_filtered, on='user_id')
        as: merged

      - name: enrich_transactions
        with: merged
        mutate:
          derive: |
            high_value = amount > ${ params.threshold }
          select: ${params.output_columns}
        as: enriched
    dumps:
      - return: enriched
        path: ../output/final_output.csv
"""

        (template_dir / "template_b.yaml").write_text(template_b_content)

        # パラメータファイル
        params_content = """\
params:
  min_amount: 1000
  threshold: 10000
  output_columns:
    - user_id
    - log_age
    - amount
    - high_value
"""
        (param_dir / "params.yaml").write_text(params_content)

        yield {"template_dir": template_dir, "param_path": param_dir / "params.yaml"}


def test_pipeline_get_dependencies(setup_virtual_project):
    pipeline = Pipeline.from_template_dir(
        dirpath=str(setup_virtual_project["template_dir"]),
        param_path=str(setup_virtual_project["param_path"]),
        derive_func_paths=[],
        transform_func_paths=[],
    )

    print("template dir:", setup_virtual_project["template_dir"])
    print("template files:", list(Path(setup_virtual_project["template_dir"]).glob("*")))
    print("param path:", setup_virtual_project["param_path"])

    deps = pipeline.get_dependencies()

    print(f"Dependencies: {deps}")

    # === template_successors ===
    template_successors = deps["template_successors"]
    assert template_successors == {"template_a": ["template_b"]}

    # === process_successors ===
    process_successors = deps["process_successors"]
    assert process_successors == {}

    # === step_successors ===
    step_successors = deps["step_successors"]
    assert step_successors["../data/tables/table_a.csv"] == ["filter_users"]
    assert step_successors["filter_users"] == ["../data/tables/output/table_a_processed.csv"]

    assert step_successors["../data/tables/output/table_a_processed.csv"] == ["merge_transactions"]
    assert step_successors["../data/tables/table_b.csv"] == ["filter_transactions"]
    assert step_successors["filter_transactions"] == ["merge_transactions"]
    assert step_successors["merge_transactions"] == ["enrich_transactions"]
    assert step_successors["enrich_transactions"] == ["../output/final_output.csv"]

    # === template_to_process ===
    assert set(deps["template_to_process"].keys()) == {"template_a", "template_b"}
    assert "process_a" in deps["template_to_process"]["template_a"]
    assert "process_b" in deps["template_to_process"]["template_b"]

    # === process_to_step ===
    process_to_step = deps["process_to_step"]

    assert process_to_step["process_a"] == ["../data/tables/table_a.csv", "filter_users"]
    assert process_to_step["process_b"] == [
        "../data/tables/output/table_a_processed.csv",
        "../data/tables/table_b.csv",
        "filter_transactions",
        "merge_transactions",
        "enrich_transactions",
    ]
