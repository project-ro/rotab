import tempfile
import shutil
import os
import yaml
from pathlib import Path
import pytest
import pandas as pd
from rotab.core.pipeline import Pipeline


@pytest.fixture
def setup_virtual_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        tables_dir = data_dir / "tables"
        output_dir = data_dir / "output"
        template_dir = root / "template"

        data_dir.mkdir()
        tables_dir.mkdir()
        output_dir.mkdir()
        template_dir.mkdir()

        # CSVファイルを配置
        pd.DataFrame({"user_id": [1, 2], "age": [25, 15]}).to_csv(tables_dir / "table_a.csv", index=False)
        pd.DataFrame({"user_id": [1, 2], "amount": [1500, 500]}).to_csv(tables_dir / "table_b.csv", index=False)

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
        filter: age > 18
        define: |
          log_age = log(age)
          age_bucket = age // 10 * 10
        select: [user_id, log_age, age_bucket]
    dumps:
      - return: table_a
        path: ../data/tables/output/table_a_processed.csv
"""

        template_b_content = """\
depends:
    - template_a.yaml
processes:
  - name: process_b
    description: template B processes
    tables:
      - name: table_a
        path: ../data/tables/output/table_a_processed.csv
      - name: table_b
        path: ../data/tables/table_b.csv
    steps:
      - name: filter_transactions
        with: table_b
        filter: amount > 1000
      - name: merge_transactions
        transform: merged = merge(left=table_a, right=table_b, on='user_id')
      - name: enrich_transactions
        with: merged
        define: high_value = amount > 10000
        select: [user_id, log_age, amount, high_value]
    dumps:
      - return: merged
        path: ../output/final_output.csv
"""

        template_a_path = template_dir / "template_a.yaml"
        template_a_path.write_text(template_a_content)

        template_b_path = template_dir / "template_b.yaml"
        template_b_path.write_text(template_b_content)

        yield template_dir


def test_pipeline_get_dependencies(setup_virtual_project):
    pipeline = Pipeline.from_template_dir(
        dirpath=str(setup_virtual_project),
        define_func_paths=[],
        transform_func_paths=[],
    )

    print("template_dir:", setup_virtual_project)
    print("files:", list(Path(setup_virtual_project).glob("*")))

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
