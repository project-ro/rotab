import os
import tempfile
import shutil
import yaml
import pytest

from rotab.loader import Loader
from rotab.loader.schema_manager import SchemaManager
from rotab.ast.step_node import MutateStep
from rotab.ast.io_node import InputNode


@pytest.fixture
def temp_dirs():
    temp_root = tempfile.mkdtemp()
    template_dir = os.path.join(temp_root, "templates")
    param_dir = os.path.join(temp_root, "params")
    schema_dir = os.path.join(temp_root, "schemas")
    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)
    yield template_dir, param_dir, schema_dir
    shutil.rmtree(temp_root)


def write_yaml(path, content):
    with open(path, "w") as f:
        yaml.dump(content, f)


@pytest.mark.parametrize(
    "filename, template_content, expected_error, match_msg",
    [
        # 依存解決失敗
        (
            "main_template.yaml",
            {"name": "main_template", "depends": ["not_exists"], "processes": []},
            ValueError,
            "Missing dependency: not_exists",
        ),
        # YAML不正
        ("bad.yaml", "- just: a list", ValueError, "Invalid YAML format"),
        # Macro未定義
        (
            "main_template.yaml",
            {
                "name": "main_template",
                "depends": [],
                "processes": [
                    {
                        "name": "p",
                        "steps": [{"name": "fail", "use": "undefined_macro", "with": "x", "as": "y", "args": {}}],
                    }
                ],
            },
            KeyError,
            "Macro `undefined_macro`",
        ),
        # パラメータ未定義
        (
            "main_template.yaml",
            {
                "name": "main_template",
                "depends": [],
                "processes": [
                    {
                        "name": "p",
                        "steps": [
                            {
                                "name": "fail",
                                "with": "x",
                                "mutate": [{"filter": "age > ${params.not_found}"}],
                                "as": "y",
                            }
                        ],
                    }
                ],
            },
            KeyError,
            "not_found",
        ),
        # スキーマ未定義
        (
            "main_template.yaml",
            {
                "name": "main_template",
                "depends": [],
                "processes": [
                    {
                        "name": "p",
                        "io": {
                            "inputs": [{"name": "x", "io_type": "csv", "path": "x.csv", "schema": "not_found"}],
                            "outputs": [{"name": "y", "io_type": "csv", "path": "y.csv", "schema": "not_found"}],
                        },
                        "steps": [],
                    }
                ],
            },
            FileNotFoundError,
            "not_found",
        ),
    ],
)
def test_loader_abnormal_cases(temp_dirs, filename, template_content, expected_error, match_msg):
    template_dir, param_dir, schema_dir = temp_dirs
    path = os.path.join(template_dir, filename)
    if isinstance(template_content, dict):
        write_yaml(path, template_content)
    else:
        with open(path, "w") as f:
            f.write(template_content)
    write_yaml(os.path.join(param_dir, "params.yaml"), {})
    write_yaml(os.path.join(schema_dir, "user.yaml"), {"columns": {"user_id": "str"}})
    schema_manager = SchemaManager(schema_dir)
    loader = Loader(template_dir, param_dir, schema_manager)
    with pytest.raises(expected_error, match=match_msg):
        loader.load()


def test_loader_full_features_with_depends_macro_and_param(temp_dirs):
    template_dir, param_dir, schema_dir = temp_dirs

    # パラメータ
    write_yaml(
        os.path.join(param_dir, "params.yaml"),
        {
            "params": {
                "min_age": 18,
                "test": True,
                "enrich_transactions": {"columns": ["user_id", "log_age", "amount", "high_value"]},
            }
        },
    )

    # スキーマ
    write_yaml(
        os.path.join(schema_dir, "user.yaml"),
        {"columns": {"user_id": "str", "age": "int", "log_age": "float", "age_bucket": "int"}},
    )
    write_yaml(os.path.join(schema_dir, "trans.yaml"), {"columns": {"user_id": "str", "amount": "int"}})
    write_yaml(
        os.path.join(schema_dir, "final_output.yaml"),
        {"columns": {"user_id": "str", "log_age": "float", "amount": "int", "high_value": "bool"}},
    )

    # user_filter.yaml
    write_yaml(
        os.path.join(template_dir, "user_filter.yaml"),
        {
            "name": "user_filter",
            "depends": [],
            "processes": [
                {
                    "name": "user_filter_proc",
                    "io": {
                        "inputs": [{"name": "user", "io_type": "csv", "path": "user.csv", "schema": "user"}],
                        "outputs": [
                            {"name": "filtered_users", "io_type": "csv", "path": "filtered.csv", "schema": "user"}
                        ],
                    },
                    "steps": [
                        {
                            "name": "filter_step",
                            "with": "user",
                            "mutate": [{"filter": "age > 18"}, {"select": ["user_id", "age"]}],
                            "as": "filtered_users",
                        }
                    ],
                }
            ],
        },
    )

    # trans_summary.yaml
    write_yaml(
        os.path.join(template_dir, "trans_summary.yaml"),
        {
            "name": "trans_summary",
            "depends": [],
            "processes": [
                {
                    "name": "trans_summary_proc",
                    "io": {
                        "inputs": [{"name": "trans", "io_type": "csv", "path": "trans.csv", "schema": "trans"}],
                        "outputs": [
                            {
                                "name": "filtered_trans",
                                "io_type": "csv",
                                "path": "filtered_trans.csv",
                                "schema": "trans",
                            }
                        ],
                    },
                    "steps": [
                        {
                            "name": "trans_filter_step",
                            "with": "trans",
                            "mutate": [{"filter": "amount > 1000"}, {"select": ["user_id", "amount"]}],
                            "as": "filtered_trans",
                        }
                    ],
                }
            ],
        },
    )

    # main_template.yaml（マクロを含む、全ての構文パターン）
    write_yaml(
        os.path.join(template_dir, "main_template.yaml"),
        {
            "name": "main_template",
            "depends": ["user_filter", "trans_summary"],
            "processes": [
                {
                    "name": "transaction_enrichment",
                    "macros": {
                        "enrich_macro": {
                            "steps": [
                                {
                                    "name": "enrich_step",
                                    "with": "${caller.with}",
                                    "mutate": [
                                        {"derive": "high_value = amount > 10000"},
                                        {"select": "${args.columns}"},
                                    ],
                                    "as": "${caller.as}",
                                }
                            ]
                        }
                    },
                    "io": {
                        "inputs": [
                            {
                                "name": "user",
                                "io_type": "csv",
                                "path": "../../output/filtered_users.csv",
                                "schema": "user",
                            },
                            {
                                "name": "trans",
                                "io_type": "csv",
                                "path": "../../output/filtered_transactions.csv",
                                "schema": "trans",
                            },
                        ],
                        "outputs": [
                            {
                                "name": "final_output",
                                "io_type": "csv",
                                "path": "../../output/final_output.csv",
                                "schema": "final_output",
                            }
                        ],
                    },
                    "steps": [
                        {
                            "name": "filter_users_main",
                            "with": "user",
                            "mutate": [
                                {"filter": "age > ${params.min_age}"},
                                {"derive": "log_age = log(age)\nage_bucket = age // 10 * 10"},
                                {"select": ["user_id", "log_age", "age_bucket"]},
                            ],
                            "as": "filtered_users",
                            "when": "${params.test}",
                        },
                        {
                            "name": "filter_transactions_main",
                            "with": "trans",
                            "mutate": [{"filter": "amount > 1000"}],
                            "as": "filtered_trans",
                        },
                        {
                            "name": "merge_transactions",
                            "with": ["filtered_users", "filtered_trans"],
                            "transform": "merge(left=filtered_users, right=filtered_trans, on='user_id')",
                            "as": "enriched",
                        },
                        {
                            "name": "enrich_macro_use",
                            "use": "enrich_macro",
                            "with": "enriched",
                            "as": "final_output",
                            "args": {"columns": "${params.enrich_transactions.columns}"},
                        },
                    ],
                }
            ],
        },
    )

    schema_manager = SchemaManager(schema_dir)

    loader = Loader(template_dir, param_dir, schema_manager)
    result = loader.load()

    # 依存順で返る
    assert [tpl.name for tpl in result] == ["user_filter", "trans_summary", "main_template"]

    # main_templateのASTチェック
    tpl = result[2]
    proc = tpl.processes[0]
    i0 = proc.inputs[0]
    assert isinstance(i0, InputNode)
    s0 = proc.steps[0]
    assert isinstance(s0, MutateStep)
    assert s0.name == "filter_users_main"
    assert s0.operations[0]["filter"] == "age > 18"
    assert "log_age = log(age)" in s0.operations[1]["derive"]
    assert "age_bucket = age // 10 * 10" in s0.operations[1]["derive"]
    assert s0.operations[2]["select"] == ["user_id", "log_age", "age_bucket"]
    s1 = proc.steps[1]
    assert s1.name == "filter_transactions_main"
    assert s1.operations[0]["filter"] == "amount > 1000"
    s2 = proc.steps[2]
    assert s2.name == "merge_transactions"
    assert s2.expr == "merge(left=filtered_users, right=filtered_trans, on='user_id')"
    # enrich_macroの展開
    s3 = proc.steps[3]
    assert s3.name == "enrich_macro_use"
    assert "high_value = amount > 10000" in s3.operations[0]["derive"]
    assert s3.operations[1]["select"] == ["user_id", "log_age", "amount", "high_value"]
