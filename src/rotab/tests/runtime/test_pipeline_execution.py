import os
import tempfile
import yaml
import pandas as pd
import textwrap

from rotab.runtime.pipeline import Pipeline


def setup_full_test_env(tmpdir: str):
    # ディレクトリ準備
    template_dir = os.path.join(tmpdir, "templates")
    param_dir = os.path.join(tmpdir, "params")
    schema_dir = os.path.join(tmpdir, "schemas")
    input_dir = os.path.join(tmpdir, "input")
    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)
    os.makedirs(input_dir)

    # テンプレート定義
    template = {
        "name": "full_test_template",
        "processes": [
            {
                "name": "enrich_and_filter",
                "macros": {
                    "standard_user_filter": {
                        "steps": [
                            {
                                "name": "filter_step",
                                "with": "${caller.with}",
                                "mutate": [
                                    {"filter": "age > ${args.min_age}"},
                                    {"derive": "log_age = log(age)\nage_bucket = age // 10 * 10"},
                                    {"select": "${args.selected_cols}"},
                                ],
                                "as": "${caller.as}",
                            }
                        ]
                    }
                },
                "io": {
                    "inputs": [
                        {"name": "user", "type": "csv", "path": os.path.join(input_dir, "user.csv"), "schema": "user"},
                        {
                            "name": "trans",
                            "type": "csv",
                            "path": os.path.join(input_dir, "transaction.csv"),
                            "schema": "trans",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "final_output",
                            "type": "csv",
                            "path": os.path.join(tmpdir, "final_output.csv"),
                            "schema": "final_output",
                        }
                    ],
                },
                "steps": [
                    {
                        "name": "filter_users_main",
                        "with": "user",
                        "use": "standard_user_filter",
                        "args": {"min_age": "${params.min_age}", "selected_cols": ["user_id", "log_age", "age_bucket"]},
                        "as": "filtered_users",
                    },
                    {
                        "name": "join_step",
                        "with": ["filtered_users", "trans"],
                        "transform": "merge_users_transactions(filtered_users, trans)",
                        "as": "final_output",
                    },
                ],
            }
        ],
    }
    with open(os.path.join(template_dir, "template.yaml"), "w") as f:
        yaml.dump(template, f)

    # パラメータ定義
    with open(os.path.join(param_dir, "params.yaml"), "w") as f:
        yaml.dump({"params": {"min_age": 20}}, f)

    # スキーマ定義
    with open(os.path.join(schema_dir, "user.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "age": "int"}}, f)
    with open(os.path.join(schema_dir, "trans.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "amount": "float"}}, f)
    with open(os.path.join(schema_dir, "final_output.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "log_age": "float", "age_bucket": "int", "amount": "float"}}, f)

    # 入力CSV
    pd.DataFrame({"user_id": ["u1", "u2", "u3"], "age": [15, 25, 35]}).to_csv(
        os.path.join(input_dir, "user.csv"), index=False
    )
    pd.DataFrame({"user_id": ["u1", "u2", "u3"], "amount": [100.0, 200.0, 300.0]}).to_csv(
        os.path.join(input_dir, "transaction.csv"), index=False
    )

    # 関数定義
    derive_path = os.path.join(tmpdir, "derive_funcs.py")
    with open(derive_path, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                import math
                def log(x):
                    return math.log(x)
            """
            ).strip()
            + "\n"
        )

    transform_path = os.path.join(tmpdir, "transform_funcs.py")
    with open(transform_path, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                import pandas as pd
                def merge_users_transactions(filtered_users, trans):
                    return pd.merge(filtered_users, trans, on="user_id")
            """
            ).strip()
            + "\n"
        )

    return template_dir, param_dir, schema_dir, derive_path, transform_path, tmpdir


def test_pipeline_execution_with_all_features():
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir, param_dir, schema_dir, derive_path, transform_path, base_dir = setup_full_test_env(tmpdir)

        pipeline = Pipeline.from_setting(
            template_dir=template_dir,
            param_dir=param_dir,
            schema_dir=schema_dir,
            derive_func_path=derive_path,
            transform_func_path=transform_path,
        )
        pipeline.run(execute=True, dag=False)

        output_path = os.path.join(base_dir, "final_output.csv")
        assert os.path.exists(output_path), "final_output.csv not generated"

        df = pd.read_csv(output_path)
        assert list(df.columns) == ["user_id", "log_age", "age_bucket", "amount"]
        assert len(df) == 2  # u2 and u3 only (age > 20)
        assert set(df["user_id"]) == {"u2", "u3"}
