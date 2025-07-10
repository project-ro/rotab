import os
import tempfile
import yaml
import pandas as pd
import textwrap
import pytest

from rotab.core.pipeline import Pipeline


def setup_full_test_env(tmpdir: str):
    template_dir = os.path.join(tmpdir, "templates")
    param_dir = os.path.join(tmpdir, "params")
    schema_dir = os.path.join(tmpdir, "schemas")
    input_dir = os.path.join(template_dir, "input")
    source_dir = os.path.join(tmpdir, "runspace")

    os.makedirs(template_dir)
    os.makedirs(param_dir)
    os.makedirs(schema_dir)
    os.makedirs(input_dir)
    os.makedirs(source_dir)

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
                                    {
                                        "derive": textwrap.dedent(
                                            """\
                                            log_age = ${args.log_func}(age)
                                            age_bucket = age // 10 * 10
                                            """
                                        ).strip()
                                    },
                                    {"select": "${args.selected_cols}"},
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
                            "path": "input/user.csv",
                            "schema": "user",
                        },
                        {
                            "name": "trans",
                            "io_type": "csv",
                            "path": "input/transaction.csv",
                            "schema": "trans",
                        },
                    ],
                    "outputs": [
                        {
                            "name": "final_output",
                            "io_type": "csv",
                            "path": "final_output.csv",
                            "schema": "final_output",
                        }
                    ],
                },
                "steps": [
                    {
                        "name": "filter_users_main",
                        "with": "user",
                        "use": "standard_user_filter",
                        "args": {
                            "min_age": "${params.min_age}",
                            "selected_cols": ["user_id", "log_age", "age_bucket"],
                            "log_func": "${params.log_func}",
                        },
                        "as": "filtered_users",
                    },
                    {
                        "name": "join_step",
                        "with": ["filtered_users", "trans"],
                        "transform": "${params.merge_func}(filtered_users, trans)",
                        "as": "final_output",
                    },
                ],
            }
        ],
    }
    with open(os.path.join(template_dir, "template.yaml"), "w") as f:
        yaml.dump(template, f)

    with open(os.path.join(param_dir, "params.yaml"), "w") as f:
        yaml.dump({"params": {"min_age": 20}}, f)

    with open(os.path.join(schema_dir, "user.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "age": "int"}}, f)
    with open(os.path.join(schema_dir, "trans.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "amount": "float"}}, f)
    with open(os.path.join(schema_dir, "final_output.yaml"), "w") as f:
        yaml.dump({"columns": {"user_id": "str", "log_age": "float", "age_bucket": "int", "amount": "float"}}, f)

    pd.DataFrame({"user_id": ["u1", "u2", "u3"], "age": [15, 25, 35]}).to_csv(
        os.path.join(input_dir, "user.csv"), index=False
    )
    pd.DataFrame({"user_id": ["u1", "u2", "u3"], "amount": [100.0, 200.0, 300.0]}).to_csv(
        os.path.join(input_dir, "transaction.csv"), index=False
    )

    derive_path = os.path.join(tmpdir, "derive_funcs.py")
    with open(derive_path, "w") as f:
        f.write(
            textwrap.dedent(
                """\
                import polars as pl
                import math
                from rotab.core.operation.derive_funcs import FUNC_NAMESPACE

                def custom_log_polars(x):
                    return pl.col(x).log()

                def custom_log_pandas(x):
                    return math.log(x)

                FUNC_NAMESPACE["custom_log_polars"] = custom_log_polars
                FUNC_NAMESPACE["custom_log_pandas"] = custom_log_pandas
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
                import polars as pl
                from rotab.core.operation.derive_funcs import FUNC_NAMESPACE

                def merge_users_transactions_pandas(filtered_users, trans):
                    return pd.merge(filtered_users, trans, on="user_id")

                def merge_users_transactions_polars(filtered_users, trans):
                    return filtered_users.join(trans, on="user_id", how="inner")

                FUNC_NAMESPACE["merge_users_transactions_pandas"] = merge_users_transactions_pandas
                FUNC_NAMESPACE["merge_users_transactions_polars"] = merge_users_transactions_polars
                """
            ).strip()
            + "\n"
        )

    return template_dir, param_dir, schema_dir, derive_path, transform_path, source_dir


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_pipeline_execution_with_all_features(backend):
    with tempfile.TemporaryDirectory() as tmpdir:
        template_dir, param_dir, schema_dir, derive_path, transform_path, source_dir = setup_full_test_env(tmpdir)

        param_file = os.path.join(param_dir, "params.yaml")
        with open(param_file) as f:
            params = yaml.safe_load(f)

        if backend == "pandas":
            params["params"]["log_func"] = "custom_log_pandas"
            params["params"]["merge_func"] = "merge_users_transactions_pandas"
        else:
            params["params"]["log_func"] = "custom_log_polars"
            params["params"]["merge_func"] = "merge_users_transactions_polars"

        with open(param_file, "w") as f:
            yaml.dump(params, f)

        pipeline = Pipeline.from_setting(
            template_dir=template_dir,
            source_dir=source_dir,
            param_dir=param_dir,
            schema_dir=schema_dir,
            derive_func_path=derive_path,
            transform_func_path=transform_path,
            backend=backend,
        )
        pipeline.run(execute=True, dag=False)

        output_path = os.path.join(source_dir, "data", "outputs", "final_output.csv")
        assert os.path.exists(output_path), "final_output.csv not generated"

        df = pd.read_csv(output_path)
        assert set(df.columns) == {"user_id", "log_age", "age_bucket", "amount"}
        assert len(df) == 2
        assert set(df["user_id"]) == {"u2", "u3"}
