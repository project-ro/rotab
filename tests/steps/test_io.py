import os
import re
from rotab.ast.io_node import InputNode, OutputNode
from rotab.ast.context.validation_context import ValidationContext, VariableInfo


def test_input_node_with_columns(base_context: ValidationContext):
    node = InputNode(name="user", io_type="csv", path="data/users.csv", schema_name="user")

    node.validate(base_context)
    script = node.generate_script(base_context)

    assert script == [
        "user = pd.read_csv(\"data/users.csv\", dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})"
    ]


def test_input_node_without_columns():
    context = ValidationContext(
        available_vars=set(),
        eval_scope={},
        schemas={},
    )

    node = InputNode(name="empty_df", io_type="csv", path="data/empty.csv", schema_name=None)

    node.validate(context)
    script = node.generate_script(context)
    assert script == ['empty_df = pd.read_csv("data/empty.csv")']


def test_input_node_with_wildcard_column(tmp_path, base_context: ValidationContext):
    # 一時CSVファイル作成
    file1 = tmp_path / "202401.csv"
    file2 = tmp_path / "202402.csv"
    file1.write_text("user_id,age\n1,20\n2,30")
    file2.write_text("user_id,age\n3,40\n4,50")

    # globパスとして使う
    path_pattern = str(tmp_path / "*.csv")
    basename_pattern = os.path.basename(path_pattern)
    regex_pattern = re.escape(basename_pattern).replace("\\*", "(.+)")

    # 入力ノード定義
    node = InputNode(name="user", io_type="csv", path=path_pattern, wildcard_column="yyyymm", schema_name="user")

    node.validate(base_context)
    script = node.generate_script(base_context)

    assert script == [
        "import glob, os, re",
        f"user_files = glob.glob('{path_pattern}')",
        "user_df_list = []",
        f"_regex = re.compile(r'{regex_pattern}')",
        "for _file in user_files:",
        "    _basename = os.path.basename(_file)",
        "    _match = _regex.match(_basename)",
        "    if not _match: raise ValueError(f'Unexpected filename: {_basename}')",
        "    _val = _match.group(1)",
        "    _df = pd.read_csv(_file, dtype={'user_id': 'str', 'age': 'int', 'log_age': 'float', 'age_bucket': 'int'})",
        "    _df['yyyymm'] = _val",
        "    _df['yyyymm'] = _df['yyyymm'].astype(str)",
        "    user_df_list.append(_df)",
        "user = pd.concat(user_df_list, ignore_index=True)",
    ]


def test_output_node_with_columns(base_context: ValidationContext):
    node = OutputNode(name="user", io_type="csv", path="data/users_out.csv", schema_name="user")

    node.validate(base_context)
    script = node.generate_script(base_context)

    assert script[:4] == [
        'user["user_id"] = user["user_id"].astype("str")',
        'user["age"] = user["age"].astype("int")',
        'user["log_age"] = user["log_age"].astype("float")',
        'user["age_bucket"] = user["age_bucket"].astype("int")',
    ]
    assert (
        script[4]
        == "user.to_csv(\"data/users_out.csv\", index=False, columns=['user_id', 'age', 'log_age', 'age_bucket'])"
    )


def test_output_node_without_columns():
    context = ValidationContext(
        available_vars={"no_schema_df"},
        eval_scope={},
        schemas={"no_schema_df": VariableInfo(type="dataframe", columns={})},
    )

    node = OutputNode(name="no_schema_df", io_type="csv", path="data/no_schema.csv", schema_name=None)

    node.validate(context)
    script = node.generate_script(context)
    assert script == ['no_schema_df.to_csv("data/no_schema.csv", index=False)']
