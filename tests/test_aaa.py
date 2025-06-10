import pytest
from types import SimpleNamespace
from rotab.core.operation.validator import TemplateValidator, ValidationError


# keyword風のダミー構造
def make_kw(arg, value):
    return SimpleNamespace(arg=arg, value=value)


# ダミー関数として log を定義（実際のバリデーション対象）
def log(x: float, base: float = 10) -> float:
    return x


@pytest.mark.parametrize(
    "args, keywords, expect_error",
    [
        ([], [], True),  # 引数なし → エラー
        ([1], [], False),  # 引数1つ → OK
        ([1, 2], [], False),  # 引数2つ → OK
        ([1, 2, 3], [], True),  # 多すぎ → エラー
        ([1], [make_kw("base", 20)], False),  # baseをキーワード指定 → OK
        ([1], [make_kw("base", 10), make_kw("base", 20)], True),  # 重複 → エラー
    ],
)
def test_check_function_signature(args, keywords, expect_error):
    validator = TemplateValidator(eval_scope={"log": log})
    validator._check_function_signature("log", args, keywords, path="test.path")

    if expect_error:
        assert len(validator.errors) == 1, f"Expected error but got none: args={args}, keywords={keywords}"
        print("Error:", validator.errors[0])
    else:
        assert not validator.errors, f"Unexpected error: {validator.errors}"
