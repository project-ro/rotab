import pytest
from rotab.core.operation.validator import TemplateValidator


def assert_error(cfg, expect_error):
    validator = TemplateValidator(cfg)
    validator.validate()
    if validator.errors:
        for error in validator.errors:
            print(f"- {error}")
    assert expect_error == bool(
        validator.errors
    ), f"Expected error: {expect_error}, but got: {[str(e) for e in validator.errors]}"


@pytest.mark.parametrize(
    "expression, expect_error",
    [
        ("log_age = log()", True),
        ("log_age = log(1, 2)", False),
        ("log_age = log(1, 2, 3)", True),
        ("x = y in ['aa', 'bb']", False),
        ("y = 'hoge' in x", False),
        ("x = y in z", False),
        ("log_age == log(1)", True),
        ("x =", True),
        ("x = unknown_func(1)", True),
        ("x = eval('2+2')", True),
        ("x = y = 1", True),
    ],
)
def test_derive_cases(expression, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {"inputs": [{"name": "user", "path": "x.csv"}], "outputs": [{"name": "user", "path": "y.csv"}]},
                "steps": [{"with": "user", "mutate": [{"derive": expression}], "as": "user"}],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "step, expect_error",
    [
        ("merge()", True),
        ("merge(left='a', right='b')", True),
        ("merge(left='a', right='b', on='id')", False),
        ("result = merge(left='a', right='b', on='id')", True),
    ],
)
def test_transform_cases(step, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {
                    "inputs": [{"name": "user", "path": "x.csv"}],
                    "outputs": [{"name": "user", "path": "y.csv"}],
                },
                "steps": [{"with": ["user", "user"], "transform": step, "as": "user"}],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "step, expect_error",
    [
        ({"mutate": [{"derive": "a = 1"}], "transform": "b = log(1)"}, True),
    ],
)
def test_mutate_and_transform_conflict(step, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {
                    "inputs": [{"name": "x", "path": "x.csv"}],
                    "outputs": [{"name": "x", "path": "x.csv"}],
                },
                "steps": [{"with": "x", **step}],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "tables, expect_error",
    [
        ([{"name": "x", "path": "x.csv"}, {"name": "x", "path": "y.csv"}], True),
    ],
)
def test_duplicate_table_names_case(tables, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {
                    "inputs": tables,
                    "outputs": [{"name": "x", "path": "z.csv"}],
                },
                "steps": [],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "outputs, expect_error",
    [
        ([{"name": "x", "path": "a.csv"}, {"name": "x", "path": "b.csv"}], True),
    ],
)
def test_duplicate_dump_outputs(outputs, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {"inputs": [{"name": "x", "path": "x.csv"}], "outputs": outputs},
                "steps": [{"mutate": [{"derive": "x = log(1)"}], "with": "x"}],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "path1, path2, expect_error",
    [
        ("exam|ple.csv", "out>put.csv", True),
    ],
)
def test_path_invalid_chars_case(path1, path2, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {"inputs": [{"name": "x", "path": path1}], "outputs": [{"name": "x", "path": path2}]},
                "steps": [],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "with_name, expect_error",
    [
        ("y", True),
    ],
)
def test_with_unknown_table_case(with_name, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {
                    "inputs": [{"name": "x", "path": "x.csv"}],
                    "outputs": [{"name": "x", "path": "y.csv"}],
                },
                "steps": [{"with": with_name}],
            }
        ]
    }
    assert_error(cfg, expect_error)


@pytest.mark.parametrize(
    "outputs, expect_error",
    [
        ([{"output": "y", "path": "y.csv"}], True),
    ],
)
def test_dump_unknown_output_case(outputs, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "io": {
                    "inputs": [{"name": "x", "path": "x.csv"}],
                    "outputs": outputs,
                },
                "steps": [],
            }
        ]
    }
    assert_error(cfg, expect_error)
