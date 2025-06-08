import pytest
from rotab.core.operation.validator import TemplateValidator


@pytest.mark.parametrize(
    "step, expect_error",
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
def test_new_columns_cases(step, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "user", "path": "x.csv"}],
                "steps": [{"with": "user", "new_columns": step}],
                "dumps": [{"return": "user", "path": "y.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    if expect_error:
        assert validator.errors, f"Expected error not raised for: {step}"
    else:
        assert not validator.errors, f"Unexpected error: {validator.errors}"


@pytest.mark.parametrize(
    "step, expect_error",
    [
        ("result = merge()", True),
        ("result = merge(left='a', right='b')", True),
        ("result = merge(left='a', right='b', on='id')", False),
    ],
)
def test_dataframes_cases(step, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "user", "path": "x.csv"}],
                "steps": [{"dataframes": step}],
                "dumps": [{"return": "user", "path": "y.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    if expect_error:
        assert validator.errors, f"Expected error not raised for: {step}"
    else:
        assert not validator.errors, f"Unexpected error: {validator.errors}"


@pytest.mark.parametrize(
    "step, expect_error",
    [
        ({"new_columns": "a = 1", "dataframes": "b = log(1)"}, True),
    ],
)
def test_new_columns_and_dataframes_conflict(step, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "x", "path": "x.csv"}],
                "steps": [{"with": "x", **step}],
                "dumps": [{"return": "x", "path": "x.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("Cannot use both" in e.message for e in validator.errors)


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
                "process": "p",
                "tables": tables,
                "steps": [],
                "dumps": [{"return": "x", "path": "z.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("Duplicate table name" in e.message for e in validator.errors)


@pytest.mark.parametrize(
    "dumps, expect_error",
    [
        ([{"return": "x", "path": "a.csv"}, {"return": "x", "path": "b.csv"}], True),
    ],
)
def test_duplicate_dump_returns(dumps, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "x", "path": "x.csv"}],
                "steps": [{"dataframes": "x = log(1)"}],
                "dumps": dumps,
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("Duplicate dump return" in e.message for e in validator.errors)


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
                "process": "p",
                "tables": [{"name": "x", "path": path1}],
                "steps": [],
                "dumps": [{"return": "x", "path": path2}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("Invalid characters in path" in e.message for e in validator.errors)


@pytest.mark.parametrize(
    "step, expect_error",
    [
        ({"with": "x", "columns": ["undefined_column"]}, False),
    ],
)
def test_columns_with_undefined_column_case(step, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "x", "path": "x.csv"}],
                "steps": [step],
                "dumps": [{"return": "x", "path": "y.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("columns" in e.message.lower() for e in validator.errors)


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
                "process": "p",
                "tables": [{"name": "x", "path": "x.csv"}],
                "steps": [{"with": with_name}],
                "dumps": [{"return": "x", "path": "y.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("with" in e.message.lower() for e in validator.errors)


@pytest.mark.parametrize(
    "dumps, expect_error",
    [
        ([{"return": "y", "path": "y.csv"}], True),
    ],
)
def test_dump_unknown_return_case(dumps, expect_error):
    cfg = {
        "processes": [
            {
                "process": "p",
                "tables": [{"name": "x", "path": "x.csv"}],
                "steps": [],
                "dumps": dumps,
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()
    assert expect_error == any("return" in e.message.lower() for e in validator.errors)
