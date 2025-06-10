import pytest
from rotab.core.operation.validator import TemplateValidator


@pytest.mark.parametrize(
    "expression, expect_error",
    [
        ("log_age = log()", True),
        # ("log_age = log(1, 2)", False),
        # ("log_age = log(1, 2, 3)", True),
        # ("x = y in ['aa', 'bb']", False),
        # ("y = 'hoge' in x", False),
        # ("x = y in z", False),
        # ("log_age == log(1)", True),
        # ("x =", True),
        # ("x = unknown_func(1)", True),
        # ("x = eval('2+2')", True),
        # ("x = y = 1", True),
    ],
)
def test_derive_cases(expression, expect_error):
    cfg = {
        "processes": [
            {
                "name": "p",
                "tables": [{"name": "user", "path": "x.csv"}],
                "steps": [{"with": "user", "mutate": [{"derive": expression}], "as": "user"}],
                "dumps": [{"output": "user", "path": "y.csv"}],
            }
        ]
    }
    validator = TemplateValidator(cfg)
    validator.validate()

    print("ERRORS:")
    for error in validator.errors:
        print(f" - {error.message}")

    assert expect_error == bool(validator.errors), f"Unexpected error state for: {expression}"


# @pytest.mark.parametrize(
#     "step, expect_error",
#     [
#         ("result = merge()", True),
#         ("result = merge(left='a', right='b')", True),
#         ("result = merge(left='a', right='b', on='id')", False),
#     ],
# )
# def test_transform_cases(step, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "user", "path": "x.csv"}],
#                 "steps": [{"with": ["user", "user"], "transform": step}],
#                 "dumps": [{"output": "user", "path": "y.csv"}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == bool(validator.errors), f"Unexpected error state for: {step}"


# @pytest.mark.parametrize(
#     "step, expect_error",
#     [
#         ({"mutate": [{"derive": "a = 1"}], "transform": "b = log(1)"}, True),
#     ],
# )
# def test_mutate_and_transform_conflict(step, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": "x.csv"}],
#                 "steps": [{"with": "x", **step}],
#                 "dumps": [{"output": "x", "path": "x.csv"}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("Cannot use both" in e.message for e in validator.errors)


# @pytest.mark.parametrize(
#     "tables, expect_error",
#     [
#         ([{"name": "x", "path": "x.csv"}, {"name": "x", "path": "y.csv"}], True),
#     ],
# )
# def test_duplicate_table_names_case(tables, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": tables,
#                 "steps": [],
#                 "dumps": [{"output": "x", "path": "z.csv"}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("Duplicate table name" in e.message for e in validator.errors)


# @pytest.mark.parametrize(
#     "dumps, expect_error",
#     [
#         ([{"output": "x", "path": "a.csv"}, {"output": "x", "path": "b.csv"}], True),
#     ],
# )
# def test_duplicate_dump_outputs(dumps, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": "x.csv"}],
#                 "steps": [{"mutate": [{"derive": "x = log(1)"}], "with": "x"}],
#                 "dumps": dumps,
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("Duplicate dump output" in e.message for e in validator.errors)


# @pytest.mark.parametrize(
#     "path1, path2, expect_error",
#     [
#         ("exam|ple.csv", "out>put.csv", True),
#     ],
# )
# def test_path_invalid_chars_case(path1, path2, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": path1}],
#                 "steps": [],
#                 "dumps": [{"output": "x", "path": path2}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("Invalid characters in path" in e.message for e in validator.errors)


# @pytest.mark.parametrize(
#     "step, expect_error",
#     [
#         ({"with": "x", "mutate": [{"columns": ["undefined_column"]}]}, False),
#     ],
# )
# def test_columns_with_undefined_column_case(step, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": "x.csv"}],
#                 "steps": [step],
#                 "dumps": [{"output": "x", "path": "y.csv"}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("columns" in e.message.lower() for e in validator.errors)


# @pytest.mark.parametrize(
#     "with_name, expect_error",
#     [
#         ("y", True),
#     ],
# )
# def test_with_unknown_table_case(with_name, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": "x.csv"}],
#                 "steps": [{"with": with_name}],
#                 "dumps": [{"output": "x", "path": "y.csv"}],
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("with" in e.message.lower() for e in validator.errors)


# @pytest.mark.parametrize(
#     "dumps, expect_error",
#     [
#         ([{"output": "y", "path": "y.csv"}], True),
#     ],
# )
# def test_dump_unknown_output_case(dumps, expect_error):
#     cfg = {
#         "processes": [
#             {
#                 "name": "p",
#                 "tables": [{"name": "x", "path": "x.csv"}],
#                 "steps": [],
#                 "dumps": dumps,
#             }
#         ]
#     }
#     validator = TemplateValidator(cfg)
#     validator.validate()
#     assert expect_error == any("output" in e.message.lower() for e in validator.errors)
