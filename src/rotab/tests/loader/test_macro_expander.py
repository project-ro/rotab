import pytest
from rotab.loader import MacroExpander


@pytest.fixture
def macro_definitions():
    return {
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
    }


@pytest.mark.parametrize(
    "input_steps, expected",
    [
        (
            [
                {
                    "name": "filter_users_macro",
                    "use": "standard_user_filter",
                    "with": "user",
                    "as": "filtered_users",
                    "args": {"min_age": 30, "selected_cols": ["user_id", "log_age", "age_bucket"]},
                }
            ],
            [
                {
                    "name": "filter_step",
                    "with": "user",
                    "mutate": [
                        {"filter": "age > 30"},
                        {"derive": "log_age = log(age)\nage_bucket = age // 10 * 10"},
                        {"select": ["user_id", "log_age", "age_bucket"]},
                    ],
                    "as": "filtered_users",
                }
            ],
        )
    ],
)
def test_macro_expansion_success(macro_definitions, input_steps, expected):
    expander = MacroExpander(macro_definitions)
    result = expander.expand(input_steps)
    assert result == expected


@pytest.mark.parametrize(
    "macros, input_steps, expected_exception",
    [
        # マクロ未定義
        ({}, [{"name": "bad_macro", "use": "nonexistent_macro", "with": "df", "as": "out", "args": {}}], KeyError),
        # args が足りない
        (
            {
                "standard_user_filter": {
                    "steps": [
                        {
                            "name": "step",
                            "with": "${caller.with}",
                            "as": "${caller.as}",
                            "mutate": [{"filter": "age > ${args.min_age}"}],
                        }
                    ]
                }
            },
            [{"name": "missing_args", "use": "standard_user_filter", "with": "df", "as": "out"}],
            KeyError,
        ),
        # args が dict でない
        (
            {"macro": {"steps": [{"name": "step", "with": "${caller.with}"}]}},
            [{"name": "invalid_args", "use": "macro", "with": "df", "as": "out", "args": "not a dict"}],
            TypeError,
        ),
        # 展開対象の steps が list でない
        (
            {"macro": {"steps": {"name": "step", "with": "${caller.with}"}}},
            [{"name": "macro_step", "use": "macro", "with": "df", "as": "out", "args": {}}],
            ValueError,
        ),
    ],
)
def test_macro_expansion_failure(macros, input_steps, expected_exception):
    expander = MacroExpander(macros)
    with pytest.raises(expected_exception):
        expander.expand(input_steps)
