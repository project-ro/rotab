import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import polars as pl
import fsspec
from core.parse import parse
from core.operation.derive_funcs_polars import *
from core.operation.transform_funcs_polars import *
from custom_functions.derive_funcs import *
from custom_functions.transform_funcs import *


def step_filter_users_user_filter(user):
    filtered_users = user
    filtered_users = filtered_users.filter(parse('age < 30'))
    filtered_users = filtered_users.with_columns(parse('age_group = age // 10'))
    filtered_users = filtered_users.select(['user_id', 'age', 'age_group'])
    return filtered_users


def user_filter():
    """Filter users under 30"""
    user = pl.scan_csv("data/inputs/user.csv", dtypes={"id": pl.Utf8, "user_id": pl.Utf8, "age": pl.Int64})
    filtered_users = step_filter_users_user_filter(user)
    filtered_users = filtered_users.with_columns(pl.col("user_id").cast(pl.Utf8))
    filtered_users = filtered_users.with_columns(pl.col("age").cast(pl.Int64))
    filtered_users = filtered_users.with_columns(pl.col("age_group").cast(pl.Int64))
    with fsspec.open("data/outputs/filtered_users.csv", "w") as f:
        filtered_users.collect(streaming=True).write_csv(f)
    return filtered_users


if __name__ == "__main__":
    user_filter()

