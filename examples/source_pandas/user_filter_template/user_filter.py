import os
import pandas as pd
from rotab.core.operation.derive_funcs_pandas import *
from rotab.core.operation.transform_funcs_pandas import *


def step_filter_users_user_filter(user):
    filtered_users = user.copy()
    filtered_users = filtered_users.query('age < 30').copy()
    filtered_users["age_group"] = filtered_users.apply(lambda row: row["age"] // 10, axis=1)
    filtered_users = filtered_users[["user_id", "age", "age_group"]]
    return filtered_users


def user_filter():
    """Filter users under 30"""
    user = pd.read_csv("data/inputs/user.csv", dtype={'id': 'str', 'user_id': 'str', 'age': 'int'})
    filtered_users = step_filter_users_user_filter(user)
    filtered_users["user_id"] = filtered_users["user_id"].astype("str")
    filtered_users["age"] = filtered_users["age"].astype("int")
    filtered_users["age_group"] = filtered_users["age_group"].astype("int")
    filtered_users.to_csv("data/outputs/filtered_users.csv", index=False, columns=['user_id', 'age', 'age_group'])
    return filtered_users


if __name__ == "__main__":
    user_filter()

