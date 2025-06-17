import os
import pandas as pd
from rotab.core.operation.derive_funcs import *
from rotab.core.operation.transform_funcs import *
from custom_functions import derive_funcs, transform_funcs


def step_filter_step_enrich_and_filter(user):
    filtered_users = user.copy()
    filtered_users = filtered_users.query('age > 20')
    filtered_users["log_age"] = filtered_users.apply(lambda row: log(row["age"]), axis=1)
    filtered_users["age_bucket"] = filtered_users.apply(lambda row: row["age"] // 10 * 10, axis=1)
    filtered_users = filtered_users[["user_id", "log_age", "age_bucket"]]
    return filtered_users


def step_join_step_enrich_and_filter(filtered_users, trans):
    final_output = merge_users_transactions(filtered_users, trans)
    return final_output


def enrich_and_filter():
    filtered_users = step_filter_step_enrich_and_filter(user)
    final_output = step_join_step_enrich_and_filter(filtered_users, trans)


if __name__ == "__main__":
    enrich_and_filter()

