import os
import pandas as pd
from rotab.core.operation.derive_funcs import *
from rotab.core.operation.transform_funcs import *
from custom_functions.derive_funcs import *
from custom_functions.transform_funcs import *


def step_filter_users_main_enrich_and_filter(user):
    filtered_users = user.copy()
    filtered_users = filtered_users.query('age > 20').copy()
    filtered_users["log_age"] = filtered_users.apply(lambda row: custom_log(row["age"]), axis=1)
    filtered_users["age_bucket"] = filtered_users.apply(lambda row: row["age"] // 10 * 10, axis=1)
    filtered_users = filtered_users[["user_id", "log_age", "age_bucket"]]
    return filtered_users


def step_join_step_enrich_and_filter(filtered_users, trans):
    final_output = merge_users_transactions(filtered_users, trans)
    return final_output


def enrich_and_filter():
    user = pd.read_csv("/tmp/tmpn7dog1oo/input/user.csv", dtype={'age': 'int', 'user_id': 'str'})
    trans = pd.read_csv("/tmp/tmpn7dog1oo/input/transaction.csv", dtype={'amount': 'float', 'user_id': 'str'})
    filtered_users = step_filter_users_main_enrich_and_filter(user)
    final_output = step_join_step_enrich_and_filter(filtered_users, trans)
    final_output["age_bucket"] = final_output["age_bucket"].astype("int")
    final_output["amount"] = final_output["amount"].astype("float")
    final_output["log_age"] = final_output["log_age"].astype("float")
    final_output["user_id"] = final_output["user_id"].astype("str")
    final_output.to_csv("/tmp/tmpn7dog1oo/final_output.csv", index=False, columns=['age_bucket', 'amount', 'log_age', 'user_id'])
    return final_output


if __name__ == "__main__":
    enrich_and_filter()

