import os
import pandas as pd
from rotab.core.operation.derive_funcs import *
from rotab.core.operation.transform_funcs import *


def step_filter_users_main_transaction_enrichment(user):
    if True:
        filtered_users = user.copy()
        filtered_users = filtered_users.query('age > 18').copy()
        filtered_users["log_age"] = filtered_users.apply(lambda row: log(row["age"]), axis=1)
        filtered_users["age_bucket"] = filtered_users.apply(lambda row: row["age"] // 10 * 10, axis=1)
        filtered_users = filtered_users[["user_id", "log_age", "age_bucket"]]
    return filtered_users


def step_filter_transactions_main_transaction_enrichment(trans):
    filtered_trans = trans.copy()
    filtered_trans = filtered_trans.query('amount > 1000').copy()
    return filtered_trans


def step_merge_transactions_transaction_enrichment(filtered_users, filtered_trans):
    enriched = merge(left=filtered_users, right=filtered_trans, on='user_id')
    return enriched


def step_enrich_transactions_transaction_enrichment(enriched):
    final_output = enriched.copy()
    final_output["high_value"] = final_output.apply(lambda row: row["amount"] > 10000, axis=1)
    final_output = final_output[["user_id", "log_age", "amount", "high_value"]]
    return final_output


def transaction_enrichment():
    """This process enriches user transactions by filtering users based on age and
    transactions based on amount, then merging the two datasets."""
    user = pd.read_csv("data/outputs/filtered_users.csv", dtype={'id': 'str', 'user_id': 'str', 'age': 'int', 'age_group': 'int'})
    trans = pd.read_csv("data/outputs/filtered_transactions.csv", dtype={'id': 'str', 'user_id': 'str', 'amount': 'int'})
    filtered_users = step_filter_users_main_transaction_enrichment(user)
    filtered_trans = step_filter_transactions_main_transaction_enrichment(trans)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users, filtered_trans)
    final_output = step_enrich_transactions_transaction_enrichment(enriched)
    final_output["user_id"] = final_output["user_id"].astype("str")
    final_output["log_age"] = final_output["log_age"].astype("float")
    final_output["amount"] = final_output["amount"].astype("int")
    final_output["high_value"] = final_output["high_value"].astype("bool")
    final_output.to_csv("data/outputs/final_output.csv", index=False, columns=['user_id', 'log_age', 'amount', 'high_value'])
    return final_output


if __name__ == "__main__":
    transaction_enrichment()

