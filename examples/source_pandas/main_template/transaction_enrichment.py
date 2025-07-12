import os
import pandas as pd
from rotab.core.operation.derive_funcs_pandas import *
from rotab.core.operation.transform_funcs_pandas import *


def step_filter_users_main_transaction_enrichment(filtered_users):
    if True:
        filtered_users_main = filtered_users.copy()
        filtered_users_main = filtered_users_main.query('age > 18').copy()
        filtered_users_main["log_age"] = filtered_users_main.apply(lambda row: log(row["age"]), axis=1)
        filtered_users_main["age_bucket"] = filtered_users_main.apply(lambda row: row["age"] // 10 * 10, axis=1)
        filtered_users_main = filtered_users_main[["user_id", "log_age", "age_bucket"]]
    return filtered_users_main


def step_filter_transactions_main_transaction_enrichment(filtered_transactions):
    filtered_trans = filtered_transactions.copy()
    filtered_trans = filtered_trans.query('amount > 1000').copy()
    return filtered_trans


def step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans):
    enriched = merge(left=filtered_users_main, right=filtered_trans, on='user_id')
    return enriched


def step_enrich_transactions_transaction_enrichment(enriched):
    final_output = enriched.copy()
    final_output["high_value"] = final_output.apply(lambda row: row["amount"] > 10000, axis=1)
    final_output = final_output[["user_id", "log_age", "amount", "high_value"]]
    return final_output


def transaction_enrichment():
    """This process enriches user transactions by filtering users based on age and
    transactions based on amount, then merging the two datasets."""
    filtered_users = pd.read_csv("data/outputs/filtered_users.csv", dtype={'user_id': 'str', 'age': 'int', 'age_group': 'int'})
    filtered_transactions = pd.read_csv("data/outputs/filtered_transactions.csv", dtype={'user_id': 'str', 'amount': 'int', 'is_large': 'bool'})
    filtered_users_main = step_filter_users_main_transaction_enrichment(filtered_users)
    filtered_trans = step_filter_transactions_main_transaction_enrichment(filtered_transactions)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans)
    final_output = step_enrich_transactions_transaction_enrichment(enriched)
    final_output["user_id"] = final_output["user_id"].astype("str")
    final_output["log_age"] = final_output["log_age"].astype("float")
    final_output["amount"] = final_output["amount"].astype("int")
    final_output["high_value"] = final_output["high_value"].astype("bool")
    final_output.to_csv("data/outputs/final_output.csv", index=False, columns=['user_id', 'log_age', 'amount', 'high_value'])
    return final_output


if __name__ == "__main__":
    transaction_enrichment()

