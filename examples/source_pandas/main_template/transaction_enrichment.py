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
    filtered_trans["high_value"] = filtered_trans.apply(lambda row: row["int"](row["abs"](row["amount"])) > 5000, axis=1)
    return filtered_trans


def step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans):
    enriched = merge(left=filtered_users_main, right=filtered_trans, on='user_id')
    return enriched


def step_derive_segment_transaction_enrichment(enriched):
    enriched_with_segment = enriched.copy()
    enriched_with_segment["segment"] = enriched_with_segment.apply(lambda row: row["str"](row["age_bucket"]) + row["str"](row["log_age"]), axis=1)
    return enriched_with_segment


def step_groupby_segment_transaction_enrichment(enriched_with_segment):
    final_output = groupby_agg(table=enriched_with_segment, by="segment", aggregations={"amount": "mean", "high_value": "sum"})

    return final_output


def transaction_enrichment():
    """This process enriches user transactions by filtering users based on age and
    transactions based on amount, then merging the two datasets and aggregating by segment."""
    filtered_users = pd.read_csv("data/outputs/filtered_users.csv", dtype={'user_id': 'str', 'age': 'int', 'age_group': 'int'})
    filtered_transactions = pd.read_csv("data/outputs/filtered_transactions.csv", dtype={'user_id': 'str', 'amount': 'int', 'is_large': 'bool'})
    filtered_users_main = step_filter_users_main_transaction_enrichment(filtered_users)
    filtered_trans = step_filter_transactions_main_transaction_enrichment(filtered_transactions)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans)
    enriched_with_segment = step_derive_segment_transaction_enrichment(enriched)
    final_output = step_groupby_segment_transaction_enrichment(enriched_with_segment)
    final_output["segment"] = final_output["segment"].astype("str")
    final_output["amount"] = final_output["amount"].astype("float")
    final_output["high_value"] = final_output["high_value"].astype("int")
    final_output.to_csv("data/outputs/final_output.csv", index=False, columns=['segment', 'amount', 'high_value'])
    return final_output


if __name__ == "__main__":
    transaction_enrichment()

