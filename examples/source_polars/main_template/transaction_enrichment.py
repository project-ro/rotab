import os
import polars as pl
import fsspec
from rotab.core.parse.parse import parse
from rotab.core.operation.derive_funcs_polars import *
from rotab.core.operation.transform_funcs_polars import *
from custom_functions.derive_funcs import *
from custom_functions.transform_funcs import *


def step_filter_users_main_transaction_enrichment(filtered_users):
    if True:
        filtered_users_main = filtered_users
        filtered_users_main = filtered_users_main.filter(parse('age > 18'))
        filtered_users_main = filtered_users_main.with_columns(parse("""
            log_age = log(age)
            age_bucket = age // 10 * 10
            """))
        filtered_users_main = filtered_users_main.select(['user_id', 'log_age', 'age_bucket'])
    return filtered_users_main


def step_filter_transactions_main_transaction_enrichment(filtered_transactions):
    filtered_trans = filtered_transactions
    filtered_trans = filtered_trans.filter(parse('amount > 1000'))
    filtered_trans = filtered_trans.with_columns(parse("""
        high_value = int(abs(amount)) > 5000
        """))
    return filtered_trans


def step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans):
    enriched = merge(left=filtered_users_main, right=filtered_trans, on='user_id')
    return enriched


def step_derive_segment_transaction_enrichment(enriched):
    enriched_with_segment = enriched
    enriched_with_segment = enriched_with_segment.with_columns(parse("""
        segment = str(age_bucket) + str(log_age)
        """))
    return enriched_with_segment


def step_groupby_segment_transaction_enrichment(enriched_with_segment):
    final_output = groupby_agg(table=enriched_with_segment, by="segment", aggregations={"amount": "mean", "high_value": "sum"})

    return final_output


def transaction_enrichment():
    """This process enriches user transactions by filtering users based on age and
    transactions based on amount, then merging the two datasets and aggregating by segment."""
    filtered_users = pl.scan_csv("data/outputs/filtered_users.csv", dtypes={"user_id": pl.Utf8, "age": pl.Int64, "age_group": pl.Int64})
    filtered_transactions = pl.scan_csv("data/outputs/filtered_transactions.csv", dtypes={"user_id": pl.Utf8, "amount": pl.Int64, "is_large": pl.Boolean})
    filtered_users_main = step_filter_users_main_transaction_enrichment(filtered_users)
    filtered_trans = step_filter_transactions_main_transaction_enrichment(filtered_transactions)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users_main, filtered_trans)
    enriched_with_segment = step_derive_segment_transaction_enrichment(enriched)
    final_output = step_groupby_segment_transaction_enrichment(enriched_with_segment)
    final_output = final_output.with_columns(pl.col("segment").cast(pl.Utf8))
    final_output = final_output.with_columns(pl.col("amount").cast(pl.Float64))
    final_output = final_output.with_columns(pl.col("high_value").cast(pl.Int64))
    with fsspec.open("data/outputs/final_output.csv", "w") as f:
        final_output.collect(streaming=True).write_csv(f)
    return final_output


if __name__ == "__main__":
    transaction_enrichment()

