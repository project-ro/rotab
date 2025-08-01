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


def step_summarize_transactions_trans_summary(trans):
    filtered_transactions = trans
    filtered_transactions = filtered_transactions.filter(parse('amount > 0'))
    filtered_transactions = filtered_transactions.with_columns(parse('is_large = amount > 5000'))
    filtered_transactions = filtered_transactions.select(['user_id', 'amount', 'is_large'])
    return filtered_transactions


def trans_summary():
    """Summarize transaction amounts"""
    trans = pl.scan_csv("data/inputs/transaction.csv")
    filtered_transactions = step_summarize_transactions_trans_summary(trans)
    filtered_transactions = filtered_transactions.with_columns(pl.col("user_id").cast(pl.Utf8))
    filtered_transactions = filtered_transactions.with_columns(pl.col("amount").cast(pl.Int64))
    filtered_transactions = filtered_transactions.with_columns(pl.col("is_large").cast(pl.Boolean))
    with fsspec.open("data/outputs/filtered_transactions.csv", "w") as f:
        filtered_transactions.collect(streaming=True).write_csv(f)
    return filtered_transactions


if __name__ == "__main__":
    trans_summary()

