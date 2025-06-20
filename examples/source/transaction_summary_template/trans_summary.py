import os
import pandas as pd
from rotab.core.operation.derive_funcs import *
from rotab.core.operation.transform_funcs import *


def step_summarize_transactions_trans_summary(trans):
    filtered_transactions = trans.copy()
    filtered_transactions = filtered_transactions.query('amount > 0').copy()
    filtered_transactions["is_large"] = filtered_transactions.apply(lambda row: row["amount"] > 5000, axis=1)
    filtered_transactions = filtered_transactions[["user_id", "amount", "is_large"]]
    return filtered_transactions


def trans_summary():
    """Summarize transaction amounts"""
    trans = pd.read_csv("data/inputs/transaction.csv", dtype={'id': 'str', 'user_id': 'str', 'amount': 'int'})
    filtered_transactions = step_summarize_transactions_trans_summary(trans)
    filtered_transactions.to_csv("data/outputs/filtered_transactions.csv", index=False)
    return filtered_transactions


if __name__ == "__main__":
    trans_summary()

