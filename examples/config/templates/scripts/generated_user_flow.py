import pandas as pd
import os
import importlib.util
from rotab.core.operation.derive_funcs import *
from rotab.core.operation.transform_funcs import *


spec = importlib.util.spec_from_file_location('derive_funcs', r'/home/yutaitatsu/rotab/custom_functions/derive_funcs.py')
custom_derive_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_derive_funcs)

spec = importlib.util.spec_from_file_location('transform_funcs', r'/home/yutaitatsu/rotab/custom_functions/transform_funcs.py')
custom_transform_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_transform_funcs)



# STEPS FUNCTIONS:


def step_filter_users_user_filter(user):
    """Step: filter_users """
    filtered_users = user.copy()
    filtered_users = filtered_users.query('age < 30')
    filtered_users["age_group"] = filtered_users.apply(lambda row: row['age'] // 10, axis=1)
    filtered_users = filtered_users[['user_id', 'age', 'age_group']]
    return filtered_users


def step_summarize_transactions_trans_summary(trans):
    """Step: summarize_transactions """
    filtered_transactions = trans.copy()
    filtered_transactions = filtered_transactions.query('amount > 0')
    filtered_transactions["is_large"] = filtered_transactions.apply(lambda row: row['amount'] > 5000, axis=1)
    filtered_transactions = filtered_transactions[['user_id', 'amount', 'is_large']]
    return filtered_transactions


def step_filter_users_main_transaction_enrichment(user):
    """Step: filter_users_main """
    filtered_users = user.copy()
    if True:
        filtered_users = filtered_users.query('age > 18')
        filtered_users["log_age"] = filtered_users.apply(lambda row: log(row['age']), axis=1)
        filtered_users["age_bucket"] = filtered_users.apply(lambda row: row['age'] // 10 * 10, axis=1)
        filtered_users = filtered_users[['user_id', 'log_age', 'age_bucket']]
        return filtered_users


def step_filter_transactions_main_transaction_enrichment(trans):
    """Step: filter_transactions_main """
    filtered_trans = trans.copy()
    filtered_trans = filtered_trans.query('amount > 1000')
    return filtered_trans


def step_merge_transactions_transaction_enrichment(filtered_users ,filtered_trans):
    """Step: merge_transactions """
    enriched = filtered_users ,filtered_trans.copy()
    enriched = merge(left=filtered_users, right=filtered_trans, on='user_id')
    return enriched


def step_enrich_transactions_transaction_enrichment(enriched):
    """Step: enrich_transactions """
    final_output = enriched.copy()
    final_output["high_value"] = final_output.apply(lambda row: row['amount'] > 10000, axis=1)
    final_output = final_output[['user_id', 'log_age', 'amount', 'high_value']]
    return final_output



# PROCESSES FUNCTIONS:


def process_user_filter():
    """Filter users under 30"""
    # load tables
    user = pd.read_csv(r'../../data/user.csv')
    # process steps
    filtered_users = step_filter_users_user_filter(user)
    # dump output
    path = os.path.abspath(r'../../output/filtered_users.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filtered_users.to_csv(path, index=False)


def process_trans_summary():
    """Summarize transaction amounts"""
    # load tables
    trans = pd.read_csv(r'../../data/transaction.csv')
    # process steps
    filtered_transactions = step_summarize_transactions_trans_summary(trans)
    # dump output
    path = os.path.abspath(r'../../output/filtered_transactions.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    filtered_transactions.to_csv(path, index=False)


def process_transaction_enrichment():
    # load tables
    user = pd.read_csv(r'../../output/filtered_users.csv')
    trans = pd.read_csv(r'../../output/filtered_transactions.csv')
    # process steps
    filtered_users = step_filter_users_main_transaction_enrichment(user)
    filtered_trans = step_filter_transactions_main_transaction_enrichment(trans)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users ,filtered_trans)
    final_output = step_enrich_transactions_transaction_enrichment(enriched)
    # dump output
    path = os.path.abspath(r'../../output/final_output.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_output["user_id"] = final_output["user_id"].astype(str)
    validate_table_schema(final_output, columns=[
        {
            "name": "user_id",
            "dtype": "str",
            "description": "Unique identifier for each user"
        },
        {
            "name": "log_age",
            "dtype": "float",
            "description": "Logarithm of the user's age"
        },
        {
            "name": "amount",
            "dtype": "int",
            "description": "Amount of the transaction"
        },
        {
            "name": "high_value",
            "dtype": "bool",
            "description": "Flag indicating if the transaction amount is greater than 10000"
        }
    ])
    final_output.to_csv(path, index=False)


if __name__ == '__main__':
    process_user_filter()
    process_trans_summary()
    process_transaction_enrichment()