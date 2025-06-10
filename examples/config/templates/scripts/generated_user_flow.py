import pandas as pd
import os
import importlib.util
from rotab.core.operation.new_columns_funcs import *
from rotab.core.operation.dataframes_funcs import *


spec = importlib.util.spec_from_file_location('new_columns_funcs', r'/home/yutaitatsu/rotab/custom_functions/new_columns_funcs.py')
custom_new_columns_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_new_columns_funcs)

spec = importlib.util.spec_from_file_location('dataframes_funcs', r'/home/yutaitatsu/rotab/custom_functions/dataframes_funcs.py')
custom_dataframes_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(custom_dataframes_funcs)



# STEPS FUNCTIONS:


def step_filter_users_user_filter(user):
    """Step: filter_users """
    user = user.query('age < 30')
    user["age_group"] = user.apply(lambda row: row['age'] // 10, axis=1)
    return user


def step_summarize_transactions_trans_summary(trans):
    """Step: summarize_transactions """
    trans = trans.query('amount > 0')
    trans["is_large"] = trans.apply(lambda row: row['amount'] > 5000, axis=1)
    return trans


def step_filter_users_main_transaction_enrichment(user):
    """Step: filter_users_main """
    if True:
        filtered_users = user.copy()
        filtered_users = user.query('age > 18')
        filtered_users["log_age"] = user.apply(lambda row: log(row['age']), axis=1)
        filtered_users["age_bucket"] = user.apply(lambda row: row['age'] // 10 * 10, axis=1)
        return filtered_users


def step_filter_transactions_main_transaction_enrichment(trans):
    """Step: filter_transactions_main """
    trans = trans.query('amount > 1000')
    return trans


def step_merge_transactions_transaction_enrichment(filtered_users, merge, trans):
    """Step: merge_transactions"""
    return merge(left=filtered_users, right=trans, on='user_id')

def step_enrich_transactions_transaction_enrichment(enriched):
    """Step: enrich_transactions """
    enriched["high_value"] = enriched.apply(lambda row: row['amount'] > 10000, axis=1)
    return enriched



# PROCESSES FUNCTIONS:


def process_user_filter():
    """Filter users under 30"""
    # load tables
    user = pd.read_csv(r'../../data/user.csv', dtype={'id': str, 'user_id': str, 'age': int, 'age_group': int})
    # process steps
    user = step_filter_users_user_filter(user)
    # dump output
    path = os.path.abspath(r'../../output/filtered_users.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    user.to_csv(path, index=False)


def process_trans_summary():
    """Summarize transaction amounts"""
    # load tables
    trans = pd.read_csv(r'../../data/transaction.csv', dtype={'id': str, 'user_id': str, 'amount': float})
    # process steps
    trans = step_summarize_transactions_trans_summary(trans)
    # dump output
    path = os.path.abspath(r'../../output/filtered_transactions.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trans.to_csv(path, index=False)


def process_transaction_enrichment():
    """Enrich user data with transaction details"""
    # load tables
    user = pd.read_csv(r'../../output/filtered_users.csv', dtype={'id': str, 'user_id': str, 'age': int, 'age_group': int})
    trans = pd.read_csv(r'../../output/filtered_transactions.csv', dtype={'id': str, 'user_id': str, 'amount': float})
    # process steps
    filtered_users = step_filter_users_main_transaction_enrichment(user)
    trans = step_filter_transactions_main_transaction_enrichment(trans)
    enriched = step_merge_transactions_transaction_enrichment(filtered_users, merge, trans)
    enriched = step_enrich_transactions_transaction_enrichment(enriched)
    # dump output
    if enriched is not None:
        path = os.path.abspath(r'../../output/final_output.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        validate_table_schema(enriched, columns=[
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
                "name": "age_bucket",
                "dtype": "int",
                "description": "Age bucket of the user, calculated as age // 10 * 10"
            },
            {
                "name": "amount",
                "dtype": "float",
                "description": "Amount of the transaction"
            },
            {
                "name": "high_value",
                "dtype": "bool",
                "description": "Flag indicating if the transaction amount is greater than 10000"
            }
        ])
        enriched.to_csv(path, index=False)


if __name__ == '__main__':
    process_user_filter()
    process_trans_summary()
    process_transaction_enrichment()