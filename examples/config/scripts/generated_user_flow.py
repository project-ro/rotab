import pandas as pd
from rotab.core.operation.define_funcs import *
from rotab.core.operation.transform_funcs import *
import importlib.util
import os
spec = importlib.util.spec_from_file_location('define_funcs', r'/home/yutaitatsu/rotab/custom_functions/define_funcs.py')
define_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(define_funcs)
globals().update({k: v for k, v in define_funcs.__dict__.items() if callable(v) and not k.startswith('__')})
spec = importlib.util.spec_from_file_location('transform_funcs', r'/home/yutaitatsu/rotab/custom_functions/transform_funcs.py')
transform_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform_funcs)
globals().update({k: v for k, v in transform_funcs.__dict__.items() if callable(v) and not k.startswith('__')})


# STEPS FUNCTIONS:

def step_filter_users_user_filter(user):
    """Step: filter_users """
    user = user.query('age < 30').copy()
    user.loc[:, 'age_group'] = user.apply(lambda row: row['age'] // 10, axis=1)
    user = user[['user_id', 'age', 'age_group']]
    return user

def step_summarize_transactions_trans_summary(trans):
    """Step: summarize_transactions """
    trans = trans.query('amount > 0').copy()
    trans.loc[:, 'is_large'] = trans.apply(lambda row: row['amount'] > 5000, axis=1)
    trans = trans[['user_id', 'amount', 'is_large']]
    return trans

def step_filter_users_main_transaction_enrichment(user):
    """Step: filter_users_main """
    user = user.query('age > 18').copy()
    user.loc[:, 'log_age'] = user.apply(lambda row: log(row['age']), axis=1)
    user.loc[:, 'age_bucket'] = user.apply(lambda row: row['age'] // 10 * 10, axis=1)
    user = user[['user_id', 'log_age', 'age_bucket']]
    return user

def step_filter_transactions_main_transaction_enrichment(trans):
    """Step: filter_transactions_main """
    trans = trans.query('amount > 1000').copy()
    return trans

def step_merge_transactions_transaction_enrichment(merge, trans, user):
    """Step: merge_transactions"""
    return merge(left=user, right=trans, on='user_id')

def step_enrich_transactions_transaction_enrichment(enriched):
    """Step: enrich_transactions """
    enriched.loc[:, 'high_value'] = enriched.apply(lambda row: row['amount'] > 10000, axis=1)
    enriched = enriched[['user_id', 'log_age', 'amount', 'high_value']]
    return enriched

# PROCESSES FUNCTIONS:

def process_user_filter():
    """Filter users under 30"""
    # load tables
    user = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/user.csv')

    # process steps
    user = step_filter_users_user_filter(user)

    # dump output
    path = os.path.abspath(r'../output/filtered_users.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    user.to_csv(path, index=False)

def process_trans_summary():
    """Summarize transaction amounts"""
    # load tables
    trans = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/transaction.csv')

    # process steps
    trans = step_summarize_transactions_trans_summary(trans)

    # dump output
    path = os.path.abspath(r'../output/filtered_transactions.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trans.to_csv(path, index=False)

def process_transaction_enrichment():
    """Enrich user data with transaction details"""
    # load tables
    user = pd.read_csv(r'/home/yutaitatsu/rotab/examples/output/filtered_users.csv')
    trans = pd.read_csv(r'/home/yutaitatsu/rotab/examples/output/filtered_transactions.csv')

    # process steps
    user = step_filter_users_main_transaction_enrichment(user)
    trans = step_filter_transactions_main_transaction_enrichment(trans)
    enriched = step_merge_transactions_transaction_enrichment(merge, trans, user)
    enriched = step_enrich_transactions_transaction_enrichment(enriched)

    # dump output
    path = os.path.abspath(r'../output/final_output.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enriched.to_csv(path, index=False)


if __name__ == '__main__':
    process_user_filter()
    process_trans_summary()
    process_transaction_enrichment()