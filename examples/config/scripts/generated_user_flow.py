import pandas as pd
from rotab.core.operation.define_funcs import *
from rotab.core.operation.transform_funcs import *
import importlib.util
spec = importlib.util.spec_from_file_location('define_funcs', r'/home/yutaitatsu/rotab/custom_functions/define_funcs.py')
define_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(define_funcs)
globals().update({k: v for k, v in define_funcs.__dict__.items() if callable(v) and not k.startswith('__')})
spec = importlib.util.spec_from_file_location('transform_funcs', r'/home/yutaitatsu/rotab/custom_functions/transform_funcs.py')
transform_funcs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transform_funcs)
globals().update({k: v for k, v in transform_funcs.__dict__.items() if callable(v) and not k.startswith('__')})

# [PROCESS_1] Filter users under 30

# load tables
user = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/user.csv')

# process steps
user = user.query('age < 30')
user['age_group'] = user.apply(lambda row: row['age'] // 10, axis=1)
user = user[['user_id', 'age', 'age_group']]

# dump output
import os
path = os.path.abspath(r'../output/filtered_users.csv')
os.makedirs(os.path.dirname(path), exist_ok=True)
user.to_csv(path, index=False)
# [PROCESS_2] Summarize transaction amounts

# load tables
trans = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/transaction.csv')

# process steps
trans = trans.query('amount > 0')
trans['is_large'] = trans.apply(lambda row: row['amount'] > 5000, axis=1)
trans = trans[['user_id', 'amount', 'is_large']]

# dump output
import os
path = os.path.abspath(r'../output/filtered_transactions.csv')
os.makedirs(os.path.dirname(path), exist_ok=True)
trans.to_csv(path, index=False)
# [PROCESS_3] Enrich user data with transaction details

# load tables
user = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/user.csv')
trans = pd.read_csv(r'/home/yutaitatsu/rotab/examples/data/transaction.csv')

# process steps
user = user.query('age > 18')
user['log_age'] = user.apply(lambda row: log(row['age']), axis=1)
user['age_bucket'] = user.apply(lambda row: row['age'] // 10 * 10, axis=1)
user = user[['user_id', 'log_age', 'age_bucket']]

trans = trans.query('amount > 1000')

enriched = merge(left=user, right=trans, on='user_id')

enriched['high_value'] = enriched.apply(lambda row: row['amount'] > 10000, axis=1)
enriched = enriched[['user_id', 'log_age', 'amount', 'high_value']]

# dump output
import os
path = os.path.abspath(r'../output/final_output.csv')
os.makedirs(os.path.dirname(path), exist_ok=True)
enriched.to_csv(path, index=False)