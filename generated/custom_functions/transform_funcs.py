import pandas as pd
def merge_users_transactions(filtered_users, trans):
    return pd.merge(filtered_users, trans, on="user_id")
