import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from user_filter_template.user_filter import user_filter
from transaction_summary_template.trans_summary import trans_summary
from main_template.transaction_enrichment import transaction_enrichment


if __name__ == '__main__':
    user_filter()
    trans_summary()
    transaction_enrichment()
