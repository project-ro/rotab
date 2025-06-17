import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from full_test_template.enrich_and_filter import enrich_and_filter


if __name__ == '__main__':
    enrich_and_filter()
