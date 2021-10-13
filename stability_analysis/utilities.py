import math
import sqlite3
from typing import List, Tuple, Any


def get_db_table_list(db_path: str):
    conn = sqlite3.connect(db_path)
    res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for name in res:
        print(name[0])

def filter_multi_list(filter_by: List, lists: List[List], filter_by_val: Any) -> Tuple[List, List[List]]:
    filtered_filter_by_list = []
    filtered_lists = [[] for _ in lists]
    for i in range(len(filter_by)):
        if filter_by[i] == filter_by_val:
            filtered_filter_by_list.append(filter_by[i])
            for filter_list, input_list in zip(filtered_lists, lists):
                filter_list.append(input_list[i])
    return filtered_filter_by_list, filtered_lists

def safe_div(a, b):
    try:
        value = a / b
    except ZeroDivisionError:
        value = math.nan
    return value
