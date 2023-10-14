from utils.globals import *
from utils.content import get_column_str

import os
import json

split_map = {
    "validation": "dev.json",
    "dev": "dev.json",
    "train": "train_spider.json"
}

def read_table_info():
    return json.load(open(os.path.join(SPIDER_ROOT, "tables.json")))

def load_spider(split):
    return json.load(open(os.path.join(SPIDER_ROOT, split_map[split])))

def get_col_mappings(tinfo, original=True):
    """
    Transforms the read table info into the mapping of db_id to a (map of
    table name to column_names). Using `original` is preferred since the original
    column names are what the queries use.
    """
    mapping = {}
    for db in tinfo:
        db_id = db['db_id']
        if original:
            tbl_names = db['table_names_original']
        else:
            tbl_names = db['table_names']
        tbl_map = {name : [] for name in tbl_names}
        if original:
            for tbl_idx, col_name in db['column_names_original']:
                if tbl_idx < 0:
                    continue
                tbl_map[tbl_names[tbl_idx]].append(col_name)
        else:
            for tbl_idx, col_name in db['column_names']:
                if tbl_idx < 0:
                    continue
                tbl_map[tbl_names[tbl_idx]].append(col_name)
        mapping[db_id] = tbl_map
    return mapping

def get_addendum(cmap, db_id, question=None, with_content=False, \
    lower=True, random_no=0, colname_map={}):
    """
    Returns the additional information to be added to the end of a query. It is of the form
    | db_id | table1: col11, col12, ..., col1{n1} | table2: col21, ..., col2{n2} | ... |
    """
    if with_content and random_no == 0:
        assert (question is not None)
    addendum = " | {}".format(db_id)
    for tbl_name in cmap[db_id]:
        addendum += " | {} :".format(tbl_name)
        first = True
        for col_name in cmap[db_id][tbl_name]:
            if not first:
                addendum += ","
            if with_content:
                addendum += " " + get_column_str(question, db_id, tbl_name, \
                    col_name, random_no=random_no)
            else:
                if col_name in colname_map:
                    col_name = colname_map[col_name]
                addendum += " " + col_name
            first = False
    return addendum.lower() if lower else addendum

if __name__ == '__main__':
    pass