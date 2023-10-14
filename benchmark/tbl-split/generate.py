import os
import json
import random

SPIDER_DIR = '/home/adithya/sem8/t2s-repo/spider'
SYNONYMS_PATH = 'two_col_synonyms.json'

tables = json.load(open(os.path.join(SPIDER_DIR, 'tables.json')))
spider = { 
    "validation": json.load(open(os.path.join(SPIDER_DIR, 'dev.json'))),
    "train": json.load(open(os.path.join(SPIDER_DIR, 'train_spider.json')))
}
values = json.load(open(os.path.join(SPIDER_DIR, 'values-lower.json')))

tables_x = None
tab_set_lower = {}
col_set_lower = {}

class Column:
    def __init__(self, column_name, column_type, examples=None):
        self.column_name = column_name
        self.column_type = column_type
        self.examples = examples
        
    def set_examples(self, examples):
        self.examples = [str(x).strip() if \
            not isinstance(x, str) else ('"'+x+'"') for x in examples]
        
    def stringify(self, with_content=False, max_examples=4, return_type=False, \
        lower=True):
        if return_type:
            col_string = "{}[{}]".format(self.column_name, self.column_type)
        else:
            col_string = "{}".format(self.column_name)
        if lower:
            col_string = col_string.lower()
        if with_content:
            if len(self.examples) <= max_examples:
                content_string = ", ".join(self.examples)
            else:
                content_string = ", ".join(random.choices(self.examples, \
                    k=max_examples))
            col_string += " (" + content_string + ")"
        return col_string

class Table:
    def __init__(self, table_name, primary_key):
        self.table_name = table_name
        self.primary_key = primary_key
        self.columns = []
        self.column_name_lower_to_idx = {}
    
    def set_examples(self, examples):
        for i in range(len(self.columns)):
            col_name = self.columns[i].column_name.lower()
            if col_name in examples:
                self.columns[i].set_examples(examples[col_name])
        
    def add_column(self, column):
        self.column_name_lower_to_idx[column.column_name.lower()] = \
            len(self.columns)
        self.columns.append(column)
        
    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def stringify(self, with_content=False, max_examples=4, \
        with_primary_key=False, return_type=False, split_col=None, lower=True):
        col_strings = [column.stringify(with_content=with_content, \
            max_examples=max_examples, return_type=return_type, lower=lower) for \
            column in self.columns]
        extra_part = None
        if split_col is not None:
            pk_idx = self.column_name_lower_to_idx[self.primary_key.lower()]
            pk_string = self.columns[pk_idx].stringify(with_content=with_content, \
                max_examples=max_examples, return_type=return_type, lower=lower)
            new_strings = [pk_string]
            col_strings_copy = col_strings.copy()
            col_strings = []
            for column, col_string in zip(self.columns, col_strings_copy):
                col_strings.append(col_string)
                if column.column_name.lower() == split_col.lower():
                    new_strings.append(col_string)
                    random.shuffle(new_strings)
                    extra_part = ", ".join(new_strings)
        random.shuffle(col_strings)
        col_part = ", ".join(col_strings)
        table_name = self.table_name.lower() if lower else self.table_name
        primary_key = "NONE" if self.primary_key is None else self.primary_key
        primary_key = "[Primary Key = {}] : ".format(primary_key)
        if with_primary_key:
            tbl_string = table_name + primary_key + col_part
        else:
            tbl_string = table_name + " : " + col_part
        if extra_part is not None:
            tbl_string += " | " + table_name + "_" + split_col
            if with_primary_key:
                tbl_string += primary_key
            else:
                tbl_string += " : "
            tbl_string += extra_part
        return tbl_string

class Database:
    def __init__(self, db_id):
        self.db_id = db_id
        self.tables = {}
    
    def set_examples(self, examples):
        for table_name in self.tables:
            if table_name.lower() in examples:
                self.tables[table_name].set_examples(examples[table_name.lower()])
        
    def add_table(self, table):
        self.tables[table.table_name] = table
        
    def add_tables(self, tables):
        for table in tables:
            self.add_table(table)
            
    def stringify(self, with_content=False, max_examples=5, \
        with_primary_key=False, return_type=False, split_map={}, lower=True):
        tbl_strings = [table.stringify(with_content=with_content, \
            max_examples=max_examples, with_primary_key=with_primary_key, \
            return_type=return_type, split_col=split_map[table.table_name.lower()] \
            if table.table_name.lower() in split_map else None, lower=lower) \
            for _, table in self.tables.items()]
        return " | ".join(tbl_strings)

def build_db_map(tables, lower=True):
    db_map = {}
    for table in tables:
        db_id = table['db_id']
        database = Database(db_id)
        table_names = table['table_names_original']
        column_names = [x[1] for x in table['column_names_original']][1:]
        table_pkeys = [None for _ in range(len(table_names))]
        column_tbls = [x[0] for x in table['column_names_original']][1:]
        column_types = table['column_types'][1:]
        for i in table['primary_keys']:
            table_pkeys[column_tbls[i-1]] = column_names[i-1]
        table_cols = [[] for _ in range(len(table_names))]
        for cname, ctbl, ctype in zip(column_names, column_tbls, column_types):
            table_cols[ctbl].append(Column(cname, ctype))
        
        for tname, tcols, tpkey in zip(table_names, table_cols, table_pkeys):
            table = Table(tname, tpkey)
            table.add_columns(tcols)
            database.add_table(table)
        if db_id.lower() in values:
            database.set_examples(values[db_id.lower()])
        if lower:
            db_map[db_id.lower()] = database
        else:
            db_map[db_id] = database
    return db_map

db_map_lower = build_db_map(tables, lower=True)

def get_schema(db_id, with_content=False, max_examples=3, \
        with_primary_key=False, return_type=False, split_map={}, lower=True):
    addendum = db_map_lower[db_id.lower()].stringify(
        with_content=with_content, max_examples=max_examples, \
        with_primary_key=with_primary_key, return_type=return_type, \
        split_map=split_map, lower=lower
    )
    return addendum

def lower(s):
    s = s.replace("``", "`")
    lowers = ""
    current_quote = None
    for c in s:
        if current_quote is None:
            lowers += c.lower()
            if c in ['"', '\'', '`']:
                current_quote = c
        else:
            lowers += c
            if c == current_quote:
                current_quote = None
    return lowers

def normalize_sql(sql):
    if sql[-1] == ';':
        sql = sql[:-1]
    sql = " ".join(lower(sql).strip().split())
    sql = sql.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        sql = sql.replace(kword+" (", kword+"(")
    return sql.replace(") ,", "),")

def get_posssible_tokens_to_replace(sql, db_id):
    sql = normalize_sql(sql)
    sql = sql.replace(",", " , ").replace("(", "( ").replace(")", " )").split()
    if 'join' in sql:
        return None, []
    assert (sql[0] == 'select' and 'from' in sql)   
    selected_cols = []
    tbl_idx = sql.index('from') + 1
    tbl = sql[tbl_idx]
    table_map = db_map_lower[db_id.lower()].tables
    table_map = {x.lower(): table_map[x] for x in table_map}
    try:
        table = table_map[tbl]
    except:
        return None, {}
    pkey = table.primary_key
    if pkey is None:
        return None, []
    pkey = pkey.lower()
    columns_lower = [column.column_name.lower() for column in table.columns]
    selected_cols = []
    for tok in sql[:tbl_idx]:
        if tok in columns_lower and tok != pkey:
            selected_cols.append(tok)
    return tbl, list(set(selected_cols))

def modify_sql(sql, db_id, tbl, split_col):
    sql = normalize_sql(sql)
    sql = sql.replace(",", " , ").replace("(", " ( ").replace(")", " ) ").split()
    table_map = db_map_lower[db_id.lower()].tables
    table_map = {x.lower(): table_map[x] for x in table_map}
    table = table_map[tbl]
    pkey = table.primary_key.lower()
    columns_lower = [column.column_name.lower() for column in table.columns]
    
    modified = []
    for token in sql:
        if token == tbl:
            string = "{} as t1 join {} as t2 on t1.{} = t2.{}".format(tbl, \
                tbl+"_"+split_col, pkey, pkey)
            modified.append(string)
        elif token == split_col:
            modified.append("t2.{}".format(split_col))
        elif token in columns_lower:
            modified.append("t1.{}".format(token))
        else:
            modified.append(token)
    modified = " ".join(modified)
    modified = modified.replace(" ,", ",").replace("( ", "(").replace(" )", \
        ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        modified = modified.replace(kword+" (", kword+"(")
    return modified

def get_reverse_map(extra_map):
    reverse_map = {}
    for tbl in extra_map:
        tbl_reverse_map = {}
        for col, mapping in extra_map[tbl].items():
            tbl_reverse_map[mapping] = col
        reverse_map[tbl] = tbl_reverse_map
    return reverse_map

def save_splits(split=['train', 'validation']):
    if type(split) == list:
        for s in split:
            save_splits(s)
        return
    original = spider[split]
    modified = []
    for example in original:
        tbl, selected_cols = get_posssible_tokens_to_replace(example['query'], \
            example['db_id'])
        if tbl is None or len(selected_cols) <= 1:
            continue
        table_map = db_map_lower[example['db_id'].lower()].tables
        table_map = {x.lower(): table_map[x] for x in table_map}
        table = table_map[tbl]
        pkey = table.primary_key.lower()
        example['primary_key'] = {tbl: pkey}
        example['orig_query'] = normalize_sql(example['query'])
        for split_col in selected_cols:
            new_example = example.copy()
            new_query = modify_sql(new_example['query'], new_example['db_id'], \
                tbl, split_col)
            split_map = {tbl: split_col}
            new_example['split_map'] = split_map
            new_example['query1'] = example['orig_query']
            new_example['query2'] = new_query
            schema_with_content = get_schema(new_example['db_id'], \
                with_content=True, split_map=split_map)
            new_example['schema_with_content'] = schema_with_content
            schema_without_content = ""
            skip = False
            for c in schema_with_content:
                if c == '(':
                    skip = True
                elif c == ')':
                    skip = False
                elif not skip:
                    schema_without_content += c
            new_example['schema_without_content'] = schema_without_content
            modified.append(new_example)
    json.dump(modified, open(split+".json", 'w+'))
        
if __name__ == '__main__':
    save_splits()