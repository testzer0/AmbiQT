import os
import json
import random
from tqdm import tqdm

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

synonyms = json.load(open(SYNONYMS_PATH))
synonyms_lower = {}
for db_id in synonyms:
    synonyms_lower[db_id.lower()] = \
        {x.lower(): synonyms[db_id][x] for x in synonyms[db_id]}
    
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
        with_primary_key=False, return_type=False, lower=True):
        col_strings = [column.stringify(with_content=with_content, \
            max_examples=max_examples, return_type=return_type, lower=lower) \
            for column in self.columns]
        random.shuffle(col_strings)
        col_part = ", ".join(col_strings)
        table_name = self.table_name.lower() if lower else self.table_name
        primary_key = "NONE" if self.primary_key is None else self.primary_key
        primary_key = "[Primary Key = {}] : ".format(primary_key)
        if with_primary_key:
            tbl_string = table_name + primary_key + col_part
        else:
            tbl_string = table_name + " : " + col_part
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
        with_primary_key=False, return_type=False, extras={}, lower=True):
        tbl_strings = [table.stringify(with_content=with_content, \
            max_examples=max_examples, with_primary_key=with_primary_key, \
            return_type=return_type, lower=lower) for _, table in \
            self.tables.items() if table.table_name.lower() not in extras]
        tables_lower = {}
        if len(extras) > 0:
            tables_lower = {x.lower(): self.tables[x] for x in self.tables}
        for tbl_name in extras:
            tbl_string = tables_lower[tbl_name].stringify(\
                with_content=with_content, max_examples=max_examples, \
                with_primary_key=with_primary_key, return_type=return_type, \
                lower=lower)
            idx = tbl_string.find(":")
            for alt_name in extras[tbl_name]:
                alt_string = alt_name + tbl_string[idx-1:]
                tbl_strings.append(alt_string)
        random.shuffle(tbl_strings)
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

def get_schema(db_id, with_content=False, max_examples=3, \
        with_primary_key=False, return_type=False, lower=True, \
        extras={}):
    addendum = db_map_lower[db_id.lower()].stringify(
        with_content=with_content, max_examples=max_examples, \
        with_primary_key=with_primary_key, return_type=return_type, \
        extras=extras, lower=lower
    )
    return addendum

def replace_table_name_in_sql(sql, orig_name, new_name):
    sql = normalize_sql(sql)
    for op in ['from', 'join', 'intersect', 'union', 'except']:
        sql = sql.replace(op+" "+orig_name+" ", op+" "+new_name+" ")
        sql = sql.replace(op+" "+orig_name+")", op+" "+new_name+")")
    return lower(sql)

def modify_example(original, idx=None):
    db_id = original['db_id']
    sql = normalize_sql(original['query'])
    table_names_in_db = [table.lower() for table in \
        db_map_lower[db_id.lower()].tables]
    tables = []
    for t in table_names_in_db:
        if (" "+t+" ") in sql or (" "+t+")") in sql:
            tables.append(t)
    if len(tables) == 0:
        return []
    examples = []
    original['orig_query'] = sql
    for table in tables:
        example = original.copy()
        extras = {table: synonyms_lower[db_id.lower()][table]}
        example['extra_table_map'] = extras
        alt_1 = replace_table_name_in_sql(sql, table, extras[table][0])
        alt_2 = replace_table_name_in_sql(sql, table, extras[table][1])
        example['query1'] = alt_1
        example['query2'] = alt_2
        example['orig_example_idx'] = idx
        schema_with_content = get_schema(db_id, with_content=True, \
            extras=extras)
        example['schema_with_content'] = schema_with_content
        schema_without_content = ""
        skip = False
        for c in schema_with_content:
            if c == '(':
                skip = True
            elif c == ')':
                skip = False
            elif not skip:
                schema_without_content += c
        example['schema_without_content'] = schema_without_content.strip()
        examples.append(example)
    return examples

def save_splits(splits=['train', 'validation']):
    for split in splits:
        modified = []
        original = spider[split]
        for i, example in tqdm(enumerate(original)):
            modified += modify_example(example, i)
        print("{} examples in the {} split.".format(len(modified), split))
        json.dump(modified, open(split+".json", 'w+'))
        
if __name__ == '__main__':        
    save_splits()