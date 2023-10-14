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

def normalize_map(mp, full=True):
    if type(mp) == dict:
        return {a.lower(): normalize_map(mp[a]) for a in mp}
    elif type(mp) == list:
        return [normalize_map(e) for e in mp]
    else:
        return mp.lower() if full else True
    
synonyms_lower = normalize_map(synonyms)

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
        with_primary_key=False, extra_map={}, return_type=False, \
        lower=True):
        col_strings = [column.stringify(with_content=with_content, \
            max_examples=max_examples, return_type=return_type, lower=lower) \
            for column in self.columns if column.column_name.lower() not in \
            extra_map]
        for col in extra_map:
            idx = self.column_name_lower_to_idx[col.lower()]
            stringified = self.columns[idx].stringify(with_content=with_content, \
                max_examples=max_examples, return_type=return_type, lower=lower)
            for alt in extra_map[col]:
                if return_type:
                    s = alt + stringified[stringified.find('['):]
                else:
                    s = alt + stringified[stringified.find(' '):]
                col_strings.append(s)
        random.shuffle(col_strings)
        col_part = ", ".join(col_strings)
        table_name = self.table_name.lower() if lower else self.table_name
        primary_key = "NONE" if self.primary_key is None else self.primary_key
        primary_key = "[Primary Key = {}] : ".format(primary_key)
        if lower:
            primary_key = primary_key.lower()
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
        with_primary_key=False, extra_map={}, return_type=False, lower=True):
        tbl_strings = [table.stringify(with_content=with_content, \
            max_examples=max_examples, with_primary_key=with_primary_key, \
            extra_map=extra_map[table.table_name.lower()] if \
            table.table_name.lower() in extra_map else {}, \
            return_type=return_type, lower=lower) for _, table in \
            self.tables.items()]
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
        with_primary_key=False, extra_map={}, return_type=False, lower=True):
    addendum = db_map_lower[db_id.lower()].stringify(
        with_content=with_content, max_examples=max_examples, \
        with_primary_key=with_primary_key, extra_map=extra_map, \
        return_type=return_type, lower=lower
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

def generate_alias_map(tokens):
    alias = {}
    for i in range(len(tokens)):
        if tokens[i] == 'as':
            alias[tokens[i+1].lower()] = tokens[i-1].lower()
    return alias

def get_table_for_selected_column(tokens, idx, alias={}):
    col_token = tokens[idx]
    if len(col_token) > 2 and col_token[2] == '.':
        return alias[col_token[:2]]
    from_idx = tokens.index('from')
    return tokens[from_idx+1]

def extract_column_name(s):
    if '(' in s and ')' in s:
        return extract_column_name(s[s.find('(')+1:s.find(')')])
    if ' ' in s:
        return extract_column_name(s.split(' ')[-1])
    if '.' in s:
        return s[s.rfind('.')+1:]
    return s

def get_potential_replacements(sql):
    sql = normalize_sql(sql)
    tokens = sql.replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ').split()
    alias_map = generate_alias_map(tokens)
    cols = sql[sql.find('select')+len('select'):sql.find('from')].strip().split(',')
    cols = [extract_column_name(col.strip()) for col in cols]
    replacements = []
    for i, token in enumerate(tokens):
        if token == 'from':
            break
        if len(token) > 2 and token[2] == '.':
            token = token[3:]
        if token in cols and token != '*':
            replacements.append((token, get_table_for_selected_column(tokens, i, \
                alias_map)))
    return list(set(replacements))

def replace_column(sql, table, column, new_column):
    tokens = normalize_sql(sql).replace(',', ' , ').replace('(', ' ( ').replace(')', ' ) ').split()
    alias = generate_alias_map(tokens)
    for i in range(len(tokens)):
        if (tokens[i] == column or (len(tokens[i]) > 2 and tokens[i][2] == '.' \
            and tokens[i][3:] == column)) and get_table_for_selected_column(tokens, \
            i, alias) == table:
            tokens[i] = new_column if '.' not in tokens[i] else \
                tokens[i][:3] + new_column
    sql = normalize_sql(" ".join(tokens))
    return sql

def get_alternatives(sql, db_id, table, column):
    db_id = db_id.lower()
    table = table.lower()
    column = column.lower()
    sql = normalize_sql(sql)
    try:
        alternatives = synonyms_lower[db_id][table][column]
    except:
        return None, None, None
    extra_map = {table: {column: alternatives}}
    alt_1 = replace_column(sql, table, column, alternatives[0])
    alt_2 = replace_column(sql, table, column, alternatives[1])
    return alt_1, alt_2, extra_map

def modify_example(example, idx=None):
    modified = []
    example['orig_query'] = normalize_sql(example['query'])
    replacements = get_potential_replacements(example['query'])
    for column, table in replacements:
        new_example = example.copy()
        alt_1, alt_2, extra_map = get_alternatives(example['query'], \
            example['db_id'], table, column)
        if alt_1 is None:
            continue
        new_example.pop('query')
        new_example['query1'] = alt_1
        new_example['query2'] = alt_2
        new_example['extra_map'] = extra_map
        schema_with_content = get_schema(new_example['db_id'], extra_map=extra_map, \
            with_content=True)
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
        new_example['schema_without_content'] = schema_without_content.strip()
        new_example['orig_example_idx'] = idx
        modified.append(new_example)
    return modified

def save_splits(splits=['train', 'validation']):
    for split in splits:
        original = spider[split]
        modified = []
        for i, example in tqdm(enumerate(original)):
            modified += modify_example(example, i)
        print("{} examples in the {} split.".format(len(modified), split))
        json.dump(modified, open(split+".json", 'w+'))

if __name__ == '__main__':        
    save_splits()