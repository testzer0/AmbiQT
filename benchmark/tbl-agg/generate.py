import os
import json
import random

SPIDER_DIR = '/home/adithya/sem8/t2s-repo/spider'

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
        with_primary_key=False, return_type=False, extra_tbl_string=None, \
        lower=True):
        tbl_strings = [table.stringify(with_content=with_content, \
            max_examples=max_examples, with_primary_key=with_primary_key, \
            return_type=return_type, lower=lower) \
            for _, table in self.tables.items()]
        if extra_tbl_string is not None:
            tbl_strings.append(extra_tbl_string)
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

def get_new_schema(new_table_name, col_names, raw_col_names, tables, \
    db_id, max_examples=3):
    col_strings = []
    aggs_start = ["sum_", "avg_", "min_", "max_"]
    tbl_map = {}
    tables = [t.lower() for t in tables]
    ttt = db_map_lower[db_id.lower()].tables
    ttt = {x.lower(): ttt[x] for x in ttt}
    for col in col_names:
        col = col.lower()
        relevant_col = col if col in raw_col_names else col[4:]
        tbl = None
        if relevant_col == "number":
            options = list(range(1,11))
            examples = [str(e) for e in random.choices(options, k=3)]
            cstring = col + "({})".format(", ".join(examples)) 
            col_strings.append(cstring)
            continue
        if relevant_col in tbl_map:
            tbl = tbl_map[relevant_col] 
        else:
            for table in tables:
                if relevant_col in ttt[table].column_name_lower_to_idx:
                    tbl_map[relevant_col] = table
                    break
            assert(relevant_col in tbl_map)
            tbl = tbl_map[relevant_col]
        ind = ttt[tbl].column_name_lower_to_idx[relevant_col]
        cstring = ttt[tbl].columns[ind].stringify(with_content=True, \
            max_examples=max_examples)
        cstring = col + " " + cstring[cstring.find('('):]
        col_strings.append(cstring) 
    random.shuffle(col_strings)
    tbl_string = "{} : {}".format(new_table_name, ", ".join(col_strings))
    return db_map_lower[db_id.lower()].stringify(with_content=True, \
        max_examples=3, extra_tbl_string=tbl_string)

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
    if len(sql) == 0:
        return ""
    if sql[-1] == ';':
        sql = sql[:-1]
    sql = " ".join(lower(sql).strip().split())
    sql = sql.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        sql = sql.replace(kword+" (", kword+"(")
    return sql.replace(") ,", "),")

def get_posssible_tokens_to_replace_old(sql, db_id):
    sql = normalize_sql(sql)
    sql = sql.replace(",", " , ").replace("(", "( ").replace(")", " )").split()
    if ' where ' in sql and 'having' in sql:
        return None, {}
    assert (sql[0] == 'select' and 'from' in sql)
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
        return None, {}
    pkey = pkey.lower()
    columns_lower = [column.column_name.lower() for column in table.columns]
    selected_cols = {}
    aggr_func=['avg(', 'max(', 'min(', 'sum(']
    for i, tok in enumerate(sql[:tbl_idx]):
        if tok in aggr_func:
            if sql[i+1] != "distinct":
                if sql[i+1] not in selected_cols:
                    selected_cols[sql[i+1]]=[]
                selected_cols[sql[i+1]].append(tok[:-1])
            else:
                if sql[i+1] not in selected_cols:
                    selected_cols[sql[i+2]]=[]
                selected_cols[sql[i+2]].append(tok[:-1])
    return tbl, selected_cols

def get_col_and_agg(sel):
    if '(' in sel:
        assert(sel.count('(') == 1)
        agg = sel[:sel.find('(')].strip()
        col = sel[sel.find('(')+1:sel.find(')')].strip()
    else:
        agg = None
        col = sel
    if '.' in col:
        col = col[col.rfind('.')+1:]
    return col, agg

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_replacements(sql, db_id):
    sql = normalize_sql(sql)
    sql = sql.replace(",", " , ").replace("(", " ( ").replace(")", " ) ")
    if ' where ' in sql and ' having ' in sql:
        return [None for _ in range(6)]
    if sql.count('select ') > 1 or 'distinct ' in sql:
        return [None for _ in range(6)]
    assert (sql.startswith('select') and 'from' in sql)
    table_map = db_map_lower[db_id.lower()].tables
    table_map = {x.lower(): table_map[x] for x in table_map}
    sel_portion = sql[sql.find('select')+len('select'):sql.find('from')].strip()
    selections = [col.strip() for col in sel_portion.split(',')]
    aggs = ['count', 'sum', 'avg', 'min', 'max']
    
    cols = []
    agg_cols = []
    extra_cols = []
    schema_cols = []
    where = ""
    for sel in selections:
        col, agg = get_col_and_agg(sel)
        if ' ' in col:
            return [None for _ in range(6)]
        if agg is None:
            if col == '*':
                return [None for _ in range(6)]
            cols.append(col)
            schema_cols.append(col)
        elif col == '*':
            cols.append('number')
            schema_cols.append('number')
        else:
            cols.append(agg+"_"+col)
            if col not in agg_cols:
                agg_cols.append(col)
    tbl_list = []
    columns_lower = []
    for token in sql.split():
        if token.lower() in table_map and token.lower() not in tbl_list:
            tbl_list.append(token.lower())
            columns_lower += [column.column_name.lower() for \
                column in table_map[token.lower()].columns]
    
    last_ind = len(sql)
    if ' where' in sql:
        last_ind = sql.rfind(' where')
        extra = sql[last_ind+len(' where'):]
    elif ' having' in sql:
        last_ind = sql.rfind(' having')
        extra = sql[last_ind+len(' where'):]
    else:
        extra = ""
    extra = extra.strip().split()
    i = 0
    while i < len(extra):
        if where != "":
            where += " " 
        token = extra[i]
        if len(token) > 2 and token[2] == '.' and token[0] == 't' and \
            token[1].isnumeric():
            token = token[token.rfind('.')+1:].strip()
            if token not in cols and token not in extra_cols:
                extra_cols.append(token)
                schema_cols.append(token)
            where += token
            i += 1
        elif token in aggs and extra[i+1] == '(':
            if token == 'count' and extra[i+2] == '*':
                col = 'number'
                if col not in cols and col not in extra_cols:
                    extra_cols.append(col)
                    schema_cols.append(col)
                where += col
            else:
                col = extra[i+2]
                if '.' in col:
                    col = col[col.rfind('.')+1:].strip()
                if col not in agg_cols:
                    agg_cols.append(col)
                where += token+"_"+col
            i += 4
        elif token in col_set_lower:
            if col not in cols and col not in extra_cols:
                extra_cols.append(col)
                schema_cols.append(col)
            where += token
            i += 1
        else:
            where += token
            i += 1
    if len(agg_cols) == 0:
        return [None for _ in range(6)]
    where = normalize_sql(where)
    select = ", ".join(cols)
    all_cols = schema_cols.copy()
    all_raw_cols = schema_cols.copy()
    for agg_col in agg_cols:
        all_cols += [agg+"_"+agg_col for agg in aggs[1:]]
        all_raw_cols.append(agg_col)
    new_table_name = "_".join(tbl_list) + "_" + "_".join(agg_cols)
        
    return select, where, all_cols, all_raw_cols, new_table_name, tbl_list

def save_splits(split=['train', 'validation']):
    if type(split) == list:
        for s in split:
            save_splits(s)
        return
    original = spider[split]
    modified = []
    for example in original[:]:
        select, where, all_cols, all_raw_cols, \
            new_table_name, tables = get_replacements(example['query'], \
            example['db_id'])
        if select is None:
            continue
        example['orig_query'] = normalize_sql(example['query'])
        new_query = "select {} from {}".format(select, new_table_name)
        if where.strip() != "":
            new_query += " where " + where
        new_query = normalize_sql(new_query)
        example['query1'] = example['orig_query']
        example['query2'] = new_query
        example['all_cols'] = all_cols
        example['all_raw_cols'] = all_raw_cols
        example['new_table_name'] = new_table_name
        tbls = []
        table_map = db_map_lower[example['db_id'].lower()].tables
        table_map = {x.lower(): table_map[x] for x in table_map}
        for tbl in tables:
            table = table_map[tbl]
            pkey = table.primary_key.lower() if table.primary_key is not None \
                else None
            tbls.append([tbl, pkey])
        example['tables_with_pkeys'] = tbls
        schema_with_content = get_new_schema(new_table_name, \
            all_cols, all_raw_cols, tables, example['db_id'])
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
        example['schema_without_content'] = schema_without_content
        modified.append(example)
    json.dump(modified, open(split+".json", 'w+'))
        
if __name__ == '__main__':
    save_splits()