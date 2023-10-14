from utils.globals import *
import json

tables = None
tab_set_lower = {}
col_set_lower = {}

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
    if len(sql) > 0 and sql[-1] == ';':
        sql = sql[:-1]
    sql = " ".join(lower(sql).strip().split())
    sql = sql.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        sql = sql.replace(kword+" (", kword+"(")
    return sql.replace(") ,", "),")

def normalize_spaces(text):
    return " ".join(text.split())

def extract_sql(sql):
    if '|' in sql:
        sql = sql[sql.find('|')+1:].strip()
    if '@' in sql:
        sql = sql[sql.find('@')+1:].strip()
    return sql

def templatize_sql(sql, db_id):
    global tables, tab_set_lower, col_set_lower
    sql = sql.replace("!=", " !=")
    sql = " ".join(sql.split())
    if tables is None:
        tables = json.load(open("spider/tables.json"))
        tables = {x['db_id']: x for x in tables}
    if db_id not in tab_set_lower:
        tab_set_lower[db_id] = set(x.lower() for x in \
            tables[db_id]["table_names_original"])
        col_set_lower[db_id] = set(x[1].lower() for x in \
            tables[db_id]["column_names_original"])
    if sql[-1] == ';':
        sql = sql[:-1]
    sql = sql.replace("''", "\"").replace(",", " , ").\
        replace("(", " ( ").replace(")", " ) ").split()
    template = []
    current_quote = None
    for token in sql:
        if current_quote is not None:
            if token[-1] == current_quote:
                current_quote = None
                template.append("string")
        else:
            if token[0] in ['\'', '"', '`']:
                if len(token) > 1 and token[-1] == token[0]:
                    template.append("string")
                else:
                    current_quote = token[0]
            elif token.isnumeric():
                template.append("number")
            else:
                if len(token) > 3 and token[0] in ('t', 'T') and token[2] == '.':
                    token_pref = token[3:].lower()
                else:
                    token_pref = token.lower()
                if token_pref in tab_set_lower[db_id]:
                    template.append("table")
                elif token_pref in col_set_lower[db_id]:
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    template = template.lower()
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return template

def get_num_join_equivalents(sql):
    sql = " ".join(sql.split()).replace("( SELECT", "(SELECT")
    n = 1
    for joiner in ['JOIN', 'INTERSECT', 'UNION', 'EXCEPT', '(SELECT']:
        n += sql.count(joiner)
    return n

def remove_db_prefix_from_sql(sql):
    if '|' in sql:
        sql = sql[sql.find('|')+1:].strip()
    return sql

if __name__ == '__main__':
    pass