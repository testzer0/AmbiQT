import sys
sys.path.append("../")

from utils.globals import *
from utils.sql import normalize_sql, normalize_spaces
from chatgpt.helper import get_reply_with_retries

import os
import json
import random
import time
from tqdm import tqdm

import argparse


tables = json.load(open(os.path.join(SPIDER_ROOT, 'tables.json')))
values = json.load(open(CONTENT_PATH))

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
        
    def stringify(self, with_content=False, max_examples=4):
        col_string = "{}[{}] ".format(self.column_name, self.column_type)
        if with_content:
            if len(self.examples) <= max_examples:
                content_string = ", ".join(self.examples)
            else:
                content_string = ", ".join(random.choices(self.examples, \
                    k=max_examples))
            col_string += "(" + content_string + ")"
        return col_string

class Table:
    def __init__(self, table_name, primary_key):
        self.table_name = table_name
        self.primary_key = primary_key
        self.columns = []
    
    def set_examples(self, examples):
        for i in range(len(self.columns)):
            col_name = self.columns[i].column_name.lower()
            if col_name in examples:
                self.columns[i].set_examples(examples[col_name])
        
    def add_column(self, column):
        self.columns.append(column)
        
    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def stringify(self, with_content=False, max_examples=4, \
        with_primary_key=False):
        col_strings = [column.stringify(with_content=with_content, \
            max_examples=max_examples) for column in self.columns]
        col_part = ", ".join(col_strings)
        if with_primary_key:
            tbl_string = self.table_name + "[Primary Key = {}] : ".format( \
                self.primary_key if self.primary_key is not None else \
                "NONE") + col_part
        else:
            tbl_string = self.table_name + " : " + col_part
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
        with_primary_key=False):
        tbl_strings = [table.stringify(with_content=with_content, \
            max_examples=max_examples, with_primary_key=with_primary_key) \
            for _, table in self.tables.items()]
        return " | ".join(tbl_strings)

def build_db_map(tables):
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
        db_map[db_id] = database
    return db_map

db_map = build_db_map(tables)
   
def normalize_chatgpt_sql(sql):
    sql = sql.replace('\n', ' ').replace('\t', ' ')
    return normalize_sql(sql)

def try_extract_json(string, check=False):
    i = 0 
    j = len(string)-1
    while i < len(string) and string[i] != '{':
        i += 1
    while j >= 0 and string[j] != '}':
        j -= 1
    if i >= j:
        return None
    try:
        j = json.loads(string[i:j+1])
    except:
        return None
    if check:
        for key in ["table", "new_column", "example_entries", "alternate_query_1", \
            "alternate_query_2", "alternate_question"]:
            if key not in j:
                return None
        if len(j["example_entries"]) == 0 or type(j["example_entries"][0]) not in \
            [str, int]:
            return None
    return j

def templatize_sql(sql, db_id, new_column=""):
    global tables_x, tab_set_lower, col_set_lower
    if tables_x is None:
        tables_x = {x['db_id']: x for x in tables}
    if db_id not in tab_set_lower:
        tab_set_lower[db_id] = set(x.lower() for x in \
            tables_x[db_id]["table_names_original"])
        col_set_lower[db_id] = set(x[1].lower() for x in \
            tables_x[db_id]["column_names_original"])
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
                elif token_pref in col_set_lower[db_id] or token_pref.lower() == new_column.lower():
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return template

def save_two_table_synonyms(out_path='two_tbl_synonyms.json'):
    base_prompt = open(os.path.join(PROMPT_ROOT, \
        'two_table_synonyms.txt')).read().strip()
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))
    base_prompt = [part.strip() for part in base_prompt.split('--[SNIP]--')]
    directive = base_prompt[0]
    eg_user = base_prompt[1]
    eg_cgpt = base_prompt[2]
    base_query = base_prompt[3]
    for db_id in tqdm(db_map):
        if db_id in out:
            continue
        stored = {}
        table_names = [table for table in db_map[db_id].tables]
        query = base_query.replace('[DB_ID]', db_id).replace("[TABLES_STRING]", \
            ", ".join(['"'+name+'"' for name in table_names]))
        for tname in table_names:
            prompt = query.replace("[TABLE_NAME]", tname)
            cnt = 0
            while True:
                if cnt > 0:
                    if cnt >= 10:
                        print("Failed {} times - aborting".format(cnt))
                        exit(0)
                    time.sleep(3)
                reply = get_reply_with_retries(directive, prompt, \
                    past_interactions=[(eg_user, eg_cgpt)])
                reply = reply.replace('\'', '"')
                saved_reply = reply
                success = True
                try:
                    reply = json.loads(reply)
                    if len(reply) <= 1 or type(reply[0]) != str or \
                        type(reply[1]) != str  or \
                        any([c in reply[0] for c in [' ', '\'', '"']]) or \
                        any([c in reply[1] for c in [' ', '\'', '"']]):
                        success = False
                    else:
                        reply = reply[:2]
                except:
                    success = False                 
                if not success:
                    print("Reply is not well formed:")
                    print(saved_reply)
                    print("Retrying...")
                    cnt += 1
                    continue
                stored[tname] = reply
                break
        out[db_id] = stored
        json.dump(out, open(out_path, 'w+'))

def save_two_column_synonyms(out_path='two_col_synonyms.json'):
    base_prompt = open(os.path.join(PROMPT_ROOT, \
        'two_col_synonyms.txt')).read().strip()
    out = {}
    if os.path.exists(out_path):
        out = json.load(open(out_path))
    base_prompt = [part.strip() for part in base_prompt.split('--[SNIP]--')]
    directive = base_prompt[0]
    eg_user = base_prompt[1]
    eg_cgpt = base_prompt[2]
    base_query = base_prompt[3]
    for db_id in tqdm(db_map):
        per_db_map = {}
        if db_id in out:
            continue
        for table_name in db_map[db_id].tables:
            query = base_query.replace('[DB_ID]', db_id).replace("[TABLE_NAME]", \
                table_name)
            column_names = [column.column_name for column in \
                db_map[db_id].tables[table_name].columns]
            column_string = ", ".join("\"{}\"".format(column_name) for column_name \
                in column_names)
            prompt = query.replace("[COLUMN_NAMES]", column_string)
            cnt = 0
            while True:
                if cnt > 0:
                    if cnt >= 10:
                        print("Failed {} times - aborting".format(cnt))
                        exit(0)
                    time.sleep(3)
                reply = get_reply_with_retries(directive, prompt, \
                    past_interactions=[(eg_user, eg_cgpt)])
                reply = reply.replace('\'', '"')
                saved_reply = reply
                success = True
                try:
                    reply = json.loads(reply)
                    for column_name in column_names:
                        if column_name not in reply or len(reply[column_name]) < 2:
                            success = False
                            break
                        else:
                            reply[column_name] = reply[column_name][:2]
                except:
                    success = False                 
                if not success:
                    print("Reply is not well formed:")
                    print(saved_reply)
                    print("Retrying...")
                    cnt += 1
                    continue
                per_db_map[table_name] = reply
                break
        out[db_id] = per_db_map
        json.dump(out, open(out_path, 'w+'))

def remove_last_comma(response):
    if ']' not in response:
        return response
    i = response.rfind(']')
    i -= 1
    while i >= 0 and response[i] in [' ', '\n', '\t']:
        i -= 1
    if i < 0 or response[i] != ',':
        return response
    return response[:i] + response[i+1:]

def save_benchmark_outputs(task_name, split='validation', resume=True, \
    in_path=None, out_path=None):
    print("Saving outputs for {}-{}...\n".format(task_name, split))
    task_dir = os.path.join(BENCH_ROOT, task_name)
    assert out_path is not None, "Need an output path!"
    
    if "/" not in out_path:
        out_path = "./" + out_path    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if in_path is None:
        in_path = os.path.join(task_dir, split+".json")
        
    base_prompt = open(os.path.join(PROMPT_ROOT, 'benchmark.txt')).read().strip()
    if resume and os.path.exists(out_path):
        out = json.load(open(out_path))
    else:
        out = json.load(open(in_path))   
    base_prompt = [part.strip() for part in base_prompt.split('--SNIP--')]
    directive = base_prompt[0]
    user1 = base_prompt[1]
    cgpt1 = base_prompt[2]
    base_prompt = base_prompt[3]
    for i in tqdm(range(len(out))):
        if "chatgpt_out" in out[i]:
            continue
        db_id = out[i]['db_id']
        question = normalize_spaces(out[i]['question'])
        schema = out[i]['schema_with_content']
        prompt = base_prompt.replace('[DB_ID]', db_id).replace("[SCHEMA]", \
            schema).replace("[QUESTION]", question)
        cnt = 0
        while True:
            if cnt > 0:
                if cnt >= 10:
                    print("Failed {} times - aborting".format(cnt))
                    exit(0)
                time.sleep(3)
            reply = get_reply_with_retries(directive, prompt, \
                past_interactions=[(user1, cgpt1)])
            saved_reply = reply
            success = True
            try:
                reply = reply[reply.find('{'):(1+reply.rfind('}'))]
                reply = remove_last_comma(reply)
                reply = json.loads(reply)
                if "queries" not in reply:
                    success = False
                else:
                    reply = [normalize_chatgpt_sql(sql) for sql in reply['queries']]
            except Exception as e:
                success = False   
                print(e)              
            if len(reply) < 3:
                print("Too few queries!")
                success = False
            if success:
                break
            else:
                print("Reply is not well formed:")
                print(saved_reply)
                print("Retrying...")
                cnt += 1
        out[i]["chatgpt_out"] = reply
        if ((i+1)%5) == 0:
            json.dump(out, open(out_path, 'w+'), indent=4)
            
    json.dump(out, open(out_path, 'w+'), indent=4)

def save_spider_outputs(resume=True, split="dev", out_path=None):
    assert out_path is not None, "Output path must be specified!"
    print("Saving outputs for SPIDER-{}...".format(split))
    if "/" not in out_path:
        out_path = "./" + out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    base_prompt = open(os.path.join(PROMPT_ROOT, 'benchmark.txt')).read().strip()
    if resume and os.path.exists(out_path):
        out = json.load(open(out_path))
    else:
        out = json.load(open(os.path.join(SPIDER_ROOT, "{}.json".format("spider-train" if \
            split == "train" else split))))
        
    base_prompt = [part.strip() for part in base_prompt.split('--SNIP--')]
    directive = base_prompt[0]
    user1 = base_prompt[1]
    cgpt1 = base_prompt[2]
    base_prompt = base_prompt[3]
    for i in tqdm(range(len(out))):
        if "chatgpt_out" in out[i]:
            continue
        db_id = out[i]['db_id']
        question = normalize_spaces(out[i]['question'])
        schema = db_map[db_id].stringify(with_content=True, max_examples=3)
        prompt = base_prompt.replace('[DB_ID]', db_id).replace("[SCHEMA]", \
            schema).replace("[QUESTION]", question)
        cnt = 0
        while True:
            if cnt > 0:
                if cnt >= 10:
                    print("Failed {} times - aborting".format(cnt))
                    exit(0)
                time.sleep(3)
            reply = get_reply_with_retries(directive, prompt, \
                past_interactions=[(user1, cgpt1)])
            saved_reply = reply
            success = True
            try:
                reply = reply[reply.find('{'):(1+reply.rfind('}'))]
                reply = remove_last_comma(reply)
                reply = json.loads(reply)
                if "queries" not in reply:
                    success = False
                else:
                    reply = [normalize_chatgpt_sql(sql) for sql in reply['queries']]
            except Exception as e:
                success = False   
                print(e)              
            if len(reply) < 3:
                print("Too few queries!")
                success = False
            if success:
                break
            else:
                print("Reply is not well formed:")
                print(saved_reply)
                print("Retrying...")
                cnt += 1
        out[i]["chatgpt_out"] = reply
        if ((i+1)%5) == 0:
            json.dump(out, open(out_path, 'w+'), indent=4)
            
    json.dump(out, open(out_path, 'w+'), indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-m", help="Must be one of save-two-column-synonyms, " + \
        "save-two-table-synonyms, eval", default="eval")
    parser.add_argument("--split", "-s", help="SPIDER split (to use instead of --in_path), or " + \
        "benchmark split", default=None)
    parser.add_argument("--in-path", "-i", help="Input file if not passing split", default=None)
    parser.add_argument("--out-path", "-o", help="Where should the output be saved? (JSON)", \
        default=None)
    parser.add_argument("--resume", "-r", help="The script saves outputs and stops after repeated " + \
        "failure. Set this to true if you'd like to continue from where you left off.", \
        action='store_true')

    args = parser.parse_args()

    accepted_modes = ["save-two-column-synonyms", "save-two-table-synonyms", "eval"]
    assert args.mode in accepted_modes, "Invalid mode, must be one of " + ", ".join(accepted_modes)
    assert args.out_path is not None, "Output must go somewhere!"
    if args.mode == "eval":
        assert (args.split is not None or args.in_path is not None), "At least one of --split or " + \
            "--in-path must be specified!"
    
    return args
    
def main():
    args = parse_args()
    if args.mode == "save-two-column-synonyms":
        save_two_column_synonyms(out_path=args.out_path)
    elif args.mode == "save-two-table-synonyms":
        save_two_table_synonyms(out_path=args.out_path)
    elif args.split in ['train', 'validation', 'dev']:
        if args.split == 'validation':
            args.split = 'dev'
        save_spider_outputs(resume=args.resume, split=args.split, \
            out_path=args.out_path)
    elif args.split in ['col-synonyms', 'tbl-synonyms', 'tbl-split', 'tbl-agg']:
        save_benchmark_outputs(task_name=args.split, split="validation", \
            resume=args.resume, in_path=None, out_path=args.out_path)
    else:
        save_benchmark_outputs(task_name=args.split, split=None, \
            resume=args.resume, in_path=args.in_path, out_path=args.out_path)

if __name__ == '__main__':
    main()