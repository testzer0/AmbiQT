from utils.globals import *
from utils.data import *
from utils.sql import normalize_sql, extract_sql
from utils.model_utils import *

import os
import torch

from transformers import LogitsProcessor
import torch.nn.functional as F
from tqdm import tqdm

tables = None
tab_set_lower = {}
col_set_lower = {}

# We need a new definition because new columns exist per-entry
def templatize_sql(sql, db_id, overwrite=False):
    global tables, tab_set_lower, col_set_lower
    sql = normalize_sql(sql)
    if overwrite or (tables is None):
        tables = json.load(open(os.path.join(SPIDER_ROOT, "tables.json")))
        tables = {x['db_id']: x for x in tables}
    if db_id not in tab_set_lower:
        tab_set_lower[db_id] = set(x.lower() for x in \
            tables[db_id]["table_names_original"])
        col_set_lower[db_id] = set(x[1].lower() for x in \
            tables[db_id]["column_names_original"])
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
                elif token_pref in col_set_lower[db_id] or token_pref == '*':
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    template = template.lower()
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return normalize_sql(template)

def templatize_sql_from_map(sql, tables, columns):
    sql = normalize_sql(sql)
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
                if token_pref in tables:
                    template.append("table")
                elif token_pref in columns or token_pref == '*':
                    template.append("column")
                else:
                    template.append(token)
    template = " ".join(template)
    template = template.replace(" ,", ",").replace("( ", "(").replace(" )", ")")
    template = template.lower()
    for kword in ["count", "avg", "sum", "min", "max"]:
        template = template.replace(kword+" (", kword+"(")
    template = template.replace(") ,", "),")
    return normalize_sql(template)

def extract_tables_and_columns(schema):
    # Without content
    tables = []
    columns = []
    for tbl_string in schema.strip().split('|'):
        tbl_string = tbl_string.split(":")
        tables.append(tbl_string[0].strip().lower())
        for col_string in tbl_string[1].strip().split(','):
            columns.append(col_string.strip().lower())
    return tables, columns

def disallowed(template, output, template_pred, column=True, table=False, \
    enforce_adherence=True):
    if "|" in output:
        output = output[output.rfind('|')+1:].strip()
    if "@" in output:
        output = output[output.rfind('@')+1:].strip()
    template_pred_previous = "" if " " not in template_pred else \
        template_pred[:template_pred.rfind(" ")]
    if enforce_adherence and not template.startswith(template_pred_previous):
        return -1
    if ' join ' in template and len(output) > 0:
        last_token = output.split(" ")[-1].lower()
        if column and (len(last_token) == 3 and last_token.startswith("t") and \
            last_token[2] == '.'): 
            return 0
    if template.startswith(template_pred) and len(template_pred) > 0:
        last_token = output.split(" ")[-1].lower()
        portion = template[len(template_pred):].strip()
        if ' join ' in template:
            if column and (len(last_token) == 3 and last_token.startswith("t") and \
                last_token[2] == '.'): 
                return 0
        else:
            if column and portion.startswith('column'):
                return 0
        if table and portion.startswith('table'):
            return 0
    return 1

class ControlSplitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, template, db_id, column=True, table=False, \
        schema_without_content=None):
        self.tokenizer = tokenizer
        self.template = template
        self.db_id = db_id
        self.column = column
        self.table = table
        if schema_without_content is None:
            self.columns = None
            self.tables = None
            self.templatize = lambda sql: templatize_sql(extract_sql(sql), \
                self.db_id)
            self.allowed_tokens = None
        else:
            self.tables, self.columns = extract_tables_and_columns( \
                schema_without_content)
            self.templatize = lambda sql: templatize_sql_from_map( \
                extract_sql(sql), self.tables, self.columns)
            self.table_tokens = [self.tokenizer.encode(table)[0] for \
                table in self.tables]
            self.column_tokens = [self.tokenizer.encode(column)[0] for \
                column in self.columns+['*']]
            self.column_tokens += [self.tokenizer.encode("t1."+column)[3] for \
                column in self.columns]
            self.allowed_tokens = []
            if column:
                self.allowed_tokens += self.column_tokens
            if table:
                self.allowed_tokens += self.table_tokens
            self.allowed_tokens = torch.LongTensor(list(set(\
                self.allowed_tokens)))
        
    def __call__(self, input_ids: torch.LongTensor, \
        scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] >= 128:
            return scores
        
        decoded = self.tokenizer.batch_decode(input_ids, \
            skip_special_tokens=True, clean_up_tokenization_spaces=True)
        disallowed_positions = torch.Tensor([disallowed(self.template, d, \
            self.templatize(d), column=self.column, table=self.table) for \
            d in decoded]).unsqueeze(1)
        idxes = torch.argmax(scores, dim=1)
        oh = F.one_hot(idxes, num_classes=scores.shape[1]).\
            to(scores.dtype).to(scores.device)
        oh[oh == 0] = -float("inf")
        
        scores_disallowed = oh.to(scores.device)
        disallowed_positions = disallowed_positions.to(scores.device)
        disallowed_positions = disallowed_positions.repeat(1, scores.shape[1])
        scores[disallowed_positions == 1] = scores_disallowed[disallowed_positions == 1]
        scores[disallowed_positions == -1] = -float("inf")
        
        if self.allowed_tokens is not None:
            aoh = F.one_hot(self.allowed_tokens, \
                num_classes=scores.shape[1]).to(scores.dtype).to(scores.device)
            aoh = torch.sum(aoh, axis=0).unsqueeze(0).repeat(scores.shape[0], 1)
            scores[(disallowed_positions + aoh) == 0] = -float("inf")
        
        return scores

def get_output_controlled(model, tokenizer, model_input, template, \
    db_id, column=True, table=False, schema_without_content=None, \
    beam_width=10, num_outputs=1, device=None):
    if device is None:
        device = get_device()
    encoded = tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids 
    cs_logits_processor = ControlSplitLogitsProcessor(tokenizer, template, \
        db_id, column=column, table=table, \
        schema_without_content=schema_without_content)
    logits_processor = [cs_logits_processor]
    with torch.no_grad():
        outputs = model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=num_outputs, \
            max_length=512, logits_processor=logits_processor)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [x[x.find('|')+1:].strip() if '|' in x else x for x in outputs]
    outputs = [normalize_sql(x) for x in outputs]
    templates = [cs_logits_processor.templatize(x) for x in outputs]
    template_portion = template[template.rfind('@')+1:].strip()
    outputs = [output for output, ptemplate in zip(outputs, templates) if \
        ptemplate == template_portion]
    outputs = [outputs[i] for i in range(len(outputs)) if outputs[i] not in \
        outputs[:i]]
    return outputs

def get_output_beam_search(model, tokenizer, model_input, beam_width=10, \
    num_outputs=1, device=None, **kwargs):
    if device is None:
        device = get_device()
    encoded = tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids 
    with torch.no_grad():
        outputs = model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=num_outputs, \
            max_length=512)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [x[x.find('|')+1:].strip() if '|' in x else x for x in outputs]
    outputs = [normalize_sql(x) for x in outputs]
    outputs = [outputs[i] for i in range(len(outputs)) if outputs[i] not in \
        outputs[:i]]
    return outputs

def save_filled_in_templates(in_path, out_path, beam_width=10, with_content=False, \
    num_outputs=10, column=True, table=True, controlled=True, \
    checkpt_path=os.path.join(CHECKPOINT_ROOT, "flan-template-fill"), **kwargs):
    device = get_device()
    model = get_model(checkpt_dir=checkpt_path, device=device)
    tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
    data = json.load(open(in_path))
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    cmap = None
    if 'schema_without_content' not in data[0]:
        cmap = get_col_mappings(read_table_info())
    for i in tqdm(range(len(data))):
        db_id = data[i]['db_id']
        question = data[i]['question']
        if 'schema_without_content' in data[i]:
            schema = data[i]['schema_without_content']
        else:
            addendum = get_addendum(cmap, db_id, with_content=with_content, \
                lower=True)
            addendum = addendum[addendum.find('|')+1:]
            schema = addendum[addendum.find('|')+1:].strip()
        filled = []
        for template in data[i]['template_outs']:
            if '@' in template:
                template = template[template.rfind('@')+1:].strip()
            template = " ".join(template.split())
            model_input = "template fill: {} | {} | {} @ {}".format(question, \
                db_id, schema, template)
            output_fn = get_output_controlled if controlled else get_output_beam_search
            filled.append(output_fn(model, tokenizer=tokenizer, \
                model_input=model_input, template=template, \
                db_id=data[i]['db_id'], column=column, table=table, \
                schema_without_content=schema, beam_width=beam_width, \
                num_outputs=num_outputs, device=device))
        data[i]['template_filled'] = filled
        data[i].pop('template_outs')
        if i % 5 == 1:
            json.dump(data, open(out_path, 'w+'), indent=4)
    json.dump(data, open(out_path, 'w+'), indent=4)

if __name__ == '__main__':
    pass