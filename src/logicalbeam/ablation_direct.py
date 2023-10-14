from utils.globals import *
from utils.data import *
from utils.sql import normalize_sql, normalize_spaces, remove_db_prefix_from_sql
from utils.model_utils import *

from logicalbeam.template_gen import get_output_with_prefix

import os
import torch

from transformers import LogitsProcessor
import torch.nn.functional as F
from tqdm import tqdm

t2s_model = None
t2s_tokenizer = None
global_cmap = None

tables = None
tab_set_lower = {}
col_set_lower = {}

def convert_text_to_sql_annotated(text, db_id="", num_outputs=1, \
    beam_width=4, with_content=False, colname_map={}, prefix="", \
    addendum=None, checkpt_path=None):
    global t2s_model, t2s_tokenizer, global_cmap
    device = get_device()
    if t2s_model is None:
        if checkpt_path is None:
            checkpt_path = os.path.join(CHECKPOINT_ROOT, "t2s-annotated-nc")
        t2s_model = get_model(device=device, checkpt_dir=checkpt_path)
        t2s_tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
        t2s_model.eval()
        global_cmap = get_col_mappings(read_table_info())
    if addendum is None:
        addendum = get_addendum(global_cmap, db_id, question=text, \
            with_content=with_content, colname_map=colname_map)
    model_input = "semantic parse: " + normalize_spaces(text).replace(" , ", \
        ", ") + addendum
    if prefix == "":
        encoded = t2s_tokenizer(model_input, max_length=512, truncation=True, \
            return_tensors="pt").input_ids 
        
        with torch.no_grad():
            outputs = t2s_model.generate(encoded.to(device), \
                num_beams=beam_width, num_return_sequences=num_outputs, \
                max_length=512)
        outputs = t2s_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    else:
        outputs = get_output_with_prefix(t2s_model, t2s_tokenizer, \
            model_input, prefix, beam_width=beam_width, num_outputs=num_outputs)
    
    outputs = [remove_db_prefix_from_sql(o) for o in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    
    return outputs

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

def disallowed(tokens, allowed_tokens):
    d = []
    for i in range(tokens.shape[0]):
        d.append(0 if tokens[i].detach().item() in allowed_tokens else 1)
    return torch.Tensor(d)

class DiverseLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, db_id, column=True, table=False, \
        schema_without_content="", prefix_tokens=[]):
        self.tokenizer = tokenizer
        self.db_id = db_id
        self.column = column
        self.table = table
        self.n_prefix_tokens = len(prefix_tokens)
        self.prefix_tokens = prefix_tokens
        self.tables, self.columns = extract_tables_and_columns( \
            schema_without_content)
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
        self.allowed_tokens_list = list(set(\
            self.allowed_tokens))
        self.allowed_tokens = torch.LongTensor(self.allowed_tokens_list)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) \
        -> torch.FloatTensor:
        if input_ids.shape[-1] >= 128:
            return scores
        if input_ids.shape[-1] < self.n_prefix_tokens:
            idxes = torch.LongTensor([self.prefix_tokens[input_ids.shape[-1]]])
            oh = F.one_hot(idxes, num_classes=scores.shape[1]).to(scores.dtype)
            oh = oh.repeat(scores.shape[0], 1)
            scores[oh == 0] = -float("inf")
            return scores
        idxes = torch.argmax(scores, dim=1)
        disallowed_positions = disallowed(idxes, self.allowed_tokens_list).\
            unsqueeze(1)
        oh = F.one_hot(idxes, num_classes=scores.shape[1]).\
            to(scores.dtype).to(scores.device)
        oh[oh == 0] = -float("inf")
        
        scores_disallowed = oh.to(scores.device)
        disallowed_positions = disallowed_positions.to(scores.device)
        disallowed_positions = disallowed_positions.repeat(1, scores.shape[1])
        scores[disallowed_positions == 1] = scores_disallowed[\
            disallowed_positions == 1]
        
        aoh = F.one_hot(self.allowed_tokens, num_classes=scores.shape[1]).\
            to(scores.dtype).to(scores.device)
        aoh = torch.sum(aoh, axis=0).unsqueeze(0).repeat(scores.shape[0], 1)
        scores[(disallowed_positions + aoh) == 0] = -float("inf")
        
        return scores

def get_prefix(db_id, joins, selects):
    return "{} | {} joins @ {} selects @".format(db_id, joins, selects)

def get_output_controlled(model, tokenizer, question, joins, selects, db_id, \
    schema_without_content=None, column=True, table=True, beam_width=10, \
    num_outputs=3, device=None):
    if device is None:
        device = get_device()
    
    prefix = get_prefix(db_id, joins, selects)
    model_input = "semantic parse: " + normalize_spaces(question).replace(" , ", \
        ", ") + " | {} | {}".format(db_id, schema_without_content)
    
    encoded = tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids
    
    prefix_encoded = tokenizer(prefix, max_length=512, truncation=True, \
        return_tensors="pt").input_ids[0].tolist()
    idx = prefix_encoded.index(1)
    prefix_tokens = prefix_encoded[:idx]
    if prefix_tokens[0] != 0:
        prefix_tokens = [0] + prefix_tokens
     
    cs_logits_processor = DiverseLogitsProcessor(tokenizer, db_id, \
        column=column, table=table, \
        schema_without_content=schema_without_content, \
        prefix_tokens=prefix_tokens)
    logits_processor = [cs_logits_processor]
    with torch.no_grad():
        outputs = model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=num_outputs, \
            max_length=512, logits_processor=logits_processor)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    outputs = [x[x.rfind('@')+1:].strip() if '@' in x else x for x in outputs]
    outputs = [normalize_sql(x) for x in outputs]
    
    return outputs

def convert_text_to_prefix(text, db_id="", beam_width=10, \
    with_content=False, colname_map={}, addendum=None, checkpt_path=None):
    """
    Get the most preferred output, and then extract its prefix (number of joins & selects).
    """
    global t2s_model, t2s_tokenizer, global_cmap
    device = get_device()
    if t2s_model is None:
        if checkpt_path is None:
            checkpt_path = os.path.join(CHECKPOINT_ROOT, "t2s-annotated-nc")
        t2s_model = get_model(device=device, checkpt_dir=checkpt_path)
        t2s_tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
        t2s_model.eval()
        global_cmap = get_col_mappings(read_table_info())
    if addendum is None:
        addendum = get_addendum(global_cmap, db_id, question=text, \
            with_content=with_content, colname_map=colname_map)
    model_input = "semantic parse: " + normalize_spaces(text).replace(" , ", \
        ", ") + addendum
    encoded = t2s_tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids 
    
    with torch.no_grad():
        outputs = t2s_model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=1, \
            max_length=512)
    output = t2s_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    output = output[output.find('|')+1:].strip()
    
    join_string = output[:output.find('@')].strip()
    output = output[output.find('@')+1:].strip()
    select_string = output[:output.find('@')]
    joins = int(join_string.split(" ")[0])
    selects = int(select_string.split(" ")[0])
    
    return joins, selects

def save_direct_t2s_with_plan_without_branching_control(split_or_path='validation', \
    with_content=False, out_path=None, beam_width=10, checkpt_path=None):
    """
    question --> plan | sql, but decoding is unconstrained in the `sql` part.
    """
    if split_or_path in ['train', 'validation']:
        dataset = load_spider(split_or_path)
    else:
        dataset = json.load(open(split_or_path))
        split_or_path = os.path.basename(split_or_path)
        split_or_path = split_or_path[:split_or_path.rfind('.')]
    assert out_path is not None, "Output must go somewhere!"
    print("Saving the Text-to-SQL outputs for the {} split...".format(split_or_path))
    outputs = []
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        top = convert_text_to_sql_annotated(entry['question'], \
                entry['db_id'], with_content=with_content, \
                colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                num_outputs=1, beam_width=beam_width, checkpt_path=checkpt_path)
        outs = []
        top = top.split('@')
        oj = int(top[0].strip().split(' ')[0])
        osx = int(top[1].strip().split(' ')[0])
        num_outputs = min(3, beam_width)
        if num_outputs == 1:
            outs.append(convert_text_to_sql_annotated(entry['question'], \
                entry['db_id'], with_content=with_content, \
                colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                num_outputs=num_outputs, beam_width=beam_width, \
                prefix="{} joins @ {} selects @".format(oj, osx), checkpt_path=checkpt_path))
        else:
            outs += convert_text_to_sql_annotated(entry['question'], \
                entry['db_id'], with_content=with_content, \
                colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                num_outputs=num_outputs, beam_width=beam_width, \
                prefix="{} joins @ {} selects @".format(oj, osx), checkpt_path=checkpt_path)
        if oj >= 4:
            njs = [oj-1, oj-2]
        else:
            njs = [oj-1, oj+1]
        for nj in njs:
            if nj >= 0:
                num_outputs = min(beam_width, 3 if oj == 0 else 2)
                if num_outputs == 1:
                    outs.append(convert_text_to_sql_annotated(\
                        entry['question'], entry['db_id'], \
                        with_content=with_content, \
                        colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                        num_outputs=num_outputs, beam_width=beam_width, \
                        prefix="{} joins @ {} selects @".format(nj, osx), checkpt_path=checkpt_path))
                else:
                    outs += convert_text_to_sql_annotated(\
                        entry['question'], entry['db_id'], \
                        with_content=with_content, \
                        colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                        num_outputs=num_outputs, beam_width=beam_width, \
                        prefix="{} joins @ {} selects @".format(nj, osx), checkpt_path=checkpt_path)
        for ns in [osx-1, osx+1]:
            if ns > 0:
                nout = convert_text_to_sql_annotated(entry['question'], \
                    entry['db_id'], with_content=with_content, \
                    colname_map=(entry['syn_map'] if 'syn_map' in entry else {}), \
                    num_outputs=1, beam_width=beam_width, \
                    prefix="{} joins @ {} selects @".format(oj, ns), checkpt_path=checkpt_path)
                outs.append(nout)
        entry['t2s_outs'] = outs
        outputs.append(entry)
    json.dump(outputs, open(out_path, 'w+'), indent=4)

def save_direct_t2s_with_plan_with_branching_control(split_or_path, out_path, beam_width=10, \
    column=True, table=True, checkpt_path=None):
    """
    question --> plan | sql, decoding in the `sql` part is constrained to whitelist of schema tokens.
    """
    global t2s_model, t2s_tokenizer, global_cmap
    device = get_device()
    if t2s_model is None:
        if checkpt_path is None:
            checkpt_dir = os.path.join(CHECKPOINT_ROOT, "t2s-annotated-nc")
        t2s_model = get_model(device=device, checkpt_dir=checkpt_path)
        t2s_tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
        t2s_model.eval()
        global_cmap = get_col_mappings(read_table_info())
    if split_or_path in ['train', 'validation']:
        dataset = load_spider(split_or_path)
    else:
        dataset = json.load(open(split_or_path))
        split_or_path = os.path.basename(split_or_path)
        split_or_path = split_or_path[:split_or_path.rfind('.')]
    print("Saving the direct outputs for the {} split...".format(split_or_path))
    outputs = []
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        addendum = " | {} | {}".format(entry['db_id'], \
            entry['schema_without_content'])
        oj, osx = convert_text_to_prefix(entry['question'], \
            entry['db_id'], beam_width, addendum=addendum)
        outs = []
        outs.append(get_output_controlled(t2s_model, t2s_tokenizer, \
            entry['question'], oj, osx, entry['db_id'], \
            schema_without_content=entry['schema_without_content'], \
            column=column, table=table, beam_width=beam_width, \
            num_outputs=2, device=device))
        if oj >= 4:
            njs = [oj-1, oj-2]
        else:
            njs = [oj-1, oj+1]
        for nj in njs:
            if nj >= 0:
                outs.append(get_output_controlled(t2s_model, t2s_tokenizer, \
                    entry['question'], nj, osx, entry['db_id'], \
                    schema_without_content=entry['schema_without_content'], \
                    column=column, table=table, beam_width=beam_width, \
                    num_outputs=2, device=device))
        for ns in [osx-1, osx+1]:
            if ns > 0:
                outs.append(get_output_controlled(t2s_model, t2s_tokenizer, \
                    entry['question'], oj, ns, entry['db_id'], \
                    schema_without_content=entry['schema_without_content'], \
                    column=column, table=table, beam_width=beam_width, \
                    num_outputs=2, device=device))
        entry["flan_direct_outs"] = outs
        outputs.append(entry)
        if (i+1) % 5 == 0:
            json.dump(outputs, open(out_path, 'w+'), indent=4)
    json.dump(outputs, open(out_path, 'w+'), indent=4)

if __name__ == '__main__':
    pass