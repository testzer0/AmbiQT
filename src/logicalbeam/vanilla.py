from utils.globals import *
from utils.data import *
from utils.sql import normalize_spaces, remove_db_prefix_from_sql
from utils.model_utils import *

import os
import torch

from tqdm import tqdm

t2s_model = None
t2s_tokenizer = None
global_cmap = None

def convert_text_to_sql(text, db_id="", num_outputs=1, beam_width=4, \
    checkpt_path=None, version=VERSION, with_content=True, colname_map={}, \
    addendum=None):
    global t2s_model, t2s_tokenizer, global_cmap
    device = get_device()
    if t2s_model is None:
        if checkpt_path is not None:
            t2s_model = get_model(device=device, checkpt_dir=checkpt_path)
            t2s_tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
        else:
            t2s_model = get_model_picard(device=device, version=version)
            t2s_tokenizer = get_tokenizer_picard(version=version)
        t2s_model.eval()
        global_cmap = get_col_mappings(read_table_info())
    if addendum is None:
        addendum = get_addendum(global_cmap, db_id, question=text, \
            with_content=with_content, colname_map=colname_map)
    model_input = normalize_spaces(text).replace(" , ", ", ") + addendum
    if checkpt_path is not None and 'flan' in checkpt_path:
        model_input = "semantic parse: " + model_input
    encoded = t2s_tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids 
    
    with torch.no_grad():
        outputs = t2s_model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=num_outputs, \
            max_length=512)
        outputs = t2s_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs = [remove_db_prefix_from_sql(sql) for sql in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs

def save_t2s_outputs(split='validation', with_content=False, \
    version=VERSION, checkpt_path=None, out_path=None):
    if split in ['train', 'validation']:
        dataset = load_spider(split)
    else:
        dataset = json.load(open(split))
        split = os.path.basename(split)
        split = split[:split.rfind('.')]
    assert out_path is not None, "Output has to go somewhere!"
    print("Saving the Text-to-SQL outputs for the {} split...".format(split))
    outputs = []
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        addendum_key = "schema_with_content" if with_content else \
            "schema_without_content"
        if addendum_key in entry:
            addendum = " | {} | {}".format(entry['db_id'], entry[addendum_key])
            model_output = convert_text_to_sql(entry['question'], \
                version=version, checkpt_path=checkpt_path, addendum=addendum)
        else:
            model_output = convert_text_to_sql(entry['question'], \
                entry['db_id'], version=version, \
                with_content=with_content, checkpt_path=checkpt_path, \
                colname_map=(entry['syn_map'] if 'syn_map' in entry \
                else {}))
        entry['t2s_out'] = model_output
        outputs.append(entry)
    json.dump(outputs, open(out_path, 'w+'), indent=4)

def save_t2s_outputs_many(split='validation', with_content=False, \
    checkpt_path=None, out_path=None, beam_width=10, num_outputs=5, \
    version=VERSION):
    if split in ['train', 'validation']:
        dataset = load_spider(split)
    else:
        dataset = json.load(open(split))
        split = os.path.basename(split)
        split = split[:split.rfind('.')]
    if out_path is None:
        out_path = "tmp/{}-out.json".format(split)
    print("Saving the Text-to-SQL outputs for the {} split...".format(split))
    outputs = []
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        addendum_key = "schema_with_content" if with_content else \
            "schema_without_content"
        if addendum_key in entry:
            addendum = " | {} | {}".format(entry['db_id'], entry[addendum_key])
            model_output = convert_text_to_sql(entry['question'], \
                version=version, checkpt_path=checkpt_path, \
                addendum=addendum, num_outputs=num_outputs, \
                beam_width=beam_width)
        else:
            model_output = convert_text_to_sql(entry['question'], \
                entry['db_id'], version=version, \
                with_content=with_content, checkpt_path=checkpt_path, \
                colname_map=(entry['syn_map'] if 'syn_map' in entry \
                else {}), num_outputs=num_outputs, beam_width=beam_width)
        entry['t2s_outs'] = model_output
        outputs.append(entry)
    json.dump(outputs, open(out_path, 'w+'), indent=4)