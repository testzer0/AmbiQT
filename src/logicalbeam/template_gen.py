from utils.globals import *
from utils.data import *
from utils.sql import normalize_spaces, remove_db_prefix_from_sql
from utils.model_utils import *

import os
import torch

from transformers import LogitsProcessor
import torch.nn.functional as F
from tqdm import tqdm

template_model = None
template_tokenizer = None
global_cmap = None

class EnforcePrefixLogitsProcessor(LogitsProcessor):
    def __init__(self, tokens):
        self.n_tokens = len(tokens)
        self.tokens = tokens
        
    def __call__(self, input_ids: torch.LongTensor, \
        scores: torch.FloatTensor) -> torch.FloatTensor:
        current_index = input_ids.shape[-1]
        if current_index >= self.n_tokens:
            return scores
        idxes = torch.LongTensor([self.tokens[current_index]])
        oh = F.one_hot(idxes, num_classes=scores.shape[1]).to(scores.dtype)
        oh[oh == 0] = -float("inf")
        scores = oh.repeat(scores.shape[0], 1).to(scores.device)
        return scores

def get_output_with_prefix(model, tokenizer, model_input, prefix, \
    beam_width=10, num_outputs=1, device=None):
    if device is None:
        device = get_device()
    encoded = tokenizer(model_input, max_length=512, truncation=True, \
        return_tensors="pt").input_ids 
    prefix_encoded = tokenizer(prefix, max_length=512, truncation=True, \
        return_tensors="pt").input_ids[0].tolist()
    idx = prefix_encoded.index(1)
    tokens = prefix_encoded[:idx]
    if tokens[0] != 0:
        tokens = [0] + tokens
    logits_processor = [EnforcePrefixLogitsProcessor(tokens)]
    with torch.no_grad():
        outputs = model.generate(encoded.to(device), \
            num_beams=beam_width, num_return_sequences=num_outputs, \
            max_length=512, logits_processor=logits_processor)
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs

def convert_text_to_template(text, db_id="", num_outputs=1, beam_width=4, \
    checkpt_path=None, with_content=False, colname_map={}, prefix="", \
    addendum=None):
    global template_model, template_tokenizer, global_cmap
    device = get_device()
    if template_model is None:
        if checkpt_path is None:
            template_model = get_model(device=device)
            template_tokenizer = get_tokenizer()
        else:
            template_model = get_model(device=device, checkpt_dir=checkpt_path)
            template_tokenizer = get_tokenizer(checkpt_dir=checkpt_path)
        template_model.eval()
        global_cmap = get_col_mappings(read_table_info())
    if addendum is None:
        addendum = get_addendum(global_cmap, db_id, question=text, \
            with_content=with_content, colname_map=colname_map)
    model_input = "template generation: " + normalize_spaces(text).replace(" , ", \
        ", ") + addendum
    if prefix == "":
        encoded = template_tokenizer(model_input, max_length=512, truncation=True, \
            return_tensors="pt").input_ids 
        
        with torch.no_grad():
            outputs = template_model.generate(encoded.to(device), \
                num_beams=beam_width, num_return_sequences=num_outputs, \
                max_length=512)
        outputs = template_tokenizer.batch_decode(outputs, \
            skip_special_tokens=True)
    else:
        outputs = get_output_with_prefix(template_model, template_tokenizer, \
            model_input, prefix, beam_width=beam_width, num_outputs=num_outputs)
    
    outputs = [remove_db_prefix_from_sql(o) for o in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    
    return outputs

def save_templates_beamsearch(split='validation', with_content=False, \
    checkpt_path=None, out_path="None", beam_width=10, num_outputs=5):
    """
    text -> template with beam search
    """
    if split in ['train', 'validation']:
        dataset = load_spider(split)
    else:
        dataset = json.load(open(split))
        split = os.path.basename(split)
        split = split[:split.rfind('.')]
    assert out_path is not None, "Output has to go somewhere!"
    print("Saving the Text-to-Template outputs for the {} split...".format(split))
    outputs = []
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        addendum_key = "schema_with_content" if with_content else \
            "schema_without_content"
        if addendum_key in entry:
            addendum = " | {} | {}".format(entry['db_id'], entry[addendum_key])
            model_output = convert_text_to_template(entry['question'], \
                checkpt_path=checkpt_path, addendum=addendum, \
                num_outputs=num_outputs, beam_width=beam_width)
        else:
            model_output = convert_text_to_template(entry['question'], \
                entry['db_id'], with_content=with_content, \
                checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                'syn_map' in entry else {}), num_outputs=num_outputs, \
                beam_width=beam_width)
        entry['template_outs'] = model_output
        outputs.append(entry)
    json.dump(outputs, open(out_path, 'w+'), indent=4)

def save_templates_logicalbeam(split='validation', \
    with_content=False, checkpt_path=os.path.join(CHECKPOINT_ROOT, \
    "flan-template-addn"), out_path=None, beam_width=1):
    """
    text -> template with prefix enforcement
    """
    if split in ['train', 'validation']:
        dataset = load_spider()[split]
    else:
        dataset = json.load(open(split))
        split = os.path.basename(split)
        split = split[:split.rfind('.')]
    assert out_path is not None, "Output must go somewhere!"
    print("Saving the Text-to-Template outputs for the {} split...".format(split))
    outputs = []
    for i in tqdm(range(len(dataset))):
        entry = dataset[i]
        top = convert_text_to_template(entry['question'], \
                entry['db_id'], with_content=with_content, \
                checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                'syn_map' in entry else {}), num_outputs=1, \
                beam_width=beam_width)
        outs = []
        top = top.split('@')
        oj = int(top[0].strip().split(' ')[0])
        osx = int(top[1].strip().split(' ')[0])
        num_outputs = min(beam_width, 3)
        if num_outputs == 1:
            outs.append(convert_text_to_template(entry['question'], \
                entry['db_id'], with_content=with_content, \
                checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                'syn_map' in entry else {}), num_outputs=num_outputs, \
                beam_width=beam_width, prefix="{} joins @ {} selects @".\
                format(oj, osx)))
        else:
            outs += convert_text_to_template(entry['question'], \
                entry['db_id'], with_content=with_content, \
                checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                'syn_map' in entry else {}), num_outputs=num_outputs, \
                beam_width=beam_width, prefix="{} joins @ {} selects @".\
                format(oj, osx))
        if oj >= 4:
            njs = [oj-1, oj-2]
        else:
            njs = [oj-1, oj+1]
        for nj in njs:
            if nj >= 0:
                num_outputs = min(beam_width, 3 if oj == 0 else 2)
                if num_outputs == 1:
                    outs.append(convert_text_to_template(entry['question'], \
                        entry['db_id'], with_content=with_content, \
                        checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                        'syn_map' in entry else {}), num_outputs=num_outputs, \
                        beam_width=beam_width, prefix="{} joins @ {} selects @".\
                        format(nj, osx)))
                else:
                    outs += convert_text_to_template(entry['question'], \
                        entry['db_id'], with_content=with_content, \
                        checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                        'syn_map' in entry else {}), num_outputs=num_outputs, \
                        beam_width=beam_width, prefix="{} joins @ {} selects @".\
                        format(nj, osx))
        for ns in [osx-1, osx+1]:
            if ns > 0:
                nout = convert_text_to_template(entry['question'], \
                    entry['db_id'], with_content=with_content, \
                    checkpt_path=checkpt_path, colname_map=(entry['syn_map'] if \
                    'syn_map' in entry else {}), num_outputs=1, \
                    beam_width=beam_width, prefix="{} joins @ {} selects @".\
                    format(oj, ns))
                outs.append(nout)
        entry['template_outs'] = outs
        outputs.append(entry)
        if (i+1)%5 == 0:
            json.dump(outputs, open(out_path, 'w+'), indent=4)
    json.dump(outputs, open(out_path, 'w+'), indent=4)


if __name__ == '__main__':
    pass