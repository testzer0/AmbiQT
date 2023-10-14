from utils.globals import *
from transformers import T5Tokenizer, T5ForConditionalGeneration

import re
import os

VERSION = "t5-3b"

def to_device_multiple(device, *args):
    return [arg.to(device) for arg in args]

def get_model_picard(device=None, version=VERSION):
    if version == "t5-3b":
        pversion = "tscholak/cxmefzzi"
    elif version == "t5-large":
        pversion = "tscholak/1wnr382e"
    else:
        pversion = "tscholak/1zha5ono"
    if device is None:
        device = get_device()
    model = T5ForConditionalGeneration.from_pretrained(pversion)
    return model.to(device)

def get_model(device=None, checkpt_dir=os.path.join(CHECKPOINT_ROOT, \
    "flan-template-addn")):
    model = T5ForConditionalGeneration.from_pretrained(checkpt_dir)
    if device is None:
        device = get_device()
    return model.to(device)

def get_tokenizer(model_max_length=512, \
    checkpt_dir=os.path.join(CHECKPOINT_ROOT, "flan-template-addn")):
    return T5Tokenizer.from_pretrained(checkpt_dir, \
        model_max_length=model_max_length)

def get_tokenizer_picard(model_max_length=512, version=VERSION):
    if version == "t5-3b":
        pversion = "tscholak/cxmefzzi"
    elif version == "t5-large":
        pversion = "tscholak/1wnr382e"
    else:
        pversion = "tscholak/1zha5ono"
    return T5Tokenizer.from_pretrained(pversion, model_max_length=model_max_length)

if __name__ == '__main__':
    pass