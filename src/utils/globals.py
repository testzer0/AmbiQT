"""
Adithya Bhaskar, 2022.
This file lists all global variables that are not part of a configuration file.
The purpose is to ease routine operations by having some common paths and variables defined.
"""

import os
import torch

REPO_ROOT = "/home/adithya/sem8/t2s-repo/"
PROMPT_ROOT = os.path.join(REPO_ROOT, "prompts")
BENCH_ROOT = os.path.join(REPO_ROOT, "benchmark")
CODE_ROOT = os.path.join(REPO_ROOT, "src")
SPIDER_ROOT = os.path.join(REPO_ROOT, "spider")
CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "checkpoints")

TEMPLATE_CHECKPT_DIR = os.path.join(CHECKPOINT_ROOT,  "template")
TEMPLATE_FIN_CHECKPT_DIR = os.path.join(CHECKPOINT_ROOT, "template-fillin")
DB_DIR = os.path.join(REPO_ROOT, "db-content", "database")
CONTENT_PATH = os.path.join(REPO_ROOT, "db-content", "values-lower.json")

GPU_NUMBER = 0

def set_gpu_number(gpu_number):
    global GPU_NUMBER
    GPU_NUMBER = gpu_number

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:{}".format(GPU_NUMBER))
    else:
        return torch.device("cpu")

if __name__ == '__main__':
    pass