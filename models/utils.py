import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler

from transformers import AdamW, get_linear_schedule_with_warmup

import random
import numpy as np
import logging

THERSHOLDS = [i * 0.1 for i in range(11)]

def load_nli_examples(file_path, do_lower_case):
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split('\t')
            if do_lower_case:
                e = InputExample(fields[0].lower(), fields[1].lower(), fields[2])
            else:
                e = InputExample(fields[0], fields[1], fields[2])
            examples.append(e)
    return examples

def get_logger(name):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H%M%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    return logger

def get_optimizer(model, t_total, args):
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.dam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_toal)
    
    return optimizer, scheduler