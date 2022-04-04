import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

import os
import random
import numpy as np
from tqdm import tqdm, trange

ENTAILMENT = 'entailment'
NON_ENTAILMENT = 'non_entailment'

class DNNC:
    def __init__(self,
                 path: str,
                 args):
        
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu")
        
        self.label_list = [ENTAILMENT, NON_ENTAILMENT]
        self.num_labels = len(self.label_list)
        
        self.config = AutoConfig.from_pretrained(self.args.bert_model, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.bert_model)
        
        if path is not None:
            state_dict = torch.load(path+"/pytorch_model.bin")
            self.model = AutoModelForSequenceClassification.from_pretrained(path, state_dict=state_dict, config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.args.bert_model, config=self.config)
        self.model.to(self.device)