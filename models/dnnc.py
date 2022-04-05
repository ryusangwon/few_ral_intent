import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .utils import get_optimizer

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
    
    def train(self, train_examples, dev_examples, file_path=None):
        
        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)
        
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        
        num_train_steps = int(len(train_examples)/train_batch_size/self.args.gradient_accumulation_steps * self.args.num_train_epochs)
        
        optimizer, scheduler = get_optimizer(self.model, num_train_steps, self.args)
        
        best_dev_accuracy = -1.0
        
        train_features, label_distribution = self.convert_examples_to_features(train_examples, train=True)
        train_dataloader = get_train_dataloader(train_features, train_batch_size)
    
    def convert_examples_to_features(self, examples, train):
        