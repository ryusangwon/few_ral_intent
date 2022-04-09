import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification

from .utils import get_optimizer, get_train_dataloader

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
        label_map = {label: i for i, label in enumerate(self.label_list)}
        is_roberta = True if "roberta" in self.config.architectures[0].lower() else False

        if train:
            label_distribution = torch.FloatTensor(len(label_map)).zero_()
        else:
            label_distribution = None

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(example.text_a)
            tokens_b = self.tokenizer.tokenize(example.text_b)

            if is_roberta:
                truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 4)
            else:
                truncate_seq_pair(tokens_a, tokens_b, self.args.max_seq_length - 3)

            tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
            segment_ids = [0] * len(tokens)

            if is_roberta:
                tokens_b = [self.tokenizer.sep_token] + tokens_b + [self.tokenizer.sep_token]
                segment_ids += [0] * len(tokens_b)
            else:
                tokens_b = tokens_b + [self.tokenizer.sep_token]
                segment_ids += [1] * len(tokens_b)
            tokens += tokens_b

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (self.args.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == self.args.max_seq_length
            assert len(input_mask) == self.args.max_seq_length
            assert len(segment_ids) == self.args.max_seq_length

            if example.label is None:
                label_id = -1
            else:
                label_id = label_map[example.label]

            if train:
                label_distribution[label_id] += 1.0

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))

        if train:
            label_distribution = label_distribution / label_distribution.sum()
            return features, label_distribution
        else:
            return features