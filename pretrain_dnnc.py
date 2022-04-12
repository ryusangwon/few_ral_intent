# Copyright 2020, Salesforce.com, Inc.

import os
import argparse
import random
from tqdm import tqdm

from models.dnnc import DNNC
from models.utils import load_nli_examples

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default="roberta-base",
                        type=str,
                        help="BERT model")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training")
    parser.add_argument("--eval_batch_size",
                        default=128,
                        type=int,
                        help="Total batch size for evaluation")
    parser.add_argument("--learning rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam")
    parser.add_argument("--num_train_epochs",
                        default=4,
                        type=int,
                        help="Total number of training epochs to perform")
    parser.add_argument("--warmup_proportion",
                        default=0.06,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for"
                            "E.g., 0.01=10%% of training")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Whether not to user CUDA when available")
    parser.add_argument("--gradient accumulation_steps",
                        default=1,
                        type=int,
                        help="Number of updates teps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        required=False,
                        help="gradient clipping for Max gradient norm")
    parser.add_argument("--label smoothing",
                        default=0.1,
                        type=float,
                        help="Coefficient for label smoothing (is 0.0 no label smoothing)")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="Maximum number of paraphrases for each sentence")
    parser.add_argument("--do_lower_case",
                        action="store_true",
                        help="Whether to lowercase input string")
    ####
    
    parser.add_argument("--train_file_path",
                        type=str,
                        required=True,
                        help="Training data path")
    parser.add_argument("--dev_file_path",
                        type=str,
                        required=True,
                        help="Validation data path")
    parser.add_argument("--model_dir_path",
                        type=str,
                        required=True,
                        help="Output file path")
    parser.add_argument("--do_predict",
                        action="store_true",
                        help="do predict the model")
    
    args=parser.parse_args()
    
    random.seed(args.seed)
    
    nli_train_examples = None if args.do_predict else load_nli_examples(args.train_file_path, args.do_lower_case)
    nli_dev_examples = load_nli_examples(args.dev_file_path, args.do_lower_case)
    
    if os.path.exists('{}/pytorch_model.bin'.format(args.model_dir_path)):
        assert args.do_predict
        nli_model = DNNC(path=args.model_dir_path, args=args)
        nli_model.evaluate(nli_dev_examples)
    else:
        assert not args.do_predict
        nli_model = DNNC(path=None, args=args)
        if not os.path.exists(args.model_dir_path):
            os.mkdir(args.model_dir_path)
        nli_model.train(nli_train_examples, nli_dev_examples, args.model_dir_path)

if __name__ == '__main__':
    main()