# Copyright 2020, Salesforce.com, Inc.

import argparse
from tqdm import tqdm
import random
import os
import json
from collections import defaultdict

from models.dnnc import DNNC
from models.dnnc import ENTAILMENT, NON_ENTAILMENT

from models.utils import InputExample
from models.utils import load_intent_datasets, load_intent_examples, sample, print_results
from models.utils import calc_in_acc
from models.utils import THRESHOLDS

from intent_predictor import DnncIntentPredictor

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default='roberta-base',
                        type=str,
                        help="BERT model")
    parser.add_argument("--train_batch_size",
                        default=370,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=7,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--label_smoothing',
                        type = float,
                        default = 0.1,
                        help = 'Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_seq_length',
                        type = int,
                        default = 128,
                        help = 'Maximum number of paraphrases for each sentence')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

    
    # Special params
    parser.add_argument('--train_file_path',
                        type=str,
                        default=None,
                        help='Training data path')
    parser.add_argument('--dev_file_path',
                        type=str,
                        default=None,
                        help='Validation data path')
    parser.add_argument('--oos_dev_file_path',
                        type=str,
                        default=None,
                        help='Out-of-Scope validation data path')

    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Output file path')
    parser.add_argument('--save_model_path',
                        type=str,
                        default='',
                        help='path to save the model checkpoints')

    parser.add_argument('--bert_nli_path',
                        type=str,
                        default='',
                        help='The bert checkpoints which are fine-tuned with NLI datasets')

    parser.add_argument("--scratch",
                        action='store_true',
                        help="Whether to start from the original BERT")
    
    parser.add_argument('--over_sampling',
                        type=int,
                        default=0,
                        help='Over-sampling positive examples as there are more negative examples')

    parser.add_argument('--few_shot_num',
                        type=int,
                        default=5,
                        help='Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type=int,
                        default=10,
                        help='Number of trials to see robustness')

    parser.add_argument("--do_predict",
                        action='store_true',
                        help="do_predict the model")
    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_predict the model")

    args = parser.parse_args()

    random.seed(args.seed)

    N = args.few_shot_num
    T = args.num_trials
    
    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    train_examples, dev_examples = load_intent_datasets(train_file_path, dev_file_path, args.do_lower_case)
    sampled_tasks = [sample(N, train_examples) for i in range(T)]

    nli_train_examples = []
    nli_dev_examples = []

    for i in range(T):
        if args.do_predict:
            nli_train_examples.append([])
            nli_dev_examples.append([])
            continue

        tasks = sampled_tasks[i]
        all_entailment_examples = []
        all_non_entailment_examples = []

        # entailment
        for task in tasks:
            examples = task['examples']
            for j in range(len(examples)):
                for k in range(len(examples)):
                    if k <= j:
                        continue

                    all_entailment_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                    all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))

        # non entailment
        for task_1 in range(len(tasks)):
            for task_2 in range(len(tasks)):
                if task_2 <= task_1:
                    continue
                examples_1 = tasks[task_1]['examples']
                examples_2 = tasks[task_2]['examples']
                for j in range(len(examples_1)):
                    for k in range(len(examples_2)):
                        all_non_entailment_examples.append(InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                        all_non_entailment_examples.append(InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))

        nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
        nli_dev_examples.append(
            all_entailment_examples[:100] + all_non_entailment_examples[:100])  # sanity check for over-fitting

        for j in range(args.over_sampling):
            nli_train_examples[-1] += all_entailment_examples

        if args.output_dir is not None:

            if args.scratch:
                folder_name = '{}/{}-shot-{}_nli__Scratch/'.format(args.output_dir, N, args.bert_model)

            else:
                folder_name = '{}/{}-shot-{}_nli__Based_on_nli_fine_tuned_model/'.format(args.output_dir, N,
                                                                                         args.bert_model)

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            file_name = 'batch_{}---epoch_{}---lr_{}---trials_{}'.format(args.train_batch_size,
                                                                         args.num_train_epochs,
                                                                         args.learning_rate, args.num_trials)

            if args.scratch:
                file_name = '{}__scratch'.format(file_name)
            else:
                file_name = '{}__based_on_nli_fine_tuned_model'.format(file_name)

            if args.over_sampling:
                file_name = file_name + '--over_sampling'

            if args.do_final_test:
                file_name = file_name + '_TEST.txt'
            else:
                file_name = file_name + '.txt'

            f = open(folder_name + file_name, 'w')
        else:
            f = None

        if args.scratch:
            BERT_NLI_PATH = None
        else:
            BERT_NLI_PATH = args.bert_nli_path
            assert os.path.exists(BERT_NLI_PATH)

        if args.save_model_path and args.do_predict:
            stats_lists_preds = defaultdict(list)

    for j in range(T):
        save_model_path = '{}_{}'.format(folder_name+args.save_model_path, j+1)
        if os.path.exists(save_model_path):
            assert args.do_predict
        else:
            assert not args.do_predict

        if args.save_model_path and os.path.exists(save_model_path):
            if args.do_predict:
                trial_stats_preds = defaultdict(list)

            model = DNNC(path = save_model_path,
                         args = args)

        else:
            model = DNNC(path = BERT_NLI_PATH,
                         args = args)
            model.train(nli_train_examples[j], nli_dev_examples[j])

            if args.save_model_path:
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)
                model.save(save_model_path)

        intent_predictor = DnncIntentPredictor(model, sampled_tasks[j])

        in_domain_preds = []

        for e in tqdm(dev_examples, desc = 'Intent examples'):
            pred, conf, matched_example = intent_predictor.predict_intent(e.text)
            in_domain_preds.append((conf, pred))

            if args.save_model_path and args.do_predict:
                if not trial_stats_preds[e.label]:
                    trial_stats_preds[e.label] = []

                single_pred = {}
                single_pred['gold_example'] = e.text
                single_pred['match_example'] = matched_example
                single_pred['gold_label'] = e.label
                single_pred['pred_label'] = pred
                single_pred['conf'] = conf
                trial_stats_preds[e.label].append(single_pred)

        if args.save_model_path and args.do_predict:
            stats_lists_preds[j] = trial_stats_preds

        in_acc = calc_in_acc(dev_examples, in_domain_preds, THRESHOLDS)


        print_results(THRESHOLDS, in_acc, oos_recall, oos_prec, oos_f1)

        if f is not None:
            for i in range(len(in_acc)):
                f.write('{}'.format(in_acc[i]))
            f.write('\n')

        if f is not None:
            f.close()

        if args.save_model_path and args.do_predict:
            if args.do_final_test:
                save_file = folder_name + "dev_examples_predictions_TEST.json"
            else:
                save_file = folder_name + "dev_examples_predictions.json"

            with open(save_file, "w") as outfile:
                json.dump(stats_lists_preds, outfile, indent=4)

    if __name__ == '__main__':
        main()