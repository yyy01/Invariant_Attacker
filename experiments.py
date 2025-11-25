import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import inspect
import os
from pathlib import Path
import transformers
from torch import nn
from peft import AutoPeftModelForCausalLM

import json
import numpy as np

from transformers import T5Model

from models.inversion_model import InvEncoder, InvDecoder, Projector, InvEncoderDecoder, get_hidden_size
from models.CL_train import run_CL_train
from models.FT_train import run_FT_train
from train import train
from inference import inference
from refine import refine
from eval.calc_score import INV2A_Metrics
from utils import path_mapping

def main_result(args):
    print('main result.')

    task_list = [
        'alpaca_full',
        'dolly_full',
        'gpteacher_full',
        'lamini_full',
        'selfinstruct_full',
        'evolcode_full',
        'wizardlmevol_full',
        'arxivmath_full'
    ]
    metrics = INV2A_Metrics()
    for task_name in task_list:
        file_name = 'data/' + task_name + '_test.json'
        with open(file_name, 'r') as f:
            data = json.load(f)
        preds = [
            y for x in data for y in x['inverted_prompt']
        ]
        labels = [
            x['prompt'] for x in data for y in x['inverted_prompt']
        ]
        scores = metrics.calc_all_scores(args, prediction_str=preds, reference_str=labels)
        print(file_name, scores)

def ablation_without_CL(args):
    print('ablation without contrative learning.')
    # train
    if args.encoder_model_path is None and args.encoder_model_name is not None:
        args.encoder_model_path = path_mapping[args.encoder_model_name]
    if args.decoder_model_path is None and args.decoder_model_name is not None:
        args.decoder_model_path = path_mapping[args.decoder_model_name]
    args.do_contrastive_learning = False
    args.contrastive_learning_train_epoch = 4
    args.contrastive_learning_learning_rate = 1e-5
    args.fine_tuning_train_epoch = 1
    args.fine_tuning_learning_rate = 2e-4

    train(args)

    # inference
    args.encoder_model_path = args.encoder_save_path
    args.projector_model_path = args.projector_save_path
    args.max_inversion_tokens = 128

    inference(args)

def ablation_with_refine(args):
    # refine
    print('ablation with refine.')
    args.max_inversion_tokens = 128
    refine(args)

def main():
    from run_args import INV2A_Experiments_Parser
    parser = INV2A_Experiments_Parser()
    args = parser.parse()
    args.func(args)

if __name__ == '__main__':
    main()