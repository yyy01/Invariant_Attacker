import torch
import torch
import inspect
import os
from pathlib import Path
import transformers

import json
        
from models.CL_train import run_CL_train
from models.FT_train import run_FT_train
from models.FT_train_system import run_FT_train_system
from utils import path_mapping

def input_template(x):
    return '[inversion]' + x + '[/inversion]'

def train(args):
    if args.encoder_model_path is None:
        args.encoder_model_path = path_mapping[args.encoder_model_name]
    if args.decoder_model_path is None:
        args.decoder_model_path = path_mapping[args.decoder_model_name]

    new_dir = Path(args.model_save_path)
    new_dir.mkdir(exist_ok=True)
    args.encoder_save_path = args.model_save_path + '/InvEncoder'
    args.projector_save_path = args.model_save_path + '/projector.pt'

    transformers.set_seed(args.seed)
    torch.manual_seed(args.seed)  
    torch.cuda.manual_seed_all(args.seed)  # GPU

    # do CL
    if args.do_contrastive_learning:
        print('Contrastive Learning ...')
        run_CL_train(args)
        args.encoder_model_path = args.encoder_save_path

    # do FT
    if not args.task == 'user':
        print('Fine Tuning ...')
        args.projector_model_path = None
        run_FT_train(args)

    # do FT system
    if args.task == 'system':
        print('System Fine Tuning ...')
        args.projector_model_path = None
        run_FT_train_system(args)


def main():
    from run_args import INV2A_Train_Parser
    parser = INV2A_Train_Parser()
    args = parser.parse()
    train(args)

if __name__ == '__main__':
    main()