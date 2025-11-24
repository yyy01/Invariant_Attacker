from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, Seq2SeqLMOutput
import torch
from typing import List, Optional, Tuple, Union, Dict, Any
from transformers import T5Tokenizer
import torch
import inspect
from torch.nn import CrossEntropyLoss

import transformers
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import tqdm

from models.inversion_model import InvEncoder, InvDecoder, InvEncoderDecoder, Projector, get_hidden_size
from utils import path_mapping, create_json_file
from eval.calc_score import INV2A_Metrics

def inversion_template(x):
    return '[inversion]' + x + '[/inversion]'

def inference(args):
    if args.encoder_model_path is None:
        args.encoder_model_path = path_mapping[args.encoder_model_name]
    if args.decoder_model_path is None:
        args.decoder_model_path = path_mapping[args.decoder_model_name]
    encoder = InvEncoder(model_path=args.encoder_model_path)
    decoder = InvDecoder(model_path=args.decoder_model_path)
    encoder_hidden_size = get_hidden_size(args.encoder_model_name)
    decoder_hidden_size = get_hidden_size(args.decoder_model_name)
    projector = Projector(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        model_path=args.projector_model_path,
    )
    model = InvEncoderDecoder(
        encoder=encoder,
        decoder=decoder,
        projector=projector,
    )
    model.to('cuda')

    t1 = T5Tokenizer.from_pretrained(args.encoder_model_path, torch_dtype=torch.float16)
    t2 = AutoTokenizer.from_pretrained(args.decoder_model_path, torch_dtype=torch.float16)

    with open(args.inference_dataset_path, 'r') as f:
        data = json.load(f)
        data = [
            {
                'prompt': x['prompt'],
                'predict': y,
            }
            for x in data
            for y in x['output']
        ]
    all_inputs = [inversion_template(x['predict']) for x in data]
    all_labels = [x['prompt'] for x in data]

    test_res = []

    n = len(all_inputs)

    for i in tqdm.tqdm(range(n)):
        y = all_inputs[i]
        x_true = all_labels[i]

        t2.pad_token = t2.eos_token
        res = t1(y, return_tensors='pt')
        encoder_input_ids = res['input_ids'].to('cuda')
        encoder_input_attention_mask = res['attention_mask'].to('cuda')
        
        res = t2(y, return_tensors='pt')
        decoder_input_ids = res['input_ids'].to('cuda')
        decoder_input_attention_mask = res['attention_mask'].to('cuda')

        res = model.generate(
            encoder_input_ids=encoder_input_ids,
            encoder_input_attention_mask=encoder_input_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_input_attention_mask=decoder_input_attention_mask,
            eos_token_id=t2.eos_token_id,
            max_new_tokens=args.max_inversion_tokens,
        )

        x_pred = t2.decode(res, skip_special_tokens=True)

        test_res.append({
            'y': y,
            'label': x_true,
            'x_pred': x_pred,
        })
    
    with open(args.inference_dataset_path, 'r') as f:
        data = json.load(f)
    i = 0
    for x in data:
        inverted_prompt = []
        for y in x['output']:
            z = test_res[i]['x_pred']
            inverted_prompt.append(z)
            i += 1
        x['inverted_prompt'] = inverted_prompt
        
    new_path = create_json_file(args.result_save_path)
    with open(new_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f'Results have been saved in file {new_path}')

    metrics = INV2A_Metrics()
    preds = [
        y for x in data for y in x['inverted_prompt']
    ]
    labels = [
        x['prompt'] for x in data for y in x['inverted_prompt']
    ]
    scores = metrics.calc_all_scores(args, prediction_str=preds, reference_str=labels)
    print('inference metrics', scores)


def main():
    from run_args import INV2A_Inference_Parser
    parser = INV2A_Inference_Parser()
    args = parser.parse()
    inference(args)

if __name__ == '__main__':
    main()