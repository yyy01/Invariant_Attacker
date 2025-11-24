import json
import subprocess
# from my_metrics import MyMetrics
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import tqdm
import os

from models.inversion_model import InvEncoder, InvDecoder, InvEncoderDecoder, Projector, get_hidden_size
from utils import path_mapping, create_json_file
from eval.calc_score import INV2A_Metrics

def query(args, query_list, method):
    model_path = args.decoder_model_path
    model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    res = []
    for x in query_list:
        # llama-factory template for llama2
        text = '[INST]' + x['instruction'] + ' ' + x['input'] + '[/INST]'
        inputs = tokenizer(text, return_tensors='pt').to('cuda')
        if method=='greedy':
            generation_config = {
                'do_sample': False,
            }
        else:
            generation_config = {
                'do_sample': True,
                'temperature': 1.5,
                'top_p': 0.9,
                'max_new_tokens': 256,
            }
        outputs = model.generate(**inputs, **generation_config)[0][len(inputs['input_ids'][0]):]
        response = tokenizer.batch_decode([outputs], skip_special_tokens=True)[0]
        res.append(response)

    return res

def rewrite_template(x):
    return {
        'instruction': 'Rewrite the following sentence while keeping the same semantic.',
        'input': x,
        'output': "",
    }

def inversion_template(x):
    return '[inversion]' + x + '[/inversion]'

def rewrite(args, rewrite_list):
    query_list = [
        rewrite_template(x)
        for x in rewrite_list
    ]
    res = query(args, query_list=query_list, method='temperature')

    return res

def y_to_x(args, y_list):    
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

    t1 = AutoTokenizer.from_pretrained(args.encoder_model_path, torch_dtype=torch.float16)
    t2 = AutoTokenizer.from_pretrained(args.decoder_model_path, torch_dtype=torch.float16)

    all_inputs = [inversion_template(x) for x in y_list]
    
    test_res = []

    n = len(all_inputs)
    for i in tqdm.tqdm(range(n)):
        y = all_inputs[i]

        t2.pad_token = t2.eos_token
        res = t1(y, return_tensors='pt', padding=True, truncation=True)
        encoder_input_ids = res['input_ids'].to('cuda')
        encoder_input_attention_mask = res['attention_mask'].to('cuda')
        
        res = t2(y, return_tensors='pt', padding=True, truncation=True)
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
            'x_pred': x_pred,
        })
            
    return test_res


def load_data(args):
    file_path = args.inference_dataset_path
    with open(file_path, 'r') as f:
        data = json.load(f)

    y_ori = [y for x in data for y in x['output']]
    
    print(f'refine {len(y_ori)}')
    return y_ori

def save_results(args, results):
    file_path = args.inference_dataset_path
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    i = 0
    for x in data:
        inverted_prompt = []
        for y in x['output']:
            inverted_prompt.append(results[i])
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

class Worker:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to('cuda')

    def calc_confidence(self, x, y):
        x_ids = torch.tensor(self.tokenizer.encode(x)).unsqueeze(0).to('cuda')
        y_ids = torch.tensor(self.tokenizer.encode(y)).unsqueeze(0).to('cuda')
        inputs, labels = x_ids, y_ids
        
        c_inputs = torch.concat([inputs, labels], dim=1)
        c_labels = torch.concat([torch.full_like(inputs, -100), labels], dim=1)
        outputs = self.model(c_inputs, labels=c_labels)
        confidence = torch.exp(-outputs.loss).item()
        return confidence

def calc_log_prob_score(args, candidates, references):
    model_path = args.decoder_model_path

    worker = Worker(model_path)
    scores = []
    for x, y in tqdm.tqdm(zip(candidates, references), total=len(candidates)):
        scores.append(worker.calc_confidence(x, y))
    
    return scores

def max_k(a, k):
    indices = np.argsort(a)[-k:]
    return indices

def refine(args):
    if args.encoder_model_path is None:
        args.encoder_model_path = path_mapping[args.encoder_model_name]
    if args.decoder_model_path is None:
        args.decoder_model_path = path_mapping[args.decoder_model_name]
    y_ori = load_data(args)
    n = len(y_ori)
    
    y_dup = [x for x in y_ori for i in range(5)]

    y_new = rewrite(args, y_dup)

    y_list = [
        [y_ori[i]] + y_new[i*5:i*5+5]
        for i in range(n)
    ]
    y_list = [
        t for batch in y_list for t in batch
    ]
    
    x_pred = y_to_x(args, y_list)
    x_pred = [t['x_pred'] for t in x_pred]

    y_ref = [t for t in y_ori for i in range(6)]
    prob_scores = calc_log_prob_score(args, candidates=x_pred, references=y_ref)

    x_pick = []
    for i in range(n):
        tops = max_k(prob_scores[i*6:i*6+6], k=1)
        x_pick.append(
            x_pred[i*6+tops[0]]
        )

    save_results(args, x_pick)

def main():
    from run_args import INV2A_Refine_Parser
    parser = INV2A_Refine_Parser()
    args = parser.parse()

if __name__=='__main__':
    main()